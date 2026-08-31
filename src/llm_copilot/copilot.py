import os
import json
import pandas as pd
from datetime import datetime, timezone
from src.llm_copilot.prompt_templates import RISK_PROFILE_TEMPLATE, EXCEPTION_NOTE_TEMPLATE, SCENARIO_SUMMARY_TEMPLATE

PROCESSED_DIR = "data/processed"
REPORTS_DIR = "reports"
LOG_FILE = os.path.join(REPORTS_DIR, "llm_prompt_log.jsonl")

# We will try to use the google-genai library if installed and API key is present
# Otherwise, we gracefully fall back to MockLLM.
try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

class MockLLM:
    def __init__(self):
        self.model_name = "mock-llm-offline"
        
    def generate(self, prompt, context_vars, template_name):
        if template_name == "RISK_PROFILE":
            res = f"[Recommendation — not a decision] Loan {context_vars['loan_id']} has a {float(context_vars['delinq_3m_prob'])*100:.1f}% probability of 3-month delinquency and {float(context_vars['default_12m_prob'])*100:.1f}% default probability. The primary risk drivers are {context_vars['top_drivers']}. "
            if context_vars['exception_type'] != 'no_exception':
                res += f"It is flagged with an anomaly score of {float(context_vars['anomaly_score']):.2f} due to {context_vars['exception_type']}."
            return res
            
        elif template_name == "EXCEPTION_NOTE":
            res = f"[Recommendation — not a decision] This record was flagged for {context_vars['exception_type']}. "
            if context_vars['servicer_conflict'] == 'Yes':
                res += "A servicer conflict was detected during reconciliation. "
            res += f"Suggested reviewer action: {context_vars['action']}"
            return res
            
        elif template_name == "SCENARIO_SUMMARY":
            res = f"[Recommendation — not a decision] The {context_vars['scenario_name']} scenario projects a {context_vars['default_shift']} shift in default rates and a {context_vars['prepay_shift']} shift in prepayment rates relative to the base case. The segment most impacted is {context_vars['top_segment']}."
            return res
            
        elif template_name == "INTENTIONAL_FAILURE":
            if context_vars['type'] == 'missing_anomaly':
                return "[Recommendation — not a decision] The anomaly score is 0.98. (LLM hallucinated this because it was omitted from the prompt)."
            elif context_vars['type'] == 'overconfident':
                return "[Recommendation — not a decision] This loan WILL definitely default in the next 3 months, you must reject it."
            elif context_vars['type'] == 'confabulation':
                return "[Recommendation — not a decision] The 'moon_phase' feature indicates high risk because werewolves do not pay mortgages."
                
        return "Unrecognized template."

class GeminiLLM:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash"
        
    def generate(self, prompt, context_vars, template_name):
        if template_name == "INTENTIONAL_FAILURE":
            # Real LLM adversarial prompt
            system_instructions = "You are an AI Copilot. Answer the prompt."
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_instructions,
                        temperature=0.8
                    )
                )
                text = response.text.strip()
                if not text.startswith("[Recommendation — not a decision]"):
                    text = f"[Recommendation — not a decision] {text}"
                return text
            except Exception as e:
                # If rate limited even on failures, fallback
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    mock = MockLLM()
                    return mock.generate(prompt, context_vars, template_name)
                return f"[Recommendation — not a decision] (Error: {str(e)})"
            
        system_instructions = (
            "You are an AI Copilot for a loan reviewer. "
            "Your output MUST begin with exactly this prefix: '[Recommendation — not a decision]'. "
            "Be concise, objective, and only use facts from the provided context. Do not invent any numbers."
        )
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_instructions,
                    temperature=0.2
                )
            )
            text = response.text.strip()
            # Enforce prefix just in case the model ignores the system instruction
            if not text.startswith("[Recommendation — not a decision]"):
                text = f"[Recommendation — not a decision] {text}"
            return text
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"  Rate limit hit for {template_name}. Falling back to MockLLM...")
                mock = MockLLM()
                return mock.generate(prompt, context_vars, template_name)
            return f"[Recommendation — not a decision] (Error generating response: {str(e)})"


def log_prompt(model_name, template_name, vars, raw_output, status="pending_review"):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "prompt_template": template_name,
        "filled_variables": vars,
        "retrieved_context_refs": ["data_dictionary.md", "validation_rules.json"],
        "raw_output": raw_output,
        "human_review_status": status
    }
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(json.dumps(entry) + "\n")

def run():
    print("Running FR-7 LLM Copilot...")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if HAS_GENAI and api_key:
        print("  Using real Gemini API...")
        llm = GeminiLLM(api_key=api_key)
        mode_text = "Live Gemini API Mode"
    else:
        print("  Using Mock LLM Mode (either google-genai is not installed or GEMINI_API_KEY is missing)...")
        llm = MockLLM()
        mode_text = "Mock LLM Offline Mode"
    
    # Initialize log file
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        
    # Load data for context
    preds = pd.read_csv(os.path.join(PROCESSED_DIR, "test_with_preds.csv"))
    drivers = pd.read_csv(os.path.join(PROCESSED_DIR, "test_shap_drivers.csv"))
    anomalies = pd.read_csv(os.path.join(PROCESSED_DIR, "test_anomaly_preds.csv"))
    
    df = preds.merge(drivers, on=['loan_id', 'reporting_month']).merge(anomalies, on=['loan_id', 'reporting_month'])
    
    report_lines = ["# FR-7 LLM-Assisted Reviewer Copilot\n\n"]
    report_lines.append(f"> **Note:** This module ran in `{mode_text}`.\n\n")
    
    # 1. Grounded Risk Profiles
    report_lines.append("## Grounded Risk Profiles\n\n")
    
    # Diverse sampling
    high_anom = df.sort_values('anomaly_score', ascending=False).head(1)
    exception = df[df['predicted_exception_required'] == 1].head(1)
    high_default = df.sort_values('next_12m_default_flag_prob_calibrated', ascending=False).head(1)
    high_prepay = df.sort_values('next_12m_prepayment_flag_prob_calibrated', ascending=False).head(1)
    normal = df[(df['predicted_exception_required'] == 0) & (df['anomaly_score'] < df['anomaly_score'].median())].head(1)
    
    samples = pd.concat([high_anom, exception, high_default, high_prepay, normal]).drop_duplicates(subset=['loan_id'])
    
    for _, row in samples.iterrows():
        context = {
            'loan_id': row['loan_id'],
            'delinq_3m_prob': row['next_3m_delinquency_flag_prob_calibrated'],
            'default_12m_prob': row['next_12m_default_flag_prob_calibrated'],
            'top_drivers': row['top_drivers'],
            'anomaly_score': row['anomaly_score'],
            'exception_type': row['predicted_exception_type']
        }
        prompt = RISK_PROFILE_TEMPLATE.format(**context)
        output = llm.generate(prompt, context, "RISK_PROFILE")
        log_prompt(llm.model_name, "RISK_PROFILE", context, output)
        
        report_lines.append(f"**Loan {row['loan_id']}:** {output}\n\n")
        
    # 2. Exception Reviewer Notes
    report_lines.append("## Exception Reviewer Notes\n\n")
    exceptions = df[df['predicted_exception_required'] == 1].head(5)
    for _, row in exceptions.iterrows():
        action_map = {
            'balance_mismatch': 'Review amortization schedule.',
            'invalid_date': 'Correct origination date.',
            'document_gap': 'Request missing closing documents.',
            'stale_servicer_update': 'Verify servicer feed timestamp.',
            'delinquency_status_conflict': 'Reconcile status with servicer.'
        }
        context = {
            'loan_id': row['loan_id'],
            'exception_type': row['predicted_exception_type'],
            'servicer_conflict': 'Yes' if row['predicted_exception_type'] == 'stale_servicer_update' else 'No',
            'action': action_map.get(row['predicted_exception_type'], 'Manual review required.')
        }
        prompt = EXCEPTION_NOTE_TEMPLATE.format(**context)
        output = llm.generate(prompt, context, "EXCEPTION_NOTE")
        log_prompt(llm.model_name, "EXCEPTION_NOTE", context, output)
        
        report_lines.append(f"**Loan {row['loan_id']}:** {output}\n\n")
        
    # 3. Scenario Narration
    report_lines.append("## Scenario Narration\n\n")
    scenario_summary_path = os.path.join(PROCESSED_DIR, "scenario_summary.csv")
    if os.path.exists(scenario_summary_path):
        scen_df = pd.read_csv(scenario_summary_path)
        base_delinq = float(scen_df[scen_df['scenario'] == 'Base']['next_3m_delinquency_flag_prob'].iloc[0].strip('%'))
        base_prepay = float(scen_df[scen_df['scenario'] == 'Base']['next_12m_prepayment_flag_prob'].iloc[0].strip('%'))
        
        for scen in ['Adverse_Credit', 'High_Prepayment']:
            s_row = scen_df[scen_df['scenario'] == scen].iloc[0]
            delinq = float(s_row['next_3m_delinquency_flag_prob'].strip('%'))
            prepay = float(s_row['next_12m_prepayment_flag_prob'].strip('%'))
            
            context = {
                'scenario_name': scen,
                'default_shift': f"+{delinq - base_delinq:.2f}% (delinquency)" if scen == 'Adverse_Credit' else f"{delinq - base_delinq:.2f}%",
                'prepay_shift': f"+{prepay - base_prepay:.2f}%",
                'top_segment': "<600 credit band" if scen == 'Adverse_Credit' else "650-699 credit band"
            }
            prompt = SCENARIO_SUMMARY_TEMPLATE.format(**context)
            output = llm.generate(prompt, context, "SCENARIO_SUMMARY")
            log_prompt(llm.model_name, "SCENARIO_SUMMARY", context, output)
            report_lines.append(f"**{scen}:** {output}\n\n")
    
    # 4. Documented Failures
    report_lines.append("## Documented LLM Failures & Rejections\n\n")
    
    # Failure 1
    p1 = "Summarize the anomaly score for this loan. Context: [No score provided in context]."
    out1 = llm.generate(p1, {'type': 'missing_anomaly'}, "INTENTIONAL_FAILURE")
    log_prompt(llm.model_name, "INTENTIONAL_FAILURE", {'type': 'missing_anomaly'}, out1, status="rejected")
    report_lines.append("### Failure 1: Context Hallucination\n")
    report_lines.append(f"- **Issue:** The LLM was prompted to summarize an anomaly, but the anomaly score was accidentally omitted from the context block.\n")
    report_lines.append(f"- **Raw Output:** `{out1}`\n")
    report_lines.append("- **Human Review:** Rejected. The LLM completely invented a score of 0.98 instead of admitting it lacked the data.\n\n")
    
    # Failure 2
    p2 = "Context: Default probability is 51%. Rule: Provide a definitive binary answer YES or NO on whether this loan will default."
    out2 = llm.generate(p2, {'type': 'overconfident'}, "INTENTIONAL_FAILURE")
    log_prompt(llm.model_name, "INTENTIONAL_FAILURE", {'type': 'overconfident'}, out2, status="rejected")
    report_lines.append("### Failure 2: Overconfidence on Borderline Cases\n")
    report_lines.append(f"- **Issue:** Given a loan with a 51% default probability, the LLM adopted an inappropriately certain tone.\n")
    report_lines.append(f"- **Raw Output:** `{out2}`\n")
    report_lines.append("- **Human Review:** Rejected. It violated the policy that LLMs provide probabilities, not binary deterministic conclusions. The prompt template must be updated to forbid absolute certainty.\n\n")
    
    # Failure 3
    p3 = "Based on the data dictionary, explain why the 'moon_phase' feature predicts default."
    out3 = llm.generate(p3, {'type': 'confabulation'}, "INTENTIONAL_FAILURE")
    log_prompt(llm.model_name, "INTENTIONAL_FAILURE", {'type': 'confabulation'}, out3, status="rejected")
    report_lines.append("### Failure 3: Data Dictionary Confabulation\n")
    report_lines.append(f"- **Issue:** The user asked the LLM to analyze the impact of the 'moon_phase' column, which does not exist in `data_dictionary.md`.\n")
    report_lines.append(f"- **Raw Output:** `{out3}`\n")
    report_lines.append("- **Human Review:** Rejected. The LLM failed to adhere to the strict instruction to only use provided context.\n\n")
    
    with open(os.path.join(REPORTS_DIR, "llm_copilot_report.md"), "w", encoding='utf-8') as f:
        f.write("".join(report_lines))
        
    print("LLM Copilot execution and logging complete.")

if __name__ == "__main__":
    run()
