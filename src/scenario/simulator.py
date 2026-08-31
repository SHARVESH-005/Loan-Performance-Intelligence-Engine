import os
import pandas as pd
import numpy as np
import joblib

DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "data/processed/models"
REPORTS_DIR = "reports"

def load_models():
    models = {
        'next_3m_delinquency_flag': joblib.load(os.path.join(MODELS_DIR, "next_3m_delinquency_flag_calibrated.joblib")),
        'next_6m_delinquency_flag': joblib.load(os.path.join(MODELS_DIR, "next_6m_delinquency_flag_calibrated.joblib")),
        'next_12m_default_flag': joblib.load(os.path.join(MODELS_DIR, "next_12m_default_flag_calibrated.joblib")),
        'next_12m_prepayment_flag': joblib.load(os.path.join(MODELS_DIR, "next_12m_prepayment_flag_calibrated.joblib")),
    }
    with open(os.path.join(PROCESSED_DIR, "features.csv"), "r") as f:
        features = [line.strip() for line in f]
    return models, features

def perturb_features(df, scenario):
    df_sim = df.copy()
    
    # unemployment_rate -> affects DPD
    if scenario['unemployment_rate'] > 5.0:
        # High unemployment pushes DPD up
        bump = (scenario['unemployment_rate'] - 5.0) * 5
        df_sim['days_past_due'] = df_sim['days_past_due'] + bump
        if 'prev_dpd' in df_sim.columns:
            df_sim['prev_dpd'] = df_sim['prev_dpd'] + bump
            
    elif scenario['unemployment_rate'] < 5.0:
        bump = (5.0 - scenario['unemployment_rate']) * 2
        df_sim['days_past_due'] = np.maximum(0, df_sim['days_past_due'] - bump)
        if 'prev_dpd' in df_sim.columns:
            df_sim['prev_dpd'] = np.maximum(0, df_sim['prev_dpd'] - bump)
            
    # hpi_change -> affects balance and LTV mapping
    if scenario['hpi_change'] != 0:
        multiplier = 1.0 + (scenario['hpi_change'] / 100.0)
        df_sim['current_balance'] = df_sim['current_balance'] / multiplier
        if 'prev_balance' in df_sim.columns:
            df_sim['prev_balance'] = df_sim['prev_balance'] / multiplier
            
        # LTV shift
        if scenario['hpi_change'] < 0:
            df_sim['ltv_band_ord'] = np.minimum(5, df_sim['ltv_band_ord'] + 1)
        elif scenario['hpi_change'] > 0:
            df_sim['ltv_band_ord'] = np.maximum(0, df_sim['ltv_band_ord'] - 1)
            
    # interest_rate_change -> affects interest_rate
    if scenario['interest_rate_change'] != 0:
        df_sim['interest_rate'] = df_sim['interest_rate'] + scenario['interest_rate_change']
        df_sim['rate_spread'] = df_sim['rate_spread'] + scenario['interest_rate_change']
        
    return df_sim

def run():
    print("Running FR-5 Scenario Simulation...")
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    scenarios_df = pd.read_csv(os.path.join(DATA_DIR, "macro_scenarios.csv"))
    
    models, features = load_models()
    
    results = []
    
    for _, row in scenarios_df.iterrows():
        scenario_name = row['scenario_name']
        print(f"  Simulating {scenario_name}...")
        
        sim_df = perturb_features(test_df, row)
        X_sim = sim_df[features].copy()
        
        # Cast categorical columns to category dtype for LightGBM
        for c in X_sim.select_dtypes(include=['object', 'string']).columns:
            X_sim[c] = X_sim[c].astype('category')
            
        # Keep original identifying info
        res = sim_df[['loan_id', 'credit_score_band_ord']].copy()
        res['scenario'] = scenario_name
        
        for target, model in models.items():
            preds = model.predict_proba(X_sim)[:, 1]
            res[target + '_prob'] = preds
            
        results.append(res)
        
    all_res = pd.concat(results, ignore_index=True)
    
    # Generate report
    report_lines = ["# FR-5 Scenario & Stress Simulation\n\n"]
    report_lines.append("## Macro Scenarios Applied\n\n")
    report_lines.append(scenarios_df.to_markdown(index=False) + "\n\n")
    
    report_lines.append("## Projected Portfolio Rates\n\n")
    
    # Aggregate by scenario
    agg_cols = ['next_3m_delinquency_flag_prob', 'next_12m_default_flag_prob', 'next_12m_prepayment_flag_prob']
    overall = all_res.groupby('scenario')[agg_cols].mean().reset_index()
    
    # Format as percentages
    for c in agg_cols:
        overall[c] = (overall[c] * 100).round(2).astype(str) + "%"
        
    overall.to_csv(os.path.join(PROCESSED_DIR, "scenario_summary.csv"), index=False)
        
    report_lines.append(overall.to_markdown(index=False) + "\n\n")
    
    report_lines.append("## Segment Breakdown (by Credit Score Band)\n\n")
    # 0="<600", 1="600-649", 2="650-699", 3="700-749", 4="750-799", 5="800+"
    band_map = {0: "<600", 1: "600-649", 2: "650-699", 3: "700-749", 4: "750-799", 5: "800+"}
    all_res['credit_score'] = all_res['credit_score_band_ord'].map(band_map)
    
    segment = all_res.groupby(['scenario', 'credit_score'])[agg_cols].mean().reset_index()
    for c in agg_cols:
        segment[c] = (segment[c] * 100).round(2).astype(str) + "%"
        
    report_lines.append(segment.to_markdown(index=False) + "\n\n")
    
    # Driver Explanation
    report_lines.append("## Scenario Drivers\n\n")
    report_lines.append("What drives the changes under each stress scenario?\n\n")
    
    # Calculate data-driven drivers by finding which segment had the largest absolute shift vs Base
    base_seg = segment[segment['scenario'] == 'Base'].set_index('credit_score')
    
    for scenario in ['Adverse_Credit', 'High_Prepayment']:
        scen_seg = segment[segment['scenario'] == scenario].set_index('credit_score')
        
        # Strip '%' and convert to float for calculation
        base_delinq = base_seg['next_3m_delinquency_flag_prob'].str.rstrip('%').astype(float)
        scen_delinq = scen_seg['next_3m_delinquency_flag_prob'].str.rstrip('%').astype(float)
        delinq_shift = (scen_delinq - base_delinq)
        
        base_prepay = base_seg['next_12m_prepayment_flag_prob'].str.rstrip('%').astype(float)
        scen_prepay = scen_seg['next_12m_prepayment_flag_prob'].str.rstrip('%').astype(float)
        prepay_shift = (scen_prepay - base_prepay)
        
        max_delinq_seg = delinq_shift.idxmax()
        max_delinq_val = delinq_shift.max()
        
        max_prepay_seg = prepay_shift.idxmax()
        max_prepay_val = prepay_shift.max()
        
        if scenario == 'Adverse_Credit':
            report_lines.append(f"- **{scenario}:** The most impacted segment for delinquency risk is `{max_delinq_seg}` (shifted by +{max_delinq_val:.2f}%). The scenario perturbation (higher unemployment, lower HPI) primarily affected features `days_past_due` and `ltv_band_ord`.\n")
        elif scenario == 'High_Prepayment':
            report_lines.append(f"- **{scenario}:** The most impacted segment for prepayment risk is `{max_prepay_seg}` (shifted by +{max_prepay_val:.2f}%). The scenario perturbation (lower interest rates, higher HPI) primarily affected `interest_rate` and `rate_spread`.\n")
    
    report_lines.append("\n> **Note:** The `High_Prepayment` scenario may show a muted prepayment response if the underlying synthetic dataset lacked a strong historical relationship between interest rate spreads and prepayment events during training. This is a known data limitation.\n\n")
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "scenario_report.md"), "w") as f:
        f.write("".join(report_lines))
        
    print("Scenario simulation complete.")

if __name__ == "__main__":
    run()
