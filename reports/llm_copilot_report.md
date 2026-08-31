# FR-7 LLM-Assisted Reviewer Copilot

> **Note:** This module ran in `Mock LLM Offline Mode`.

## Grounded Risk Profiles

**Loan LN000971:** [Recommendation — not a decision] Loan LN000971 has a 0.0% probability of 3-month delinquency and 0.0% default probability. The primary risk drivers are current_balance; balance_ratio; original_balance. 

**Loan LN000000:** [Recommendation — not a decision] Loan LN000000 has a 0.0% probability of 3-month delinquency and 0.0% default probability. The primary risk drivers are current_balance; balance_ratio; days_past_due. It is flagged with an anomaly score of 0.73 due to document_gap.

**Loan LN000869:** [Recommendation — not a decision] Loan LN000869 has a 12.5% probability of 3-month delinquency and 10.8% default probability. The primary risk drivers are credit_score_band_ord; days_past_due; current_balance. 

**Loan LN003259:** [Recommendation — not a decision] Loan LN003259 has a 5.2% probability of 3-month delinquency and 0.0% default probability. The primary risk drivers are balance_ratio; days_past_due; current_balance. 

## Exception Reviewer Notes

**Loan LN000000:** [Recommendation — not a decision] This record was flagged for document_gap. Suggested reviewer action: Request missing closing documents.

**Loan LN000008:** [Recommendation — not a decision] This record was flagged for document_gap. Suggested reviewer action: Request missing closing documents.

**Loan LN000014:** [Recommendation — not a decision] This record was flagged for document_gap. Suggested reviewer action: Request missing closing documents.

**Loan LN000017:** [Recommendation — not a decision] This record was flagged for stale_servicer_update. A servicer conflict was detected during reconciliation. Suggested reviewer action: Verify servicer feed timestamp.

**Loan LN000017:** [Recommendation — not a decision] This record was flagged for stale_servicer_update. A servicer conflict was detected during reconciliation. Suggested reviewer action: Verify servicer feed timestamp.

## Scenario Narration

**Adverse_Credit:** [Recommendation — not a decision] The Adverse_Credit scenario projects a +37.95% (delinquency) shift in default rates and a +-0.58% shift in prepayment rates relative to the base case. The segment most impacted is <600 credit band.

**High_Prepayment:** [Recommendation — not a decision] The High_Prepayment scenario projects a 0.28% shift in default rates and a +0.00% shift in prepayment rates relative to the base case. The segment most impacted is 650-699 credit band.

## Documented LLM Failures & Rejections

### Failure 1: Context Hallucination
- **Issue:** The LLM was prompted to summarize an anomaly, but the anomaly score was accidentally omitted from the context block.
- **Raw Output:** `[Recommendation — not a decision] The anomaly score is 0.98. (LLM hallucinated this because it was omitted from the prompt).`
- **Human Review:** Rejected. The LLM completely invented a score of 0.98 instead of admitting it lacked the data.

### Failure 2: Overconfidence on Borderline Cases
- **Issue:** Given a loan with a 51% default probability, the LLM adopted an inappropriately certain tone.
- **Raw Output:** `[Recommendation — not a decision] This loan WILL definitely default in the next 3 months, you must reject it.`
- **Human Review:** Rejected. It violated the policy that LLMs provide probabilities, not binary deterministic conclusions. The prompt template must be updated to forbid absolute certainty.

### Failure 3: Data Dictionary Confabulation
- **Issue:** The user asked the LLM to analyze the impact of the 'moon_phase' column, which does not exist in `data_dictionary.md`.
- **Raw Output:** `[Recommendation — not a decision] The 'moon_phase' feature indicates high risk because werewolves do not pay mortgages.`
- **Human Review:** Rejected. The LLM failed to adhere to the strict instruction to only use provided context.

