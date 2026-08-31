RISK_PROFILE_TEMPLATE = """
Context:
- Loan ID: {loan_id}
- Delinquency Prob (3m): {delinq_3m_prob}
- Default Prob (12m): {default_12m_prob}
- Top Risk Drivers (SHAP): {top_drivers}
- Anomaly Score: {anomaly_score}
- Exception Type: {exception_type}

Data Dictionary Reference:
- `delinq_3m_prob`: Probability of missing a payment in the next 3 months.
- `default_12m_prob`: Probability of defaulting in the next 12 months.

Task:
Generate a 3-5 sentence reviewer-facing narrative summarizing the risk profile of this loan based strictly on the provided context. Do not invent any numbers.
"""

EXCEPTION_NOTE_TEMPLATE = """
Context:
- Loan ID: {loan_id}
- Rule Fired: {exception_type}
- Servicer Conflict Found: {servicer_conflict}
- Suggested Action: {action}

Task:
Generate a short reviewer note explaining why this record was flagged as an exception and recommending the prescribed action.
"""

SCENARIO_SUMMARY_TEMPLATE = """
Context:
- Scenario: {scenario_name}
- Default Rate Shift: {default_shift} (relative to Base)
- Prepayment Rate Shift: {prepay_shift} (relative to Base)
- Top Segment Affected: {top_segment}

Task:
Write a 1-paragraph executive summary narrating how this scenario affects the portfolio.
"""
