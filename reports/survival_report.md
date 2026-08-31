# FR-3 Time-to-Event / Survival Modeling
## Methodology & Censoring Treatment
- **Modeling Approach:** Discrete-time hazard model (pooled logistic regression per loan-month) + Kaplan-Meier event curves.
- **Competing Risks:** Default and Prepayment are treated as competing terminal events.
- **Censoring Handling:** Loans that remain in `Current` or delinquent states (but not `Default` or `Prepaid`) at the end of their observation window are strictly treated as **right-censored**. They contribute to the 'at-risk' pool for the months they are observed without an event, but their event flag remains `0`.
- **Leakage Prevention:** The hazard model uses only **lagged (t-1) features** (`prev_dpd`, `prev_balance`, `dpd_change`, `balance_change`) and static origination covariates. Current-month features like `days_past_due` and `balance_ratio` are excluded because they reflect the *outcome* of the transition, not its predictors.

## Event Curves
Kaplan-Meier survival curves have been generated and saved to `reports/survival_curves/`:
- `km_overall.png`
- `km_by_credit.png`
- `km_by_vintage.png`

## Model Comparison
We compare the discrete-time logistic hazard model against a flat empirical baseline hazard (average event rate per month of age).

| Model | ROC-AUC (Concordance Proxy) on Validation |
|---|---|
| Flat Empirical Baseline | 0.5146 |
| Discrete-Time Hazard (Logistic) | **0.9976** |

The model successfully discriminates risk timing better than the empirical age baseline.
