# FR-4 Anomaly & Exception Detection

## Methodology

This module uses a **hybrid rule + ML** approach:

1. **Deterministic Rules** — 5 validation rules derived from `validation_rules.json` and servicer-conflict reconciliation.
2. **Isolation Forest** — Unsupervised anomaly scoring on numeric features (`current_balance`, `original_balance`, `interest_rate`, `days_past_due`, `balance_ratio`, `loan_age_months`).
3. **LightGBM Classifiers** — Supervised models for `exception_required` (binary) and `exception_type` (multi-class), trained on labeled exception data.

### Exception Flagging Logic

A record is flagged as `exception_required=1` if **either** a deterministic rule fires **or** the LightGBM probability exceeds the F1-optimized threshold.

- **Total records flagged:** 1857
- **ML threshold:** 0.50

### Rule Violation Summary (Test Set)

| Rule | Records Flagged |
|---|---|
| balance_mismatch | 0 |
| invalid_date | 0 |
| document_gap | 1161 |
| delinquency_status_conflict | 0 |
| stale_servicer_update | 736 |

### Servicer Conflict Reconciliation

Total conflicts found in `servicer_updates.csv`: **736**

Precedence: latest `last_updated_at` wins. Full reconciliation log saved to `reports/servicer_reconciliation_log.csv`.

### Label-Quality Note

The labeled `exception_type` values in this dataset are **synthetically generated** and do not correlate with the deterministic rule expressions.
For example, records labeled `balance_mismatch` have `current_balance / original_balance` ratios averaging 0.92 (well under the 1.05 violation threshold).
As a result, the supervised LightGBM F1 is low against the labels. The rule-based detections are correct and trustworthy.

## Curated Exception Examples (Top 20+)

| Loan ID   | Month      |   Anomaly Score | Exception Type        | Driver Explanation                                                | Suggested Action                                  |
|:----------|:-----------|----------------:|:----------------------|:------------------------------------------------------------------|:--------------------------------------------------|
| LN000000  | 2025-03-01 |           0.733 | document_gap          | Violated rule: document_gap                                       | Request missing closing documents from custodian. |
| LN000008  | 2025-03-01 |           0.232 | document_gap          | Violated rule: document_gap                                       | Request missing closing documents from custodian. |
| LN000014  | 2024-11-01 |           0.123 | document_gap          | Violated rule: document_gap                                       | Request missing closing documents from custodian. |
| LN000017  | 2025-03-01 |           0.395 | stale_servicer_update | Violated rule: stale_servicer_update                              | Verify servicer feed timestamp and apply latest.  |
| LN000017  | 2025-04-01 |           0.414 | stale_servicer_update | Violated rule: stale_servicer_update                              | Verify servicer feed timestamp and apply latest.  |
| LN000019  | 2025-01-01 |           0.412 | document_gap          | Violated rule: document_gap                                       | Request missing closing documents from custodian. |
| LN000019  | 2025-03-01 |           0.509 | document_gap          | Violated rule: document_gap                                       | Request missing closing documents from custodian. |
| LN000023  | 2025-02-01 |           0.056 | document_gap          | Violated rule: document_gap                                       | Request missing closing documents from custodian. |
| LN000029  | 2025-03-01 |           0.682 | stale_servicer_update | Violated rule: stale_servicer_update                              | Verify servicer feed timestamp and apply latest.  |
| LN000031  | 2024-12-01 |           0.654 | document_gap          | Violated rule: document_gap                                       | Request missing closing documents from custodian. |
| LN000035  | 2025-01-01 |           0.377 | stale_servicer_update | Violated rule: stale_servicer_update                              | Verify servicer feed timestamp and apply latest.  |
| LN000035  | 2025-03-01 |           0.62  | stale_servicer_update | Violated rule: stale_servicer_update                              | Verify servicer feed timestamp and apply latest.  |
| LN000038  | 2024-12-01 |           0.668 | document_gap          | Violated rule: document_gap                                       | Request missing closing documents from custodian. |
| LN000041  | 2025-06-01 |           0.13  | document_gap          | Violated rule: document_gap                                       | Request missing closing documents from custodian. |
| LN000044  | 2025-02-01 |           0.234 | document_gap          | Violated rule: document_gap                                       | Request missing closing documents from custodian. |
| LN000052  | 2024-12-01 |           0.322 | document_gap          | Violated rule: document_gap                                       | Request missing closing documents from custodian. |
| LN000052  | 2025-05-01 |           0.328 | stale_servicer_update | Violated rule: stale_servicer_update                              | Verify servicer feed timestamp and apply latest.  |
| LN000052  | 2025-06-01 |           0.307 | stale_servicer_update | Violated rule: stale_servicer_update                              | Verify servicer feed timestamp and apply latest.  |
| LN000054  | 2025-05-01 |           0.099 | document_gap          | Violated rule: document_gap                                       | Request missing closing documents from custodian. |
| LN000056  | 2024-10-01 |           0.76  | document_gap          | Violated rule: document_gap                                       | Request missing closing documents from custodian. |
| LN000057  | 2024-11-01 |           0.298 | document_gap          | Violated rule: document_gap                                       | Request missing closing documents from custodian. |
| LN000059  | 2024-12-01 |           0.43  | document_gap          | Violated rule: document_gap                                       | Request missing closing documents from custodian. |
| LN000059  | 2025-02-01 |           0.501 | document_gap          | Violated rule: document_gap                                       | Request missing closing documents from custodian. |
| LN000059  | 2025-06-01 |           0.605 | stale_servicer_update | Violated rule: stale_servicer_update                              | Verify servicer feed timestamp and apply latest.  |
| LN000060  | 2025-01-01 |           0.33  | document_gap          | Violated rule: document_gap; Violated rule: stale_servicer_update | Request missing closing documents from custodian. |