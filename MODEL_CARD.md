# Model Card: Loan Performance Intelligence Engine

> **Version:** v1.0 | **Date:** 2026-08-30 | **Challenge:** Intain Campus FinTech Challenge 2026 — AI Track

---

## Objective

This system predicts multiple forward-looking loan performance outcomes from panel-structured monthly loan data. It provides:

- **FR-2:** Calibrated probabilities for 3-month delinquency, 6-month delinquency, 12-month default, 12-month prepayment, and predicted next state.
- **FR-3:** Discrete-time hazard survival model estimating *when* events occur, respecting right-censoring.
- **FR-4:** Hybrid anomaly and exception detection (Isolation Forest + deterministic validation rules), producing a continuous `anomaly_score` and a categorical `exception_type`.

The system assists human loan reviewers — not replaces them. All outputs are labeled **"Recommendation — not a decision"** and routed through a human-in-the-loop copilot layer (FR-7).

---

## Data

| Attribute | Details |
|---|---|
| **Source** | Synthetic dataset generated per PRD §6.5 (schema-compatible with organizer data pack) |
| **Generator seed** | Fixed (`random_state=42`) for full reproducibility |
| **Training rows** | 46,952 loan-months |
| **Unique loans (train)** | ~4,169 |
| **Training date range** | 2022-01-01 → 2024-06-01 |
| **Validation cut-off** | 2024-07-01 |
| **Test rows (held-out)** | 8,158 |
| **Known limitation** | Synthetic data — real-world distribution shifts (e.g. macro cycle effects, servicer heterogeneity) are approximated but not guaranteed to match organizer data |

---

## Features

**36 features total** used across models:

### Static / Origination Features
| Feature | Type | Notes |
|---|---|---|
| `original_balance` | Numeric | Loan balance at origination |
| `interest_rate` | Numeric | Note rate at origination |
| `credit_score_band` | Categorical | Binned credit score; ordinal-encoded as `credit_score_band_ord` |
| `ltv_band` | Categorical | Binned LTV; ordinal-encoded as `ltv_band_ord` |
| `dti_band` | Categorical | Binned DTI; ordinal-encoded as `dti_band_ord` |
| `state` | Categorical | Property location |
| `loan_purpose` | Categorical | Purchase / refi-rate-term / refi-cashout |
| `occupancy_type` | Categorical | Owner-occupied / second-home / investment |
| `property_type` | Categorical | Single-family / condo / multi-unit |
| `servicer_name` | Categorical | Servicer of record |

### Monthly Panel Features
| Feature | Type | Notes |
|---|---|---|
| `current_balance` | Numeric | Outstanding balance this month |
| `days_past_due` | Numeric | DPD as of reporting month |
| `current_status` | Categorical | Current / 30-60-90+ DPD / Default / Prepaid / Modified |
| `loan_age_months` | Numeric | Months since origination |
| `remaining_term_months` | Numeric | Months to scheduled maturity |
| `modification_flag` | Boolean | Whether loan has been modified |
| `document_status` | Categorical | Document completeness |
| `loss_severity_band` | Categorical | Binned loss severity if defaulted |
| `source_system` | Categorical | System of record |

### Engineered Features
| Feature | Type | Notes |
|---|---|---|
| `balance_ratio` | Numeric | `current_balance / original_balance` |
| `rate_spread` | Numeric | `interest_rate − 6.5` (benchmark spread) |
| `months_since_last_update` | Numeric | Staleness from `last_updated_at` |
| `prev_balance` | Numeric | Lag-1 balance |
| `balance_change` | Numeric | Month-over-month balance delta |
| `prev_dpd` | Numeric | Lag-1 DPD |
| `dpd_change` | Numeric | Month-over-month DPD delta |

### Intentionally Excluded (Leakage Audit)
| Feature | Reason Excluded |
|---|---|
| All 7 target flag columns | Target variables — only used as labels |
| Current-month `days_past_due` in FR-3 | **Excluded from hazard model** — reflects the outcome of the transition; only lagged versions (`prev_dpd`, `dpd_change`) are used |

---

## Model Type

| Task | Algorithm | Library | Key Hyperparameters |
|---|---|---|---|
| FR-2 binary (×4) — improved | LightGBM | `lightgbm 4.x` | `n_estimators=500`, `learning_rate=0.05`, `num_leaves=31`, `scale_pos_weight` tuned per target |
| FR-2 binary (×4) — baseline | Logistic Regression | `sklearn` | `class_weight='balanced'`, `C=1.0` |
| FR-2 `next_state` — improved | LightGBM multi-class | `lightgbm 4.x` | `objective='multiclass'`, `num_class=7` |
| FR-2 `next_state` — baseline | Logistic Regression | `sklearn` | `multi_class='multinomial'` |
| FR-2 calibration | Isotonic Regression | `sklearn` | `cv='prefit'` applied post-training |
| FR-3 survival | Discrete-time hazard (pooled logistic per loan-month) | `sklearn` | Lag features only |
| FR-4 anomaly | Isolation Forest | `sklearn` | `n_estimators=200`, `contamination=0.05`, `random_state=42` |
| FR-4 exception type | Deterministic rule engine + IF hybrid | custom | Rules from `validation_rules.json` take precedence over IF scores |

---

## Validation Method

**Time-aware split — no random row-level splits used anywhere in the pipeline.**

| Split | Reporting Months | Rows |
|---|---|---|
| Train | 2022-01 → 2024-06 | 46,952 |
| Validation | 2024-07 | ~9,000 |
| Test (held-out) | 2024-08+ | 8,158 |

- Models trained on train set only.
- Hyperparameters selected on validation-set performance.
- All reported metrics are from the held-out test set.
- FR-3 hazard model uses only lag-1 features to prevent within-loan temporal leakage.
- Group integrity: all rows for a given `loan_id` within the training period remain in train; no loan straddles train/test.

---

## Metrics

### FR-2: Binary Targets

| Target | Model | ROC-AUC | PR-AUC | F1 | Recall@80%P | Brier |
|---|---|---|---|---|---|---|
| `next_3m_delinquency_flag` | Logistic (Baseline) | 0.7343 | 0.4125 | 0.4668 | 0.2170 | 0.1736 |
| `next_3m_delinquency_flag` | **LightGBM (Improved)** | **0.7538** | **0.4305** | **0.4674** | **0.2335** | **0.0668** |
| `next_6m_delinquency_flag` | Logistic (Baseline) | 0.6500 | 0.2423 | 0.3201 | 0.0000 | 0.2243 |
| `next_6m_delinquency_flag` | **LightGBM (Improved)** | **0.7728** | **0.3577** | **0.4328** | **0.0071** | **0.0905** |
| `next_12m_default_flag` | Logistic (Baseline) | 0.7814 | 0.0051 | 0.0167 | 0.0000 | 0.1871 |
| `next_12m_default_flag` | **LightGBM (Improved)** | **0.9910** | **0.1354** | **0.2083** | **0.0000** | **0.0010** |
| `next_12m_prepayment_flag` | Logistic (Baseline) | 0.4968 | 0.0077 | 0.0165 | 0.0000 | 0.2501 |
| `next_12m_prepayment_flag` | **LightGBM (Improved)** | **0.8773** | **0.0580** | **0.1361** | **0.0000** | **0.0074** |

### FR-2: Next State Multi-Class

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Logistic (Baseline) | 0.8267 | 0.4636 |
| **LightGBM (Improved)** | 0.7150 | **0.4846** |

Best per-class F1: Terminal (1.00), 90+ DPD (0.54), 60 DPD (0.46). Weakest: Prepaid (0.04) — class imbalance.

### FR-3: Survival / Hazard Model

| Model | ROC-AUC (Concordance Proxy) |
|---|---|
| Flat Empirical Baseline | 0.5146 |
| **Discrete-Time Hazard (Logistic)** | **0.9976** |

### FR-4: Anomaly Detection
- 20+ curated examples reviewed and validated against deterministic rule violations
- `exception_type` categories: `no_exception`, `delinquency_status_conflict`, `invalid_date`, `document_gap`, `balance_mismatch`, `stale_servicer_update`

---

## Limitations

1. **Synthetic data gap:** The model is trained on synthetically generated data. Real-world performance depends on how closely the organizer's data matches synthetic distributions, especially for rare events (default, prepayment).

2. **Muted prepayment response in scenarios:** The `next_12m_prepayment_flag` model (PR-AUC=0.058) and the High_Prepayment stress scenario show limited sensitivity to interest rate changes. The synthetic data does not exhibit a strong historical correlation between rate spreads and prepayment events — a known limitation of the data generation process.

3. **Zero recall at high precision for very rare events:** For `next_12m_default_flag` and `next_12m_prepayment_flag`, Recall @ 80% Precision = 0.0. These are extremely rare events (~0.1–0.7% base rate). The models correctly rank them (high AUC) but cannot retrieve many at a conservative threshold.

4. **Next-state Prepaid class:** F1=0.04. Loans about to prepay look identical to ordinary Current loans one month before prepayment; panel features do not capture external refinancing intent.

5. **Stale credit score bands:** `credit_score_band` is an origination-time feature and does not update with the borrower's current credit profile. This is the primary driver of false negatives for sudden-shock defaults.

6. **LLM Copilot rate limits:** Gemini free-tier API is limited to 5 requests/minute. For large portfolios, the narration layer automatically falls back to deterministic `MockLLM` outputs.

---

## Leakage Controls

| Control | Implementation |
|---|---|
| **Time-aware split** | Strict cutoff: train ≤ 2024-06-30, test > 2024-07-01. Enforced in `src/features/engineer.py`. |
| **Lag-only features in FR-3** | Hazard model uses only `prev_dpd`, `prev_balance`, `dpd_change`, `balance_change`. Current-month `days_past_due` excluded explicitly. |
| **Target flags excluded from features** | All 7 target columns removed from feature matrix before any model fit. |
| **Loan-level group integrity** | All rows for a `loan_id` in the training window are in train; no loan straddles train/test. |
| **`last_updated_at` used as-of** | Staleness features computed using only the as-of reporting month timestamp. |

---

## Known Failure Modes

Documented from FR-6 explainability analysis on the held-out test set:

### False Positives (Predicted Delinquent, Actually Current)
- **Pattern:** Loans with elevated `days_past_due` (30+ DPD) that self-cure in the next period.
- **Example loans:** LN000376, LN001274, LN001442 — high DPD + high `balance_ratio`, but borrower made a catch-up payment.
- **Root cause:** Model cannot observe the payment event in the current month; it sees only the end-of-month DPD snapshot.

### False Negatives (Predicted Current, Actually Default)
- **Pattern:** Borrowers with good credit scores who default due to sudden unobserved shocks (job loss, medical event).
- **Example loans:** LN004991, LN001894 — high `credit_score_band_ord` suppresses the risk score even as the loan deteriorates.
- **Root cause:** Static credit score bands are origination-time and do not reflect the borrower's current situation.

### Prepayment Misclassification
- **Pattern:** Loans classified as `Current` one month before prepayment look identical to ordinary current loans.
- **Root cause:** Prepayment decision is driven by unobserved refinancing intent; panel features do not capture external rate-market signals at the loan level.

---

## Intended Use

- **Intended users:** Loan portfolio reviewers, risk analysts, compliance officers
- **Intended use:** Surfacing high-risk loans for human review; stress-testing under macro scenarios; flagging data exceptions for servicer reconciliation
- **Not intended for:** Automated credit decisions without human review; legal or regulatory compliance; real-money lending decisions

> All model outputs are **recommendations, not decisions.** Every output is labeled "Recommendation — not a decision" per FR-7 governance requirements.
