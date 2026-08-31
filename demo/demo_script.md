# Demo Script — Loan Performance Intelligence Engine

> **5 minutes · 15 steps · Just follow this top to bottom**

---

## ⚙️ Before You Start

In your terminal (with `.venv` active), run the full pipeline once so all reports are fresh:
```
python -m src.run_pipeline --stage all
```
Open VS Code with the `e:\Intain` folder. Keep this script visible on a second screen or phone.

---

## Step 1 — Dataset & Targets `[0:00 – 0:20]`

**Show:** Terminal  
**Run:**
```
python -c "import pandas as pd; df = pd.read_csv('data/processed/train.csv'); print('Rows:', len(df)); print('Loans:', df['loan_id'].nunique()); print('Range:', df['reporting_month'].min(), '->', df['reporting_month'].max())"
```
**Say:** *"We have 46,952 monthly loan records across 4,169 unique loans from 2022 to 2024. The system predicts 5 outcomes: 3-month delinquency, 6-month delinquency, 12-month default, prepayment, and next loan state."*

---

## Step 2 — Data Profiling Report `[0:20 – 0:40]`

**Show:** Open `reports/data_intelligence_report.md`  
**Say:** *"FR-1 profiles every field — distributions, missingness by servicer and state, and a data quality score per record. The batch quality rollup is at the bottom."*

---

## Step 3 — Top Data-Quality Issues `[0:40 – 1:00]`

**Show:** Scroll to the drift / outlier section in the same report  
📍 `data_intelligence_report.md` line 59 → `Missing_Doc: 3158` | line 143 → `469 outliers (1.00%)`  
**Say:** *"The report highlights a few key issues: over 3,100 records have missing documents (about 5%), and there are stale servicer updates where the reporting month and last update timestamp conflict. The Isolation forest also flags 1% of the training data as multivariate outliers. These issues directly drive the exception flags in FR-4."*

---

## Step 4 — Feature Engineering `[1:00 – 1:20]`

**Show:** Open `src/features/engineer.py`, scroll to the lag features block  
**Say:** *"We engineer four lag features — previous DPD, previous balance, and their month-over-month changes — to capture momentum. Plus balance ratio, rate spread, and staleness. 36 features total, feeding every downstream model."*

---

## Step 5 — Time-Aware Split `[1:20 – 1:35]`

**Show:** Show the `perform_time_split` function in `engineer.py`  
**Say:** *"The split is strictly time-based — train ends June 2024, test starts October 2024. No random shuffling. This is the most critical step to prevent data leakage in panel data."*

---

## Step 6 — Baseline Model `[1:35 – 1:55]`

**Show:** Open `reports/model_performance_report.md`, scroll to the Logistic Regression rows  
📍 `model_performance_report.md` line 6 → 3m delinq AUC `0.7343` | line 18 → 12m default PR-AUC `0.0051`  
**Say:** *"Our baseline is logistic regression — simple and interpretable. AUC of 0.73 for 3-month delinquency, but PR-AUC of just 0.005 for 12-month default. Naive models collapse on rare events."*

---

## Step 7 — Improved Model `[1:55 – 2:15]`

**Show:** Show the LightGBM rows in the same report  
📍 `model_performance_report.md` line 19 → AUC `0.9910`, Brier `0.0010` | line 18 → baseline Brier `0.1871`  
**Say:** *"The improved model is a tuned LightGBM with isotonic calibration. Default AUC jumps to 0.991. Brier score drops 190 times — from 0.187 to 0.001. Much better calibrated probabilities."*

---

## Step 8 — Survival Model `[2:15 – 2:40]`

**Show:** Open `reports/survival_report.md`, then open `reports/survival_curves/km_by_credit.png`  
📍 `survival_report.md` line 19 → Baseline `0.5146` | line 20 → Hazard `0.9976`  
**Say:** *"FR-3 is a discrete-time hazard model that predicts when an event will happen, not just if. It handles right-censored loans correctly. AUC of 0.997 versus 0.51 for the flat baseline. The KM curves show sub-600 credit score loans fail twice as fast."*

---

## Step 9 — Anomaly Detection `[2:40 – 3:00]`

**Show:** Open `reports/anomaly_examples.md`, scroll to the Rule Violation Summary table then curated examples  
📍 `anomaly_examples.md` line 15 → total `1857` | line 24 → `document_gap 1161` | line 26 → `stale_servicer_update 736` | line 44 → LN000000 score `0.733`  
**Say:** *"FR-4 combines Isolation Forest with rule-based checks. 1,857 records flagged total — 1,161 document gaps and 736 stale servicer updates. Here in the curated examples, LN000000 has anomaly score 0.73 and is flagged as document_gap."*

---

## Step 10 — Scenario Simulation `[3:00 – 3:25]`

**Show:** Open `reports/scenario_report.md`, show the projected rates table  
📍 `scenario_report.md` line 15 → Adverse `47.6%` | line 16 → Base `9.65%` | line 46 → 600-649 `+52.44%` | line 49 → muted prepayment note  
**Say:** *"FR-5 runs three stress scenarios. Under Adverse Credit — 8.5% unemployment, HPI down 10% — delinquency rate spikes from 9.6% to 47.6%. The 600-649 credit band is hit hardest at +52 points. Prepayment response is muted due to the synthetic data not capturing rate-spread sensitivity — we document this limitation explicitly."*

---

## Step 11 — Local Explanation `[3:25 – 3:50]`

**Show:** Open `reports/explainability/shap_waterfall_LN000000.png` full screen  
**Say:** *"FR-6 uses SHAP to explain every prediction. This waterfall shows exactly why LN000000 is high risk — days_past_due and DTI band are the top drivers. We also have global SHAP summaries for all three models, plus documented false positive and false negative patterns."*

---

## Step 12 — LLM Reviewer Note `[3:50 – 4:10]`

**Show:** Open `reports/llm_copilot_report.md`, show the Grounded Risk Profiles section  
**Say:** *"FR-7 is an LLM copilot powered by Gemini. It reads the SHAP values, anomaly score, and probabilities, then writes a plain-English reviewer note. The LLM narrates numbers we give it — it never invents them. Every output is labeled: Recommendation — not a decision."*

---

## Step 13 — LLM Failure Examples `[4:10 – 4:30]`

**Show:** Scroll to the Documented LLM Failures section of the same report  
**Say:** *"We document three intentional failures. One: the LLM hallucinated an anomaly score when the field was missing from context. Two: it said a loan WILL default — rejected for being too certain. Three: it explained a feature that doesn't exist. All caught, logged, and corrected."*

---

## Step 14 — Submission File `[4:30 – 4:45]`

**Show:** Terminal. Run:
```
python -c "import pandas as pd; df = pd.read_csv('submission/submission.csv'); print('Shape:', df.shape); print('Nulls:', df.isna().sum().sum()); print(df[['loan_id','next_3m_delinquency_prob','anomaly_score','exception_type','recommended_action']].head(2).to_string())"
```
**Say:** *"23,894 rows, 15 columns, zero nulls. Every loan has probabilities, anomaly score, exception type, SHAP drivers, and an LLM recommendation. Fully reproducible with one command."*

---

## Step 15 — AI Development Log `[4:45 – 5:00]`

**Show:** Open `AI_DEVELOPMENT_LOG.md`, scroll through log entries and Lessons Learned  
**Say:** *"FR-8 is the AI Development Log — every prompt, every accepted and rejected output, throughout the build. Not retroactive. It shows where the AI got things wrong and how we corrected it. That's the project — thank you."*


