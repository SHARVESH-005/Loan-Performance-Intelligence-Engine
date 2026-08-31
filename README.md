# Loan Performance Intelligence Engine

> **Intain Campus FinTech Challenge 2026 — AI Track**  
> Solo build · Deadline: Aug 31, 2026 · Python 3.11+

A data-science-first system for loan-level portfolio analysis. It takes panel-structured loan performance data and produces: a quantified data-quality picture, multi-outcome performance predictions (delinquency, default, prepayment, next-state), time-to-event survival curves, anomaly/exception flags, scenario-based stress projections, SHAP-based explanations, and a governed LLM copilot that narrates — but never generates — the underlying numbers.

---

## Architecture

```
Raw Data Pack (8 files)
        │
        ▼
FR-1: Data Intelligence & Profiling  ──► data_intelligence_report.md
        │
        ▼
Feature Engineering  (src/features/engineer.py)
        │
        ▼
Time-Aware Train / Val / Test Split  (cutoff: 2024-07-01)
    ┌───┴──────────────────┐──────────────────┐
    ▼                      ▼                  ▼
FR-2: Prediction       FR-3: Survival      FR-4: Anomaly
  Models (LightGBM)      (Hazard Model)      Detection (IF)
    │                      │                  │
    └──────────────────────┤                  │
                           ▼                  │
                    FR-6: Explainability ◄─────┘
                    (SHAP global/local)
                           │
                    FR-5: Scenario Simulation
                           │
                           ▼
                    FR-7: LLM Reviewer Copilot
                    (Gemini API + MockLLM fallback)
                           │
                           ▼
               submission.csv + Reports + Model Card
```

FR-8 (AI Development Log) tracks all stages continuously.

---

## Repository Structure

```
loan-performance-intelligence-engine/
├── README.md                        ← this file
├── AI_DEVELOPMENT_LOG.md            ← FR-8: agentic coding evidence
├── MODEL_CARD.md                    ← model details, metrics, limitations
├── requirements.txt                 ← pinned dependencies
├── PRD.md                           ← full product requirements document
├── data/
│   ├── raw/                         ← organizer data pack (or synthetic)
│   └── processed/                   ← engineered features, splits, model outputs
├── src/
│   ├── run_pipeline.py              ← single entry point
│   ├── data/
│   │   ├── profiler.py              ← FR-1: basic profiling
│   │   ├── advanced_profiler.py     ← FR-1: drift, associations, quality scores
│   │   └── synthetic_generator.py  ← synthetic data generation (§6.5)
│   ├── features/
│   │   └── engineer.py              ← feature engineering + time-aware split
│   ├── models/
│   │   ├── baseline.py              ← initial LightGBM baseline
│   │   ├── logistic_baseline.py     ← FR-2: logistic regression baseline
│   │   ├── improved.py              ← FR-2: tuned LightGBM
│   │   ├── calibration.py           ← FR-2: isotonic calibration
│   │   ├── metrics_report.py        ← FR-2: §12 metrics table
│   │   └── submission.py            ← assembles final submission.csv
│   ├── survival/
│   │   └── discrete_hazard.py       ← FR-3: discrete-time hazard model
│   ├── anomaly/
│   │   └── detector.py              ← FR-4: Isolation Forest + rule engine
│   ├── scenario/
│   │   └── simulator.py             ← FR-5: macro stress scenarios
│   ├── explainability/
│   │   └── explainer.py             ← FR-6: SHAP global/local, FP/FN analysis
│   └── llm_copilot/
│       ├── copilot.py               ← FR-7: LLM orchestrator
│       └── prompt_templates.py      ← FR-7: grounded prompt templates
├── reports/
│   ├── data_intelligence_report.md  ← FR-1 output
│   ├── model_performance_report.md  ← FR-2 §12 metrics
│   ├── survival_report.md           ← FR-3 output
│   ├── anomaly_examples.md          ← FR-4: 20+ curated examples
│   ├── scenario_report.md           ← FR-5 output
│   ├── explainability_report.md     ← FR-6 output
│   ├── llm_copilot_report.md        ← FR-7 output
│   ├── llm_prompt_log.jsonl         ← FR-7: append-only prompt log
│   └── servicer_reconciliation_log.csv
├── submission/
│   └── submission.csv               ← final scored output (23,894 rows)
└── demo/
    └── demo_script.md               ← 5-minute demo walkthrough (§16)
```

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- `pip` (or `pip` inside a virtual environment)

### Step 1 — Clone and create virtual environment
```bash
git clone <repo-url>
cd loan-performance-intelligence-engine
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### Step 2 — Install pinned dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Data setup

**Option A — Use organizer data pack:**  
Place all 8 provided files directly into `data/raw/`.

**Option B — Use synthetic data (default):**  
The pipeline auto-generates synthetic data on first run if `data/raw/` is empty:
```bash
python -m src.data.synthetic_generator
```

### Step 4 — Set Gemini API key (optional, for FR-7 live LLM)
```bash
# Windows PowerShell:
$env:GEMINI_API_KEY = "your-api-key-here"
```
If not set, FR-7 automatically falls back to a deterministic `MockLLM` — the pipeline runs without an API key.

---

## Running the Pipeline

### Full end-to-end run
```bash
python -m src.run_pipeline --stage all
```

### Run individual stages
```bash
python -m src.run_pipeline --stage day1   # FR-1: profiling + feature engineering
python -m src.run_pipeline --stage day2   # FR-2: prediction models + metrics
python -m src.run_pipeline --stage day3   # FR-3: survival + FR-4: anomaly detection
python -m src.run_pipeline --stage day4   # FR-5: scenarios + FR-6: SHAP + FR-7: copilot
```

> All stages produce a fresh `submission/submission.csv` at the end.

### Reproducibility guarantees
- All random seeds are fixed at `random_state=42` throughout.
- No internet access is required beyond the optional Gemini API call.
- The pipeline is idempotent — re-running overwrites outputs deterministically.

---

## Deliverables

All §11 required deliverables are present and linked below:

| Deliverable | Location | FR |
|---|---|---|
| GitHub repository | This repo | — |
| `submission.csv` | [`submission/submission.csv`](submission/submission.csv) | FR-2/FR-4 |
| Model card | [`MODEL_CARD.md`](MODEL_CARD.md) | §11 |
| Data intelligence report | [`reports/data_intelligence_report.md`](reports/data_intelligence_report.md) | FR-1 |
| Model performance report | [`reports/model_performance_report.md`](reports/model_performance_report.md) | FR-2 |
| Survival report | [`reports/survival_report.md`](reports/survival_report.md) | FR-3 |
| Anomaly examples (20+) | [`reports/anomaly_examples.md`](reports/anomaly_examples.md) | FR-4 |
| Scenario report | [`reports/scenario_report.md`](reports/scenario_report.md) | FR-5 |
| Explainability report | [`reports/explainability_report.md`](reports/explainability_report.md) | FR-6 |
| LLM copilot demo | [`reports/llm_copilot_report.md`](reports/llm_copilot_report.md) | FR-7 |
| LLM prompt log | [`reports/llm_prompt_log.jsonl`](reports/llm_prompt_log.jsonl) | FR-7 |
| AI Development Log | [`AI_DEVELOPMENT_LOG.md`](AI_DEVELOPMENT_LOG.md) | FR-8 |
| Demo script | [`demo/demo_script.md`](demo/demo_script.md) | §16 |

---

## Submission File Schema

`submission/submission.csv` contains 23,894 rows and 15 columns:

| Column | Description |
|---|---|
| `loan_id` | Loan identifier |
| `reporting_month` | As-of month |
| `next_3m_delinquency_prob` | Calibrated 3-month delinquency probability |
| `next_6m_delinquency_prob` | Calibrated 6-month delinquency probability |
| `next_12m_default_prob` | Calibrated 12-month default probability |
| `next_12m_prepayment_prob` | Calibrated 12-month prepayment probability |
| `predicted_next_state` | Most likely next loan state |
| `confidence` | Overall model confidence score |
| `anomaly_score` | Continuous Isolation Forest anomaly score |
| `exception_probability` | Probability of exception required |
| `exception_required_flag` | Binary exception flag |
| `exception_type` | Category: no_exception / delinquency_status_conflict / invalid_date / document_gap / balance_mismatch / stale_servicer_update |
| `top_drivers` | Top SHAP feature drivers (semicolon-separated) |
| `recommended_action` | LLM-generated reviewer action (labeled as recommendation) |
| `model_version` | Model artifact version tag |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Time-aware split, never random** | Panel data — random row-level splits cause target leakage across months |
| **LightGBM for all FR-2 targets** | One library, one API; handles imbalance via `scale_pos_weight`; fast iteration |
| **Isotonic calibration** | Converts raw scores to empirical probabilities; required for Brier score and copilot use |
| **Discrete-time hazard model (FR-3)** | Reuses the same feature stack as FR-2; handles right-censoring correctly |
| **Hybrid rule + IF anomaly (FR-4)** | Deterministic rules from `validation_rules.json` take precedence; IF adds continuous score |
| **MockLLM fallback (FR-7)** | Ensures pipeline runs to completion even without an API key or during rate limiting |
| **Lag-only features in survival model** | Prevents within-loan temporal leakage — current-month DPD excluded from FR-3 |

---

## Disqualifier Audit (§4.3)

- ✅ Core predictions come from trained ML models (LightGBM / Logistic Regression), not LLM API calls
- ✅ Time-aware split enforced — no random row-level splits
- ✅ No target-leaking features (all 7 target columns excluded from feature matrix)
- ✅ Pipeline runs end-to-end from raw data to `submission.csv`
- ✅ All reported metrics come from actual held-out evaluation runs
- ✅ No fabricated or cherry-picked results
- ✅ All LLM outputs grounded in retrieved structured context
- ✅ Every LLM artifact labeled "Recommendation — not a decision"
