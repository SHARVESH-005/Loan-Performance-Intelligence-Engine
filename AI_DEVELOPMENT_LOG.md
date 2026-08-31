# AI Development Log

## Tools Used
- Antigravity IDE (Gemini 3.1 Pro High) - Code generation, refactoring, planning, and debugging.
- Anthropic Claude / Google Gemini API (Mock/Live via Copilot) - Used in FR-7 for risk narrative generation.

## Representative Prompts
1. [Initial setup] -> Created repository structure and synthetic data generator.
2. [Model Calibration] -> Add isotonic calibration to LightGBM models to output true probabilities instead of raw scores.
3. [Fix Target Leakage] -> The survival model is learning concurrently from current-month status. Shift the features to t-1 to fix target leakage.
4. [Explainability] -> Use shap.TreeExplainer to extract the top 3 drivers for each record and save them to a file.

## Accepted AI Output Examples
- Generated repository structure and synthetic data generator script.
- Successfully implemented time-series splitting to avoid lookahead bias.
- Successfully implemented discrete hazard models using pooled logistic regression.
- Correctly identified target leakage in the initial survival model and refactored features to use `.shift(1)`.
- Successfully implemented SHAP TreeExplainer for global and local explainability.
- Successfully implemented offline MockLLM to bypass DNS/internet restrictions in the environment.

## Rejected / Corrected AI Output Examples
- **Rejected:** Initial anomaly detection approach tried to tune LightGBM for binary F1 on the `exception_type` label.
  - **Correction:** The user explicitly corrected the agent, noting that the ground-truth `exception_type` labels were *synthetic noise* and uncorrelated with the features. The agent was directed to build a hybrid rules+ML system where deterministic rules take precedence.
- **Rejected:** LLM copilot generated responses with absolute certainty (e.g. "WILL default").
  - **Correction:** Updated system instructions to forbid absolute certainty and strictly enforce the `[Recommendation — not a decision]` prefix constraint.

## Human Review Process
- All generated code is reviewed against the PRD requirements before being committed.
- Implementation plans are generated as `implementation_plan.md` and explicitly reviewed/approved by the human user before execution begins.
- Outputs (SHAP plots, Scenario reports) are manually reviewed via markdown artifacts.

## Approximate AI-Generated Code Share
- 95% AI-generated, 5% Human-guided (architectural corrections, environment debugging).

## Lessons Learned
- **Data Reality:** Always analyze the data before modeling. Trying to fit an ML model on pure synthetic noise wastes time; plotting feature distributions against the target early on is critical.
- **Environment Constraints:** Designing modular code allows for easy fallback mechanisms (e.g., MockLLM) when external APIs are unavailable due to network restrictions.
- **Target Leakage in Survival Models:** Month-to-month state transitions require strict lagged feature engineering; otherwise, the model trivially predicts the current state.

## Log Entries

| Date | Tool | Prompt (summary) | Accepted? | Notes |
|---|---|---|---|---|
| 2026-08-26 | Antigravity IDE | Project setup and synthetic data generation | Yes | Initial setup — repo skeleton, requirements.txt, synthetic generator |
| 2026-08-27 | Antigravity IDE | Day 1: Basic profiling, feature engineering, first submission.csv | Yes | End-to-end thin slice complete; first valid submission produced |
| 2026-08-27 | Antigravity IDE | Day 2: Advanced profiling and calibrated LightGBM baselines | Yes | Isotonic calibration, time-aware split, §12 metrics report |
| 2026-08-27 | Antigravity IDE | Day 3: Survival models and Anomaly Detection | Partial | User intervened to correct labeling noise issue — agent pivoted to hybrid rule-based system where deterministic rules take precedence over IF scores |
| 2026-08-28 | Antigravity IDE | Day 4 (initial): Scenarios, Explainability, LLM Copilot | Yes | Implemented SHAP, macro perturbation, and LLM pipeline with offline mock fallback |
| 2026-08-28 | Gemini API | FR-7: Generate risk profile summaries for test loans | Yes | 14 real API calls logged; Gemini 2.5 Flash used; outputs all labeled [Recommendation — not a decision] |
| 2026-08-30 | Antigravity IDE | Day 4 Audit: Identify gaps between implementation and PRD acceptance criteria | Partial | 9 issues found across FR-5, FR-6, FR-7 |
| 2026-08-30 | Antigravity IDE | FR-5 fix: Replace hardcoded scenario driver narrative with data-driven computation | Yes | Now computes which credit-score segment shifted most; adds prepayment limitation disclaimer |
| 2026-08-30 | Antigravity IDE | FR-6 fix: Add SHAP plots for default and prepayment models, not just delinquency | Yes | PRD §8 FR-6 req 5 explicitly requires covering all prediction targets — this was missed in initial build |
| 2026-08-30 | Antigravity IDE | FR-6 fix: Vary FP/FN hypotheses based on actual top SHAP driver per example | Yes | Previous version used identical boilerplate text for all examples — not PRD-compliant |
| 2026-08-30 | Antigravity IDE | FR-7 fix: Implement scenario narration by reading scenario_summary.csv | Yes | SCENARIO_SUMMARY_TEMPLATE existed in prompt_templates.py but was never invoked — dead code |
| 2026-08-30 | Antigravity IDE | FR-7 fix: Sample diverse loans (high-anomaly, exception, high-default, normal) instead of head(5) | Yes | All 5 previous samples were the same loan (LN000000) across different months — non-compliant |
| 2026-08-30 | Antigravity IDE | FR-7 fix: Add rate-limit fallback and real adversarial prompts for failure documentation | Yes | 429 RESOURCE_EXHAUSTED errors now trigger automatic MockLLM fallback instead of crashing |
| 2026-08-30 | Antigravity IDE | Populate MODEL_CARD.md with real metrics, features, validation, limitations | Yes | Was 29 lines of TBD stubs; now 209 lines with actual pipeline outputs |
| 2026-08-30 | Antigravity IDE | Rewrite README.md with architecture, deliverable links, run order, disqualifier audit | Yes | Was 16-line skeleton; now full reproducibility guide |
| 2026-08-30 | Antigravity IDE | Write demo/demo_script.md with all 15 PRD §16 steps timed and scripted | Yes | Includes exact commands, files to show, and narration for each step |

## Rejected / Corrected AI Output Examples — Additional Day 4

- **Rejected:** FR-7 initial copilot sampled `df.head(5)` for risk profiles — all 5 rows were the same loan (LN000000) from different months. Agent corrected to diverse-sampling logic.
  - **Correction:** Sort by anomaly score, exception flag, default probability, prepayment probability, and normal — take one of each.

- **Rejected:** FR-5 scenario driver explanation was hardcoded static prose unrelated to actual model outputs.
  - **Correction:** Compute actual per-segment probability shifts dynamically, find the segment with max absolute delta, report that.

- **Rejected:** FR-6 SHAP coverage was limited to a single model (`next_3m_delinquency_flag`). PRD explicitly requires coverage of all prediction targets.
  - **Correction:** Added SHAP summary plots for `next_12m_default_flag` and `next_12m_prepayment_flag`, using the same TreeExplainer pattern with conditional file-existence check.

- **Accepted with caveat:** LLM Copilot's `SCENARIO_SUMMARY_TEMPLATE` was written in `prompt_templates.py` during Day 4 initial build but was never called from `copilot.py`. This dead code went unnoticed until the Day 4 audit.

## Lessons Learned — Updated

- **Data Reality:** Always analyze the data before modeling. Trying to fit an ML model on pure synthetic noise wastes time; plotting feature distributions against the target early on is critical.
- **Environment Constraints:** Designing modular code allows for easy fallback mechanisms (e.g., MockLLM) when external APIs are unavailable due to network restrictions.
- **Target Leakage in Survival Models:** Month-to-month state transitions require strict lagged feature engineering; otherwise, the model trivially predicts the current state.
- **Dead Code is Invisible:** Feature code that exists but is never called (like SCENARIO_SUMMARY_TEMPLATE) doesn't fail any tests — it just silently fails to satisfy PRD requirements. Auditing against the PRD acceptance criteria after implementation is essential.
- **PRD Acceptance Criteria > Code Review:** Reading your own code looks correct. Reading the PRD acceptance criteria against your outputs catches gaps the code itself won't reveal (e.g., "cover all models" vs. the one model you implemented first).
- **Diverse Test Cases Matter:** Using `df.head(n)` for demonstration samples is a common mistake that produces non-representative outputs. Always build sampling logic that captures the extreme cases judges are most likely to look for.
