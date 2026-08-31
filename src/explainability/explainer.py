import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

PROCESSED_DIR = "data/processed"
MODELS_DIR = "data/processed/models"
REPORTS_DIR = "reports"
EXP_DIR = os.path.join(REPORTS_DIR, "explainability")

def run():
    print("Running FR-6 Explainability Analysis...")
    os.makedirs(EXP_DIR, exist_ok=True)
    
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    preds_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test_with_preds.csv"))
    
    with open(os.path.join(PROCESSED_DIR, "features.csv"), "r") as f:
        features = [line.strip() for line in f]
        
    X_test = test_df[features].copy()
    for c in X_test.select_dtypes(include=['object', 'string']).columns:
        X_test[c] = X_test[c].astype('category')
        
    # We will focus on the primary risk target for row-level drivers
    primary_target = 'next_3m_delinquency_flag'
    model_path = os.path.join(MODELS_DIR, f"{primary_target}_lgb_improved.joblib")
    
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Skipping explainability.")
        return
        
    model = joblib.load(model_path)
    
    # SHAP explainer
    print("  Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    # Top drivers per row
    print("  Extracting top drivers per record...")
    # shape of shap_values.values is (N, num_features)
    # We want the top 3 features by absolute SHAP value for each row
    vals = shap_values.values
    feature_names = np.array(features)
    
    # Get indices of top 3 absolute values for each row
    top_indices = np.argsort(np.abs(vals), axis=1)[:, -3:]
    
    # Reverse to get highest impact first
    top_indices = top_indices[:, ::-1]
    
    drivers_list = []
    for i in range(len(test_df)):
        top_feats = feature_names[top_indices[i]]
        drivers_list.append("; ".join(top_feats))
        
    drivers_df = test_df[['loan_id', 'reporting_month']].copy()
    drivers_df['top_drivers'] = drivers_list
    drivers_df.to_csv(os.path.join(PROCESSED_DIR, "test_shap_drivers.csv"), index=False)
    
    # Global summary plot for primary target
    print("  Generating SHAP summary plot for delinquency...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(EXP_DIR, "shap_summary_delinquency.png"), bbox_inches='tight')
    plt.close()
    
    # Global summary plots for other targets
    for other_target, short_name in [('next_12m_default_flag', 'default'), ('next_12m_prepayment_flag', 'prepayment')]:
        other_path = os.path.join(MODELS_DIR, f"{other_target}_lgb_improved.joblib")
        if os.path.exists(other_path):
            print(f"  Generating SHAP summary plot for {short_name}...")
            other_model = joblib.load(other_path)
            other_explainer = shap.TreeExplainer(other_model)
            other_shap = other_explainer(X_test)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(other_shap, X_test, show=False)
            plt.savefig(os.path.join(EXP_DIR, f"shap_summary_{short_name}.png"), bbox_inches='tight')
            plt.close()
    
    # Local waterfall plot for an anomalous exception loan
    anomaly_preds_path = os.path.join(PROCESSED_DIR, "test_anomaly_preds.csv")
    if os.path.exists(anomaly_preds_path):
        anom_df = pd.read_csv(anomaly_preds_path)
        # Find a row that is flagged as exception
        exceptions = anom_df[anom_df['predicted_exception_required'] == 1]
        if len(exceptions) > 0:
            target_idx = exceptions.index[0]
            loan_id = exceptions.iloc[0]['loan_id']
            
            print(f"  Generating local waterfall plot for {loan_id}...")
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(shap_values[target_idx], show=False)
            plt.savefig(os.path.join(EXP_DIR, f"shap_waterfall_{loan_id}.png"), bbox_inches='tight')
            plt.close()
    
    # FP/FN Analysis
    print("  Running FP/FN analysis...")
    # Find FP and FN for the primary target
    # test.csv has the true labels for the next 3m flag
    if primary_target in test_df.columns:
        true_labels = test_df[primary_target]
        pred_probs = preds_df[f"{primary_target}_prob_calibrated"]
        pred_labels = (pred_probs > 0.5).astype(int)
        
        fp_mask = (true_labels == 0) & (pred_labels == 1)
        fn_mask = (true_labels == 1) & (pred_labels == 0)
        
        fp_indices = np.where(fp_mask)[0]
        fn_indices = np.where(fn_mask)[0]
        
        # Sort by how wrong they were
        fp_sorted = fp_indices[np.argsort(pred_probs.iloc[fp_indices])[::-1]] # highest prob first
        fn_sorted = fn_indices[np.argsort(pred_probs.iloc[fn_indices])] # lowest prob first
        
        fp_examples = []
        for idx in fp_sorted[:3]:
            drivers = drivers_list[idx]
            if 'days_past_due' in drivers:
                hyp = "High DPD suggests delinquency, but borrower cured."
            elif 'balance' in drivers:
                hyp = "High balance ratio increased risk score, but external factors prevented actual delinquency."
            else:
                hyp = "Feature values mirror historical defaults, but external factors prevented actual delinquency."
                
            fp_examples.append({
                'Loan ID': test_df.iloc[idx]['loan_id'],
                'Probability': f"{pred_probs.iloc[idx]:.2f}",
                'Top Drivers': drivers,
                'Hypothesized Cause': hyp
            })
            
        fn_examples = []
        for idx in fn_sorted[:3]:
            drivers = drivers_list[idx]
            if 'credit_score' in drivers:
                hyp = "Good credit score suppressed risk, missing a sudden unobserved shock (e.g. job loss)."
            elif 'ltv' in drivers:
                hyp = "Favorable LTV hid true risk of borrower's precarious cash flow."
            else:
                hyp = "Missing risk signal or sudden unobserved shock not reflected in panel features."
                
            fn_examples.append({
                'Loan ID': test_df.iloc[idx]['loan_id'],
                'Probability': f"{pred_probs.iloc[idx]:.2f}",
                'Top Drivers': drivers,
                'Hypothesized Cause': hyp
            })
    else:
        fp_examples = []
        fn_examples = []
        
    # Write report
    report_lines = ["# FR-6 Explainability & Responsible AI Report\n\n"]
    report_lines.append("## Global Feature Importance\n\n")
    report_lines.append("The SHAP summary plots below illustrate the global impact of features on the primary risk targets.\n\n")
    report_lines.append("### 3-Month Delinquency\n")
    report_lines.append("![SHAP Summary Delinquency](explainability/shap_summary_delinquency.png)\n\n")
    
    if os.path.exists(os.path.join(EXP_DIR, "shap_summary_default.png")):
        report_lines.append("### 12-Month Default\n")
        report_lines.append("![SHAP Summary Default](explainability/shap_summary_default.png)\n\n")
        
    if os.path.exists(os.path.join(EXP_DIR, "shap_summary_prepayment.png")):
        report_lines.append("### 12-Month Prepayment\n")
        report_lines.append("![SHAP Summary Prepayment](explainability/shap_summary_prepayment.png)\n\n")
    
    if 'loan_id' in locals():
        report_lines.append("## Local Single-Loan Explanation\n\n")
        report_lines.append(f"A detailed SHAP waterfall plot for an anomalous loan (`{loan_id}`) demonstrating how specific feature values push the baseline risk up or down.\n\n")
        report_lines.append(f"![SHAP Waterfall Plot](explainability/shap_waterfall_{loan_id}.png)\n\n")
        
    report_lines.append("## Uncertainty & Confidence Reporting\n\n")
    report_lines.append("Model confidence is explicitly surfaced in two ways:\n")
    report_lines.append("1. **Isotonic Calibration:** Raw scores are mapped to true empirical probabilities, ensuring a `0.80` score means an 80% real-world event rate.\n")
    report_lines.append("2. **Multi-Model Agreement:** The final `confidence` score penalizes predictions where the suite of temporal models (3m, 6m, 12m) produce contradictory risk trajectories.\n\n")
    
    report_lines.append("## Error Analysis: False Positives & False Negatives\n\n")
    
    if fp_examples:
        report_lines.append("### Top False Positives (Predicted Default, Actual Current)\n\n")
        report_lines.append(pd.DataFrame(fp_examples).to_markdown(index=False) + "\n\n")
        
    if fn_examples:
        report_lines.append("### Top False Negatives (Predicted Current, Actual Default)\n\n")
        report_lines.append(pd.DataFrame(fn_examples).to_markdown(index=False) + "\n\n")
        
    with open(os.path.join(REPORTS_DIR, "explainability_report.md"), "w") as f:
        f.write("".join(report_lines))
        
    print("Explainability analysis complete.")

if __name__ == "__main__":
    run()
