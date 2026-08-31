import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss, precision_recall_curve
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = "data/processed"
MODELS_DIR = os.path.join(PROCESSED_DIR, "models")
REPORTS_DIR = "reports"

def recall_at_precision(y_true, y_prob, target_precision=0.80):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # Find the smallest threshold where precision is >= target_precision
    valid_idx = np.where(precision >= target_precision)[0]
    if len(valid_idx) == 0:
        return 0.0
    return np.max(recall[valid_idx])

def get_lr_preds(model_path, valid_df, features):
    if not os.path.exists(model_path):
        return None
    model = joblib.load(model_path)
    
    # Needs scaling and one-hot encoding
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(valid_df[c])]
    cat_cols = [c for c in features if c not in num_cols]
    
    df = valid_df.copy()
    medians = df[num_cols].median() # Using valid medians for simplicity in report gen
    df[num_cols] = df[num_cols].fillna(medians)
    
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)
        # load lr features
        with open(os.path.join(PROCESSED_DIR, "lr_features.csv"), "r") as f:
            lr_feats = f.read().splitlines()
        df = df.reindex(columns=lr_feats, fill_value=0)
    else:
        lr_feats = num_cols
        
    X = df[lr_feats]
    
    if len(model.classes_) == 2:
        return model.predict_proba(X)[:, 1]
    else:
        return model.predict(X)

def get_optimal_f1(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores)
    return f1_scores[best_idx]

def run():
    print("Generating §12 Metrics Report...")
    valid_df = pd.read_csv(os.path.join(PROCESSED_DIR, "valid.csv"))
    
    with open(os.path.join(PROCESSED_DIR, "features.csv"), "r") as f:
        features = f.read().splitlines()
        
    cat_cols = [c for c in features if valid_df[c].dtype == 'O' or valid_df[c].nunique() < 20 and not pd.api.types.is_numeric_dtype(valid_df[c])]
    for col in cat_cols:
        valid_df[col] = valid_df[col].astype('category')
        
    X_valid_lgb = valid_df[features]
    
    binary_targets = ['next_3m_delinquency_flag', 'next_6m_delinquency_flag', 'next_12m_default_flag', 'next_12m_prepayment_flag']
    
    report_lines = ["# FR-2 Model Performance Report (§12 Metrics)\n\n"]
    
    for target in binary_targets:
        report_lines.append(f"### Target: `{target}`\n")
        report_lines.append("| Model | ROC-AUC | PR-AUC | F1 Score | Recall @ 80% Precision | Brier Score |\n")
        report_lines.append("|---|---|---|---|---|---|\n")
        
        y_valid = valid_df[target]
        if len(np.unique(y_valid)) < 2:
            report_lines.append("Skipped due to single class in validation set.\n\n")
            continue
            
        # 1. LR Baseline
        lr_path = os.path.join(MODELS_DIR, f"{target}_lr_model.joblib")
        lr_probs = get_lr_preds(lr_path, valid_df, features)
        if lr_probs is not None:
            roc = roc_auc_score(y_valid, lr_probs)
            pr = average_precision_score(y_valid, lr_probs)
            f1 = get_optimal_f1(y_valid, lr_probs)
            rec80 = recall_at_precision(y_valid, lr_probs, 0.80)
            brier = brier_score_loss(y_valid, lr_probs)
            report_lines.append(f"| Logistic Regression (Baseline) | {roc:.4f} | {pr:.4f} | {f1:.4f} | {rec80:.4f} | {brier:.4f} |\n")
            
        # 2. Improved LightGBM (Calibrated)
        lgb_path = os.path.join(MODELS_DIR, f"{target}_calibrated.joblib")
        if os.path.exists(lgb_path):
            model = joblib.load(lgb_path)
            lgb_probs = model.predict_proba(X_valid_lgb)[:, 1]
            roc = roc_auc_score(y_valid, lgb_probs)
            pr = average_precision_score(y_valid, lgb_probs)
            f1 = get_optimal_f1(y_valid, lgb_probs)
            rec80 = recall_at_precision(y_valid, lgb_probs, 0.80)
            brier = brier_score_loss(y_valid, lgb_probs)
            report_lines.append(f"| LightGBM Tuned+Calibrated (Improved) | **{roc:.4f}** | **{pr:.4f}** | **{f1:.4f}** | **{rec80:.4f}** | **{brier:.4f}** |\n")
            
        report_lines.append("\n")
        
    # Multi-class
    target = 'next_state'
    report_lines.append(f"### Target: `{target}` (Multi-class)\n")
    report_lines.append("| Model | Accuracy | Macro-F1 |\n")
    report_lines.append("|---|---|---|\n")
    y_valid = valid_df[target]
    
    # 1. LR
    lr_path = os.path.join(MODELS_DIR, f"next_state_lr_model.joblib")
    lr_preds = get_lr_preds(lr_path, valid_df, features)
    if lr_preds is not None:
        acc = (lr_preds == y_valid).mean()
        mf1 = f1_score(y_valid, lr_preds, average='macro')
        report_lines.append(f"| Logistic Regression (Baseline) | {acc:.4f} | {mf1:.4f} |\n")
        
    # 2. LGB
    lgb_path = os.path.join(MODELS_DIR, f"next_state_lgb_improved.joblib")
    if os.path.exists(lgb_path):
        model = joblib.load(lgb_path)
        lgb_preds = model.predict(X_valid_lgb)
        acc = (lgb_preds == y_valid).mean()
        mf1 = f1_score(y_valid, lgb_preds, average='macro')
        report_lines.append(f"| LightGBM Tuned (Improved) | **{acc:.4f}** | **{mf1:.4f}** |\n\n")
        
        # Add per-class precision/recall and confusion matrix
        from sklearn.metrics import classification_report, confusion_matrix
        
        report_lines.append("#### Per-Class Classification Report (LightGBM)\n```text\n")
        report_lines.append(classification_report(y_valid, lgb_preds, zero_division=0))
        report_lines.append("\n```\n\n#### Confusion Matrix\n")
        
        classes = sorted(np.unique(y_valid))
        cm = confusion_matrix(y_valid, lgb_preds, labels=classes)
        cm_df = pd.DataFrame(cm, index=[f"True {c}" for c in classes], columns=[f"Pred {c}" for c in classes])
        report_lines.append(cm_df.to_markdown())
        report_lines.append("\n\n")
        
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, "model_performance_report.md")
    with open(out_path, "w") as f:
        f.write("".join(report_lines))
        
    print(f"Metrics report saved to {out_path}")

if __name__ == "__main__":
    run()
