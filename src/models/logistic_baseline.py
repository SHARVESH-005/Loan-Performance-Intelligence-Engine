import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import joblib

PROCESSED_DIR = "data/processed"
MODELS_DIR = os.path.join(PROCESSED_DIR, "models")

def run():
    print("Training Logistic Regression Baselines...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    valid_df = pd.read_csv(os.path.join(PROCESSED_DIR, "valid.csv"))
    
    with open(os.path.join(PROCESSED_DIR, "features.csv"), "r") as f:
        features = f.read().splitlines()
        
    binary_targets = ['next_3m_delinquency_flag', 'next_6m_delinquency_flag', 'next_12m_default_flag', 'next_12m_prepayment_flag']
    keep_cols = features + binary_targets + ['next_state']
    
    train_df = train_df[keep_cols]
    valid_df = valid_df[keep_cols]
        
    # Logistic regression requires numeric inputs, no NaNs
    # For baseline, we'll fill NaN with median and one-hot encode categoricals
    print("Preprocessing for Logistic Regression...")
    
    # Identify cat/num
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(train_df[c])]
    cat_cols = [c for c in features if c not in num_cols]
    
    # Impute missing numeric with median
    medians = train_df[num_cols].median()
    train_df[num_cols] = train_df[num_cols].fillna(medians)
    valid_df[num_cols] = valid_df[num_cols].fillna(medians)
    
    # Scale numeric
    scaler = StandardScaler()
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    valid_df[num_cols] = scaler.transform(valid_df[num_cols])
    
    # One-hot encode cat_cols
    if cat_cols:
        train_df = pd.get_dummies(train_df, columns=cat_cols, dummy_na=True)
        valid_df = pd.get_dummies(valid_df, columns=cat_cols, dummy_na=True)
        
        # Align columns
        model_features = [c for c in train_df.columns if c not in ['loan_id', 'reporting_month'] and not c.startswith('next_')]
        valid_df = valid_df.reindex(columns=train_df.columns, fill_value=0)
    else:
        model_features = num_cols
        
    X_train = train_df[model_features]
    X_valid = valid_df[model_features]
    
    binary_targets = ['next_3m_delinquency_flag', 'next_6m_delinquency_flag', 'next_12m_default_flag', 'next_12m_prepayment_flag']
    
    for target in binary_targets:
        print(f"Training LR for {target}...")
        y_train = train_df[target]
        y_valid = valid_df[target]
        
        model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict_proba(X_valid)[:, 1]
        
        auc = roc_auc_score(y_valid, preds)
        pr_auc = average_precision_score(y_valid, preds)
        
        print(f"[{target}] LR Validation ROC-AUC: {auc:.4f}, PR-AUC: {pr_auc:.4f}")
        joblib.dump(model, os.path.join(MODELS_DIR, f"{target}_lr_model.joblib"))
        
    # Multi-class
    print(f"Training LR for next_state...")
    y_train = train_df['next_state']
    y_valid = valid_df['next_state']
    
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    acc = model.score(X_valid, y_valid)
    preds_cls = model.predict(X_valid)
    mf1 = f1_score(y_valid, preds_cls, average='macro')
    
    print(f"[next_state] LR Validation Accuracy: {acc:.4f}, Macro-F1: {mf1:.4f}")
    joblib.dump(model, os.path.join(MODELS_DIR, "next_state_lr_model.joblib"))
    
    # Save the feature list for LR (since it's one-hot encoded and different from LightGBM)
    with open(os.path.join(PROCESSED_DIR, "lr_features.csv"), "w") as f:
        f.write("\n".join(model_features))
        
    print("Logistic Regression Baselines complete.")

if __name__ == "__main__":
    run()
