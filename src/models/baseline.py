import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import joblib

PROCESSED_DIR = "data/processed"
MODELS_DIR = "data/processed/models"

def load_data():
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    valid_df = pd.read_csv(os.path.join(PROCESSED_DIR, "valid.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    
    # Read features
    with open(os.path.join(PROCESSED_DIR, "features.csv"), "r") as f:
        features = [line.strip() for line in f.readlines()]
        
    # Categorical columns need to be of type category for lightgbm
    cat_cols = ['state', 'loan_purpose', 'occupancy_type', 'property_type', 'servicer_name', 'source_system', 'document_status']
    cat_cols = [c for c in cat_cols if c in features]
    
    for df in [train_df, valid_df, test_df]:
        for col in cat_cols:
            df[col] = df[col].astype('category')
            
    return train_df, valid_df, test_df, features, cat_cols

def train_binary_model(target, train_df, valid_df, features, cat_cols):
    print(f"Training baseline for {target}...")
    X_train, y_train = train_df[features], train_df[target]
    X_valid, y_valid = valid_df[features], valid_df[target]
    
    clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        class_weight='balanced', # Crucial for rare events
        random_state=42
    )
    
    clf.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )
    
    preds_prob = clf.predict_proba(X_valid)[:, 1]
    preds = clf.predict(X_valid)
    
    try:
        auc = roc_auc_score(y_valid, preds_prob)
    except ValueError:
        auc = float('nan')
        
    try:
        f1 = f1_score(y_valid, preds)
    except ValueError:
        f1 = float('nan')
        
    print(f"[{target}] Validation ROC-AUC: {auc:.4f}, F1: {f1:.4f}")
    
    return clf

def train_multiclass_model(target, train_df, valid_df, features, cat_cols):
    print(f"Training baseline for {target} (multi-class)...")
    
    # Convert string states to integers for LGBM
    classes = train_df[target].unique()
    class_map = {c: i for i, c in enumerate(classes)}
    
    y_train = train_df[target].map(class_map)
    y_valid = valid_df[target].map(class_map)
    X_train = train_df[features]
    X_valid = valid_df[features]
    
    clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        objective='multiclass',
        class_weight='balanced',
        random_state=42
    )
    
    clf.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )
    
    preds = clf.predict(X_valid)
    acc = accuracy_score(y_valid, preds)
    f1 = f1_score(y_valid, preds, average='macro')
    print(f"[{target}] Validation Accuracy: {acc:.4f}, Macro-F1: {f1:.4f}")
    
    return clf, class_map

def run():
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_df, valid_df, test_df, features, cat_cols = load_data()
    
    targets = [
        'next_3m_delinquency_flag', 'next_6m_delinquency_flag', 
        'next_12m_default_flag', 'next_12m_prepayment_flag'
    ]
    
    models = {}
    for target in targets:
        models[target] = train_binary_model(target, train_df, valid_df, features, cat_cols)
        joblib.dump(models[target], os.path.join(MODELS_DIR, f"{target}_baseline.pkl"))
        
        # Save test probabilities for submission
        test_df[f"{target}_prob"] = models[target].predict_proba(test_df[features])[:, 1]
        
    # Multi-class target
    next_state_model, class_map = train_multiclass_model('next_state', train_df, valid_df, features, cat_cols)
    joblib.dump({'model': next_state_model, 'class_map': class_map}, os.path.join(MODELS_DIR, "next_state_baseline.pkl"))
    
    # Save predicted state names for test set
    inv_class_map = {v: k for k, v in class_map.items()}
    test_preds = next_state_model.predict(test_df[features])
    test_df['predicted_next_state'] = [inv_class_map[p] for p in test_preds]
    test_df['next_state_confidence'] = np.max(next_state_model.predict_proba(test_df[features]), axis=1)
    
    # Save back test.csv with predictions attached so submission script can just format it
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test_with_preds.csv"), index=False)
    print(f"Models and test predictions saved to {MODELS_DIR} and {PROCESSED_DIR}/test_with_preds.csv")

if __name__ == "__main__":
    run()
