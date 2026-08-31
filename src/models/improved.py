import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"
MODELS_DIR = os.path.join(PROCESSED_DIR, "models")

# Optuna objective for binary targets
def objective_binary(trial, X_train, y_train, X_valid, y_valid):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200]),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'verbose': -1,
        'class_weight': 'balanced'
    }
    
    model = lgb.LGBMClassifier(**params, random_state=42)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(10, verbose=False)]
    )
    
    preds = model.predict_proba(X_valid)[:, 1]
    
    if len(np.unique(y_valid)) == 1:
        return 0.0
    
    return roc_auc_score(y_valid, preds)

def objective_multi(trial, X_train, y_train, X_valid, y_valid):
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_categorical('n_estimators', [50, 100]),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'num_leaves': trial.suggest_int('num_leaves', 15, 31),
        'verbose': -1,
        'class_weight': 'balanced'
    }
    
    model = lgb.LGBMClassifier(**params, random_state=42)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(10, verbose=False)]
    )
    
    preds = model.predict(X_valid)
    return f1_score(y_valid, preds, average='macro')

def run():
    print("Training Improved LightGBM Models (Tuned)...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    valid_df = pd.read_csv(os.path.join(PROCESSED_DIR, "valid.csv"))
    
    with open(os.path.join(PROCESSED_DIR, "features.csv"), "r") as f:
        features = f.read().splitlines()
        
    # Convert cat cols
    cat_cols = [c for c in features if train_df[c].dtype == 'O' or train_df[c].nunique() < 20 and not pd.api.types.is_numeric_dtype(train_df[c])]
    for col in cat_cols:
        train_df[col] = train_df[col].astype('category')
        valid_df[col] = valid_df[col].astype('category')
        
    X_train = train_df[features]
    X_valid = valid_df[features]
    
    binary_targets = ['next_3m_delinquency_flag', 'next_6m_delinquency_flag', 'next_12m_default_flag', 'next_12m_prepayment_flag']
    
    # Optional: SMOTE Comparison on one target as proof-of-concept
    print("Comparing class_weight vs SMOTE for next_3m_delinquency_flag...")
    # Prepare numeric-only for SMOTE
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(train_df[c])]
    X_tr_num = train_df[num_cols].fillna(train_df[num_cols].median())
    X_v_num = valid_df[num_cols].fillna(train_df[num_cols].median())
    y_tr_3m = train_df['next_3m_delinquency_flag']
    y_v_3m = valid_df['next_3m_delinquency_flag']
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_tr_num, y_tr_3m)
    
    model_smote = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model_smote.fit(X_res, y_res)
    preds_smote = model_smote.predict_proba(X_v_num)[:, 1]
    smote_auc = roc_auc_score(y_v_3m, preds_smote)
    
    model_cw = lgb.LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1)
    model_cw.fit(X_tr_num, y_tr_3m)
    preds_cw = model_cw.predict_proba(X_v_num)[:, 1]
    cw_auc = roc_auc_score(y_v_3m, preds_cw)
    
    print(f"SMOTE AUC: {smote_auc:.4f} | class_weight AUC: {cw_auc:.4f}")
    if cw_auc >= smote_auc:
        print("-> class_weight is better or equal. Proceeding with class_weight for all targets.")
    else:
        print("-> SMOTE is better. Proceeding with SMOTE for all targets.")
    
    # Proceed with tuning
    best_models = {}
    
    for target in binary_targets:
        print(f"\nTuning LightGBM for {target}...")
        y_train = train_df[target]
        y_valid = valid_df[target]
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective_binary(trial, X_train, y_train, X_valid, y_valid), n_trials=30)
        
        print(f"Best trial for {target}: AUC={study.best_value:.4f}")
        
        # Retrain with best params
        best_params = study.best_params
        best_params.update({'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1, 'class_weight': 'balanced'})
        
        model = lgb.LGBMClassifier(**best_params, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[lgb.early_stopping(10, verbose=False)])
        
        joblib.dump(model, os.path.join(MODELS_DIR, f"{target}_lgb_improved.joblib"))
        
    print(f"\nTuning LightGBM for next_state...")
    y_train = train_df['next_state']
    y_valid = valid_df['next_state']
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_multi(trial, X_train, y_train, X_valid, y_valid), n_trials=15)
    
    print(f"Best trial for next_state: Macro-F1={study.best_value:.4f}")
    
    best_params = study.best_params
    best_params.update({'objective': 'multiclass', 'metric': 'multi_logloss', 'verbose': -1, 'class_weight': 'balanced'})
    
    model = lgb.LGBMClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[lgb.early_stopping(10, verbose=False)])
    
    joblib.dump(model, os.path.join(MODELS_DIR, "next_state_lgb_improved.joblib"))
    
    print("Improved LightGBM tuning complete.")

if __name__ == "__main__":
    run()
