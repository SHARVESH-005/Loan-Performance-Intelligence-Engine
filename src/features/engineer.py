import os
import pandas as pd
import numpy as np

DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def load_data():
    static_df = pd.read_csv(os.path.join(DATA_DIR, "loan_static_attributes.csv"))
    train_df = pd.read_csv(os.path.join(DATA_DIR, "loan_monthly_performance_train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "loan_monthly_performance_test.csv"))
    
    # Combine train and test temporarily for consistent feature engineering
    panel_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Drop overlapping columns from static_df before merging (except loan_id)
    cols_to_drop = [c for c in static_df.columns if c in panel_df.columns and c != 'loan_id']
    static_df_clean = static_df.drop(columns=cols_to_drop)
    
    # Merge static attributes
    full_df = pd.merge(panel_df, static_df_clean, on="loan_id", how="left")
    
    return full_df

def engineer_features(df):
    print("Engineering features...")
    # Numeric features
    df['balance_ratio'] = df['current_balance'] / (df['original_balance'] + 1e-6)
    
    median_rate = df['interest_rate'].median()
    df['rate_spread'] = df['interest_rate'] - median_rate
    
    # Dates
    df['reporting_month_dt'] = pd.to_datetime(df['reporting_month'])
    df['last_updated_at_dt'] = pd.to_datetime(df['last_updated_at'])
    df['months_since_last_update'] = (df['reporting_month_dt'].dt.year - df['last_updated_at_dt'].dt.year) * 12 + (df['reporting_month_dt'].dt.month - df['last_updated_at_dt'].dt.month)
    
    # Ordinal encodings
    credit_map = {"<600": 0, "600-649": 1, "650-699": 2, "700-749": 3, "750-799": 4, "800+": 5}
    df['credit_score_band_ord'] = df['credit_score_band'].map(credit_map)
    
    ltv_map = {"<60%": 0, "60-70%": 1, "70-80%": 2, "80-90%": 3, "90-95%": 4, ">95%": 5}
    df['ltv_band_ord'] = df['ltv_band'].map(ltv_map)
    
    dti_map = {"<30%": 0, "30-40%": 1, "40-50%": 2, ">50%": 3}
    df['dti_band_ord'] = df['dti_band'].map(dti_map)
    
    # Lag features (per-loan, sorted by time) — critical for month-to-month differentiation
    df = df.sort_values(['loan_id', 'reporting_month_dt']).reset_index(drop=True)
    df['prev_balance'] = df.groupby('loan_id')['current_balance'].shift(1)
    df['balance_change'] = df['current_balance'] - df['prev_balance']
    df['prev_dpd'] = df.groupby('loan_id')['days_past_due'].shift(1)
    df['dpd_change'] = df['days_past_due'] - df['prev_dpd']
    
    # Fill NaN for first observation of each loan (no previous month exists)
    df['prev_balance'] = df['prev_balance'].fillna(df['original_balance'])
    df['balance_change'] = df['balance_change'].fillna(0)
    df['prev_dpd'] = df['prev_dpd'].fillna(0)
    df['dpd_change'] = df['dpd_change'].fillna(0)
    
    # Categorical encodings (One-hot for models that need it, or we leave as categorical for LightGBM)
    # LightGBM handles categorical natively, so we just convert to category dtype
    cat_cols = ['state', 'loan_purpose', 'occupancy_type', 'property_type', 'servicer_name', 'source_system', 'document_status']
    for col in cat_cols:
        df[col] = df[col].astype('category')
        
    return df

def perform_time_split(df):
    print("Performing time-aware split...")
    t1 = pd.to_datetime("2024-06-01")
    t2 = pd.to_datetime("2024-09-01")
    
    train_mask = df['reporting_month_dt'] <= t1
    valid_mask = (df['reporting_month_dt'] > t1) & (df['reporting_month_dt'] <= t2)
    test_mask = df['reporting_month_dt'] > t2
    
    train_df = df[train_mask].copy()
    valid_df = df[valid_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"Train rows: {len(train_df)}, Valid rows: {len(valid_df)}, Test rows: {len(test_df)}")
    return train_df, valid_df, test_df

def check_leakage(df):
    print("Running leakage audit...")
    # Ensure targets are not used as features inadvertently
    target_cols = [
        'next_3m_delinquency_flag', 'next_6m_delinquency_flag', 
        'next_12m_default_flag', 'next_12m_prepayment_flag', 
        'next_state', 'exception_required', 'exception_type'
    ]
    
    # The current_status and default/prepayment flags from current month are valid as features 
    # (they represent current state), but predicting NEXT state shouldn't include next state labels.
    
    exclude_cols = ['loan_id', 'reporting_month', 'origination_month', 'last_updated_at', 'reporting_month_dt', 'last_updated_at_dt', 
                    'credit_score_band', 'ltv_band', 'dti_band', 'current_status', 'loss_severity_band',
                    'prepayment_flag', 'default_flag', 'modification_flag', 'source_system']
    features = [c for c in df.columns if c not in target_cols and c not in exclude_cols]
    
    # Also ensure we only pass numeric/bool/category types to lightgbm
    valid_dtypes = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'bool', 'category']
    features = [c for c in features if df[c].dtype.name in valid_dtypes or pd.api.types.is_numeric_dtype(df[c])]
    
    return features, target_cols

def run():
    full_df = load_data()
    full_df = engineer_features(full_df)
    
    features, target_cols = check_leakage(full_df)
    
    train_df, valid_df, test_df = perform_time_split(full_df)
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    valid_df.to_csv(os.path.join(PROCESSED_DIR, "valid.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    
    # Save feature list for models to use
    pd.Series(features).to_csv(os.path.join(PROCESSED_DIR, "features.csv"), index=False, header=False)
    pd.Series(target_cols).to_csv(os.path.join(PROCESSED_DIR, "targets.csv"), index=False, header=False)
    
    print(f"Processed data saved to {PROCESSED_DIR}")

if __name__ == "__main__":
    run()
