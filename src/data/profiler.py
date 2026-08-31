import os
import pandas as pd
import numpy as np

DATA_DIR = "data/raw"
REPORT_DIR = "reports"

def load_data():
    static_df = pd.read_csv(os.path.join(DATA_DIR, "loan_static_attributes.csv"))
    panel_df = pd.read_csv(os.path.join(DATA_DIR, "loan_monthly_performance_train.csv"))
    
    # Merge static attributes into panel for a complete view
    full_df = pd.merge(panel_df, static_df, on="loan_id", how="left", suffixes=("", "_static"))
    
    # Drop duplicated columns from merge if any
    cols_to_drop = [c for c in full_df.columns if c.endswith("_static")]
    full_df.drop(columns=cols_to_drop, inplace=True)
    
    return static_df, panel_df, full_df

def profile_distributions(df):
    report_lines = ["### Distributions\n"]
    
    target_cols = ['next_3m_delinquency_flag', 'next_6m_delinquency_flag', 'next_12m_default_flag', 'next_12m_prepayment_flag', 'next_state']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in target_cols]
    
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    cat_cols = [c for c in cat_cols if c not in target_cols]
    
    report_lines.append("#### Numeric Feature Fields\n")
    if len(numeric_cols) > 0:
        desc = df[numeric_cols].describe(percentiles=[.05, .25, .5, .75, .95]).T
        desc['skew'] = df[numeric_cols].skew()
        report_lines.append(desc[['mean', '50%', 'std', 'min', 'max', '5%', '95%', 'skew']].round(3).to_markdown())
        report_lines.append("\n\n")
        
    report_lines.append("#### Categorical Feature Fields\n")
    for col in cat_cols:
        val_counts = df[col].value_counts(dropna=False)
        cardinality = len(val_counts)
        top5 = val_counts.head(5).to_dict()
        rare_share = (val_counts[val_counts < len(df) * 0.01].sum() / len(df)) * 100
        
        report_lines.append(f"**{col}**: {cardinality} unique values, {rare_share:.1f}% rare categories (<1%). Top 5: {top5}\n\n")
        
    report_lines.append("#### Target Fields\n")
    for col in [c for c in target_cols if c in df.columns]:
        val_counts = df[col].value_counts(dropna=False, normalize=True) * 100
        report_lines.append(f"**{col}**: {val_counts.to_dict()}\n\n")
        
    return "".join(report_lines)

def profile_missingness(df):
    report_lines = ["### Missingness\n"]
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing %', ascending=False)
    
    if len(missing_df) > 0:
        report_lines.append(missing_df.round(2).to_markdown())
    else:
        report_lines.append("No missing values found across all columns.")
        
    report_lines.append("\n\n")
    return "".join(report_lines)

def run_basic_checks(df):
    report_lines = ["### Basic Date & Logic Checks\n"]
    
    if 'origination_month' in df.columns and 'reporting_month' in df.columns:
        df['orig_month_dt'] = pd.to_datetime(df['origination_month'])
        df['rep_month_dt'] = pd.to_datetime(df['reporting_month'])
        violations = (df['orig_month_dt'] > df['rep_month_dt']).sum()
        report_lines.append(f"- `origination_month > reporting_month` violations: **{violations}**\n")
        
    if 'current_balance' in df.columns and 'original_balance' in df.columns:
        violations = (df['current_balance'] > df['original_balance'] * 1.05).sum()
        report_lines.append(f"- `current_balance > original_balance` (by >5%) violations: **{violations}**\n")
        
    if 'days_past_due' in df.columns and 'current_status' in df.columns:
        # Check simple inconsistency
        # E.g. current status is Current, but DPD > 0
        violations = ((df['current_status'] == 'Current') & (df['days_past_due'] > 0)).sum()
        report_lines.append(f"- Status 'Current' but DPD > 0 violations: **{violations}**\n")
        
    report_lines.append("\n\n")
    return "".join(report_lines)

def generate_report():
    print("Loading data for profiling...")
    _, _, full_df = load_data()
    
    print("Generating profiling report...")
    report = ["# Data Intelligence & Profiling Report (FR-1 minimal)\n\n"]
    
    report.append(profile_missingness(full_df))
    report.append(profile_distributions(full_df))
    report.append(run_basic_checks(full_df))
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "data_intelligence_report.md")
    
    with open(report_path, "w") as f:
        f.write("".join(report))
        
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    generate_report()
