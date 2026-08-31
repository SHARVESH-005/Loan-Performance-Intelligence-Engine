import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from mlxtend.frequent_patterns import apriori, association_rules

PROCESSED_DIR = "data/processed"
REPORT_DIR = "reports"

def calculate_psi(expected, actual, buckets=10):
    # Handle empty or zero-variance inputs
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
        
    min_val = min(np.min(expected), np.min(actual))
    max_val = max(np.max(expected), np.max(actual))
    
    if min_val == max_val:
        return 0.0  # No variance, no drift
        
    breakpoints = np.linspace(min_val, max_val, buckets + 1)
    
    # Add small epsilon to edge bins to ensure inclusivity
    breakpoints[0] -= 1e-5
    breakpoints[-1] += 1e-5
    
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    
    def sub_psi(e_perc, a_perc):
        if a_perc == 0:
            a_perc = 0.0001
        if e_perc == 0:
            e_perc = 0.0001
        return (e_perc - a_perc) * np.log(e_perc / a_perc)
    
    psi_value = sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(len(expected_percents)))
    return psi_value

def profile_drift(train_df, test_df, numeric_cols):
    report_lines = ["\n### Train vs. Test Drift\n"]
    drift_records = []
    
    for col in numeric_cols:
        tr_data = train_df[col].dropna()
        te_data = test_df[col].dropna()
        if len(tr_data) == 0 or len(te_data) == 0:
            continue
            
        ks_stat, p_value = ks_2samp(tr_data, te_data)
        try:
            psi_val = calculate_psi(tr_data.values, te_data.values)
        except:
            psi_val = np.nan
            
        drift_records.append({
            'Feature': col,
            'KS Stat': ks_stat,
            'KS p-value': p_value,
            'PSI': psi_val
        })
        
    drift_df = pd.DataFrame(drift_records)
    if len(drift_df) > 0:
        drift_df['High Drift (PSI>0.2)'] = drift_df['PSI'] > 0.2
        report_lines.append(drift_df.round(4).to_markdown())
    else:
        report_lines.append("No numeric columns available for drift calculation.")
        
    report_lines.append("\n")
    return "".join(report_lines)

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    if min((kcorr-1), (rcorr-1)) == 0:
        return 0
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def profile_correlations(df, numeric_cols, cat_cols):
    report_lines = ["\n### Correlation & Association\n"]
    
    # Numeric (Pearson)
    report_lines.append("#### Top Numeric Correlations (Pearson)\n")
    corr = df[numeric_cols].corr()
    corr_unstacked = corr.unstack().reset_index()
    corr_unstacked.columns = ['Feature 1', 'Feature 2', 'Correlation']
    corr_unstacked = corr_unstacked[corr_unstacked['Feature 1'] != corr_unstacked['Feature 2']]
    corr_unstacked['Abs_Corr'] = corr_unstacked['Correlation'].abs()
    top_corr = corr_unstacked.sort_values(by='Abs_Corr', ascending=False).drop_duplicates(subset=['Abs_Corr']).head(5)
    report_lines.append(top_corr[['Feature 1', 'Feature 2', 'Correlation']].round(3).to_markdown(index=False))
    report_lines.append("\n\n")
    
    # Numeric (Spearman)
    report_lines.append("#### Top Numeric Rank Correlations (Spearman)\n")
    corr_sp = df[numeric_cols].corr(method='spearman')
    corr_sp_unstacked = corr_sp.unstack().reset_index()
    corr_sp_unstacked.columns = ['Feature 1', 'Feature 2', 'Correlation']
    corr_sp_unstacked = corr_sp_unstacked[corr_sp_unstacked['Feature 1'] != corr_sp_unstacked['Feature 2']]
    corr_sp_unstacked['Abs_Corr'] = corr_sp_unstacked['Correlation'].abs()
    top_corr_sp = corr_sp_unstacked.sort_values(by='Abs_Corr', ascending=False).drop_duplicates(subset=['Abs_Corr']).head(5)
    report_lines.append(top_corr_sp[['Feature 1', 'Feature 2', 'Correlation']].round(3).to_markdown(index=False))
    report_lines.append("\n\n")
    
    # Categorical
    report_lines.append("#### Top Categorical Associations (Cramér's V)\n")
    cat_assoc = []
    cat_cols_clean = [c for c in cat_cols if df[c].nunique() > 1 and df[c].nunique() < 20]
    for i in range(len(cat_cols_clean)):
        for j in range(i+1, len(cat_cols_clean)):
            c1, c2 = cat_cols_clean[i], cat_cols_clean[j]
            v = cramers_v(df[c1], df[c2])
            cat_assoc.append({'Feature 1': c1, 'Feature 2': c2, "Cramér's V": v})
            
    if cat_assoc:
        cat_df = pd.DataFrame(cat_assoc).sort_values(by="Cramér's V", ascending=False).head(5)
        report_lines.append(cat_df.round(3).to_markdown(index=False))
    else:
        report_lines.append("Not enough categorical features to compute associations.")
    report_lines.append("\n")
    
    return "".join(report_lines)

def profile_outliers(df, numeric_cols):
    report_lines = ["\n### Outlier Detection\n"]
    
    key_cols = [c for c in ['original_balance', 'current_balance', 'interest_rate', 'days_past_due'] if c in df.columns]
    
    report_lines.append("#### Univariate Outliers (IQR method)\n")
    outlier_counts = []
    for col in key_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_counts.append({'Feature': col, 'Outlier Count': outliers, '% Outliers': (outliers/len(df))*100})
        
    out_df = pd.DataFrame(outlier_counts)
    report_lines.append(out_df.round(2).to_markdown(index=False))
    report_lines.append("\n\n")
    
    report_lines.append("#### Multivariate Outliers (Isolation Forest)\n")
    if len(key_cols) > 0:
        data_clean = df[key_cols].fillna(df[key_cols].median())
        clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        preds = clf.fit_predict(data_clean)
        num_outliers = (preds == -1).sum()
        report_lines.append(f"- Isolation Forest detected **{num_outliers}** outliers ({(num_outliers/len(df))*100:.2f}% of training data) across `{key_cols}`.\n")
    
    return "".join(report_lines)

def compute_quality_scores(df):
    report_lines = ["\n### Data Quality Scores\n"]
    
    # 1. Completeness (fraction of non-nulls)
    completeness = 1 - (df.isnull().sum(axis=1) / df.shape[1])
    
    # 2. Validity (basic logic)
    validity = pd.Series(1.0, index=df.index)
    if 'origination_month' in df.columns and 'reporting_month' in df.columns:
        invalid_dates = pd.to_datetime(df['origination_month']) > pd.to_datetime(df['reporting_month'])
        validity.loc[invalid_dates] -= 0.5
        
    # 3. Consistency (DPD vs Status)
    consistency = pd.Series(1.0, index=df.index)
    if 'current_status' in df.columns and 'days_past_due' in df.columns:
        inconsistent = (df['current_status'] == 'Current') & (df['days_past_due'] > 0)
        consistency.loc[inconsistent] -= 0.5
        
    # Ensure bounds [0, 1]
    validity = validity.clip(0, 1)
    consistency = consistency.clip(0, 1)
    
    # Composite Score
    df['data_quality_score'] = (completeness * 0.4) + (validity * 0.3) + (consistency * 0.3)
    
    report_lines.append(f"- **Overall Mean Data Quality Score:** {df['data_quality_score'].mean():.4f}\n")
    
    # Rollup by state
    if 'state' in df.columns:
        rollup = df.groupby('state')['data_quality_score'].mean().reset_index().sort_values(by='data_quality_score')
        report_lines.append("\n#### Bottom 5 States by Data Quality\n")
        report_lines.append(rollup.head(5).to_markdown(index=False))
        
    report_lines.append("\n")
    return "".join(report_lines)
    
def profile_association_rules(df):
    report_lines = ["\n### Association-Rule Mining (Apriori)\n"]
    cat_cols = [c for c in ['state', 'current_status', 'loan_purpose', 'occupancy_type'] if c in df.columns]
    
    if len(cat_cols) > 0:
        # Convert to one-hot for apriori
        df_cat = pd.get_dummies(df[cat_cols].dropna())
        try:
            frequent_itemsets = apriori(df_cat, min_support=0.05, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.1)
            
            # Filter trivial rules (we want RHS to be 1 item)
            rules['rhs_len'] = rules['consequents'].apply(lambda x: len(x))
            rules = rules[rules['rhs_len'] == 1]
            
            top_rules = rules.sort_values(by='lift', ascending=False).head(5)
            
            if not top_rules.empty:
                report_lines.append("Top 5 Association Rules by Lift:\n")
                
                fmt_rules = []
                for _, row in top_rules.iterrows():
                    lhs = ", ".join(list(row['antecedents']))
                    rhs = ", ".join(list(row['consequents']))
                    fmt_rules.append({'Rule': f"If {lhs} -> {rhs}", 'Support': row['support'], 'Confidence': row['confidence'], 'Lift': row['lift']})
                    
                report_lines.append(pd.DataFrame(fmt_rules).round(3).to_markdown(index=False))
            else:
                report_lines.append("No strong association rules found.")
        except Exception as e:
            report_lines.append(f"Could not compute association rules: {str(e)}")
            
    report_lines.append("\n")
    return "".join(report_lines)

def run():
    print("Running advanced profiling (FR-1 Deepening)...")
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    
    report = ["\n---\n## Day 2 Advanced Profiling Extensions\n"]
    
    with open(os.path.join(PROCESSED_DIR, "features.csv"), "r") as f:
        features = f.read().splitlines()
        
    numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(train_df[c])]
    cat_cols = [c for c in features if c not in numeric_cols or train_df[c].nunique() < 20]
    
    report.append(profile_drift(train_df, test_df, numeric_cols))
    report.append(profile_correlations(train_df, numeric_cols, cat_cols))
    report.append(profile_outliers(train_df, numeric_cols))
    
    # Reload raw data for apriori and quality scoring as they need strings/dates
    raw_static = pd.read_csv(os.path.join("data", "raw", "loan_static_attributes.csv"))
    raw_panel = pd.read_csv(os.path.join("data", "raw", "loan_monthly_performance_train.csv"))
    raw_full = pd.merge(raw_panel, raw_static, on="loan_id", how="left", suffixes=("", "_static"))
    
    report.append(profile_association_rules(raw_full))
    report.append(compute_quality_scores(raw_full))
    
    # Append to existing report without duplicating
    report_path = os.path.join(REPORT_DIR, "data_intelligence_report.md")
    
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            existing_content = f.read()
        
        # Strip existing Day 2 section if present
        day2_header = "\n---\n## Day 2 Advanced Profiling Extensions\n"
        if day2_header in existing_content:
            existing_content = existing_content.split(day2_header)[0]
            
        with open(report_path, "w") as f:
            f.write(existing_content + "".join(report))
    else:
        with open(report_path, "w") as f:
            f.write("".join(report))
        
    print(f"Advanced profiling results appended to {report_path}")

if __name__ == "__main__":
    run()
