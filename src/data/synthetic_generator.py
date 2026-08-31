import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = "data/raw"

def generate_static_attributes(num_loans=5000):
    np.random.seed(42)
    loan_ids = [f"LN{str(i).zfill(6)}" for i in range(num_loans)]
    
    # 2022 to 2024
    origination_months = pd.date_range(start="2022-01-01", end="2024-12-01", freq="MS")
    origination = np.random.choice(origination_months, size=num_loans)
    
    original_balance = np.random.normal(300000, 100000, num_loans)
    original_balance = np.clip(original_balance, 50000, 1000000).round(2)
    
    interest_rate = np.random.normal(5.0, 1.5, num_loans)
    interest_rate = np.clip(interest_rate, 2.5, 9.0).round(3)
    
    credit_score_bands = ["<600", "600-649", "650-699", "700-749", "750-799", "800+"]
    credit_score_band = np.random.choice(credit_score_bands, size=num_loans, p=[0.05, 0.1, 0.2, 0.3, 0.25, 0.1])
    
    ltv_bands = ["<60%", "60-70%", "70-80%", "80-90%", "90-95%", ">95%"]
    ltv_band = np.random.choice(ltv_bands, size=num_loans, p=[0.1, 0.15, 0.4, 0.2, 0.1, 0.05])
    
    dti_bands = ["<30%", "30-40%", "40-50%", ">50%"]
    dti_band = np.random.choice(dti_bands, size=num_loans, p=[0.2, 0.4, 0.3, 0.1])
    
    states = ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI"]
    state = np.random.choice(states, size=num_loans)
    
    loan_purposes = ["Purchase", "Refinance-Rate-Term", "Refinance-Cashout"]
    loan_purpose = np.random.choice(loan_purposes, size=num_loans, p=[0.6, 0.2, 0.2])
    
    occupancy_types = ["Owner-Occupied", "Second Home", "Investment"]
    occupancy_type = np.random.choice(occupancy_types, size=num_loans, p=[0.8, 0.1, 0.1])
    
    property_types = ["Single-Family", "Condo", "Multi-Unit"]
    property_type = np.random.choice(property_types, size=num_loans, p=[0.75, 0.15, 0.1])
    
    df = pd.DataFrame({
        "loan_id": loan_ids,
        "origination_month": origination,
        "original_balance": original_balance,
        "interest_rate": interest_rate,
        "credit_score_band": credit_score_band,
        "ltv_band": ltv_band,
        "dti_band": dti_band,
        "state": state,
        "loan_purpose": loan_purpose,
        "occupancy_type": occupancy_type,
        "property_type": property_type,
    })
    return df

def generate_panel_data(static_df, cutoff_date="2025-06-01"):
    records = []
    cutoff = pd.to_datetime(cutoff_date)
    np.random.seed(42)
    
    status_states = ["Current", "30 DPD", "60 DPD", "90+ DPD", "Default", "Prepaid"]
    
    for _, row in static_df.iterrows():
        loan_id = row["loan_id"]
        orig_month = row["origination_month"]
        orig_bal = row["original_balance"]
        
        current_bal = orig_bal
        status = "Current"
        
        # Determine number of months to simulate
        months_to_simulate = ((cutoff.year - orig_month.year) * 12 + cutoff.month - orig_month.month) + 1
        
        # Introduce risk factor based on credit score
        risk_modifier = 1.0
        if row["credit_score_band"] == "<600": risk_modifier = 3.0
        elif row["credit_score_band"] == "600-649": risk_modifier = 2.0
        elif row["credit_score_band"] == "800+": risk_modifier = 0.5
        
        for m in range(months_to_simulate):
            rep_month = orig_month + pd.DateOffset(months=m)
            
            # Transition logic
            if status == "Current":
                probs = [0.95, 0.02 * risk_modifier, 0, 0, 0, 0.03]
            elif status == "30 DPD":
                probs = [0.4, 0.2, 0.3 * risk_modifier, 0, 0, 0.1]
            elif status == "60 DPD":
                probs = [0.1, 0.3, 0.2, 0.3 * risk_modifier, 0, 0.1]
            elif status == "90+ DPD":
                probs = [0.05, 0.05, 0.1, 0.6, 0.2 * risk_modifier, 0]
            else:
                # Terminal state
                probs = [0, 0, 0, 0, 0, 0]
                if status == "Default": probs[4] = 1.0
                if status == "Prepaid": probs[5] = 1.0
                
            probs = np.array(probs)
            probs = probs / probs.sum() # Normalize
            
            if status not in ["Default", "Prepaid"]:
                status = np.random.choice(status_states, p=probs)
                
            # Update balances
            if status == "Prepaid" or status == "Default":
                current_bal = 0.0
            else:
                # Basic amortization simulation
                current_bal = max(0, current_bal - (orig_bal / 360))
                
            dpd = 0
            if status == "30 DPD": dpd = 30
            elif status == "60 DPD": dpd = 60
            elif status == "90+ DPD": dpd = 90
            
            records.append({
                "loan_id": loan_id,
                "month_index": m + 1,
                "reporting_month": rep_month,
                "origination_month": orig_month,
                "loan_age_months": m,
                "remaining_term_months": 360 - m,
                "original_balance": orig_bal,
                "current_balance": current_bal,
                "interest_rate": row["interest_rate"],
                "credit_score_band": row["credit_score_band"],
                "ltv_band": row["ltv_band"],
                "dti_band": row["dti_band"],
                "state": row["state"],
                "loan_purpose": row["loan_purpose"],
                "occupancy_type": row["occupancy_type"],
                "property_type": row["property_type"],
                "servicer_name": np.random.choice(["Servicer A", "Servicer B", "Servicer C"]),
                "current_status": status,
                "days_past_due": dpd,
                "modification_flag": False,
                "prepayment_flag": status == "Prepaid",
                "default_flag": status == "Default",
                "loss_severity_band": "None" if status != "Default" else np.random.choice(["<10%", "10-25%", "25-50%", ">50%"]),
                "last_updated_at": rep_month + pd.DateOffset(days=15),
                "source_system": "PrimaryCore",
                "document_status": "Complete" if np.random.random() > 0.05 else "Missing_Doc"
            })
            
            if status in ["Default", "Prepaid"]:
                break
                
    return pd.DataFrame(records)

def create_targets(panel_df):
    panel_df = panel_df.sort_values(["loan_id", "reporting_month"]).reset_index(drop=True)
    
    # Next state
    panel_df["next_state"] = panel_df.groupby("loan_id")["current_status"].shift(-1)
    panel_df["next_state"] = panel_df["next_state"].fillna("Terminal")
    
    # 3m, 6m delinquency
    def check_future_delinquency(group, horizon):
        is_delinq = group["current_status"].isin(["30 DPD", "60 DPD", "90+ DPD", "Default"])
        return is_delinq.shift(-1).rolling(window=horizon, min_periods=1).max().shift(-(horizon - 1)).fillna(0).astype(int)
        
    panel_df["next_3m_delinquency_flag"] = panel_df.groupby("loan_id").apply(lambda g: check_future_delinquency(g, 3)).reset_index(level=0, drop=True)
    panel_df["next_6m_delinquency_flag"] = panel_df.groupby("loan_id").apply(lambda g: check_future_delinquency(g, 6)).reset_index(level=0, drop=True)
    
    # 12m default/prepay
    def check_future_event(group, horizon, event):
        is_event = (group["current_status"] == event)
        return is_event.shift(-1).rolling(window=horizon, min_periods=1).max().shift(-(horizon - 1)).fillna(0).astype(int)

    panel_df["next_12m_default_flag"] = panel_df.groupby("loan_id").apply(lambda g: check_future_event(g, 12, "Default")).reset_index(level=0, drop=True)
    panel_df["next_12m_prepayment_flag"] = panel_df.groupby("loan_id").apply(lambda g: check_future_event(g, 12, "Prepaid")).reset_index(level=0, drop=True)
    
    # Add exception labels (for supervised anomaly detection training if needed)
    np.random.seed(42)
    panel_df["exception_required"] = (np.random.random(len(panel_df)) < 0.02).astype(int)
    exception_types = ["None", "balance_mismatch", "stale_servicer_update", "invalid_date", "delinquency_status_conflict", "document_gap"]
    panel_df["exception_type"] = "None"
    
    mask = panel_df["exception_required"] == 1
    panel_df.loc[mask, "exception_type"] = np.random.choice(exception_types[1:], size=mask.sum())
    
    return panel_df

def generate_servicer_updates(panel_df):
    updates = panel_df.sample(frac=0.05, random_state=42).copy()
    updates = updates[["loan_id", "reporting_month", "current_balance", "current_status", "last_updated_at"]]
    updates["source_system"] = "ServicerSecondary"
    # Inject some conflicts
    updates["current_balance"] = updates["current_balance"] * np.random.uniform(0.9, 1.1, len(updates))
    updates["last_updated_at"] = updates["last_updated_at"] - pd.DateOffset(days=5) # stale
    return updates

def generate_metadata_files():
    data_dict = """# Data Dictionary
- `loan_id`: Unique loan identifier.
- `reporting_month`: As-of month.
- `origination_month`: Month loan was originated.
- `loan_age_months`: Months since origination.
- `remaining_term_months`: Scheduled months to maturity.
- `original_balance`: Balance at origination.
- `current_balance`: Outstanding balance.
- `interest_rate`: Note interest rate.
- `credit_score_band`: Binned credit score.
- `current_status`: Current payment status (Current, 30 DPD, etc.).
"""
    with open(os.path.join(DATA_DIR, "data_dictionary.md"), "w") as f:
        f.write(data_dict)
        
    validation_rules = {
        "rules": [
            {"name": "balance_consistency", "expression": "current_balance <= original_balance * 1.05", "description": "Balance should not exceed original significantly."},
            {"name": "date_validity", "expression": "origination_month <= reporting_month", "description": "Origination cannot be after reporting."},
            {"name": "document_gap", "expression": "document_status == 'Complete'", "description": "Documents must be complete."}
        ]
    }
    with open(os.path.join(DATA_DIR, "validation_rules.json"), "w") as f:
        json.dump(validation_rules, f, indent=4)
        
    macro_scenarios = pd.DataFrame({
        "scenario_name": ["Base", "Adverse_Credit", "High_Prepayment"],
        "unemployment_rate": [4.0, 8.5, 4.0],
        "hpi_change": [2.0, -10.0, 5.0],
        "interest_rate_change": [0.0, 1.5, -2.0]
    })
    macro_scenarios.to_csv(os.path.join(DATA_DIR, "macro_scenarios.csv"), index=False)
    
    submission_template = pd.DataFrame(columns=[
        "loan_id", "reporting_month", "next_3m_delinquency_prob", "next_6m_delinquency_prob",
        "next_12m_default_prob", "next_12m_prepayment_prob", "predicted_next_state",
        "exception_required_flag", "exception_type", "exception_probability",
        "anomaly_score", "top_drivers", "recommended_action", "confidence", "model_version"
    ])
    submission_template.to_csv(os.path.join(DATA_DIR, "submission_template.csv"), index=False)

def main():
    print("Generating static attributes...")
    static_df = generate_static_attributes(5000)
    static_df.to_csv(os.path.join(DATA_DIR, "loan_static_attributes.csv"), index=False)
    
    print("Generating panel data...")
    panel_df = generate_panel_data(static_df)
    
    print("Creating targets...")
    panel_df = create_targets(panel_df)
    
    # Split train/test (time-based)
    cutoff = pd.to_datetime("2024-12-01")
    train = panel_df[pd.to_datetime(panel_df["reporting_month"]) <= cutoff]
    test = panel_df[pd.to_datetime(panel_df["reporting_month"]) > cutoff]
    
    train.to_csv(os.path.join(DATA_DIR, "loan_monthly_performance_train.csv"), index=False)
    # Test set shouldn't have target labels conceptually, but for evaluation we keep them.
    # We will simulate test set having targets so we can compute metrics locally.
    test.to_csv(os.path.join(DATA_DIR, "loan_monthly_performance_test.csv"), index=False)
    
    print("Generating servicer updates...")
    updates_df = generate_servicer_updates(test)
    updates_df.to_csv(os.path.join(DATA_DIR, "servicer_updates.csv"), index=False)
    
    print("Generating metadata files...")
    generate_metadata_files()
    
    print(f"Generated data saved to {DATA_DIR}/")
    print(f"Train rows: {len(train)}, Test rows: {len(test)}")

if __name__ == "__main__":
    main()
