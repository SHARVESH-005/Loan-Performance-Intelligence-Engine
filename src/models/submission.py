import os
import pandas as pd

PROCESSED_DIR = "data/processed"
SUBMISSION_DIR = "submission"

def run():
    print("Generating first-pass submission.csv...")
    # Load test set with predictions
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test_with_preds.csv"))
    
    # Initialize with core test predictions
    preds_df = test_df.copy()
    
    # Merge FR-3 predictions
    surv_preds_path = os.path.join(PROCESSED_DIR, "test_hazard_preds.csv")
    if os.path.exists(surv_preds_path):
        surv_preds = pd.read_csv(surv_preds_path)
        # Assuming discrete_hazard outputs a probability for the current month
        # For simplicity in schema, we will map this to a custom column if needed
        pass
        
    # Merge FR-4 predictions
    anomaly_preds_path = os.path.join(PROCESSED_DIR, "test_anomaly_preds.csv")
    if os.path.exists(anomaly_preds_path):
        anomaly_preds = pd.read_csv(anomaly_preds_path)
        
        # Merge on loan_id and reporting_month
        preds_df = preds_df.merge(anomaly_preds[['loan_id', 'reporting_month', 'anomaly_score', 
                                                 'exception_probability', 'predicted_exception_required', 
                                                 'predicted_exception_type']],
                                  on=['loan_id', 'reporting_month'], how='left')
        
        # Assign values
        preds_df['anomaly_score'] = preds_df['anomaly_score'].fillna(0.0)
        preds_df['exception_probability'] = preds_df['exception_probability'].fillna(0.0)
        preds_df['exception_required_flag'] = preds_df['predicted_exception_required'].fillna(0).astype(int)
        preds_df['exception_type'] = preds_df['predicted_exception_type'].fillna('no_exception')
    else:
        # Fallback if anomaly detector hasn't run
        preds_df['anomaly_score'] = 0.0
        preds_df['exception_probability'] = 0.0
        preds_df['exception_required_flag'] = 0
        preds_df['exception_type'] = "no_exception"
        
    # Build schema
    sub_df = pd.DataFrame({
        'loan_id': preds_df['loan_id'],
        'reporting_month': preds_df['reporting_month'],
        'next_3m_delinquency_prob': preds_df['next_3m_delinquency_flag_prob_calibrated'],
        'next_6m_delinquency_prob': preds_df['next_6m_delinquency_flag_prob_calibrated'],
        'next_12m_default_prob': preds_df['next_12m_default_flag_prob_calibrated'],
        'next_12m_prepayment_prob': preds_df['next_12m_prepayment_flag_prob_calibrated'],
        'predicted_next_state': preds_df['predicted_next_state'],
        'confidence': preds_df['next_state_confidence'],
        'anomaly_score': preds_df['anomaly_score'],
        'exception_probability': preds_df['exception_probability'],
        'exception_required_flag': preds_df['exception_required_flag'],
        'exception_type': preds_df['exception_type']
    })
    
    # Merge FR-6 SHAP drivers
    drivers_path = os.path.join(PROCESSED_DIR, "test_shap_drivers.csv")
    if os.path.exists(drivers_path):
        drivers_df = pd.read_csv(drivers_path)
        sub_df = sub_df.merge(drivers_df[['loan_id', 'reporting_month', 'top_drivers']], on=['loan_id', 'reporting_month'], how='left')
    else:
        sub_df['top_drivers'] = "pending"
        
    # Map actions (FR-7)
    action_map = {
        'balance_mismatch': 'Review amortization schedule against original principal.',
        'invalid_date': 'Correct origination date in core system.',
        'document_gap': 'Request missing closing documents from custodian.',
        'delinquency_status_conflict': 'Reconcile status with servicer DPD reports.',
        'stale_servicer_update': 'Verify servicer feed timestamp and apply latest.',
        'no_exception': 'Standard monitoring.'
    }
    sub_df['recommended_action'] = sub_df['exception_type'].map(action_map).fillna('Manual review required.')
    
    sub_df['model_version'] = "v1.0-agentic-baseline"
    
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    
    sub_df.to_csv(sub_path, index=False)
    
    print(f"Submission saved to {sub_path} with {len(sub_df)} rows.")

if __name__ == "__main__":
    run()
