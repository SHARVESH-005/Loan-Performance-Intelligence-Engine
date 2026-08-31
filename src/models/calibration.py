import os
import pandas as pd
import numpy as np
import joblib
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"
MODELS_DIR = os.path.join(PROCESSED_DIR, "models")
REPORTS_DIR = "reports/calibration_curves"

def run():
    print("Calibrating Models...")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    valid_df = pd.read_csv(os.path.join(PROCESSED_DIR, "valid.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    
    with open(os.path.join(PROCESSED_DIR, "features.csv"), "r") as f:
        features = f.read().splitlines()
        
    cat_cols = [c for c in features if valid_df[c].dtype == 'O' or valid_df[c].nunique() < 20 and not pd.api.types.is_numeric_dtype(valid_df[c])]
    for col in cat_cols:
        valid_df[col] = valid_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')
        
    X_valid = valid_df[features]
    X_test = test_df[features]
    
    binary_targets = ['next_3m_delinquency_flag', 'next_6m_delinquency_flag', 'next_12m_default_flag', 'next_12m_prepayment_flag']
    
    for target in binary_targets:
        print(f"Calibrating {target}...")
        y_valid = valid_df[target]
        
        if len(np.unique(y_valid)) < 2:
            print(f"Skipping calibration for {target} due to only one class in validation.")
            continue
            
        base_model = joblib.load(os.path.join(MODELS_DIR, f"{target}_lgb_improved.joblib"))
        
        from sklearn.frozen import FrozenEstimator
        frozen_model = FrozenEstimator(base_model)
        
        # Platt Scaling (sigmoid)
        cal_platt = CalibratedClassifierCV(frozen_model, method='sigmoid', cv=2)
        cal_platt.fit(X_valid, y_valid)
        preds_platt = cal_platt.predict_proba(X_valid)[:, 1]
        brier_platt = brier_score_loss(y_valid, preds_platt)
        
        # Isotonic Regression
        cal_iso = CalibratedClassifierCV(frozen_model, method='isotonic', cv=2)
        cal_iso.fit(X_valid, y_valid)
        preds_iso = cal_iso.predict_proba(X_valid)[:, 1]
        brier_iso = brier_score_loss(y_valid, preds_iso)
        
        # Original
        preds_orig = base_model.predict_proba(X_valid)[:, 1]
        brier_orig = brier_score_loss(y_valid, preds_orig)
        
        print(f"  Brier Scores -> Orig: {brier_orig:.5f} | Platt: {brier_platt:.5f} | Isotonic: {brier_iso:.5f}")
        
        if brier_iso < brier_platt and brier_iso < brier_orig:
            best_calibrator = cal_iso
            best_preds = preds_iso
            method = "isotonic"
        elif brier_platt < brier_orig:
            best_calibrator = cal_platt
            best_preds = preds_platt
            method = "sigmoid"
        else:
            best_calibrator = base_model
            best_preds = preds_orig
            method = "none"
            
        print(f"  Selected: {method}")
        joblib.dump(best_calibrator, os.path.join(MODELS_DIR, f"{target}_calibrated.joblib"))
        
        # Plot curve
        prob_true, prob_pred = calibration_curve(y_valid, best_preds, n_bins=10)
        plt.figure(figsize=(6, 6))
        plt.plot(prob_pred, prob_true, marker='o', label=f'Best ({method})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        plt.title(f'Calibration Curve: {target}')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, f"{target}_curve.png"))
        plt.close()
        
        # Generate Test Predictions
        test_df[f"{target}_prob_calibrated"] = best_calibrator.predict_proba(X_test)[:, 1]
        
    # Multi-class predictions for next_state
    print("Generating predictions for next_state...")
    next_state_model = joblib.load(os.path.join(MODELS_DIR, "next_state_lgb_improved.joblib"))
    test_df['predicted_next_state'] = next_state_model.predict(X_test)
    
    probs = next_state_model.predict_proba(X_test)
    test_df['next_state_confidence'] = np.max(probs, axis=1)
    
    # Save test set with calibrated predictions
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test_with_preds.csv"), index=False)
    print("Calibration complete.")

if __name__ == "__main__":
    run()
