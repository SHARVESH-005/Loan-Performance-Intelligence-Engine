# FR-2 Model Performance Report (§12 Metrics)

### Target: `next_3m_delinquency_flag`
| Model | ROC-AUC | PR-AUC | F1 Score | Recall @ 80% Precision | Brier Score |
|---|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.7343 | 0.4125 | 0.4668 | 0.2170 | 0.1736 |
| LightGBM Tuned+Calibrated (Improved) | **0.7538** | **0.4305** | **0.4674** | **0.2335** | **0.0668** |

### Target: `next_6m_delinquency_flag`
| Model | ROC-AUC | PR-AUC | F1 Score | Recall @ 80% Precision | Brier Score |
|---|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.6500 | 0.2423 | 0.3201 | 0.0000 | 0.2243 |
| LightGBM Tuned+Calibrated (Improved) | **0.7728** | **0.3577** | **0.4328** | **0.0071** | **0.0905** |

### Target: `next_12m_default_flag`
| Model | ROC-AUC | PR-AUC | F1 Score | Recall @ 80% Precision | Brier Score |
|---|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.7814 | 0.0051 | 0.0167 | 0.0000 | 0.1871 |
| LightGBM Tuned+Calibrated (Improved) | **0.9910** | **0.1354** | **0.2083** | **0.0000** | **0.0010** |

### Target: `next_12m_prepayment_flag`
| Model | ROC-AUC | PR-AUC | F1 Score | Recall @ 80% Precision | Brier Score |
|---|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.4968 | 0.0077 | 0.0165 | 0.0000 | 0.2501 |
| LightGBM Tuned+Calibrated (Improved) | **0.8773** | **0.0580** | **0.1361** | **0.0000** | **0.0074** |

### Target: `next_state` (Multi-class)
| Model | Accuracy | Macro-F1 |
|---|---|---|
| Logistic Regression (Baseline) | 0.8267 | 0.4636 |
| LightGBM Tuned (Improved) | **0.7150** | **0.4846** |

#### Per-Class Classification Report (LightGBM)
```text
              precision    recall  f1-score   support

      30 DPD       0.06      0.16      0.09       267
      60 DPD       0.33      0.79      0.46       119
     90+ DPD       0.43      0.73      0.54        78
     Current       0.95      0.74      0.83      7154
     Default       0.31      0.69      0.42        16
     Prepaid       0.03      0.13      0.04       235
    Terminal       1.00      1.00      1.00       289

    accuracy                           0.72      8158
   macro avg       0.44      0.61      0.48      8158
weighted avg       0.88      0.72      0.78      8158

```

#### Confusion Matrix
|               |   Pred 30 DPD |   Pred 60 DPD |   Pred 90+ DPD |   Pred Current |   Pred Default |   Pred Prepaid |   Pred Terminal |
|:--------------|--------------:|--------------:|---------------:|---------------:|---------------:|---------------:|----------------:|
| True 30 DPD   |            42 |            52 |             35 |            112 |              2 |             24 |               0 |
| True 60 DPD   |             1 |            94 |             22 |              0 |              2 |              0 |               0 |
| True 90+ DPD  |             0 |             0 |             57 |              0 |             21 |              0 |               0 |
| True Current  |           627 |           111 |              8 |           5309 |              0 |           1099 |               0 |
| True Default  |             0 |             0 |              5 |              0 |             11 |              0 |               0 |
| True Prepaid  |            22 |            31 |              6 |            145 |              0 |             31 |               0 |
| True Terminal |             0 |             0 |              0 |              0 |              0 |              0 |             289 |

