# FR-5 Scenario & Stress Simulation

## Macro Scenarios Applied

| scenario_name   |   unemployment_rate |   hpi_change |   interest_rate_change |
|:----------------|--------------------:|-------------:|-----------------------:|
| Base            |                 4   |            2 |                    0   |
| Adverse_Credit  |                 8.5 |          -10 |                    1.5 |
| High_Prepayment |                 4   |            5 |                   -2   |

## Projected Portfolio Rates

| scenario        | next_3m_delinquency_flag_prob   | next_12m_default_flag_prob   | next_12m_prepayment_flag_prob   |
|:----------------|:--------------------------------|:-----------------------------|:--------------------------------|
| Adverse_Credit  | 47.6%                           | 0.02%                        | 0.12%                           |
| Base            | 9.65%                           | 0.04%                        | 0.7%                            |
| High_Prepayment | 9.93%                           | 0.03%                        | 0.7%                            |

## Segment Breakdown (by Credit Score Band)

| scenario        | credit_score   | next_3m_delinquency_flag_prob   | next_12m_default_flag_prob   | next_12m_prepayment_flag_prob   |
|:----------------|:---------------|:--------------------------------|:-----------------------------|:--------------------------------|
| Adverse_Credit  | 600-649        | 71.96%                          | 0.04%                        | 0.0%                            |
| Adverse_Credit  | 650-699        | 42.85%                          | 0.01%                        | 0.13%                           |
| Adverse_Credit  | 700-749        | 44.26%                          | 0.03%                        | 0.16%                           |
| Adverse_Credit  | 750-799        | 45.18%                          | 0.01%                        | 0.13%                           |
| Adverse_Credit  | 800+           | 45.05%                          | 0.03%                        | 0.13%                           |
| Adverse_Credit  | <600           | 67.0%                           | 0.03%                        | 0.0%                            |
| Base            | 600-649        | 19.52%                          | 0.18%                        | 0.57%                           |
| Base            | 650-699        | 8.43%                           | 0.01%                        | 0.86%                           |
| Base            | 700-749        | 8.44%                           | 0.02%                        | 0.73%                           |
| Base            | 750-799        | 7.85%                           | 0.01%                        | 0.71%                           |
| Base            | 800+           | 5.84%                           | 0.0%                         | 0.53%                           |
| Base            | <600           | 25.25%                          | 0.3%                         | 0.31%                           |
| High_Prepayment | 600-649        | 19.91%                          | 0.18%                        | 0.58%                           |
| High_Prepayment | 650-699        | 8.95%                           | 0.0%                         | 0.87%                           |
| High_Prepayment | 700-749        | 8.65%                           | 0.01%                        | 0.7%                            |
| High_Prepayment | 750-799        | 8.09%                           | 0.01%                        | 0.73%                           |
| High_Prepayment | 800+           | 5.51%                           | 0.0%                         | 0.54%                           |
| High_Prepayment | <600           | 26.44%                          | 0.24%                        | 0.41%                           |

## Scenario Drivers

What drives the changes under each stress scenario?

- **Adverse_Credit:** The most impacted segment for delinquency risk is `600-649` (shifted by +52.44%). The scenario perturbation (higher unemployment, lower HPI) primarily affected features `days_past_due` and `ltv_band_ord`.
- **High_Prepayment:** The most impacted segment for prepayment risk is `<600` (shifted by +0.10%). The scenario perturbation (lower interest rates, higher HPI) primarily affected `interest_rate` and `rate_spread`.

> **Note:** The `High_Prepayment` scenario may show a muted prepayment response if the underlying synthetic dataset lacked a strong historical relationship between interest rate spreads and prepayment events during training. This is a known data limitation.

