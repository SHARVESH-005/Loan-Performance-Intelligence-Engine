# Data Intelligence & Profiling Report (FR-1 minimal)

### Missingness
|                    |   Missing Count |   Missing % |
|:-------------------|----------------:|------------:|
| loss_severity_band |           63540 |       99.75 |
| exception_type     |           62455 |       98.04 |

### Distributions
#### Numeric Feature Fields
|                       |       mean |        50% |        std |     min |    max |         5% |        95% |   skew |
|:----------------------|-----------:|-----------:|-----------:|--------:|-------:|-----------:|-----------:|-------:|
| month_index           |     10.197 |      8     |      7.706 |     1   |     36 |      1     |     26     |  0.911 |
| loan_age_months       |      9.197 |      7     |      7.706 |     0   |     35 |      0     |     25     |  0.911 |
| remaining_term_months |    350.803 |    353     |      7.706 |   325   |    360 |    335     |    360     | -0.911 |
| original_balance      | 301782     | 301096     |  98408.4   | 50000   | 705255 | 134790     | 463879     |  0.047 |
| current_balance       | 283138     | 288257     | 108288     |     0   | 703296 |  94257     | 449426     | -0.368 |
| interest_rate         |      5.02  |      5.009 |      1.44  |     2.5 |      9 |      2.553 |      7.532 |  0.193 |
| days_past_due         |      2.515 |      0     |     11.924 |     0   |     90 |      0     |     30     |  5.453 |
| exception_required    |      0.02  |      0     |      0.138 |     0   |      1 |      0     |      0     |  6.939 |

#### Categorical Feature Fields
**loan_id**: 5000 unique values, 100.0% rare categories (<1%). Top 5: {'LN003761': 36, 'LN000667': 36, 'LN000606': 36, 'LN003640': 36, 'LN000596': 36}

**reporting_month**: 36 unique values, 2.1% rare categories (<1%). Top 5: {'2024-12-01': 2903, '2024-11-01': 2879, '2024-10-01': 2809, '2024-09-01': 2767, '2024-08-01': 2723}

**origination_month**: 36 unique values, 3.1% rare categories (<1%). Top 5: {'2022-01-01': 3013, '2022-08-01': 3005, '2022-04-01': 2986, '2022-03-01': 2813, '2022-02-01': 2723}

**credit_score_band**: 6 unique values, 0.0% rare categories (<1%). Top 5: {'700-749': 18529, '750-799': 15636, '650-699': 13761, '800+': 6506, '600-649': 6218}

**ltv_band**: 6 unique values, 0.0% rare categories (<1%). Top 5: {'70-80%': 25272, '80-90%': 12805, '60-70%': 9455, '90-95%': 6624, '<60%': 6258}

**dti_band**: 4 unique values, 0.0% rare categories (<1%). Top 5: {'30-40%': 24813, '40-50%': 20562, '<30%': 12222, '>50%': 6104}

**state**: 10 unique values, 0.0% rare categories (<1%). Top 5: {'MI': 6815, 'GA': 6688, 'OH': 6505, 'TX': 6401, 'FL': 6316}

**loan_purpose**: 3 unique values, 0.0% rare categories (<1%). Top 5: {'Purchase': 38897, 'Refinance-Cashout': 12931, 'Refinance-Rate-Term': 11873}

**occupancy_type**: 3 unique values, 0.0% rare categories (<1%). Top 5: {'Owner-Occupied': 50693, 'Second Home': 6549, 'Investment': 6459}

**property_type**: 3 unique values, 0.0% rare categories (<1%). Top 5: {'Single-Family': 47828, 'Condo': 9639, 'Multi-Unit': 6234}

**servicer_name**: 3 unique values, 0.0% rare categories (<1%). Top 5: {'Servicer C': 21351, 'Servicer A': 21224, 'Servicer B': 21126}

**current_status**: 6 unique values, 1.2% rare categories (<1%). Top 5: {'Current': 58150, 'Prepaid': 2046, '30 DPD': 1935, '60 DPD': 821, '90+ DPD': 588}

**modification_flag**: 1 unique values, 0.0% rare categories (<1%). Top 5: {False: 63701}

**prepayment_flag**: 2 unique values, 0.0% rare categories (<1%). Top 5: {False: 61655, True: 2046}

**default_flag**: 2 unique values, 0.3% rare categories (<1%). Top 5: {False: 63540, True: 161}

**loss_severity_band**: 5 unique values, 0.3% rare categories (<1%). Top 5: {nan: 63540, '>50%': 46, '<10%': 43, '10-25%': 42, '25-50%': 30}

**last_updated_at**: 36 unique values, 2.1% rare categories (<1%). Top 5: {'2024-12-16': 2903, '2024-11-16': 2879, '2024-10-16': 2809, '2024-09-16': 2767, '2024-08-16': 2723}

**source_system**: 1 unique values, 0.0% rare categories (<1%). Top 5: {'PrimaryCore': 63701}

**document_status**: 2 unique values, 0.0% rare categories (<1%). Top 5: {'Complete': 60543, 'Missing_Doc': 3158}

**exception_type**: 6 unique values, 2.0% rare categories (<1%). Top 5: {nan: 62455, 'stale_servicer_update': 260, 'delinquency_status_conflict': 257, 'document_gap': 249, 'invalid_date': 248}

#### Target Fields
**next_3m_delinquency_flag**: {0: 90.94990659487291, 1: 9.050093405127079}

**next_6m_delinquency_flag**: {0: 88.21996514968367, 1: 11.780034850316321}

**next_12m_default_flag**: {0: 99.58713363997425, 1: 0.4128663600257453}

**next_12m_prepayment_flag**: {0: 96.79596866611199, 1: 3.2040313338880075}

**next_state**: {'Current': 87.8259367984804, 'Terminal': 3.4646237892654743, 'Prepaid': 3.1600759799689175, '30 DPD': 2.9654165554700866, '60 DPD': 1.3516271330120406, '90+ DPD': 0.9701574543570745, 'Default': 0.26216228944600556}

### Basic Date & Logic Checks
- `origination_month > reporting_month` violations: **0**
- `current_balance > original_balance` (by >5%) violations: **0**
- Status 'Current' but DPD > 0 violations: **0**



---
## Day 2 Advanced Profiling Extensions

### Train vs. Test Drift
|    | Feature                  |   KS Stat |   KS p-value |    PSI | High Drift (PSI>0.2)   |
|---:|:-------------------------|----------:|-------------:|-------:|:-----------------------|
|  0 | month_index              |    0.3187 |       0      | 1.1385 | True                   |
|  1 | loan_age_months          |    0.3187 |       0      | 1.1385 | True                   |
|  2 | remaining_term_months    |    0.3187 |       0      | 1.1385 | True                   |
|  3 | original_balance         |    0.0166 |       0.0003 | 0.0033 | False                  |
|  4 | current_balance          |    0.0435 |       0      | 0.0154 | False                  |
|  5 | interest_rate            |    0.0137 |       0.0051 | 0.0037 | False                  |
|  6 | days_past_due            |    0.0055 |       0.7315 | 0.001  | False                  |
|  7 | balance_ratio            |    0.3098 |       0      | 0.1832 | False                  |
|  8 | rate_spread              |    0.0137 |       0.0051 | 0.0037 | False                  |
|  9 | months_since_last_update |    0      |       1      | 0      | False                  |
| 10 | credit_score_band_ord    |    0.022  |       0      | 0.0048 | False                  |
| 11 | ltv_band_ord             |    0.0039 |       0.9703 | 0.0006 | False                  |
| 12 | dti_band_ord             |    0.0207 |       0      | 0.0022 | False                  |
| 13 | prev_balance             |    0.0431 |       0      | 0.0135 | False                  |
| 14 | balance_change           |    0.071  |       0      | 0.0015 | False                  |
| 15 | prev_dpd                 |    0.0095 |       0.1151 | 0.0022 | False                  |
| 16 | dpd_change               |    0.0055 |       0.7252 | 0.0021 | False                  |

### Correlation & Association
#### Top Numeric Correlations (Pearson)
| Feature 1             | Feature 2             |   Correlation |
|:----------------------|:----------------------|--------------:|
| remaining_term_months | loan_age_months       |        -1     |
| month_index           | remaining_term_months |        -1     |
| loan_age_months       | month_index           |         1     |
| prev_balance          | original_balance      |         0.998 |
| balance_ratio         | balance_change        |         0.904 |

#### Top Numeric Rank Correlations (Spearman)
| Feature 1             | Feature 2        |   Correlation |
|:----------------------|:-----------------|--------------:|
| month_index           | loan_age_months  |         1     |
| prev_balance          | original_balance |         0.998 |
| remaining_term_months | balance_ratio    |         0.935 |
| current_balance       | prev_balance     |         0.934 |
| original_balance      | current_balance  |         0.932 |

#### Top Categorical Associations (Cramér's V)
| Feature 1             | Feature 2             |   Cramér's V |
|:----------------------|:----------------------|-------------:|
| prev_dpd              | dpd_change            |        0.585 |
| days_past_due         | prev_dpd              |        0.525 |
| days_past_due         | dpd_change            |        0.47  |
| days_past_due         | credit_score_band_ord |        0.077 |
| credit_score_band_ord | prev_dpd              |        0.075 |

### Outlier Detection
#### Univariate Outliers (IQR method)
| Feature          |   Outlier Count |   % Outliers |
|:-----------------|----------------:|-------------:|
| original_balance |             206 |         0.44 |
| current_balance  |            1759 |         3.75 |
| interest_rate    |               0 |         0    |
| days_past_due    |            2377 |         5.06 |

#### Multivariate Outliers (Isolation Forest)
- Isolation Forest detected **469** outliers (1.00% of training data) across `['original_balance', 'current_balance', 'interest_rate', 'days_past_due']`.

### Association-Rule Mining (Apriori)
Top 5 Association Rules by Lift:
| Rule                                                                                        |   Support |   Confidence |   Lift |
|:--------------------------------------------------------------------------------------------|----------:|-------------:|-------:|
| If current_status_Current, loan_purpose_Purchase, occupancy_type_Owner-Occupied -> state_TX |     0.051 |        0.114 |  1.138 |
| If loan_purpose_Purchase, occupancy_type_Owner-Occupied -> state_TX                         |     0.055 |        0.113 |  1.122 |

### Data Quality Scores
- **Overall Mean Data Quality Score:** 0.9816

#### Bottom 5 States by Data Quality
| state   |   data_quality_score |
|:--------|---------------------:|
| GA      |             0.981566 |
| NC      |             0.981577 |
| FL      |             0.981582 |
| OH      |             0.981583 |
| MI      |             0.981597 |
