AIF360 not available. Using custom fairness implementations.
- Matplotlib style set to 'seaborn-v0_8-whitegrid' with larger fonts.
======================================================================
 FAIRNESS AND TRANSPARENCY IN AI MODELS FOR CREDIT SCORING
 FINAL VERSION - Fully Aligned with Thesis Requirements
======================================================================

Execution started at: 2025-11-08 19:26:03.621212
Random seed: 42
AIF360 available: False

======================================================================
 MAIN EXECUTION PIPELINE
======================================================================

*** RUNNING FULL ANALYSIS FOR PROTECTED ATTRIBUTE: pa_Age_Under_5_Years ***

==================================================
PHASE 1: DATA LOADING AND PREPROCESSING
==================================================

Loading dataset...
Dataset loaded: 100000 rows, 145 columns

Pre-encoding target variable: loan_status
  Dropped 293 rows with unmappable target status (e.g., 'Issued').
  Target variable pre-encoded to 0/1.

==================================================
EXPLORATORY DATA ANALYSIS
==================================================

1. Dataset Shape: (99707, 145)

2. Data Types:
float64    105
object      35
int64        4
int32        1
dtype: int64

3. Missing Values (Top 10):
                                            Missing_Count  Percentage
id                                                  99707  100.000000
url                                                 99707  100.000000
member_id                                           99707  100.000000
orig_projected_additional_accrued_interest          99328   99.619886
hardship_length                                     99238   99.529622
hardship_reason                                     99238   99.529622
hardship_status                                     99238   99.529622
deferral_term                                       99238   99.529622
hardship_amount                                     99238   99.529622
hardship_start_date                                 99238   99.529622

4. Target Variable Distribution (loan_status):
0    86798
1    12909
Name: loan_status, dtype: int64
Class Imbalance Ratio: 0.149
Default Rate: 12.9% (Thesis target: 19.7%)

5. Statistical Summary (Numerical Features):
                   count          mean           std      min        25%       50%       75%         max
id                   0.0           NaN           NaN      NaN        NaN       NaN       NaN         NaN
member_id            0.0           NaN           NaN      NaN        NaN       NaN       NaN         NaN
loan_amnt        99707.0  15033.119039   9212.492956  1000.00   8000.000  12800.00  20000.00    40000.00
funded_amnt      99707.0  15029.205071   9210.416871  1000.00   8000.000  12800.00  20000.00    40000.00
funded_amnt_inv  99707.0  15015.153657   9211.729027     0.00   8000.000  12800.00  20000.00    40000.00
int_rate         99707.0     13.104645      4.845556     5.31      9.490     12.62     15.99       30.99
installment      99707.0    445.942127    268.217576    21.25    251.195    378.10    594.62     1715.42
annual_inc       99707.0  77645.416118  71377.477150     0.00  46000.000  65000.00  93000.00  9522972.00
loan_status      99707.0      0.129469      0.335720     0.00      0.000      0.00      0.00        1.00
url                  0.0           NaN           NaN      NaN        NaN       NaN       NaN         NaN
- Saved improved EDA visualization.

==================================================
DATA PREPROCESSING PIPELINE
==================================================

Implementing 4 Sequential Stages (Thesis Section 3.4):
1. Data Cleaning
2. Missing Value Imputation
3. Feature Engineering
4. Target Variable Encoding

----------------------------------------
STAGE 1: DATA CLEANING
----------------------------------------
- Removed 0 duplicate records
- Winsorized 5972 outliers at 1st/99th percentiles (as per thesis)
- Fixed 0 inconsistent loan/funded amounts
- Converted 2 negative values in total_rec_late_fee
Data Cleaning Complete: 99707 records retained

----------------------------------------
STAGE 2: MISSING VALUE IMPUTATION
----------------------------------------
Found 103 columns with missing values
  ⚠ Warning: Dropping 3 numerical columns that are 100% empty: ['id', 'member_id', 'url']
- Applying MICE imputation to 81 numerical features (as per thesis)
- Applying mode imputation to 19 categorical features
  - emp_title: Filled 7439 values with 'Teacher'
  - emp_length: Filled 6535 values with '10+ years'
  - desc: Filled 94255 values with ' '
  - title: Filled 1022 values with 'Debt consolidation'
  - last_pymnt_d: Filled 115 values with 'Feb-2019'
  - next_pymnt_d: Filled 57561 values with 'Mar-2019'
  - last_credit_pull_d: Filled 9 values with 'Feb-2019'
  - verification_status_joint: Filled 94582 values with 'Not Verified'
  - sec_app_earliest_cr_line: Filled 94936 values with 'Aug-2006'
  - hardship_type: Filled 99238 values with 'INTEREST ONLY-3 MONTHS DEFERRAL'
  - hardship_reason: Filled 99238 values with 'NATURAL_DISASTER'
  - hardship_status: Filled 99238 values with 'COMPLETED'
  - hardship_start_date: Filled 99238 values with 'Sep-2017'
  - hardship_end_date: Filled 99238 values with 'Dec-2017'
  - payment_plan_start_date: Filled 99238 values with 'Oct-2017'
  - hardship_loan_status: Filled 99238 values with 'Late (16-30 days)'
  - debt_settlement_flag_date: Filled 98245 values with 'Feb-2019'
  - settlement_status: Filled 98245 values with 'ACTIVE'
  - settlement_date: Filled 98245 values with 'Oct-2018'
Missing Value Imputation Complete: 0 missing values remaining

----------------------------------------
STAGE 3: FEATURE ENGINEERING
----------------------------------------
Cleaning 'term' column (e.g., ' 36 months' -> 36)...
- 'term' column converted to numeric.
Creating binary protected attribute for fairness testing...
- Created: pa_rent_vs_other (0=RENT, 1=MORTGAGE/OWN/OTHER)
Creating thesis-specific protected attributes...
- Created: pa_Geographic_High_Risk (1=True, 0=False)
Creating domain-relevant financial ratios...
- Created: credit_utilization_ratio
- Created: payment_burden (monthly payment / monthly income)
✓ Converted 'issue_d' to datetime
- Created: credit_history_years
- Created: pa_Age_Under_5_Years (1=True, 0=False)
Creating risk indicator features...
- Created: has_delinquency
- Created: has_public_record
- Created: high_inquiry_flag
- Created: loan_to_income_ratio
- Created: total_interest
Encoding categorical features...
- Encoded: grade → grade_numeric
- Encoded: emp_length → emp_length_years
- One-hot encoded: home_ownership → 6 features
- One-hot encoded: verification_status → 3 features
- One-hot encoded: purpose → 10 features
Standardizing numerical features...
- Standardized 11 numerical features
Feature Engineering Complete: 178 total features

----------------------------------------
STAGE 4: TARGET VARIABLE ENCODING
----------------------------------------
- Target variable encoded: loan_status
  - Class 0 (Good): 86798 samples (87.1%)
  - Class 1 (Default): 12909 samples (12.9%)
  - Imbalance Ratio: 0.149
  ⚠ Warning: Significant class imbalance detected - SMOTE will be applied

Finalizing feature set...
- Dropping 27 non-numeric/unprocessed columns.

Applying final cleanup for Inf and NaN values...
- Replaced infinite values with NaN.
- Final NaN cleanup complete using median imputation.
- Clipped extreme values to fit float32 range.

Final Data Quality Checks:
- No missing values: True
- No duplicate rows: True
- All numerical: True

==================================================
DATA PREPROCESSING COMPLETE
==================================================
Initial shape: (99707, 145)
Final shape: (99707, 151)
Features created: 150
Data reduction: 0.0%
Feature expansion: 4.1%

Stratifying split by 'loan_status' and 'pa_Age_Under_5_Years'...
Stratification groups created. Example: ['0_0.0' '0_1.0' '1_0.0' '1_1.0']...

==================================================
DATA SPLITTING (Thesis Section 3.9)
==================================================
Data Split Configuration:
  Training Set:   59823 samples (60.0%)
  Validation Set: 19942 samples (20.0%)
  Test Set:       19942 samples (20.0%)

Class Distribution:
  Train: [52078  7745] (ratio: 0.129)
  Val:   [17360  2582] (ratio: 0.129)
  Test:  [17360  2582] (ratio: 0.129)

==================================================
PHASE 2: HANDLING CLASS IMBALANCE
==================================================

Applying SMOTE to training data...
Balanced training set: (104156, 150)
Class distribution after SMOTE: [52078 52078]

==================================================
PHASE 3: MODEL DEVELOPMENT
==================================================

==================================================
TRAINING ALL MODELS
==================================================

----------------------------------------
Training Baseline Logistic Regression...
Validation Accuracy: 0.6713

Logistic Regression Performance (Test Set):
------------------------------
Accuracy:  0.6704
Precision: 0.1533
Recall:    0.3416
F1-Score:  0.2116
AUC-ROC:   0.4995
Avg Precision: 0.1293

Confusion Matrix:
[[12488  4872]
 [ 1700   882]]

----------------------------------------
Training Decision Tree (Baseline)...
Validation Accuracy: 0.9082

Decision Tree Performance (Test Set):
------------------------------
Accuracy:  0.9080
Precision: 0.6235
Recall:    0.7304
F1-Score:  0.6727
AUC-ROC:   0.9110
Avg Precision: 0.7693

Confusion Matrix:
[[16221  1139]
 [  696  1886]]

----------------------------------------
Training Random Forest...
Validation Accuracy: 0.9776

Random Forest Performance (Test Set):
------------------------------
Accuracy:  0.9780
Precision: 0.9963
Recall:    0.8335
F1-Score:  0.9076
AUC-ROC:   0.9782
Avg Precision: 0.9484

Confusion Matrix:
[[17352     8]
 [  430  2152]]

----------------------------------------
Training XGBoost...
Validation Accuracy: 0.9841

XGBoost Performance (Test Set):
------------------------------
Accuracy:  0.9835
Precision: 0.9930
Recall:    0.8784
F1-Score:  0.9322
AUC-ROC:   0.9832
Avg Precision: 0.9557

Confusion Matrix:
[[17344    16]
 [  314  2268]]

----------------------------------------
Training LightGBM...
Training until validation scores don't improve for 20 rounds
Did not meet early stopping. Best iteration is:
[187]   valid_0's binary_logloss: 0.0634883
Validation Accuracy: 0.9840

LightGBM Performance (Test Set):
------------------------------
Accuracy:  0.9837
Precision: 0.9909
Recall:    0.8819
F1-Score:  0.9332
AUC-ROC:   0.9835
Avg Precision: 0.9554

Confusion Matrix:
[[17339    21]
 [  305  2277]]

==================================================
MODEL COMPARISON
==================================================

Model Performance Comparison:
              Model  Val_Accuracy  Test_Accuracy  Precision   Recall  F1-Score  AUC-ROC
           LightGBM      0.983953       0.983653   0.990862 0.881875  0.933197 0.983486
            XGBoost      0.984054       0.983452   0.992995 0.878389  0.932182 0.983233
      Random Forest      0.977635       0.978036   0.996296 0.833462  0.907634 0.978196
      Decision Tree      0.908184       0.907983   0.623471 0.730442  0.672731 0.911028
Logistic Regression      0.671297       0.670444   0.153285 0.341596  0.211612 0.499528

==================================================
STATISTICAL SIGNIFICANCE TESTING
==================================================
Friedman Test Results:
  Chi-square statistic: 114.1067
  P-value: 0.0000
  - Significant difference detected between models (p < 0.05)
  Post-hoc analysis recommended (Nemenyi test)

==================================================
PHASE 4: EXPLAINABILITY IMPLEMENTATION
==================================================

==================================================
SHAP EXPLAINABILITY ANALYSIS
==================================================
SHAP explainer initialized

Generating SHAP explanations...

==================================================
EXPLANATION QUALITY EVALUATION
==================================================
Explanation Quality Metrics:
  Stability: 0.921
  Completeness: 0.644
  Comprehensibility: 0.850
  Fidelity: 0.920

==================================================
PHASE 5: FAIRNESS ASSESSMENT AND ENHANCEMENT
==================================================
Found protected attribute 'pa_Age_Under_5_Years' at index 112

==================================================
BIAS DETECTION FOR LOGISTIC REGRESSION
==================================================

PA_AGE_UNDER_5_YEARS Bias Analysis:
------------------------------
  Demographic Parity Difference: 0.193
  Demographic Parity Ratio: 0.595
  Equal Opportunity Difference: 0.170
  Disparate Impact: 0.732
  Satisfies 80% Rule: False
  Group 0.0: Size=19319, Approval Rate=0.717, Accuracy=0.675
  Group 1.0: Size=623, Approval Rate=0.525, Accuracy=0.526

==================================================
BIAS DETECTION FOR DECISION TREE
==================================================

PA_AGE_UNDER_5_YEARS Bias Analysis:
------------------------------
  Demographic Parity Difference: 0.006
  Demographic Parity Ratio: 0.963
  Equal Opportunity Difference: 0.012
  Disparate Impact: 0.993
  Satisfies 80% Rule: True
  Group 0.0: Size=19319, Approval Rate=0.848, Accuracy=0.908
  Group 1.0: Size=623, Approval Rate=0.843, Accuracy=0.905

==================================================
BIAS DETECTION FOR RANDOM FOREST
==================================================

PA_AGE_UNDER_5_YEARS Bias Analysis:
------------------------------
  Demographic Parity Difference: 0.006
  Demographic Parity Ratio: 0.949
  Equal Opportunity Difference: 0.037
  Disparate Impact: 0.993
  Satisfies 80% Rule: True
  Group 0.0: Size=19319, Approval Rate=0.892, Accuracy=0.978
  Group 1.0: Size=623, Approval Rate=0.886, Accuracy=0.971

==================================================
BIAS DETECTION FOR XGBOOST
==================================================

PA_AGE_UNDER_5_YEARS Bias Analysis:
------------------------------
  Demographic Parity Difference: 0.013
  Demographic Parity Ratio: 0.900
  Equal Opportunity Difference: 0.002
  Disparate Impact: 0.986
  Satisfies 80% Rule: True
  Group 0.0: Size=19319, Approval Rate=0.886, Accuracy=0.984
  Group 1.0: Size=623, Approval Rate=0.873, Accuracy=0.981

==================================================
BIAS DETECTION FOR LIGHTGBM
==================================================

PA_AGE_UNDER_5_YEARS Bias Analysis:
------------------------------
  Demographic Parity Difference: 0.015
  Demographic Parity Ratio: 0.883
  Equal Opportunity Difference: 0.006
  Disparate Impact: 0.983
  Satisfies 80% Rule: True
  Group 0.0: Size=19319, Approval Rate=0.885, Accuracy=0.984
  Group 1.0: Size=623, Approval Rate=0.870, Accuracy=0.978

==================================================
APPLYING REWEIGHING FOR BIAS MITIGATION
==================================================
Reweighing applied. Weight range: [0.914, 1.104]
Mean weight: 1.000

Model trained with reweighing

==================================================
BIAS DETECTION FOR XGBOOST WITH REWEIGHING
==================================================

PA_AGE_UNDER_5_YEARS Bias Analysis:
------------------------------
  Demographic Parity Difference: 0.011
  Demographic Parity Ratio: 0.909
  Equal Opportunity Difference: 0.008
  Disparate Impact: 0.987
  Satisfies 80% Rule: True
  Group 0.0: Size=19319, Approval Rate=0.885, Accuracy=0.984
  Group 1.0: Size=623, Approval Rate=0.873, Accuracy=0.981

==================================================
APPLYING THRESHOLD OPTIMIZATION (FOR DEMOGRAPHIC PARITY)
==================================================
Disadvantaged Group (0) Approval Rate (Target): 0.8859
Advantaged Group (1) Optimal Threshold: 0.950
  (New approval rate will be approx: 0.8859)

Evaluating fairness with new optimized thresholds...
Overall Accuracy After Optimization: 0.9832

==================================================
BIAS DETECTION FOR XGBOOST WITH THRESHOLD OPTIMIZATION
==================================================
  (Using optimized predictions)

PA_AGE_UNDER_5_YEARS Bias Analysis:
------------------------------
  Demographic Parity Difference: 0.000
  Demographic Parity Ratio: 0.998
  Equal Opportunity Difference: 0.081
  Disparate Impact: 1.000
  Satisfies 80% Rule: True
  Group 0.0: Size=19319, Approval Rate=0.886, Accuracy=0.984
  Group 1.0: Size=623, Approval Rate=0.886, Accuracy=0.971

==================================================
PHASE 6: GENERATING PARETO FRONTIER
==================================================
Preparing Pareto data for: pa_Age_Under_5_Years

==================================================
GENERATING PARETO FRONTIER PLOT for pa_Age_Under_5_Years
==================================================
  Running experiment with intervention strength: 0.00
  Running experiment with intervention strength: 0.07
  Running experiment with intervention strength: 0.14
  Running experiment with intervention strength: 0.21
  Running experiment with intervention strength: 0.29
  Running experiment with intervention strength: 0.36
  Running experiment with intervention strength: 0.43
  Running experiment with intervention strength: 0.50
  Running experiment with intervention strength: 0.57
  Running experiment with intervention strength: 0.64
  Running experiment with intervention strength: 0.71
  Running experiment with intervention strength: 0.79
  Running experiment with intervention strength: 0.86
  Running experiment with intervention strength: 0.93
  Running experiment with intervention strength: 1.00

Saved Pareto frontier plot: ../outputs/main/pareto_frontier_pa_Age_Under_5_Years.png

======================================================================
 FINAL SUMMARY AND RECOMMENDATIONS
======================================================================

1. BEST MODEL BY CRITERION:
----------------------------------------
   - Highest Predictive Performance: LightGBM (AUC-ROC: 0.983)
   - Best Explainability: SHAP with LightGBM
   - Explanation Stability: 0.921

2. KEY FAIRNESS FINDINGS:
----------------------------------------
   - Baseline Bias (XGBoost): 0.013 DP Difference
   - Mitigation 1 (Reweighing): Had minimal to no effect on bias.
   - Mitigation 2 (Thresholding): Succeeded.
     - Optimized Bias: 0.000 DP Difference
   - Cost of Fairness: Overall accuracy changed from 0.9835 to 0.9832

======================================================================
 ANALYSIS COMPLETE
 Execution completed at: 2025-11-08 20:18:01.031834
 All visualizations saved to /outputs folder
======================================================================