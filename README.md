# Fairness and Transparency in AI Models for Credit Scoring

## Liverpool John Moores University - MSc Thesis Project

This repository contains the complete implementation for my MSc thesis, "Fairness and Transparency in AI Models for Credit Scoring." The project develops, evaluates, and documents a complete pipeline for building an accurate, explainable, and verifiably fair machine learning model for credit risk assessment.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Key Results](#-key-results)
- [Visualizations](#-visualizations)
- [How to Cite](#-how-to-cite)
- [License](#-license)

## 🎯 Project Overview

This research addresses the critical challenge of developing AI-driven credit scoring models that are both highly accurate and ethically sound. As financial institutions adopt complex "black box" models (like XGBoost), they risk perpetuating historical biases and engaging in "proxy discrimination" against protected groups.

My project confronts this by reconciling three competing objectives:

1. **Accuracy:** Maximize the model's ability to predict loan defaults.
2. **Transparency:** Ensure the model's decisions can be explained to stakeholders (regulators, customers) using SHAP.
3. **Fairness:** Audit the model for bias and apply mitigation techniques to ensure equitable outcomes.

This repository provides the complete Python script to reproduce my thesis findings, from raw data to a fully-analyzed, fair, and transparent model.

## ✨ Key Features

- **High-Performance Models**: Compares 5 models, with **LightGBM** and **XGBoost** achieving **~0.983 AUC-ROC**.
- **Advanced Preprocessing**: Implements a robust pipeline including `IterativeImputer` (MICE) for missing data and a critical fix for calculating `credit_history_years` based on the loan's issue date, not the current date.
- **Imbalance Handling**: Uses **SMOTE** on the training set to correct for the low (12.9%) default rate.
- **Stratified Splitting**: Guarantees that rare protected groups (like `pa_Age_Under_5_Years`) are present in the train, validation, and test sets, enabling a valid fairness analysis.
- **Explainability**: Fully integrated **SHAP** (`TreeExplainer`) to generate clear, publication-ready plots for global feature importance (bar/dot), feature dependence, and individual waterfall explanations.
- **Custom Fairness Framework**: Includes custom-built functions for:
  - **Bias Detection**: Calculating Demographic Parity, Equal Opportunity, and Disparate Impact.
  - **Reweighing (Pre-processing)**: A custom implementation to adjust sample weights.
  - **Threshold Optimization (Post-processing)**: A custom algorithm to achieve perfect Demographic Parity.
- **Trade-off Analysis**: Generates a **Pareto Frontier** plot to visualize the precise trade-off between fairness (Disparate Impact) and accuracy (AUC-ROC).

## 🏗️ Project Structure

This project uses a simple, centralized structure. All code is contained in the `src/` directory, and all outputs are saved to the `outputs/` directory.

```
credit-scoring-fairness-thesis/
│
├── 📄 README.md                           # This file
│
├── 📁 data/
│   └── loan.csv                           # The LendingClub dataset
│
├── 📁 src/
│   ├── 🐍 credit_scoring_fairness_complete.py  # The main, complete Python script
│   └── 🐍 visualize_loan_data.py              # (Optional) Helper script for EDA
│
├── 📁 outputs/                            # All results are saved here
│   ├── 📁 main/
│   │   ├── eda_visualization.png
│   │   ├── correlation_heatmap_top25.png
│   │   ├── model_comparison.png
│   │   ├── shap_summary_dot_plot.png
│   │   ├── shap_summary_bar_plot.png
│   │   ├── shap_dependence_plot.png
│   │   ├── shap_waterfall_plot.png
│   │   └── pareto_frontier_pa_Age_Under_5_Years.png
│   └── 📁 code_results/
│       └── Results.md                     # The full console log of the last run
│
└── 📋 requirements.txt                   # Python dependencies
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or 3.9
- `pip` package manager
- `git`

### Setup Instructions

1. **Clone the repository:**

    ```bash
    git clone [https://github.com/abdulhadikanjo-tech/credit-scoring-fairness-thesis.git]
    cd credit-scoring-fairness-thesis
    ```

2. **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

    *On Windows, activate with:*

    ```bash
    .\venv\Scripts\activate
    ```

    *On macOS/Linux, activate with:*

    ```bash
    source venv/bin/activate
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Place the Data:**
    - Download the `loan.csv` dataset (from Kaggle: <https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv>).
    - Place it inside the `data/` folder.

## 🚀 Usage

All code is in a single, executable file. Navigate to the `src` directory and run the main script.

```bash
cd src/
python credit_scoring_fairness_complete.py
```

- **Runtime:** ~10-15 minutes on a standard machine (with a 100k sample size).

- **Output:** The script will run the complete pipeline, print all results to the console, and save all visualizations in the `../outputs/main/` directory.

### How to Change the Protected Attribute

To run the analysis for a different attribute (e.g., `'pa_Geographic_High_Risk'`), open `credit_scoring_fairness_complete.py` and change **one line** at the top of the `main()` function (around line 1205):

```python
# Change this line:
pa_col_name = 'pa_Age_Under_5_Years'

# To this line:
# pa_col_name = 'pa_Geographic_High_Risk'
```

## 🔬 Methodology

My research followed a 6-phase experimental design, all automated in the main script:

### 1. Data Preprocessing

- Loaded a 100,000-record sample from the LendingClub dataset.
- Cleaned data, winsorized outliers, and used **MICE (`IterativeImputer`)** to handle missing values.
- Engineered domain-specific features, most importantly calculating `credit_history_years` based on the loan's `issue_d`.
- Created binary protected attributes: `pa_Age_Under_5_Years` and `pa_Geographic_High_Risk`.

### 2. Imbalance Handling & Splitting

- Applied **SMOTE** to the training data to correct for the severe class imbalance (12.9% default rate).
- Used a **stratified split** on *both* the target variable and the protected attribute to ensure all groups were represented in the test set.

### 3. Model Development

- Trained and evaluated 5 models: Logistic Regression, Decision Tree, Random Forest, **XGBoost**, and **LightGBM**.
- Compared all models on AUC-ROC, Precision, Recall, and F1-Score.

### 4. Explainability (XAI)

- Used **SHAP** (`TreeExplainer`) on the best-performing model (LightGBM/XGBoost) to ensure full transparency.
- Generated global importance (bar/dot plots) and local explanations (waterfall plots).

### 5. Fairness Analysis

- Conducted a bias audit on the baseline models using my defined protected attributes.
- Calculated **Disparate Impact (DI)**, **Demographic Parity (DP)**, and **Equal Opportunity (EOpp)**.

### 6. Bias Mitigation & Trade-off

- Applied two custom-built interventions: **Reweighing** (pre-processing) and **Threshold Optimization** (post-processing).
- Generated a **Pareto Frontier plot** to visualize the final trade-off between model accuracy (AUC) and fairness (DI).

## 📊 Key Results

My final run (documented in `outputs/code_results/Results.md`) produced new, corrected results. After fixing the date calculation for `pa_Age_Under_5_Years`, the baseline models performed much better and were already significantly fairer than in my initial drafts.

### Model Performance (Final)

LightGBM and XGBoost were the clear winners, achieving exceptional accuracy.

| Model | AUC-ROC | Precision | Recall | F1-Score | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | **0.9835** | 0.9909 | 0.8819 | 0.9332 | 0.9837 |
| **XGBoost** | 0.9832 | 0.9930 | 0.8784 | 0.9322 | 0.9835 |
| Random Forest | 0.9782 | 0.9963 | 0.8335 | 0.9076 | 0.9780 |
| Decision Tree | 0.9110 | 0.6235 | 0.7304 | 0.6727 | 0.9080 |
| Logistic Regression| 0.4995 | 0.1533 | 0.3416 | 0.2116 | 0.6704 |

### Fairness Analysis Results (pa_Age_Under_5_Years)

The baseline XGBoost model was already **legally fair** (DI > 0.80). The interventions then perfected this fairness, demonstrating that fairness can be achieved with almost no loss of accuracy.

| Strategy | Disparate Impact (DI) | DP Difference | EOpp Difference | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline XGBoost** | **0.986** (Fair) | 0.013 | 0.002 | 0.9835 |
| XGBoost + Reweighing | 0.987 (Fair) | 0.011 | 0.008 | 0.9840 |
| **XGBoost + Thresholding**| **1.000** (Perfect) | **0.000** | 0.081 | 0.9832 |

The "Cost of Fairness" was minimal: applying Threshold Optimization to achieve perfect Demographic Parity reduced the model's overall accuracy by only **0.03%** (from 0.9835 to 0.9832).

## 📈 Visualizations

This project generates the following key visualizations, saved in `outputs/main/`:

1. **`eda_visualization.png`**: A 3x3 grid of key data distributions (loan amount, grade, etc.).
2. **`correlation_heatmap_top25.png`**: A readable heatmap of the top 25 features.
3. **`model_comparison.png`**: A 2x2 grid showing model metrics, ROC curves, and P-R curves.
4. **`shap_summary_dot_plot.png`**: The main SHAP plot showing global feature impact.
5. **`shap_waterfall_plot.png`**: An explanation for a single individual's prediction.
6. **`pareto_frontier_pa_Age_Under_5_Years.png`**: The final trade-off analysis plot.

## 📝 How to Cite

If you use this work, please cite it as:

```
Kanjo, A. (2025). Fairness and Transparency in AI Models for Credit Scoring (MSc Thesis). Liverpool John Moores University.
```

## 📜 License

This project is licensed under the MIT License.
