# Fraud Detection Pipeline for E-commerce and Credit Card Transactions

## Overview
Fraud remains one of the most critical challenges in financial technology, affecting both online commerce platforms and traditional banking systems. This project, developed at **Adey Innovations Inc.**, focuses on building a robust data preprocessing and feature engineering pipeline to support fraud detection across **two distinct transaction domains**:
- E-commerce transactions
- Credit card transactions

Although these datasets differ in structure and behavior, they share common challenges such as extreme class imbalance, noisy data, and evolving fraud patterns. This project prepares both datasets for effective machine learning–based fraud detection.

---

## Business Context
Adey Innovations Inc. provides fintech solutions that require high transaction security without compromising customer experience. Fraud detection systems must therefore:
- Minimize financial losses caused by undetected fraud
- Reduce false alerts that disrupt legitimate users
- Adapt to different transaction environments

This project addresses these needs by engineering behavioral, temporal, and geographic signals that enhance fraud separability in downstream models.

---

## Data Sources
- **E-commerce Transaction Dataset**  
  Contains user-level transaction logs with timestamps, purchase details, IP addresses, and fraud labels.

- **Credit Card Transaction Dataset**  
  Includes anonymized transaction features representing card usage behavior, where fraud cases are rare and highly imbalanced.

- **IP-to-Country Mapping Dataset**  
  Used to enrich e-commerce transactions with geographic information.

---
## Task 1
## Data Preparation and Quality Control
Both datasets underwent systematic cleaning to ensure reliability:
- Removal of duplicate records
- Correction of inconsistent data types
- Treatment of missing values based on data impact analysis

While the specific variables differ across datasets, consistent preprocessing principles were applied to maintain comparability.

---

## Exploratory Analysis
Exploratory Data Analysis was performed independently for each dataset to capture domain-specific fraud patterns:
- Distributional analysis of transaction features
- Relationship between key variables and fraud labels
- Quantification of class imbalance

Findings from EDA guided dataset-specific feature engineering strategies.

---

## Geolocation Analysis (E-commerce Data)
For e-commerce transactions, IP addresses were converted to numeric format and merged with country-level data using range-based lookup. This enabled:
- Country-level fraud rate analysis
- Identification of geographic risk concentrations
- Support for future location-aware fraud scoring

Geolocation features were not applied to the credit card dataset due to anonymization constraints.

---

## Feature Engineering

### E-commerce Transactions
- Transaction frequency and velocity per user
- Time-based features (hour of day, day of week)
- Account age indicators (time since signup)

---

## Data Transformation
To prepare both datasets for modeling:
- Numerical features were normalized to ensure stable optimization
- Categorical features (where applicable) were encoded appropriately
- Transformations were designed to avoid data leakage

---

## Handling Class Imbalance
Fraud cases represent a very small proportion of total transactions in both datasets. To address this:
- SMOTE resampling techniques were applied **only on training data**
- Class distributions before and after resampling were documented

---

## Key Outcomes
- Two clean, model-ready datasets prepared using a unified framework
- Domain-aware feature sets for e-commerce and credit card fraud
- A reproducible preprocessing pipeline suitable for multiple model types

---

## Task 2 – Model Building and Evaluation

### Baseline Model
An interpretable **Logistic Regression** model was trained as a baseline for fraud detection.  
The model was evaluated using **AUC-PR**, **F1-score**, and the **confusion matrix**, providing a transparent reference point for comparison.

### Ensemble Model
A **Random Forest** classifier was trained as the ensemble model to capture non-linear patterns in the data.  
Basic hyperparameter tuning was performed (e.g., number of trees and tree depth) to improve performance while avoiding overfitting.

### Cross-Validation
To obtain reliable performance estimates, **Stratified 5-Fold Cross-Validation** was applied.  
Mean and standard deviation of evaluation metrics were reported across folds, ensuring robustness on this highly imbalanced dataset.

### Model Comparison and Selection
Both models were evaluated using the same metrics for fair comparison.  
The Random Forest model consistently outperformed Logistic Regression, achieving higher **AUC-PR** and **F1-score**, particularly in detecting fraudulent transactions.

While Logistic Regression offers strong interpretability, the **Random Forest model was selected as the final model** due to its superior predictive performance, making it better suited for accurate fraud detection.
---
## Task 3 – Model Explainability

### Feature Importance Baseline
Built-in feature importance was extracted from the Random Forest model to establish a baseline explanation.  
The **top 10 most important features** were visualized, highlighting which features are most frequently used by the model during decision-making.

### SHAP Analysis

#### Global Explanation
SHAP (SHapley Additive exPlanations) was used to compute global feature importance through a **SHAP summary plot**.  
Due to the large size of the dataset, SHAP values were computed on a representative sample of test transactions using an approximate TreeExplainer, which is standard practice for improving computational efficiency.

#### Local Explanation
To understand individual model decisions, SHAP force plots were generated for three specific cases:
- **True Positive**: Fraud correctly identified by the model
- **False Positive**: Legitimate transaction incorrectly flagged as fraud
- **False Negative**: Fraudulent transaction missed by the model

These plots illustrate how individual features contribute to each prediction.

---

# Business Recommendations
## For Fraud Dataset
**1. Implement Enhanced Verification for New Accounts**
*   **Recommendation:** Flag all transactions from accounts less than 24-48 hours old for mandatory, multi-factor authentication (e.g., SMS or biometric verification).
*   **SHAP Insight:** `num_time_since_signup_days` is the dominant fraud driver by a wide margin. The SHAP plot shows extremely low values (very new accounts) have a massive positive impact on fraud probability.

**2. Deploy Dynamic, Time-Based Risk Scoring**
*   **Recommendation:** Automatically increase the risk score for transactions occurring during late-night/early morning hours (e.g., 10 PM to 6 AM local time) and combine this with other signals for a layered defense.
*   **SHAP Insight:** `num_hour_of_day` is the second most important SHAP feature, with higher hours (later in the day) consistently pushing predictions toward fraud, indicating a clear temporal pattern to fraudulent activity.

**3. Investigate and Refine Gender-Based Risk Profiles**
*   **Recommendation:** Conduct a focused analysis to understand *why* user sex (`cat_sex_M` and `cat_sex_F`) appears as a top-5 predictive driver in SHAP. This could reveal underlying issues (e.g., specific marketing campaigns being targeted, or data proxy effects) that need to be addressed to ensure fair and effective risk modeling.
*   **SHAP Insight:** The high SHAP importance for gender features is a surprising, counterintuitive finding that built-in importance missed. This suggests a significant, but potentially non-causal or proxy-based, pattern exists that requires business scrutiny to prevent bias and to understand the true root cause (e.g., it may be correlating with specific product lines or signup channels).

## For Credit Card Dataset
**1. Implement a Hard Rule-Based Alert for Extreme V14 Values**
*   **Recommendation:** Establish an automated rule to flag and manually review any transaction where the `scaler_V14` feature falls below a specific negative threshold (e.g., -3.5), regardless of the model's final score.
*   **SHAP Insight:** `scaler_V14` is the dominant fraud driver. The SHAP plot and case reviews show extreme negative values are almost exclusively associated with fraud (true positives and false negatives), representing a near-certain risk signal.

**2. Develop a Secondary Review Queue for Conflicting Signal Patterns**
*   **Recommendation:** Create a dedicated review queue for transactions where a strongly negative `V14` coincides with strongly positive values in other key features like `V4` or `V10`.
*   **SHAP Insight:** The false negative analysis revealed that a powerfully negative `V14` can be "overruled" by opposing signals (e.g., a high `V4`), causing misses. This specific conflict pattern requires human expert judgment to resolve.

**3. Prioritize Model Monitoring and Investigation on Feature V11**
*   **Recommendation:** Allocate data science resources to deeply analyze the underlying transaction attributes that drive the `V11` principal component, as SHAP indicates it is a critical, underrated fraud lever.
*   **SHAP Insight:** `scaler_V11` ranks 5th in SHAP importance but is absent from the model's own top 5 built-in importance. This discrepancy signals it is a highly influential yet potentially misunderstood feature; understanding its real-world meaning could unlock more precise detection rules or feature engineering.
---
## 2. Set Up the Environment

Create a virtual environment and install dependencies:

bash 
```
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
```

