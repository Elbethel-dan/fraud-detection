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
- Resampling techniques were applied **only on training data**
- Class distributions before and after resampling were documented
- Trade-offs between minority class recall and overall precision were considered

---

## Key Outcomes
- Two clean, model-ready datasets prepared using a unified framework
- Domain-aware feature sets for e-commerce and credit card fraud
- A reproducible preprocessing pipeline suitable for multiple model types

---
## 2. Set Up the Environment

Create a virtual environment and install dependencies:

bash 
```
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
```

