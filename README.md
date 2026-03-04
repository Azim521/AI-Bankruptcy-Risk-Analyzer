# AI Corporate Bankruptcy Risk Analyzer

AI-powered system that predicts the probability of corporate bankruptcy using financial ratios and machine learning.

The application analyzes financial indicators and estimates bankruptcy risk using an **XGBoost model trained on the Polish Companies Bankruptcy Dataset.**

The project demonstrates how machine learning can detect **financial distress signals several years before bankruptcy occurs.**

---

## Dashboard Preview

![Dashboard](Dashboard.png)

---

## Live Application

🔗 **Try the live app here:**  
[(Streamlit App)](https://ai-bankruptcy-risk-analyzer-azim.streamlit.app/)

---

## Problem Statement

Corporate bankruptcy prediction is an important task in **financial risk management.**

Financial institutions, investors, and regulators use predictive models to identify companies at risk of failure.

Early detection of financial distress helps organizations:

- reduce credit risk

- avoid bad investments

- manage financial exposure

This project builds an **AI system capable of predicting bankruptcy risk using financial ratios.**

---

## Dataset

Dataset used:

**Polish Companies Bankruptcy Dataset**

Source: UCI Machine Learning Repository / Kaggle

Dataset characteristics:

| Property | Value |
|--------|----------|
| Companies | 7027 |
| Features | 64 Financial Ratios |
| XGBoost | Bankruptcy (0/1) |

The dataset includes financial data from companies **1–5 years before bankruptcy**, enabling multi-year prediction analysis.

---

## Machine Learning Pipeline
**Data Preprocessing**

- Missing value handling

- Feature selection

- Consistent preprocessing across datasets

---

**Model Training**

Three machine learning models were evaluated:

- Logistic Regression

- Random Forest

- XGBoost

XGBoost produced the best results and was selected for deployment.

---

## Model Performance
**Test Performance (1-Year Prediction)**
| Metric | Score |
|--------|----------|
| ROC-AUC | 0.97 |
| Accuracy | 0.99 |

---
## Multi Year Bankruptcy Forecasting
The model was also evaluated on financial data from **2–5 years before bankruptcy.**

| Prediction Horizon |	ROC-AUC |
|---------------------|----------|
| 1 Year | 0.97 |
| 2 Years	| 0.88 |
| 3 Years	| 0.85 |
| 4 Years	| 0.87 |
| 5 Years	| 0.89 |

This shows that financial distress signals can be detected **several years before bankruptcy occurs.**

---

## Important Financial Indicators

Top financial ratios identified by the model:

- Attr24
- Attr27
- Attr13
- Attr26
- Attr23
- Attr14
- Attr34
- Attr22
- Attr16
- Attr21

These ratios correspond to indicators related to:

- profitability
- leverage
- liquidity

These factors are widely used in traditional financial risk models such as **Altman Z-Score.**

---

## Application Features

The deployed web application provides:

- bankruptcy probability prediction
- risk score (0–100)
- risk category classification
- interactive financial ratio inputs
- financial risk dashboard interface

Risk categories:

| Score	| Risk Level |
|-------|------------|
| 0–30	| Low Risk |
| 30–60	| Medium Risk |
| 60–100 | High Risk |

---

## Tech Stack

Programming Language

Python

- Libraries
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit

Deployment

Streamlit Cloud

---
## How to run Locally

Clone the repository

git clone 
  - **https://github.com/Azim521/AI-Bankruptcy-Risk-Analyzer.git**

Install dependencies

  - pip install -r requirements.txt

Run the application

  - streamlit run app.py

---

## Future Improvements

Possible extensions for this project:

- upload financial statements (CSV)
- automated ratio calculation
- SHAP explainability dashboard
- integration with financial APIs

---

Author

Azim Sadath

Aspiring Data Scientist focused on **financial analytics and machine learning systems.**
