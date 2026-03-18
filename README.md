# 🏢 AI Corporate Bankruptcy Risk Analyzer

> Predict the probability of corporate bankruptcy **years before it happens** — using real financial ratios, XGBoost, and SHAP explainability.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-bankruptcy-risk-analyzer-azim.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-orange?style=flat)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-brightgreen?style=flat)

---

## 🚀 Live Demo

👉 **[ai-bankruptcy-risk-analyzer-azim.streamlit.app](https://ai-bankruptcy-risk-analyzer-azim.streamlit.app/)**

---

## 📸 App Preview

![Dashboard](Dashboard.png)

---

## ✨ What Makes This Different

| Typical bankruptcy model | This project |
|---|---|
| Predicts only 1 year ahead | Forecasts **1–5 years** before bankruptcy |
| Generic features | Real financial ratios (profitability, leverage, liquidity) |
| Binary output only | Risk score (0–100) + risk category |
| No explainability | SHAP identifies key distress signals |
| Academic only | Deployed live Streamlit app |

---

## 📌 Overview

Built on the **Polish Companies Bankruptcy Dataset** (7,027 companies, 64 financial ratios), this system:

- Predicts bankruptcy with **97% ROC-AUC** (1-year horizon)
- Detects financial distress signals **up to 5 years in advance**
- Uses **SHAP** to identify which financial ratios are driving risk
- Converts predictions into an interpretable **0–100 risk score**
- Deployed as an interactive Streamlit dashboard with grouped tabbed inputs

---

## 📊 Multi-Year Forecasting Performance

This is the core differentiator — the model can detect financial distress signals **years before bankruptcy occurs.**

| Prediction Horizon | ROC-AUC |
|---|---|
| 1 Year before bankruptcy | **0.97** |
| 2 Years before bankruptcy | 0.88 |
| 3 Years before bankruptcy | 0.85 |
| 4 Years before bankruptcy | 0.87 |
| 5 Years before bankruptcy | 0.89 |

---

## 🗃️ Dataset

| Property | Value |
|---|---|
| Source | Polish Companies Bankruptcy Dataset (UCI / Kaggle) |
| Companies | 7,027 |
| Features | 64 Financial Ratios |
| Target | Bankruptcy (0 = Survived, 1 = Bankrupt) |
| Prediction horizons | 1–5 years before bankruptcy |

---

## 🔑 Key Financial Indicators

Top financial ratios driving the model's predictions — mapped to real-world financial categories:

| Category | Ratios Used |
|---|---|
| Profitability | Net profit / total assets, return on equity |
| Leverage | Total liabilities / total assets, debt ratio |
| Liquidity | Current ratio, working capital / total assets |
| Efficiency | Operating revenue / total assets |

These align with indicators used in traditional models like the **Altman Z-Score** and **Ohlson O-Score**.

---

## 🏗️ ML Pipeline

```
Polish Companies Dataset (1–5 year horizons)
        ↓
  Data Preprocessing  →  Missing value handling + feature selection
        ↓
  Model Comparison  →  Logistic Regression / Random Forest / XGBoost
        ↓
  XGBoost Selected  →  Best ROC-AUC across all horizons
        ↓
  SHAP Explainability  →  Feature importance per prediction
        ↓
  Risk Scoring  →  0–100 scale with Low / Medium / High categories
        ↓
  Streamlit App  →  Interactive financial ratio inputs + dashboard
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model | XGBoost |
| Explainability | SHAP |
| Visualization | Plotly (gauge charts) |
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy, Scikit-learn |

---

## 💻 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Azim521/AI-Bankruptcy-Risk-Analyzer.git
cd AI-Bankruptcy-Risk-Analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
AI-Bankruptcy-Risk-Analyzer/
├── app.py                  ← Streamlit dashboard
├── requirements.txt        ← Dependencies
├── Dashboard.png           ← App preview
└── model/
    ├── xgb_model.pkl       ← Trained XGBoost model
    └── scaler.pkl          ← Feature scaler
```

---

## 🔮 Future Improvements

- CSV upload for automated ratio calculation from financial statements
- SHAP explainability dashboard with interactive plots
- Integration with financial data APIs (Yahoo Finance, Alpha Vantage)
- Peer company benchmarking

---

## 📬 Contact

Built by **Azim Sadath**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/azim-sadath-a3ba34321/)
[![GitHub](https://img.shields.io/badge/GitHub-Azim521-black?style=flat&logo=github)](https://github.com/Azim521)
[![Email](https://img.shields.io/badge/Email-azimsadath521@gmail.com-red?style=flat&logo=gmail)](mailto:azimsadath521@gmail.com)
