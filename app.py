import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="AI Bankruptcy Risk Analyzer", layout="wide")

st.title("AI Corporate Bankruptcy Risk Analyzer")

st.write(
"""
This system predicts the probability that a company will go bankrupt
based on financial ratios using a trained XGBoost model.
"""
)

model = joblib.load("model/bankruptcy_xgb_model.pkl")
features = joblib.load("model/bankruptcy_features.pkl")

st.sidebar.header("Input Financial Ratios")

input_data = {}

for feature in features:
    input_data[feature] = st.sidebar.number_input(feature, value=0.0)

input_df = pd.DataFrame([input_data])

if st.button("Analyze Bankruptcy Risk"):

    prob = model.predict_proba(input_df)[0][1]
    score = prob * 100

    if score < 30:
        category = "Low Risk"
    elif score < 60:
        category = "Medium Risk"
    else:
        category = "High Risk"

    st.subheader("Prediction Result")

    st.write(f"Bankruptcy Probability: **{prob:.2f}**")
    st.write(f"Risk Score (0–100): **{score:.2f}**")
    st.write(f"Risk Category: **{category}**")
