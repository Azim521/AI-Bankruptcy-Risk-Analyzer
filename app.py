import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="AI Bankruptcy Risk Analyzer", layout="wide")

st.title("AI Corporate Bankruptcy Risk Analyzer")

st.write("""
This AI system predicts the probability that a company will go bankrupt
based on financial ratios using a trained XGBoost model.
""")

# Load model
model_path = os.path.join("model", "bankruptcy_xgb_model.pkl")
features_path = os.path.join("model", "bankruptcy_features.pkl")

model = joblib.load(model_path)
features = joblib.load(features_path)

# Top financial features
top_features = [
    'Attr24','Attr27','Attr13','Attr26','Attr23',
    'Attr14','Attr34','Attr22','Attr16','Attr21'
]

st.sidebar.header("Input Key Financial Ratios")

user_inputs = {}

for feature in top_features:
    user_inputs[feature] = st.sidebar.number_input(feature, value=0.0)

# Build full feature vector
input_data = {}

for feature in features:
    if feature in user_inputs:
        input_data[feature] = user_inputs[feature]
    else:
        input_data[feature] = 0

input_df = pd.DataFrame([input_data])

if st.button("Analyze Bankruptcy Risk"):

    prob = model.predict_proba(input_df)[0][1]
    score = prob * 100

    if score < 30:
        category = "Low Risk"
        color = "green"
    elif score < 60:
        category = "Medium Risk"
        color = "orange"
    else:
        category = "High Risk"
        color = "red"

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Bankruptcy Probability", f"{prob:.2f}")

    with col2:
        st.metric("Risk Score (0–100)", f"{score:.2f}")

    st.progress(score/100)

    st.markdown(f"### Risk Category: :{color}[{category}]")

    st.write("---")

    st.write("### Model Information")

    st.write("""
Model: **XGBoost**

Dataset: **Polish Companies Bankruptcy Dataset**

Features: **64 Financial Ratios**

Validation: **Multi-year forecasting (1–5 years before bankruptcy)**
""")
