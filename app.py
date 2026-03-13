import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="AI Bankruptcy Risk Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stMetric { background: #1a1d27; border-radius: 10px; padding: 16px; border: 1px solid #2a2d3e; }
    .risk-badge {
        display: inline-block; padding: 6px 20px; border-radius: 20px;
        font-weight: 700; font-size: 1.1rem; letter-spacing: 0.05em;
    }
    .risk-low    { background: #0d3320; color: #4ade80; border: 1px solid #4ade80; }
    .risk-medium { background: #3b2800; color: #fb923c; border: 1px solid #fb923c; }
    .risk-high   { background: #3b0a0a; color: #f87171; border: 1px solid #f87171; }
    h1, h2, h3 { color: #e2e8f0 !important; }
    .section-header {
        font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.12em; color: #64748b; margin-bottom: 12px; margin-top: 4px;
    }
    .stNumberInput label { font-size: 0.82rem !important; color: #94a3b8 !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.85rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Feature Name Mapping ─────────────────────────────────────────────────────
FEATURE_NAMES = {
    "Attr1":  "Net Profit / Total Assets",
    "Attr2":  "Total Liabilities / Total Assets",
    "Attr3":  "Working Capital / Total Assets",
    "Attr4":  "Current Assets / Short-term Liabilities",
    "Attr5":  "Cash Cycle Ratio (×365 days)",
    "Attr6":  "Retained Earnings / Total Assets",
    "Attr7":  "EBIT / Total Assets",
    "Attr8":  "Book Value of Equity / Total Liabilities",
    "Attr9":  "Sales / Total Assets",
    "Attr10": "Equity / Total Assets",
    "Attr11": "(Gross Profit + Extraordinary Items + Fin. Expenses) / Total Assets",
    "Attr12": "Gross Profit / Short-term Liabilities",
    "Attr13": "(Gross Profit + Depreciation) / Sales",
    "Attr14": "(Gross Profit + Interest) / Total Assets",
    "Attr15": "(Total Liabilities × 365) / (Gross Profit + Depreciation)",
    "Attr16": "(Gross Profit + Depreciation) / Total Liabilities",
    "Attr17": "Total Assets / Total Liabilities",
    "Attr18": "Gross Profit / Total Assets",
    "Attr19": "Gross Profit / Sales  (Gross Margin)",
    "Attr20": "Inventory Days  (Inventory × 365 / Sales)",
    "Attr21": "Sales Growth  (YoY)",
    "Attr22": "Operating Profit / Total Assets",
    "Attr23": "Net Profit Margin  (Net Profit / Sales)",
    "Attr24": "3-Year Gross Profit / Total Assets",
    "Attr25": "(Equity − Share Capital) / Total Assets",
    "Attr26": "(Net Profit + Depreciation) / Total Liabilities",
    "Attr27": "Operating Profit / Financial Expenses  (Interest Cover)",
    "Attr28": "Working Capital / Fixed Assets",
    "Attr29": "log(Total Assets)  — Size Proxy",
    "Attr30": "(Total Liabilities − Cash) / Sales",
    "Attr31": "(Gross Profit + Interest) / Sales",
    "Attr32": "Payables Days  (Current Liabilities × 365 / COGS)",
    "Attr33": "Operating Expenses / Short-term Liabilities",
    "Attr34": "Operating Expenses / Total Liabilities",
    "Attr35": "Profit on Sales / Total Assets",
    "Attr36": "Asset Turnover  (Total Sales / Total Assets)",
    "Attr37": "(Current Assets − Inventories) / Long-term Liabilities",
    "Attr38": "Constant Capital / Total Assets",
    "Attr39": "Profit on Sales / Sales",
    "Attr40": "Quick Ratio excl. Receivables  ((CA − Inv − Rec) / STL)",
    "Attr41": "Debt Repayment Period  (days)",
    "Attr42": "Operating Profit Margin  (Op. Profit / Sales)",
    "Attr43": "Receivables + Inventory Turnover  (days)",
    "Attr44": "Receivables Days  (Receivables × 365 / Sales)",
    "Attr45": "Net Profit / Inventory",
    "Attr46": "Quick Ratio  ((Current Assets − Inventory) / STL)",
    "Attr47": "Inventory Days by COGS  (Inventory × 365 / COGS)",
    "Attr48": "EBITDA / Total Assets",
    "Attr49": "EBITDA Margin  (EBITDA / Sales)",
    "Attr50": "Current Assets / Total Liabilities",
    "Attr51": "Short-term Liabilities / Total Assets",
    "Attr52": "Payables Days  (STL × 365 / COGS)",
    "Attr53": "Equity / Fixed Assets",
    "Attr54": "Constant Capital / Fixed Assets",
    "Attr55": "Working Capital  (absolute, PLN)",
    "Attr56": "Gross Margin  ((Sales − COGS) / Sales)",
    "Attr57": "Cash Conversion Efficiency",
    "Attr58": "Cost Ratio  (Total Costs / Total Sales)",
    "Attr59": "Long-term Liabilities / Equity",
    "Attr60": "Inventory Turnover  (Sales / Inventory)",
    "Attr61": "Receivables Turnover  (Sales / Receivables)",
    "Attr62": "STL Days  (Short-term Liabilities × 365 / Sales)",
    "Attr63": "Sales / Short-term Liabilities",
    "Attr64": "Fixed Asset Turnover  (Sales / Fixed Assets)",
}

FEATURE_HELP = {
    "Attr1":  "Measures how profitable a company is relative to its total assets. Higher = better.",
    "Attr2":  "Debt load relative to assets. Above 1.0 is insolvent. Lower = safer.",
    "Attr3":  "Liquidity buffer as a share of total assets. Positive values preferred.",
    "Attr4":  "Classic current ratio. >1 means current assets cover short-term debts.",
    "Attr5":  "Days of operating expenses coverable by liquid assets. Higher = more cushion.",
    "Attr6":  "Accumulated profit reinvested. Higher = stronger retained earnings.",
    "Attr7":  "Earnings before interest & tax divided by assets. Core profitability.",
    "Attr8":  "Equity cushion vs total debt. Higher = more solvent.",
    "Attr9":  "How efficiently assets generate sales. Higher = more productive.",
    "Attr10": "Share of assets financed by equity (not debt). Higher = less leveraged.",
    "Attr13": "Cash generation ability from sales. Higher = better cash profit margin.",
    "Attr14": "Pre-tax, pre-interest profitability on assets.",
    "Attr16": "Cash profit generated per unit of debt. Higher = safer debt service.",
    "Attr21": "Revenue growth rate vs prior year. >1 means growth.",
    "Attr22": "Operating efficiency: profit per asset dollar.",
    "Attr23": "How many cents of net profit per dollar of sales.",
    "Attr24": "Sustained 3-year profitability. Key Altman Z-Score analog.",
    "Attr26": "Cash available for debt repayment after depreciation add-back.",
    "Attr27": "How many times operating profit covers interest charges. <1 = danger.",
    "Attr34": "Operating cost burden relative to total liabilities.",
}

# ── Grouped Feature Layout ───────────────────────────────────────────────────
GROUPS = {
    "📈 Profitability": [
        "Attr1","Attr6","Attr7","Attr11","Attr13","Attr14","Attr18",
        "Attr19","Attr22","Attr23","Attr24","Attr31","Attr35","Attr39",
        "Attr42","Attr48","Attr49","Attr56",
    ],
    "💧 Liquidity": [
        "Attr3","Attr4","Attr5","Attr12","Attr28","Attr40",
        "Attr46","Attr50","Attr55","Attr57",
    ],
    "🏦 Leverage / Solvency": [
        "Attr2","Attr8","Attr10","Attr15","Attr16","Attr17","Attr25",
        "Attr26","Attr30","Attr34","Attr38","Attr41","Attr51","Attr53",
        "Attr54","Attr58","Attr59",
    ],
    "⚙️ Efficiency / Turnover": [
        "Attr9","Attr20","Attr21","Attr27","Attr29","Attr32","Attr33",
        "Attr36","Attr43","Attr44","Attr45","Attr47","Attr52","Attr60",
        "Attr61","Attr62","Attr63","Attr64",
    ],
}

TOP_FEATURES = [
    "Attr24","Attr27","Attr13","Attr26","Attr23",
    "Attr14","Attr34","Attr22","Attr16","Attr21"
]

# ── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path    = os.path.join("model", "bankruptcy_xgb_model.pkl")
    features_path = os.path.join("model", "bankruptcy_features.pkl")
    medians_path  = os.path.join("model", "feature_medians.pkl")
    m = joblib.load(model_path)
    f = joblib.load(features_path)
    # Load medians if available, else fall back to 0 for all features
    if os.path.exists(medians_path):
        med = joblib.load(medians_path)
    else:
        med = {feat: 0.0 for feat in f}
    return m, f, med

model, features, feature_medians = load_model()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 🏦 AI Corporate Bankruptcy Risk Analyzer")
st.markdown(
    "Predict the probability of corporate bankruptcy from 63 financial ratios using "
    "an XGBoost model trained on the **Polish Companies Bankruptcy Dataset** (7,027 firms)."
)
st.markdown("---")

# ── Input Mode Toggle ────────────────────────────────────────────────────────
mode = st.radio(
    "Input mode",
    ["⚡ Quick — Top 10 Key Ratios", "🔬 Full — All 63 Ratios"],
    horizontal=True,
    help="Quick mode uses only the 10 most predictive features; all others are imputed with training-set medians."
)
st.markdown("")

user_inputs = {}

if mode == "⚡ Quick — Top 10 Key Ratios":
    st.markdown("### Key Financial Ratios")
    st.caption("These 10 ratios drive the most predictive power in the model.")
    cols = st.columns(2)
    for i, feat in enumerate(TOP_FEATURES):
        label = FEATURE_NAMES.get(feat, feat)
        help_text = FEATURE_HELP.get(feat, "")
        with cols[i % 2]:
            user_inputs[feat] = st.number_input(
                label, value=0.0, format="%.4f",
                key=feat, help=help_text
            )
else:
    tab_names = list(GROUPS.keys())
    tabs = st.tabs(tab_names)
    for tab, (group_name, group_features) in zip(tabs, GROUPS.items()):
        with tab:
            st.markdown(f'<p class="section-header">{group_name.split(" ", 1)[1]}</p>', unsafe_allow_html=True)
            cols = st.columns(2)
            for i, feat in enumerate(group_features):
                label = FEATURE_NAMES.get(feat, feat)
                help_text = FEATURE_HELP.get(feat, "")
                with cols[i % 2]:
                    user_inputs[feat] = st.number_input(
                        label, value=0.0, format="%.4f",
                        key=feat, help=help_text
                    )

st.markdown("")

# ── Build Input Vector ────────────────────────────────────────────────────────
# Use median for any feature the user didn't explicitly provide
# This avoids treating "unknown" as 0, which often signals distress to the model
input_data = {
    feat: user_inputs[feat] if feat in user_inputs else feature_medians.get(feat, 0.0)
    for feat in features
}
input_df = pd.DataFrame([input_data])

# Show imputation notice in Quick mode
if mode == "⚡ Quick — Top 10 Key Ratios":
    n_imputed = len(features) - len(user_inputs)
    st.caption(
        f"ℹ️ {n_imputed} unset features will be imputed with their training-set median — "
        "representing a financially average company for those ratios."
    )

# ── Analyze Button ────────────────────────────────────────────────────────────
if st.button("🔍 Analyze Bankruptcy Risk", type="primary", use_container_width=False):

    prob  = model.predict_proba(input_df)[0][1]
    score = prob * 100

    if score < 30:
        category, badge_class, bar_color = "Low Risk",    "risk-low",    "#4ade80"
    elif score < 60:
        category, badge_class, bar_color = "Medium Risk", "risk-medium", "#fb923c"
    else:
        category, badge_class, bar_color = "High Risk",   "risk-high",   "#f87171"

    st.markdown("---")
    st.markdown("## 📊 Prediction Results")

    # ── Gauge ─────────────────────────────────────────────────────────────────
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number={"suffix": "%", "font": {"size": 36, "color": "#e2e8f0"}},
        title={"text": "Bankruptcy Risk Score", "font": {"size": 16, "color": "#94a3b8"}},
        delta={"reference": 30, "increasing": {"color": "#f87171"}, "decreasing": {"color": "#4ade80"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748b", "tickfont": {"color": "#64748b"}},
            "bar": {"color": bar_color, "thickness": 0.25},
            "bgcolor": "#1a1d27",
            "bordercolor": "#2a2d3e",
            "steps": [
                {"range": [0, 30],   "color": "#0d3320"},
                {"range": [30, 60],  "color": "#3b2800"},
                {"range": [60, 100], "color": "#3b0a0a"},
            ],
            "threshold": {
                "line": {"color": bar_color, "width": 3},
                "thickness": 0.75,
                "value": score,
            },
        }
    ))
    fig.update_layout(
        height=280, margin=dict(t=40, b=10, l=30, r=30),
        paper_bgcolor="#0f1117", font_color="#e2e8f0"
    )

    col_gauge, col_metrics = st.columns([1, 1])
    with col_gauge:
        st.plotly_chart(fig, use_container_width=True)
    with col_metrics:
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("Bankruptcy Probability", f"{prob:.1%}")
        st.metric("Risk Score (0–100)",      f"{score:.1f}")
        st.markdown(
            f'<span class="risk-badge {badge_class}">{category}</span>',
            unsafe_allow_html=True
        )
        st.markdown("")
        if score >= 60:
            st.error("⚠️ High distress signals detected. Immediate financial review recommended.")
        elif score >= 30:
            st.warning("⚡ Moderate risk signals present. Monitor key ratios closely.")
        else:
            st.success("✅ Company financials appear stable based on submitted ratios.")

    # ── Multi-Year Context ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📅 Multi-Year Model Accuracy Context")
    st.caption(
        "This model was also validated on financial data from 2–5 years before bankruptcy. "
        "These are historical ROC-AUC scores showing how early financial distress can be detected."
    )
    horizon_df = pd.DataFrame({
        "Prediction Horizon": ["1 Year (current)", "2 Years Out", "3 Years Out", "4 Years Out", "5 Years Out"],
        "ROC-AUC": [0.97, 0.88, 0.85, 0.87, 0.89],
    })
    fig2 = go.Figure(go.Bar(
        x=horizon_df["Prediction Horizon"],
        y=horizon_df["ROC-AUC"],
        marker_color=["#6366f1","#818cf8","#a5b4fc","#c7d2fe","#e0e7ff"],
        text=[f"{v:.2f}" for v in horizon_df["ROC-AUC"]],
        textposition="outside",
        textfont=dict(color="#e2e8f0"),
    ))
    fig2.update_layout(
        yaxis=dict(range=[0.75, 1.0], tickformat=".2f", gridcolor="#2a2d3e", color="#94a3b8"),
        xaxis=dict(color="#94a3b8"),
        paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
        height=300, margin=dict(t=20, b=20, l=40, r=20),
        font_color="#e2e8f0",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Top Features Reference ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔑 Top Predictive Ratios (Model Reference)")
    ref_data = [
        {"Rank": i+1, "Feature ID": f, "Financial Ratio": FEATURE_NAMES.get(f, f)}
        for i, f in enumerate(TOP_FEATURES)
    ]
    st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Model: XGBoost · Dataset: Polish Companies Bankruptcy (UCI) · "
    "1-Year ROC-AUC: 0.97 · Built by Azim Sadath"
)
