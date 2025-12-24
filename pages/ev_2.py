# =========================================================
# EV CHARGING STATION FAULT DETECTION LAB (3-TAB VERSION)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="EV Charging Station Fault Detection",
    layout="wide"
)

# =========================================================
# HIDE STREAMLIT CHROME
# =========================================================
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display:none;}
section[data-testid="stSidebar"] {display:none;}
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# GLOBAL CSS
# =========================================================
st.markdown("""
<style>
* { font-family: Inter, sans-serif; }

.header {
    font-size:34px;
    font-weight:700;
    margin-bottom:8px;
}
.section-title {
    font-size:22px;
    font-weight:600;
    margin-top:26px;
    margin-bottom:12px;
    color:#064b86;
}
.card {
    background:#fff;
    padding:20px;
    border-radius:14px;
    border:1px solid #e5e5e5;
    box-shadow:0 4px 16px rgba(0,0,0,0.07);
}
.kpi {
    background:#fff;
    padding:20px;
    border-radius:14px;
    border:1px solid #e2e2e2;
    text-align:center;
    font-size:20px;
    font-weight:600;
    color:#064b86;
}
.variable-box {
    background:#fff;
    padding:14px;
    border-radius:12px;
    border:1px solid #e5e5e5;
    text-align:center;
    font-weight:500;
    margin-bottom:8px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
logo = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display:flex;align-items:center;margin-bottom:10px;">
<img src="{logo}" width="55" style="margin-right:12px;">
<div>
<div style="font-size:32px;font-weight:700;color:#064b86;">Analytics Avenue &</div>
<div style="font-size:32px;font-weight:700;color:#064b86;">Advanced Analytics</div>
</div>
</div>
<div class="header">EV Charging Station Fault Detection Lab</div>
""", unsafe_allow_html=True)

# =========================================================
# CONSTANTS
# =========================================================
DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/ev/charging_station_fault_data.csv"

REQUIRED_COLS = [
    "charger_id","timestamp","voltage","current",
    "temperature","power_kw","session_active",
    "fault_flag","fault_type"
]

NUM_COLS = ["voltage","current","temperature","power_kw"]

# =========================================================
# HELPERS
# =========================================================
def download_df(df, name, label):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode())
    b.seek(0)
    st.download_button(label, b, name, "text/csv")

# =========================================================
# TABS (ONLY 3)
# =========================================================
tab1, tab2, tab3 = st.tabs(
    ["Overview", "Important Attributes", "Application"]
)

# =========================================================
# TAB 1 – OVERVIEW
# =========================================================
with tab1:
    st.markdown('<div class="section-title">Use Case Overview</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <b>Problem:</b><br><br>
    EV charging stations suffer from silent failures, thermal risks, and electrical
    instability that traditional rule-based monitoring fails to detect early.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="card">
        <b>What this system detects</b><br>
        • Voltage drop & instability<br>
        • Overcurrent conditions<br>
        • Thermal overheating<br>
        • Session-active but zero power delivery
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
        <b>Business Impact</b><br>
        • Reduced charger downtime<br>
        • Faster root cause isolation<br>
        • Proactive maintenance planning<br>
        • Improved driver trust & EV adoption
        </div>
        """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Charger Health</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Fault Rate</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Silent Failures</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Thermal Risk</div>", unsafe_allow_html=True)

# =========================================================
# TAB 2 – IMPORTANT ATTRIBUTES
# =========================================================
with tab2:
    st.markdown('<div class="section-title">Data Dictionary</div>', unsafe_allow_html=True)

    dict_df = pd.DataFrame([
        ("charger_id","Unique charging station identifier"),
        ("timestamp","Sensor timestamp"),
        ("voltage","Measured voltage (V)"),
        ("current","Measured current (A)"),
        ("power_kw","Delivered power (kW)"),
        ("temperature","Internal charger temperature (°C)"),
        ("session_active","Charging session active flag"),
        ("fault_flag","Binary fault indicator"),
        ("fault_type","Fault classification label")
    ], columns=["Column","Description"])

    st.dataframe(dict_df, use_container_width=True)
    download_df(dict_df, "charger_data_dictionary.csv", "Download Data Dictionary")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Independent Variables")
        for v in ["voltage","current","temperature","power_kw","session_active"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("### Dependent Variables")
        for v in ["fault_flag","fault_type"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# =========================================================
# TAB 3 – APPLICATION (EDA + ML + INSIGHTS)
# =========================================================
with tab3:
    st.markdown('<div class="section-title">Load Dataset</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Dataset option:",
        ["Default (GitHub)", "Upload CSV"],
        horizontal=True
    )

    df = None
    if mode == "Default (GitHub)":
        df = pd.read_csv(DEFAULT_URL)
    else:
        file = st.file_uploader("Upload charger dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)

    if df is None:
        st.stop()

    df.columns = df.columns.str.strip()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        st.stop()

    # =====================================================
    # FILTERS
    # =====================================================
    st.markdown('<div class="section-title">Filters</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    sel_charger = c1.multiselect("Charger ID", df["charger_id"].unique(), df["charger_id"].unique())
    sel_fault = c2.multiselect("Fault Type", df["fault_type"].unique(), df["fault_type"].unique())
    active_only = c3.checkbox("Only Active Sessions")

    df_f = df[
        df["charger_id"].isin(sel_charger) &
        df["fault_type"].isin(sel_fault)
    ]
    if active_only:
        df_f = df_f[df_f["session_active"] == 1]

    st.dataframe(df_f.head(10), use_container_width=True)

    # =====================================================
    # KPIs
    # =====================================================
    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Records", len(df_f))
    k2.metric("Fault Rate (%)", f"{df_f['fault_flag'].mean()*100:.1f}")
    k3.metric("Avg Temp (°C)", f"{df_f['temperature'].mean():.1f}")
    silent = df_f[(df_f["session_active"]==1) & (df_f["power_kw"]<5)]
    k4.metric("Silent Failure %", f"{len(silent)/len(df_f)*100:.1f}")

    # =====================================================
    # EDA
    # =====================================================
    st.markdown('<div class="section-title">Diagnostics</div>', unsafe_allow_html=True)

    st.plotly_chart(
        px.scatter(df_f, x="voltage", y="current", color="fault_type",
                   title="Voltage vs Current"),
        use_container_width=True
    )

    st.plotly_chart(
        px.line(df_f.sort_values("timestamp"),
                x="timestamp", y="temperature",
                color="fault_flag",
                title="Temperature Over Time"),
        use_container_width=True
    )

    # =====================================================
    # ML – FAULT CLASSIFICATION
    # =====================================================
    st.markdown('<div class="section-title">ML – Fault Classification</div>', unsafe_allow_html=True)

    X = df_f[["voltage","current","temperature","power_kw","session_active"]]
    y = df_f["fault_flag"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    probs = model.predict_proba(Xte)[:,1]

    metrics_df = pd.DataFrame([{
        "Model":"RandomForestClassifier",
        "ROC_AUC": round(roc_auc_score(yte, probs),3),
        "Accuracy": round(accuracy_score(yte, preds),3),
        "Rows_Used": len(yte)
    }])

    st.dataframe(metrics_df, use_container_width=True)
    download_df(metrics_df, "fault_model_metrics.csv", "Download Model Metrics")

    # =====================================================
    # UNSUPERVISED ANOMALY DETECTION
    # =====================================================
    st.markdown('<div class="section-title">Anomaly Detection</div>', unsafe_allow_html=True)

    iso = IsolationForest(contamination=0.05, random_state=42)
    df_f["anomaly"] = iso.fit_predict(X).map({1:0,-1:1})

    st.plotly_chart(
        px.scatter(df_f, x="temperature", y="power_kw",
                   color="anomaly",
                   title="Isolation Forest – Detected Anomalies"),
        use_container_width=True
    )

    # =====================================================
    # AUTOMATED INSIGHTS
    # =====================================================
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)

    insights_df = pd.DataFrame([
        ["Worst Charger", df.groupby("charger_id")["fault_flag"].mean().idxmax()],
        ["Most Common Fault", df["fault_type"].value_counts().idxmax()],
        ["Avg Temp During Fault", f"{df[df['fault_flag']==1]['temperature'].mean():.1f} °C"],
        ["Silent Failure Rate", f"{len(silent)/len(df_f)*100:.1f}%"]
    ], columns=["Insight","Value"])

    st.dataframe(insights_df, use_container_width=True)
    download_df(insights_df, "automated_insights.csv", "Download Insights")

