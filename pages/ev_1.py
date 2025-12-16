import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

import math
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="EV Dynamic Range & Anxiety Reduction Lab",
    layout="wide"
)

# Hide sidebar
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display:none;}
section[data-testid="stSidebar"] {display:none;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# GLOBAL UI CSS (enterprise lab style)
# =========================================================
st.markdown("""
<style>
* { font-family: Inter, sans-serif; }

.big-header {
    font-size:36px;
    font-weight:700;
    margin-bottom:6px;
}
.section-title {
    font-size:24px;
    font-weight:600;
    margin-top:28px;
    margin-bottom:10px;
    color:#064b86;
}
.card {
    background:#fff;
    padding:22px;
    border-radius:14px;
    border:1px solid #e5e7eb;
    box-shadow:0 4px 16px rgba(0,0,0,0.06);
    font-size:16px;
}
.card:hover {
    transform:translateY(-4px);
    box-shadow:0 12px 30px rgba(6,75,134,0.18);
}
.kpi {
    background:#fff;
    padding:22px;
    border-radius:14px;
    border:1px solid #e5e7eb;
    text-align:center;
    font-size:20px;
    font-weight:600;
    color:#064b86;
}
.variable-box {
    padding:16px;
    border-radius:12px;
    border:1px solid #e5e7eb;
    background:#fff;
    text-align:center;
    margin-bottom:10px;
    font-weight:500;
}
.stButton>button,
.stDownloadButton>button {
    background:#064b86;
    color:white;
    border:none;
    padding:10px 22px;
    border-radius:8px;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display:flex; align-items:center; margin-bottom:10px;">
    <img src="{logo_url}" width="60" style="margin-right:12px;">
    <div>
        <div style="font-size:34px; font-weight:700; color:#064b86;">Analytics Avenue &</div>
        <div style="font-size:34px; font-weight:700; color:#064b86;">Advanced Analytics</div>
    </div>
</div>
<div class="big-header">EV Dynamic Range Prediction & Anxiety Reduction Lab</div>
""", unsafe_allow_html=True)

# =========================================================
# CONSTANTS
# =========================================================
DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/ev/ev_dynamic_range_dataset.csv"

REQUIRED_COLS = [
    "vehicle_id","battery_capacity_kwh","state_of_charge_pct","state_of_health_pct",
    "battery_temp_c","vehicle_weight_kg","payload_kg","avg_speed_kmph",
    "acceleration_score","regen_braking_pct","driving_style","terrain_type",
    "elevation_gain_m","traffic_level","ambient_temp_c","rain_intensity_mm",
    "wind_speed_kmph","energy_consumption_wh_per_km","static_range_km",
    "dynamic_range_km","actual_range_km","range_error_km","range_anxiety_flag"
]

NUM_COLS = [
    "battery_capacity_kwh","state_of_charge_pct","state_of_health_pct",
    "battery_temp_c","vehicle_weight_kg","payload_kg","avg_speed_kmph",
    "acceleration_score","regen_braking_pct","elevation_gain_m",
    "ambient_temp_c","rain_intensity_mm","wind_speed_kmph",
    "energy_consumption_wh_per_km","static_range_km","dynamic_range_km",
    "actual_range_km","range_error_km"
]

CAT_COLS = ["driving_style","terrain_type","traffic_level"]

# =========================================================
# HELPERS
# =========================================================
def download_df(df, filename, label):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode())
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")

def to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# =========================================================
# TAB 1 — OVERVIEW
# =========================================================
with tab1:
    st.markdown("""
    <div class="card">
    <b>Problem:</b> EV drivers do not trust the displayed range and fear being stranded.<br><br>
    <b>Why:</b> Static OEM logic, no driving behavior modeling, terrain & weather ignored.<br><br>
    <b>Solution:</b> ML-driven, personalized, context-aware range prediction that adapts in real time.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="card">
        <b>What this lab does</b><br>
        • Predicts realistic remaining range<br>
        • Quantifies OEM over/under-promise<br>
        • Flags anxiety-prone trips in advance<br>
        • Segments drivers by behavior
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
        <b>Business impact</b><br>
        • Higher driver trust<br>
        • Fewer breakdowns<br>
        • Better charging decisions<br>
        • Faster EV adoption
        </div>
        """, unsafe_allow_html=True)

    k1,k2,k3,k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Range Accuracy</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Anxiety Reduction</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Energy Efficiency</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Driver Trust Index</div>", unsafe_allow_html=True)

# =========================================================
# TAB 2 — IMPORTANT ATTRIBUTES
# =========================================================
with tab2:
    st.markdown('<div class="section-title">Required Data Dictionary</div>', unsafe_allow_html=True)

    dict_df = pd.DataFrame([
        ("battery_capacity_kwh","Total battery capacity"),
        ("state_of_charge_pct","Current SOC %"),
        ("state_of_health_pct","Battery health"),
        ("avg_speed_kmph","Average driving speed"),
        ("terrain_type","Terrain category"),
        ("ambient_temp_c","Outside temperature"),
        ("payload_kg","Vehicle payload"),
        ("energy_consumption_wh_per_km","Energy usage rate"),
        ("static_range_km","OEM displayed range"),
        ("dynamic_range_km","ML predicted range"),
        ("actual_range_km","Actual achieved range"),
        ("range_anxiety_flag","Anxiety indicator")
    ], columns=["Column","Description"])

    st.dataframe(dict_df, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-title'>Independent Variables</div>", unsafe_allow_html=True)
        for v in NUM_COLS + CAT_COLS:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='section-title'>Dependent Variables</div>", unsafe_allow_html=True)
        for v in ["actual_range_km","range_anxiety_flag"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# =========================================================
# TAB 3 — APPLICATION
# =========================================================
with tab3:
    st.markdown('<div class="section-title">Step 1: Load Dataset</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Dataset option:",
        ["Default dataset","Upload CSV","Upload CSV + Column Mapping"],
        horizontal=True
    )

    df = None

    if mode == "Default dataset":
        df = pd.read_csv(DEFAULT_URL)

    elif mode == "Upload CSV":
        up = st.file_uploader("Upload EV dataset", type=["csv"])
        if up:
            df = pd.read_csv(up)

    else:
        up = st.file_uploader("Upload CSV to map", type=["csv"])
        if up:
            raw = pd.read_csv(up)
            mapping = {}
            for c in REQUIRED_COLS:
                mapping[c] = st.selectbox(f"Map → {c}", ["-- Select --"] + list(raw.columns))
            if st.button("Apply mapping"):
                df = raw.rename(columns={v:k for k,v in mapping.items() if v!="-- Select --"})

    if df is None:
        st.stop()

    df.columns = df.columns.str.strip()
    df = to_numeric(df, NUM_COLS)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        st.stop()

    # =====================================================
    # KPIs
    # =====================================================
    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Trips", len(df))
    k2.metric("OEM Avg Error (km)", f"{df['range_error_km'].mean():.1f}")
    k3.metric("Dynamic Avg Error (km)", f"{(df['dynamic_range_km']-df['actual_range_km']).mean():.1f}")
    k4.metric("Anxiety Rate (%)", f"{df['range_anxiety_flag'].mean()*100:.1f}%")

    # =====================================================
    # VISUALS
    # =====================================================
    st.markdown('<div class="section-title">Diagnostics</div>', unsafe_allow_html=True)
    st.plotly_chart(px.scatter(df, x="static_range_km", y="actual_range_km",
                               title="Static vs Actual Range"), use_container_width=True)
    st.plotly_chart(px.scatter(df, x="dynamic_range_km", y="actual_range_km",
                               title="Dynamic vs Actual Range"), use_container_width=True)

    # =====================================================
    # ML MODEL 1 — RANGE REGRESSION
    # =====================================================
    st.markdown('<div class="section-title">ML — Dynamic Range Prediction</div>', unsafe_allow_html=True)

    X = df[NUM_COLS + CAT_COLS]
    y = df["actual_range_km"]

    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS)
    ])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = Pipeline([
        ("prep", prep),
        ("rf", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    reg.fit(Xtr, ytr)
    preds = reg.predict(Xte)

    rmse = math.sqrt(mean_squared_error(yte, preds))
    r2 = r2_score(yte, preds)

    range_metrics = pd.DataFrame([{
        "Model":"RandomForest",
        "Target":"Actual Range (km)",
        "RMSE_km":round(rmse,2),
        "R2":round(r2,4),
        "Rows_Used":len(yte)
    }])

    st.dataframe(range_metrics, use_container_width=True)
    download_df(range_metrics, "ev_range_model_metrics.csv", "Download Range Metrics")

    # =====================================================
    # ML MODEL 2 — ANXIETY CLASSIFICATION
    # =====================================================
    st.markdown('<div class="section-title">ML — Range Anxiety Prediction</div>', unsafe_allow_html=True)

    y_cls = df["range_anxiety_flag"]
    Xtr, Xte, ytr, yte = train_test_split(X, y_cls, test_size=0.2, random_state=42)

    clf = Pipeline([
        ("prep", prep),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
    ])

    clf.fit(Xtr, ytr)
    probs = clf.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, probs)

    anxiety_metrics = pd.DataFrame([{
        "Model":"RandomForestClassifier",
        "Target":"Range Anxiety",
        "ROC_AUC":round(auc,4),
        "Rows_Used":len(yte)
    }])

    st.dataframe(anxiety_metrics, use_container_width=True)
    download_df(anxiety_metrics, "ev_anxiety_model_metrics.csv", "Download Anxiety Metrics")

    # =====================================================
    # ML MODEL 3 — DRIVER CLUSTERING
    # =====================================================
    st.markdown('<div class="section-title">ML — Driver Behavior Clustering</div>', unsafe_allow_html=True)

    cl_features = ["avg_speed_kmph","acceleration_score",
                   "regen_braking_pct","energy_consumption_wh_per_km"]

    cl_df = df[cl_features].dropna()
    km = KMeans(n_clusters=4, random_state=42)
    cl_df["Cluster"] = km.fit_predict(cl_df)

    st.plotly_chart(
        px.scatter(cl_df, x="avg_speed_kmph", y="energy_consumption_wh_per_km",
                   color="Cluster", title="Driver Behavior Clusters"),
        use_container_width=True
    )

    cluster_summary = cl_df.groupby("Cluster")[cl_features].mean().round(2).reset_index()
    st.dataframe(cluster_summary, use_container_width=True)
    download_df(cluster_summary, "ev_driver_clusters.csv", "Download Cluster Summary")

    # =====================================================
    # AUTOMATED INSIGHTS
    # =====================================================
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)

    insights = [
        ["Worst OEM over-promise terrain", df.groupby("terrain_type")["range_error_km"].mean().idxmax()],
        ["Avg SOC during anxiety (%)", f"{df[df['range_anxiety_flag']==1]['state_of_charge_pct'].mean():.1f}"],
        ["Cold weather penalty (km)", f"{df[df['ambient_temp_c']<5]['range_error_km'].mean():.1f}"]
    ]

    ins_df = pd.DataFrame(insights, columns=["Insight","Value"])
    st.dataframe(ins_df, use_container_width=True)
    download_df(ins_df, "ev_automated_insights.csv", "Download Insights")
