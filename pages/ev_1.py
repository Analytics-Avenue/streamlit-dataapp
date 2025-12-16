import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, roc_auc_score

import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="EV Range Prediction & Anxiety Reduction Lab", layout="wide")

# Hide default sidebar nav
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display:none;}
section[data-testid="stSidebar"] {display:none;}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header & Logo
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display: flex; align-items: center; margin-bottom:16px;">
    <img src="{logo_url}" width="60" style="margin-right:12px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:700;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:700;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Global CSS – SAME AS ETA LAB
# -------------------------
st.markdown("""
<style>
* { font-family: 'Inter', sans-serif; }
body, [class*="css"] { color:#000 !important; font-size:17px; }

.big-header {
    font-size: 36px !important;
    font-weight: 700 !important;
    color:#000 !important;
    margin-bottom:12px;
}

.section-title {
    font-size: 24px !important;
    font-weight: 600 !important;
    margin-top:30px;
    margin-bottom:12px;
    color:#000 !important;
    position:relative;
}
.section-title:after {
    content:"";
    position:absolute;
    bottom:-5px;
    left:0;
    height:2px;
    width:0%;
    background:#064b86;
    transition: width 0.35s ease;
}
.section-title:hover:after { width:40%; }

.card {
    background:#ffffff;
    padding:22px;
    border-radius:14px;
    border:1px solid #e6e6e6;
    font-size:16.5px;
    color:#000 !important;
    font-weight:500;
    box-shadow:0 3px 14px rgba(0,0,0,0.08);
    transition: all 0.25s ease;
}
.card:hover {
    transform:translateY(-4px);
    box-shadow:0 12px 25px rgba(6,75,134,0.18);
    border-color:#064b86;
}

.kpi {
    background:#ffffff;
    padding:22px;
    border-radius:14px;
    border:1px solid #e2e2e2;
    font-size:20px !important;
    font-weight:600 !important;
    text-align:center;
    color:#064b86 !important;
    box-shadow:0 3px 14px rgba(0,0,0,0.07);
    transition:0.25s ease;
}

.variable-box {
    padding:18px;
    border-radius:14px;
    background:white;
    border:1px solid #e5e5e5;
    box-shadow:0 2px 10px rgba(0,0,0,0.10);
    transition:0.25s ease;
    text-align:center;
    font-size:17.5px !important;
    font-weight:500 !important;
    color:#064b86 !important;
    margin-bottom:12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-header'>EV Range Prediction & Anxiety Reduction Lab</div>", unsafe_allow_html=True)

# -------------------------
# Constants
# -------------------------
DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/ev/ev_dynamic_range_dataset.csv"

REQUIRED_COLS = [
    "battery_capacity_kwh","state_of_charge_pct","state_of_health_pct",
    "avg_speed_kmph","acceleration_score","regen_braking_pct",
    "terrain_type","ambient_temp_c","rain_intensity_mm","wind_speed_kmph",
    "payload_kg","energy_consumption_wh_per_km",
    "static_range_km","dynamic_range_km","actual_range_km",
    "range_error_km","range_anxiety_flag"
]

NUM_COLS = [
    "battery_capacity_kwh","state_of_charge_pct","state_of_health_pct",
    "avg_speed_kmph","acceleration_score","regen_braking_pct",
    "ambient_temp_c","rain_intensity_mm","wind_speed_kmph",
    "payload_kg","energy_consumption_wh_per_km",
    "static_range_km","dynamic_range_km","actual_range_km","range_error_km"
]

CAT_COLS = ["terrain_type"]

def download_df(df, name, label):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=name, mime="text/csv")

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# =====================================================
# TAB 1 – OVERVIEW
# =====================================================
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <b>Problem:</b><br><br>
    EV drivers frequently distrust the displayed range and fear being stranded. 
    This anxiety comes from static range calculations that ignore real-world driving behavior, terrain, and weather.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Why Range Anxiety Happens</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Static OEM range assumptions<br>
        • No personalization by driving style<br>
        • Terrain & elevation ignored<br>
        • Weather impact not modeled<br>
        • Battery aging treated optimistically
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Analytics Solution</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Dynamic ML-based range prediction<br>
        • Driver behavior & speed-aware modeling<br>
        • Terrain, weather & payload adjustments<br>
        • Continuous learning from actual trips
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Lower Range Anxiety</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Accurate Range Display</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Fewer Breakdowns</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Higher EV Adoption</div>", unsafe_allow_html=True)

# =====================================================
# TAB 2 – IMPORTANT ATTRIBUTES
# =====================================================
with tab2:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    attr_desc = {
        "battery_capacity_kwh": "Total battery capacity",
        "state_of_charge_pct": "Current SOC percentage",
        "state_of_health_pct": "Battery health percentage",
        "avg_speed_kmph": "Average driving speed",
        "acceleration_score": "Driving aggressiveness indicator",
        "regen_braking_pct": "Regenerative braking usage",
        "terrain_type": "Terrain category",
        "ambient_temp_c": "Ambient temperature",
        "rain_intensity_mm": "Rainfall intensity",
        "wind_speed_kmph": "Wind speed",
        "payload_kg": "Additional payload weight",
        "energy_consumption_wh_per_km": "Energy consumption rate",
        "static_range_km": "OEM static range estimate",
        "dynamic_range_km": "ML predicted range",
        "actual_range_km": "Actual achieved range",
        "range_error_km": "Static vs actual error",
        "range_anxiety_flag": "Driver anxiety indicator"
    }

    dict_df = pd.DataFrame(
        [{"Attribute": k, "Description": v} for k, v in attr_desc.items()]
    )
    st.dataframe(dict_df, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        independents = [
            "battery_capacity_kwh","state_of_charge_pct","state_of_health_pct",
            "avg_speed_kmph","acceleration_score","regen_braking_pct",
            "terrain_type","ambient_temp_c","rain_intensity_mm",
            "wind_speed_kmph","payload_kg"
        ]
        for v in independents:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        dependents = [
            "dynamic_range_km",
            "actual_range_km",
            "range_error_km",
            "range_anxiety_flag"
        ]
        for v in dependents:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# =====================================================
# TAB 3 – APPLICATION
# =====================================================
with tab3:
    st.markdown('<div class="section-title">Step 1: Load dataset</div>', unsafe_allow_html=True)

    df = None
    mode = st.radio(
        "Select Dataset Option:",
        ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"],
        horizontal=True
    )

    if mode == "Default dataset":
        df = pd.read_csv(DEFAULT_URL)
        st.success("Default EV dataset loaded.")
        st.dataframe(df.head(), use_container_width=True)

    elif mode == "Upload CSV":
        uploaded = st.file_uploader("Upload EV dataset", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.dataframe(df.head(), use_container_width=True)

    else:
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded:
            raw = pd.read_csv(uploaded)
            mapping = {}
            for c in REQUIRED_COLS:
                mapping[c] = st.selectbox(f"Map → {c}", ["-- Select --"] + list(raw.columns))
            if st.button("Apply mapping"):
                df = raw.rename(columns={mapping[k]: k for k in mapping})
                st.success("Mapping applied.")
                st.dataframe(df.head(), use_container_width=True)

    if df is None:
        st.stop()

    # -------------------------
    # KPIs
    # -------------------------
    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Trips", len(df))
    k2.metric("Avg Static Error (km)", round(df["range_error_km"].mean(),2))
    k3.metric("Anxiety Rate (%)", round(df["range_anxiety_flag"].mean()*100,1))
    k4.metric("Avg Dynamic Range (km)", round(df["dynamic_range_km"].mean(),1))

    # -------------------------
    # Charts
    # -------------------------
    st.markdown('<div class="section-title">Diagnostics</div>', unsafe_allow_html=True)

    st.plotly_chart(
        px.histogram(df, x="range_error_km", title="Static Range Error Distribution"),
        use_container_width=True
    )

    st.plotly_chart(
        px.box(df, x="terrain_type", y="actual_range_km", title="Actual Range by Terrain"),
        use_container_width=True
    )

    st.plotly_chart(
        px.scatter(df, x="avg_speed_kmph", y="energy_consumption_wh_per_km",
                   color="range_anxiety_flag",
                   title="Speed vs Energy Consumption"),
        use_container_width=True
    )

    # -------------------------
    # ML – Range Prediction
    # -------------------------
    st.markdown('<div class="section-title">ML — Dynamic Range Prediction</div>', unsafe_allow_html=True)

    feat_cols = [
        "battery_capacity_kwh","state_of_charge_pct","state_of_health_pct",
        "avg_speed_kmph","acceleration_score","regen_braking_pct",
        "ambient_temp_c","payload_kg","energy_consumption_wh_per_km",
        "terrain_type"
    ]

    ml_df = df.dropna(subset=feat_cols + ["actual_range_km"])
    if len(ml_df) > 100:
        X = ml_df[feat_cols]
        y = ml_df["actual_range_km"]

        cat_cols = ["terrain_type"]
        num_cols = [c for c in feat_cols if c not in cat_cols]

        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols)
        ])

        X_t = pre.fit_transform(X)
        Xtr, Xte, ytr, yte = train_test_split(X_t, y, test_size=0.2, random_state=42)

        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(Xtr, ytr)
        preds = rf.predict(Xte)

        st.write("MAE:", round(mean_absolute_error(yte, preds),2),
                 "R²:", round(r2_score(yte, preds),3))

    download_df(df.head(1000), "ev_range_filtered_sample.csv", "Download sample")
