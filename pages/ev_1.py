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
from sklearn.cluster import KMeans

import math
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="EV Dynamic Range & Anxiety Reduction Lab", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebarNav"] {display:none;}
section[data-testid="stSidebar"] {display:none;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display:flex; align-items:center; margin-bottom:16px;">
    <img src="{logo_url}" width="60" style="margin-right:12px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:700;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:700;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# GLOBAL CSS (SAME AS ETA APP)
# =========================================================
st.markdown("""
<style>
* { font-family: 'Inter', sans-serif; }
body, [class*="css"] { color:#000 !important; font-size:17px; }

.big-header { font-size:36px; font-weight:700; margin-bottom:12px; }

.section-title {
    font-size:24px; font-weight:600;
    margin-top:30px; margin-bottom:12px;
    position:relative;
}
.section-title:after {
    content:""; position:absolute; bottom:-5px; left:0;
    height:2px; width:0%; background:#064b86;
    transition:width 0.35s ease;
}
.section-title:hover:after { width:40%; }

.card {
    background:#fff; padding:22px;
    border-radius:14px; border:1px solid #e6e6e6;
    box-shadow:0 3px 14px rgba(0,0,0,0.08);
    transition:0.25s;
}
.card:hover {
    transform:translateY(-4px);
    box-shadow:0 12px 25px rgba(6,75,134,0.18);
}

.kpi {
    background:#fff; padding:22px;
    border-radius:14px; border:1px solid #e2e2e2;
    font-size:20px; font-weight:600;
    text-align:center; color:#064b86;
    transition:0.25s;
}
.kpi:hover {
    transform:translateY(-4px);
    box-shadow:0 13px 26px rgba(6,75,134,0.20);
}

.variable-box {
    padding:18px; border-radius:14px;
    background:white; border:1px solid #e5e5e5;
    box-shadow:0 2px 10px rgba(0,0,0,0.10);
    transition:0.25s;
    text-align:center; color:#064b86;
    margin-bottom:12px;
}
.variable-box:hover {
    transform:translateY(-5px);
    box-shadow:0 12px 22px rgba(6,75,134,0.18);
}

.stButton>button, .stDownloadButton>button {
    background:#064b86; color:white;
    padding:10px 22px; border-radius:8px;
    font-weight:600;
}
.stButton>button:hover { background:#0a6eb3; transform:translateY(-3px); }

.block-container { animation:fadeIn 0.5s ease; }
@keyframes fadeIn {
    from {opacity:0; transform:translateY(10px);}
    to {opacity:1; transform:translateY(0);}
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-header'>EV Dynamic Range Prediction & Anxiety Reduction Lab</div>", unsafe_allow_html=True)

# =========================================================
# CONSTANTS
# =========================================================
DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/ev/ev_dynamic_range_dataset.csv"

NUM_COLS = [
    "battery_capacity_kwh","state_of_charge_pct","state_of_health_pct",
    "battery_temp_c","vehicle_weight_kg","payload_kg","avg_speed_kmph",
    "acceleration_score","regen_braking_pct","elevation_gain_m",
    "ambient_temp_c","rain_intensity_mm","wind_speed_kmph",
    "energy_consumption_wh_per_km","static_range_km",
    "dynamic_range_km","actual_range_km","range_error_km"
]

CAT_COLS = ["driving_style","terrain_type","traffic_level"]

# =========================================================
# HELPERS
# =========================================================
def download_df(df, name, label):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode())
    b.seek(0)
    st.download_button(label, b, name, "text/csv")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# =========================================================
# TAB 1 – OVERVIEW
# =========================================================
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <b>Problem:</b> EV drivers do not trust the displayed range and fear being stranded.<br><br>
    <b>Root causes:</b><br>
    • Static OEM range logic<br>
    • No driving behavior awareness<br>
    • Terrain & weather ignored<br><br>
    <b>Solution:</b> Dynamic ML-driven range estimation personalized per driver.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">• Predict realistic remaining range<br>• Detect OEM over-promise<br>• Flag anxiety-prone trips early</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">• Higher trust & adoption<br>• Fewer roadside failures<br>• Smarter charging behavior</div>', unsafe_allow_html=True)

# =========================================================
# TAB 2 – IMPORTANT ATTRIBUTES
# =========================================================
with tab2:
    st.markdown('<div class="section-title">Required Data Dictionary</div>', unsafe_allow_html=True)

    dict_df = pd.DataFrame([
        ("battery_capacity_kwh","Battery size","Independent"),
        ("state_of_charge_pct","Current SOC","Independent"),
        ("avg_speed_kmph","Driving speed","Independent"),
        ("terrain_type","Road terrain","Independent"),
        ("ambient_temp_c","Weather","Independent"),
        ("energy_consumption_wh_per_km","Consumption rate","Derived"),
        ("static_range_km","OEM estimate","Dependent"),
        ("dynamic_range_km","ML estimate","Dependent"),
        ("actual_range_km","Actual achieved","Target"),
        ("range_anxiety_flag","Anxiety indicator","Target")
    ], columns=["Column","Description","Role"])

    st.dataframe(dict_df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        for v in NUM_COLS + CAT_COLS:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        for v in ["actual_range_km","range_anxiety_flag"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# =========================================================
# TAB 3 – APPLICATION
# =========================================================
with tab3:
    st.markdown('<div class="section-title">Step 1: Load Dataset</div>', unsafe_allow_html=True)

    mode = st.radio("Dataset option:", ["Default dataset","Upload CSV"], horizontal=True)

    if mode == "Default dataset":
        df = pd.read_csv(DEFAULT_URL)
    else:
        file = st.file_uploader("Upload EV dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)
        else:
            st.stop()

    # =====================================================
    # KPIs
    # =====================================================
    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Trips", len(df))
    k2.metric("Avg Static Error (km)", f"{df['range_error_km'].mean():.2f}")
    k3.metric("Avg Dynamic Error (km)", f"{(df['dynamic_range_km']-df['actual_range_km']).mean():.2f}")
    k4.metric("Anxiety Rate (%)", f"{df['range_anxiety_flag'].mean()*100:.1f}%")

    # =====================================================
    # ML – RANGE REGRESSION
    # =====================================================
    st.markdown('<div class="section-title">ML – Dynamic Range Prediction</div>', unsafe_allow_html=True)

    X = df[NUM_COLS + CAT_COLS]
    y = df["actual_range_km"]

    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS)
    ])

    X_t = prep.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X_t, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(Xtr, ytr)
    preds = rf.predict(Xte)

    rmse = math.sqrt(mean_squared_error(yte, preds))
    r2 = r2_score(yte, preds)

    range_metrics = pd.DataFrame([{
        "Model":"RandomForest",
        "Target":"Actual Range (km)",
        "RMSE_km":round(rmse,2),
        "R2":round(r2,3)
    }])

    st.dataframe(range_metrics)
    download_df(range_metrics,"ev_range_model_metrics.csv","Download Range Metrics")

    # =====================================================
    # ML – ANXIETY CLASSIFICATION
    # =====================================================
    st.markdown('<div class="section-title">ML – Range Anxiety Prediction</div>', unsafe_allow_html=True)

    y_cls = df["range_anxiety_flag"]
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(Xtr, y_cls.iloc[Xtr.shape[0]*0:len(Xtr)])

    probs = clf.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte > yte.mean(), probs)

    anxiety_metrics = pd.DataFrame([{
        "Model":"RandomForestClassifier",
        "Target":"Range Anxiety",
        "ROC_AUC":round(auc,3)
    }])

    st.dataframe(anxiety_metrics)
    download_df(anxiety_metrics,"ev_anxiety_model_metrics.csv","Download Anxiety Metrics")

    # =====================================================
    # AUTOMATED INSIGHTS
    # =====================================================
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)

    insights = pd.DataFrame([
        ("Worst terrain for OEM over-promise", df.groupby("terrain_type")["range_error_km"].mean().idxmax()),
        ("Cold weather penalty (km)", f"{df[df['ambient_temp_c']<5]['range_error_km'].mean():.1f}"),
        ("Avg SOC during anxiety (%)", f"{df[df['range_anxiety_flag']==1]['state_of_charge_pct'].mean():.1f}")
    ], columns=["Insight","Value"])

    st.dataframe(insights)
    download_df(insights,"ev_insights.csv","Download Insights")
