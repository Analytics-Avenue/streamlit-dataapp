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
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
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
    <div>
        <div style="font-size:36px; font-weight:700; color:#064b86;">Analytics Avenue &</div>
        <div style="font-size:36px; font-weight:700; color:#064b86;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<h1>EV Dynamic Range Prediction & Anxiety Reduction Lab</h1>", unsafe_allow_html=True)

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

CAT_COLS = [
    "driving_style","terrain_type","traffic_level"
]

# =========================================================
# HELPERS
# =========================================================
def download_df(df, filename, label):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode())
    b.seek(0)
    st.download_button(label, b, filename, "text/csv")

def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# =========================================================
# TAB 1 – OVERVIEW
# =========================================================
with tab1:
    st.markdown("""
    <div style="padding:20px; border-radius:12px; background:#fff; border:1px solid #ddd;">
    <b>Problem:</b> EV drivers do not trust displayed range and fear being stranded.<br><br>

    <b>Why it happens:</b><br>
    • Static OEM range logic<br>
    • No driving behavior awareness<br>
    • Terrain & weather ignored<br><br>

    <b>Solution:</b><br>
    Dynamic ML-based range prediction that adapts to driver behavior,
    terrain, weather, traffic, and payload in real time.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style="padding:20px; border-radius:12px; background:#fff; border:1px solid #ddd;">
        <b>What this lab does</b><br>
        • Predicts realistic remaining range<br>
        • Quantifies OEM over/under-promise<br>
        • Flags anxiety-prone trips in advance<br>
        • Personalizes range per driver
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style="padding:20px; border-radius:12px; background:#fff; border:1px solid #ddd;">
        <b>Business impact</b><br>
        • Higher driver trust<br>
        • Fewer roadside failures<br>
        • Better charging behavior<br>
        • Increased EV adoption
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 2 – IMPORTANT ATTRIBUTES
# =========================================================
with tab2:
    st.markdown("### Required Dataset Schema")

    dict_rows = [
        ("battery_capacity_kwh","Total battery capacity","Independent"),
        ("state_of_charge_pct","Current SOC %","Independent"),
        ("state_of_health_pct","Battery health %","Independent"),
        ("avg_speed_kmph","Average speed","Independent"),
        ("terrain_type","Road terrain","Independent"),
        ("ambient_temp_c","Outside temperature","Independent"),
        ("payload_kg","Vehicle payload","Independent"),
        ("energy_consumption_wh_per_km","Energy use rate","Derived"),
        ("static_range_km","OEM estimated range","Dependent"),
        ("dynamic_range_km","ML estimated range","Dependent"),
        ("actual_range_km","Actual achieved range","Target"),
        ("range_anxiety_flag","Driver anxiety indicator","Target")
    ]

    st.dataframe(
        pd.DataFrame(dict_rows, columns=["Column","Description","Role"]),
        use_container_width=True
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Independent Variables")
        for v in NUM_COLS + CAT_COLS:
            st.markdown(f"• {v}")

    with c2:
        st.markdown("### Dependent Variables")
        st.markdown("• actual_range_km")
        st.markdown("• range_anxiety_flag")

# =========================================================
# TAB 3 – APPLICATION
# =========================================================
with tab3:
    st.markdown("## Step 1: Load Dataset")

    mode = st.radio(
        "Dataset option:",
        ["Default dataset", "Upload CSV", "Upload CSV + Column Mapping"],
        horizontal=True
    )

    df = None

    if mode == "Default dataset":
        df = pd.read_csv(DEFAULT_URL)

    elif mode == "Upload CSV":
        uploaded = st.file_uploader("Upload EV dataset", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)

    else:
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded:
            raw = pd.read_csv(uploaded)
            mapping = {}
            for c in REQUIRED_COLS:
                mapping[c] = st.selectbox(
                    f"Map → {c}",
                    ["-- Select --"] + list(raw.columns)
                )
            if st.button("Apply mapping"):
                df = raw.rename(columns={v:k for k,v in mapping.items() if v!="-- Select --"})

    if df is None:
        st.stop()

    df.columns = df.columns.str.strip()
    df = to_num(df, NUM_COLS)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        st.stop()

    # =====================================================
    # KPIs
    # =====================================================
    st.markdown("## KPIs")
    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Trips", len(df))
    k2.metric("Avg Static Error (km)", f"{df['range_error_km'].mean():.1f}")
    k3.metric("Avg Dynamic Error (km)", f"{(df['dynamic_range_km']-df['actual_range_km']).mean():.1f}")
    k4.metric("Anxiety Rate (%)", f"{df['range_anxiety_flag'].mean()*100:.1f}%")

    # =====================================================
    # CHARTS
    # =====================================================
    st.markdown("## Diagnostics")

    fig1 = px.scatter(
        df,
        x="static_range_km",
        y="actual_range_km",
        title="Static Range vs Actual Range"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(
        df,
        x="dynamic_range_km",
        y="actual_range_km",
        title="Dynamic Range vs Actual Range"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # =====================================================
    # ML MODEL 1 – RANGE PREDICTION
    # =====================================================
    st.markdown("## ML – Dynamic Range Prediction")

    features = NUM_COLS + CAT_COLS
    X = df[features]
    y = df["actual_range_km"]

    cat = [c for c in features if c in CAT_COLS]
    num = [c for c in features if c not in cat]

    prep = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)
    ])

    X_t = prep.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X_t, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(Xtr, ytr)
    preds = rf.predict(Xte)

    rmse = math.sqrt(mean_squared_error(yte, preds))
    r2 = r2_score(yte, preds)

    st.write(f"RMSE: {rmse:.2f} km | R²: {r2:.3f}")

    # =====================================================
    # ML MODEL 2 – RANGE ANXIETY CLASSIFICATION
    # =====================================================
    st.markdown("## ML – Range Anxiety Prediction")

    y_cls = df["range_anxiety_flag"]

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X_t, y_cls, test_size=0.2, random_state=42)

    clf.fit(Xtr, ytr)
    probs = clf.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, probs)

    st.write(f"ROC-AUC: {auc:.3f}")

    # =====================================================
    # ML MODEL 3 – DRIVER CLUSTERING
    # =====================================================
    st.markdown("## ML – Driver Personalization")

    cl_features = [
        "avg_speed_kmph",
        "acceleration_score",
        "regen_braking_pct",
        "energy_consumption_wh_per_km"
    ]

    cl_df = df[cl_features].dropna()
    km = KMeans(n_clusters=4, random_state=42)
    cl_df["cluster"] = km.fit_predict(cl_df)

    figc = px.scatter(
        cl_df,
        x="avg_speed_kmph",
        y="energy_consumption_wh_per_km",
        color="cluster",
        title="Driver Behavior Clusters"
    )
    st.plotly_chart(figc, use_container_width=True)

    # =====================================================
    # AUTOMATED INSIGHTS
    # =====================================================
    st.markdown("## Automated Insights")

    insights = []

    worst_terrain = df.groupby("terrain_type")["range_error_km"].mean().idxmax()
    insights.append(("Worst OEM over-promise terrain", worst_terrain))

    cold_penalty = df[df["ambient_temp_c"] < 5]["range_error_km"].mean()
    insights.append(("Cold weather avg error (km)", f"{cold_penalty:.1f}"))

    anxiety_soc = df[df["range_anxiety_flag"]==1]["state_of_charge_pct"].mean()
    insights.append(("Avg SOC during anxiety", f"{anxiety_soc:.1f}%"))

    ins_df = pd.DataFrame(insights, columns=["Insight","Value"])
    st.dataframe(ins_df, use_container_width=True)
    download_df(ins_df, "ev_insights.csv", "Download Insights")
