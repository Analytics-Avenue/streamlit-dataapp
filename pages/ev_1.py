import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="EV Range Prediction & Anxiety Reduction", layout="wide")

# -------------------------------------------------
# HIDE SIDEBAR
# -------------------------------------------------
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display:none;}
section[data-testid="stSidebar"] {display:none;}
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# GLOBAL CSS
# -------------------------------------------------
st.markdown("""
<style>
.card {
    padding:18px;
    border-radius:12px;
    background:#fff;
    border:1px solid #e6eef7;
    box-shadow:0 2px 6px rgba(0,0,0,0.05);
}
.kpi {
    padding:22px;
    border-radius:12px;
    text-align:center;
    font-weight:700;
    font-size:20px;
    color:#064b86;
    background:#fff;
    border:1px solid #e6eef7;
}
.section-title {
    font-size:22px;
    font-weight:700;
    color:#064b86;
}
.small {
    font-size:13px;
    color:#666;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
logo = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display:flex;align-items:center;">
<img src="{logo}" width="60">
<div style="margin-left:10px;">
<div style="font-size:30px;font-weight:700;color:#064b86;">Analytics Avenue &</div>
<div style="font-size:30px;font-weight:700;color:#064b86;">Advanced Analytics</div>
</div>
</div>
<h1 style="margin-top:6px;">EV Range Prediction & Anxiety Reduction</h1>
<p class="small">
Dynamic, behavior-aware EV range estimation to reduce driver anxiety and improve adoption.
</p>
""", unsafe_allow_html=True)

# -------------------------------------------------
# REQUIRED COLUMNS
# -------------------------------------------------
REQUIRED_COLS = [
    "battery_capacity_kwh","state_of_charge_pct","state_of_health_pct",
    "avg_speed_kmph","acceleration_score","regen_braking_pct",
    "terrain_type","ambient_temp_c","rain_intensity_mm","wind_speed_kmph",
    "payload_kg","energy_consumption_wh_per_km",
    "static_range_km","dynamic_range_km","actual_range_km",
    "range_error_km","range_anxiety_flag"
]

DATA_DICT = [
    ("battery_capacity_kwh","Battery capacity","Independent"),
    ("state_of_charge_pct","Current SOC %","Independent"),
    ("state_of_health_pct","Battery health %","Independent"),
    ("avg_speed_kmph","Average speed","Independent"),
    ("acceleration_score","Driving aggressiveness","Independent"),
    ("regen_braking_pct","Regen braking usage","Independent"),
    ("terrain_type","Terrain category","Independent"),
    ("ambient_temp_c","Ambient temperature","Independent"),
    ("rain_intensity_mm","Rain intensity","Independent"),
    ("wind_speed_kmph","Wind speed","Independent"),
    ("payload_kg","Extra load weight","Independent"),
    ("energy_consumption_wh_per_km","Energy usage","Derived Feature"),
    ("static_range_km","OEM static range","Baseline"),
    ("dynamic_range_km","ML estimated range","Dependent"),
    ("actual_range_km","Actual achievable range","Target"),
    ("range_error_km","Static vs actual error","Derived"),
    ("range_anxiety_flag","Anxiety indicator","Classification Target")
]

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def download_df(df, name):
    buf = BytesIO()
    buf.write(df.to_csv(index=False).encode())
    buf.seek(0)
    st.download_button("Download CSV", buf, name, "text/csv")

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Application", "Data Dictionary"])

# =================================================
# OVERVIEW TAB
# =================================================
with tab1:
    st.markdown("""
    <div class="card">
    <b>Problem</b><br>
    EV drivers distrust range estimates and fear being stranded due to static, unrealistic calculations.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-title'>Why Range Anxiety Happens</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Static OEM range calculation<br>
        • Driving behavior ignored<br>
        • Terrain & elevation not considered<br>
        • Weather impact missing<br>
        • Battery aging not factored
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='section-title'>Analytics Solution</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Dynamic range prediction models<br>
        • Personalized driver behavior modeling<br>
        • Weather, terrain & traffic integration<br>
        • Continuous learning from actual trips
        </div>
        """, unsafe_allow_html=True)

    k1,k2,k3,k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Static Range Error</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Dynamic Accuracy</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Anxiety Events</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Driver Trust Score</div>", unsafe_allow_html=True)

# =================================================
# APPLICATION TAB
# =================================================
with tab2:
    st.header("Application")

    mode = st.radio(
        "Dataset Source",
        ["Default dataset (GitHub)", "Upload CSV", "Upload CSV + Column Mapping"],
        horizontal=True
    )

    df = None

    # -------------------------
    # MODE 1: DEFAULT
    # -------------------------
    if mode == "Default dataset (GitHub)":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/ev/ev_dynamic_range_dataset.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            st.success("Default dataset loaded.")
            st.dataframe(df.head())
        except:
            st.error("Failed to load default dataset.")

    # -------------------------
    # MODE 2: UPLOAD CSV
    # -------------------------
    elif mode == "Upload CSV":
        file = st.file_uploader("Upload EV dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)
            st.dataframe(df.head())

    # -------------------------
    # MODE 3: UPLOAD + MAPPING
    # -------------------------
    else:
        file = st.file_uploader("Upload CSV for mapping", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.dataframe(raw.head())
            mapping = {}
            for c in REQUIRED_COLS:
                mapping[c] = st.selectbox(f"Map → {c}", ["-- Select --"] + list(raw.columns))
            if st.button("Apply Mapping"):
                missing = [k for k,v in mapping.items() if v=="-- Select --"]
                if missing:
                    st.error(f"Missing mappings: {missing}")
                else:
                    df = raw.rename(columns={mapping[k]:k for k in mapping})

    if df is None:
        st.stop()

    # -------------------------
    # KPIs
    # -------------------------
    st.markdown("### KPIs (Dynamic)")
    k1,k2,k3,k4 = st.columns(4)

    k1.metric("Trips", len(df))
    k2.metric("Avg Static Error (km)", round(df["range_error_km"].mean(),2))
    k3.metric("Anxiety Rate (%)", round(df["range_anxiety_flag"].mean()*100,2))
    k4.metric("Avg Dynamic Range (km)", round(df["dynamic_range_km"].mean(),1))

    # -------------------------
    # EDA
    # -------------------------
    st.markdown("## Exploratory Analysis")

    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(
            px.histogram(df, x="range_error_km", title="Static Range Error Distribution"),
            use_container_width=True
        )

    with c2:
        st.plotly_chart(
            px.box(df, x="terrain_type", y="dynamic_range_km",
                   title="Dynamic Range by Terrain"),
            use_container_width=True
        )

    st.plotly_chart(
        px.scatter(df, x="avg_speed_kmph", y="energy_consumption_wh_per_km",
                   color="range_anxiety_flag",
                   title="Speed vs Energy Consumption"),
        use_container_width=True
    )

    # -------------------------
    # ML MODELS
    # -------------------------
    st.markdown("## Machine Learning")

    # Regression
    st.markdown("### Predict Actual Range")
    features = [
        "battery_capacity_kwh","state_of_charge_pct","state_of_health_pct",
        "avg_speed_kmph","acceleration_score","regen_braking_pct",
        "ambient_temp_c","payload_kg","energy_consumption_wh_per_km"
    ]
    model_df = df.dropna(subset=features + ["actual_range_km"])
    if len(model_df) > 200:
        X = model_df[features]
        y = model_df["actual_range_km"]
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
        reg = RandomForestRegressor(n_estimators=150, random_state=42)
        reg.fit(Xtr,ytr)
        preds = reg.predict(Xte)
        st.write("R²:", round(r2_score(yte,preds),3))

    # Classification
    st.markdown("### Predict Range Anxiety")
    clf_df = df.dropna(subset=features + ["range_anxiety_flag"])
    if clf_df["range_anxiety_flag"].nunique() > 1:
        X = clf_df[features]
        y = clf_df["range_anxiety_flag"]
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
        clf = RandomForestClassifier(n_estimators=150, random_state=42)
        clf.fit(Xtr,ytr)
        prob = clf.predict_proba(Xte)[:,1]
        pred = (prob>0.5).astype(int)
        st.write("Accuracy:", round(accuracy_score(yte,pred),3),
                 "AUC:", round(roc_auc_score(yte,prob),3))

    download_df(df, "ev_range_analytics_filtered.csv")

# =================================================
# DATA DICTIONARY TAB
# =================================================
with tab3:
    st.header("Data Dictionary")

    dict_df = pd.DataFrame(DATA_DICT, columns=["Column","Description","Role"])
    st.dataframe(dict_df, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='card'><b>Independent Variables</b><hr>", unsafe_allow_html=True)
        for c in dict_df[dict_df["Role"].str.contains("Independent")]["Column"]:
            st.markdown(f"- {c}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'><b>Dependent / Target Variables</b><hr>", unsafe_allow_html=True)
        for c in dict_df[dict_df["Role"].str.contains("Target|Dependent")]["Column"]:
            st.markdown(f"- {c}")
        st.markdown("</div>", unsafe_allow_html=True)
