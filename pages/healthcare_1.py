import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import math
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Hide Sidebar
# -------------------------
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display:none;}
section[data-testid="stSidebar"] {display:none;}
.big-header {font-size:40px; font-weight:900; color:black;}
.card {
    background:#fff;border-radius:15px;padding:20px;margin-bottom:15px;
    box-shadow:0 4px 20px rgba(0,0,0,0.08);
}
.variable-box {
    padding:16px; border-radius:10px; background:#eef4ff;
    font-size:16px; font-weight:600; color:#064b86; text-align:center;
    margin-bottom:10px;
}
.section-title {
    font-size:26px; font-weight:700; margin-top:20px; margin-bottom:10px;
}
.metric-card {
    background:#eef4ff; padding:18px; border-radius:10px;
    text-align:center; font-weight:700; color:#064b86;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Healthscope Insights", layout="wide")

# -------------------------
# Logo Header
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style='display:flex; align-items:center; margin-bottom:15px;'>
    <img src="{logo_url}" width="60" style="margin-right:12px;">
    <div>
        <div style='color:#064b86; font-size:36px; font-weight:700;'>Analytics Avenue &</div>
        <div style='color:#064b86; font-size:36px; font-weight:700;'>Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Required Columns
# -------------------------
REQUIRED_HEALTH_COLS = [
    "Date", "Hospital", "Department", "Doctor", "Patient",
    "AgeGroup", "Gender", "Visits", "TreatmentCost", "Revenue",
    "RecoveryRate", "SatisfactionScore"
]

# -------------------------
# Helper Functions
# -------------------------
def ensure_datetime(df, col="Date"):
    try:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    except:
        pass
    return df

def download_df(df, filename):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button("Download CSV", b, file_name=filename)

def to_currency(x):
    try:
        return "₹ " + f"{float(x):,.2f}"
    except:
        return x

# -------------------------
# Page Title
# -------------------------
st.markdown("<div class='big-header'>Healthscope Insights</div>", unsafe_allow_html=True)
st.markdown("Enterprise healthcare analytics: patient behavior, hospital KPIs & revenue intelligence.")

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# ==========================================================
# TAB 1: OVERVIEW
# ==========================================================
with tab1:
    st.markdown("<div class='section-title'>Overview</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
    Analyze patient visits, hospital revenues, doctor performance, recovery KPIs, and satisfaction scores.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Purpose</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
    • Track patient volume & hospital load<br>
    • Evaluate doctors & departments<br>
    • Understand revenue streams<br>
    • Monitor recovery & satisfaction<br>
    • Forecast demand & outcomes
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>KPIs</div>", unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Total Visits</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Total Revenue</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Avg Recovery Rate</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Avg Satisfaction</div>", unsafe_allow_html=True)

# ==========================================================
# TAB 2: IMPORTANT ATTRIBUTES (FIXED)
# ==========================================================
with tab2:

    st.markdown("<div class='section-title'>Data Dictionary</div>", unsafe_allow_html=True)

    data_dict = pd.DataFrame({
        "Column Name": REQUIRED_HEALTH_COLS,
        "Description": [
            "Date of patient visit",
            "Hospital where treatment took place",
            "Medical department (Cardiology, Neuro, etc)",
            "Doctor handling the case",
            "Patient Identifier",
            "Age group bucket",
            "Gender of patient",
            "Visit count for that entry",
            "Treatment cost for the visit",
            "Total revenue generated",
            "Patient recovery rate in %",
            "Satisfaction score (1–5)"
        ]
    })

    st.dataframe(data_dict, use_container_width=True)

    st.markdown("<div class='section-title'>Variables Overview</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    # Independent Variables
    with c1:
        st.markdown("<h4>Independent Variables</h4>", unsafe_allow_html=True)
        for v in [
            "Hospital",
            "Department",
            "Doctor",
            "AgeGroup",
            "Gender",
            "Visits",
            "TreatmentCost"
        ]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    # Dependent Variables
    with c2:
        st.markdown("<h4>Dependent Variables</h4>", unsafe_allow_html=True)
        for v in ["Revenue", "RecoveryRate", "SatisfactionScore"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# ==========================================================
# TAB 3: APPLICATION
# ==========================================================
with tab3:

    st.header("Application")
    st.markdown("### Step 1 — Load dataset")

    mode = st.radio("Dataset option:",
                    ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"],
                    horizontal=True)

    df = None

    # Default dataset
    if mode == "Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/healthcare/healthcare.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            df = ensure_datetime(df)
            st.dataframe(df.head())
        except Exception as e:
            st.error(str(e))
            st.stop()

    # Upload CSV
    elif mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.dataframe(df.head())

    # Upload + mapping
    else:
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded:
            raw = pd.read_csv(uploaded)
            st.dataframe(raw.head())

            mapping = {}
            for col in REQUIRED_HEALTH_COLS:
                mapping[col] = st.selectbox(f"Map to {col}", ["-- Select --"] + list(raw.columns))

            if st.button("Apply Mapping"):
                if "-- Select --" in mapping.values():
                    st.error("Map all required columns.")
                else:
                    df = raw.rename(columns={mapping[k]:k for k in mapping})
                    st.success("Mapping applied!")
                    st.dataframe(df.head())

    if df is None:
        st.stop()

    # Convert types
    df = ensure_datetime(df)
    for col in ["Visits","TreatmentCost","Revenue","RecoveryRate","SatisfactionScore"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Filters
    st.markdown("### Step 2 — Filters")
    c1,c2,c3 = st.columns([2,2,1])

    hospitals = sorted(df["Hospital"].unique())
    departments = sorted(df["Department"].unique())

    with c1:
        sel_h = st.multiselect("Hospital", hospitals, default=hospitals[:2])
    with c2:
        sel_d = st.multiselect("Department", departments, default=departments[:2])
    with c3:
        date_range = st.date_input("Date Range", (df["Date"].min(), df["Date"].max()))

    filt = df.copy()
    if sel_h: filt = filt[filt["Hospital"].isin(sel_h)]
    if sel_d: filt = filt[filt["Department"].isin(sel_d)]
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filt = filt[(filt["Date"] >= start) & (filt["Date"] <= end)]

    st.dataframe(filt.head())

    # KPIs
    st.markdown("### Key Metrics")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Visits", int(filt["Visits"].sum()))
    k2.metric("Total Revenue", to_currency(filt["Revenue"].sum()))
    k3.metric("Avg Recovery Rate", f"{filt['RecoveryRate'].mean():.2f}%")
    k4.metric("Avg Satisfaction", f"{filt['SatisfactionScore'].mean():.2f}/5")

    # Charts
    st.markdown("### Revenue & Visits by Hospital")
    agg = filt.groupby("Hospital").agg({"Visits":"sum","Revenue":"sum"}).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["Hospital"],y=agg["Visits"],name="Visits"))
    fig.add_trace(go.Bar(x=agg["Hospital"],y=agg["Revenue"],name="Revenue"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Visits Over Time")
    ts = filt.groupby("Date")["Visits"].sum().reset_index()
    st.plotly_chart(px.line(ts, x="Date", y="Visits"), use_container_width=True)

    # ML Revenue Prediction
    if len(filt) >= 50:
        st.markdown("### ML: Revenue Prediction")
        X = filt[["Visits","Hospital","Department"]]
        y = filt["Revenue"]

        pre = ColumnTransformer([
            ("cat", OneHotEncoder(sparse_output=False), ["Hospital","Department"]),
            ("num", StandardScaler(), ["Visits"])
        ])

        Xt = pre.fit_transform(X)
        X_train,X_test,y_train,y_test = train_test_split(Xt,y,test_size=0.2,random_state=42)

        model = RandomForestRegressor(n_estimators=150, random_state=42)
        model.fit(X_train,y_train)

        preds = model.predict(X_test)
        rmse = math.sqrt(np.mean((preds - y_test)**2))
        r2 = model.score(X_test, y_test)

        st.write(f"RMSE: {rmse:.2f} | R²: {r2:.3f}")
