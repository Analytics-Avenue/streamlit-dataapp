import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import math
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="Healthscope Insights", layout="wide")

# Hide sidebar
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display:none;}
section[data-testid="stSidebar"] {display:none;}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# MASTER CSS (Restored)
# ==========================================================
st.markdown("""
<style>

.big-header {
    font-size:40px;
    font-weight:900;
    color:black;
}

.card {
    background:white;
    padding:20px;
    border-radius:15px;
    border:1px solid #e5e5e5;
    margin-bottom:15px;
    box-shadow:0 4px 20px rgba(0,0,0,0.08);
}

.metric-card {
    background:#eef4ff;
    padding:18px;
    border-radius:12px;
    text-align:center;
    font-weight:700;
    color:#064b86;
    box-shadow:0 3px 14px rgba(0,0,0,0.1);
    transition:0.25s;
}
.metric-card:hover {
    transform:translateY(-4px);
    box-shadow:0 10px 25px rgba(6,75,134,0.2);
}

.variable-box {
    padding:15px;
    border-radius:10px;
    background:#eef4ff;
    color:#064b86;
    margin-bottom:8px;
    font-weight:600;
    text-align:center;
}

.section-title {
    font-size:26px;
    font-weight:700;
    margin-top:18px;
    margin-bottom:12px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# Company Header
# ==========================================================
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

st.markdown(f"""
<div style="display:flex; align-items:center; margin-bottom:15px;">
    <img src="{logo_url}" width="60" style="margin-right:12px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:700;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:700;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='big-header'>Healthscope Insights</div>", unsafe_allow_html=True)
st.write("Enterprise healthcare analytics: patient trends, hospital performance, and actionable ML insights.")

# ==========================================================
# Required Columns
# ==========================================================
REQUIRED_HEALTH_COLS = [
    "Date", "Hospital", "Department", "Doctor", "Patient",
    "AgeGroup", "Gender", "Visits", "TreatmentCost", "Revenue",
    "RecoveryRate", "SatisfactionScore"
]

def ensure_datetime(df, col):
    try:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    except:
        pass
    return df

def download_df(df, name):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button("Download CSV", b, file_name=name, mime="text/csv")

def to_currency(x):
    try:
        return "₹ " + f"{float(x):,.2f}"
    except:
        return x

# ==========================================================
# TABS
# ==========================================================
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])


# ==========================================================
# TAB 1 – OVERVIEW
# ==========================================================
with tab1:

    st.markdown("<div class='section-title'>Overview</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        Analyze hospitals, departments, doctors, patient visits, recovery score, satisfaction, and cost–revenue KPIs.
        Designed for CXOs, Medical Directors, and Analytics Teams.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Purpose</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        • Patient & visit analytics<br>
        • Revenue & cost tracking<br>
        • Department performance insights<br>
        • Recovery rate & satisfaction benchmarking<br>
        • Early indicators for hospital risk and utilization
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>KPIs</div>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Total Visits</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Total Revenue</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Avg Recovery Rate</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Avg Satisfaction Score</div>", unsafe_allow_html=True)


# ==========================================================
# TAB 2 — IMPORTANT ATTRIBUTES
# ==========================================================
with tab2:

    st.markdown("<div class='section-title'>Data Dictionary</div>", unsafe_allow_html=True)

    dict_df = pd.DataFrame({
        "Column Name": REQUIRED_HEALTH_COLS,
        "Data Type": [
            "Date", "string", "string", "string", "string",
            "string", "string", "int", "float", "float",
            "float", "float"
        ],
        "Description": [
            "Appointment / visit date",
            "Hospital name",
            "Department of treatment",
            "Attending doctor",
            "Patient name / ID",
            "Age bucket",
            "Gender",
            "Number of visits",
            "Treatment cost",
            "Revenue generated",
            "Patient recovery score",
            "Patient satisfaction rating"
        ]
    })

    st.dataframe(dict_df, use_container_width=True)

    # independent vs dependent
    st.markdown("<div class='section-title'>Variables Overview</div>", unsafe_allow_html=True)

    ind, dep = st.columns(2)

    with ind:
        st.markdown("<div class='section-title'>Independent Variables</div>", unsafe_allow_html=True)
        for v in [
            "Hospital", "Department", "Doctor", "AgeGroup", "Gender",
            "Visits", "TreatmentCost"
        ]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with dep:
        st.markdown("<div class='section-title'>Dependent Variables</div>", unsafe_allow_html=True)
        for v in ["Revenue", "RecoveryRate", "SatisfactionScore"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)


# ==========================================================
# TAB 3 — APPLICATION
# ==========================================================
with tab3:

    st.markdown("<div class='section-title'>Step 1 — Load Dataset</div>", unsafe_allow_html=True)
    mode = st.radio("Choose Option:", ["Default Dataset", "Upload CSV", "Upload CSV + Mapping"], horizontal=True)

    df = None

    if mode == "Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/healthcare/healthcare.csv"
        df = pd.read_csv(URL)
        df = ensure_datetime(df, "Date")
        st.dataframe(df.head())

    elif mode == "Upload CSV":
        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)
            df = ensure_datetime(df, "Date")
            st.dataframe(df.head())

    else:
        file = st.file_uploader("Upload file to map", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.write("Preview:", raw.head())
            mapping = {}
            for col in REQUIRED_HEALTH_COLS:
                mapping[col] = st.selectbox(f"Map → {col}", ["-- Select --"] + list(raw.columns))
            if st.button("Apply Mapping"):
                missing = [k for k,v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Missing mappings: " + ", ".join(missing))
                    st.stop()
                df = raw.rename(columns={v:k for k,v in mapping.items()})
                st.success("Mapping applied")

    if df is None:
        st.stop()

    # numeric conversions
    for col in ["Visits","TreatmentCost","Revenue","RecoveryRate","SatisfactionScore"]:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # ==========================================================
    # FILTERS
    # ==========================================================
    st.markdown("<div class='section-title'>Step 2 — Filters</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2,2,2])
    hospitals = st.multiselect("Hospital", sorted(df["Hospital"].unique()))
    depts = st.multiselect("Department", sorted(df["Department"].unique()))
    date_range = st.date_input("Date Range", value=(df["Date"].min(), df["Date"].max()))

    filt = df.copy()
    if hospitals:
        filt = filt[filt["Hospital"].isin(hospitals)]
    if depts:
        filt = filt[filt["Department"].isin(depts)]
    if date_range:
        s, e = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filt = filt[(filt["Date"] >= s) & (filt["Date"] <= e)]

    st.dataframe(filt.head())

    # ==========================================================
    # KPIs
    # ==========================================================
    st.markdown("<div class='section-title'>Key Metrics</div>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Visits", int(filt["Visits"].sum()))
    k2.metric("Total Revenue", to_currency(filt["Revenue"].sum()))
    k3.metric("Avg Recovery Rate", f"{filt['RecoveryRate'].mean():.2f}%")
    k4.metric("Avg Satisfaction Score", f"{filt['SatisfactionScore'].mean():.2f}/5")

    # ==========================================================
    # CHARTS
    # ==========================================================
    st.markdown("<div class='section-title'>Hospital Performance</div>", unsafe_allow_html=True)

    agg = filt.groupby("Hospital").agg({"Visits":"sum","Revenue":"sum"}).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["Hospital"], y=agg["Visits"], name="Visits"))
    fig.add_trace(go.Bar(x=agg["Hospital"], y=agg["Revenue"], name="Revenue"))
    fig.update_layout(barmode="group", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>Visits Over Time</div>", unsafe_allow_html=True)
    ts = filt.groupby("Date")["Visits"].sum().reset_index()
    fig2 = px.line(ts, x="Date", y="Visits", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    # ==========================================================
    # ML MODEL — Revenue Prediction
    # ==========================================================
    st.markdown("<div class='section-title'>ML: Revenue Prediction</div>", unsafe_allow_html=True)

    if len(filt) >= 50:

        X = filt[["Visits","Hospital","Department"]].copy()
        y = filt["Revenue"].astype(float)

        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Hospital","Department"]),
            ("num", StandardScaler(), ["Visits"])
        ])

        X_t = pre.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)

        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)

        preds = rf.predict(X_test)
        rmse = math.sqrt(np.mean((preds - y_test)**2))
        r2 = rf.score(X_test, y_test)

        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"R² Score: {r2:.3f}")

    # ==========================================================
    # AUTOMATED INSIGHTS TABLE
    # ==========================================================
    st.markdown("<div class='section-title'>Automated Insights</div>", unsafe_allow_html=True)

    insights_list = []

    top_rev = filt.groupby("Hospital")["Revenue"].mean().idxmax()
    insights_list.append(["Top Revenue Hospital", top_rev])

    best_rec = filt.groupby("Department")["RecoveryRate"].mean().idxmax()
    insights_list.append(["Best Recovery Department", best_rec])

    low_satis = filt.groupby("Department")["SatisfactionScore"].mean().idxmin()
    insights_list.append(["Lowest Satisfaction Score Dept", low_satis])

    busy = filt.groupby("Hospital")["Visits"].sum().idxmax()
    insights_list.append(["Highest Patient Load", busy])

    trend = (
        "Increasing" if filt.sort_values("Date").Revenue.diff().mean() > 0 else "Decreasing"
    )
    insights_list.append(["Revenue Trend", trend])

    insights_df = pd.DataFrame(insights_list, columns=["Insight", "Value"])
    st.dataframe(insights_df, use_container_width=True)

    download_df(insights_df, "health_insights.csv")
