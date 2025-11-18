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

st.set_page_config(page_title="Healthscope Insights", layout="wide")

# -------------------------
# Company Logo + Name
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display: flex; align-items: center;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0; padding:0;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0; padding:0;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Required columns for project
# -------------------------
REQUIRED_HEALTH_COLS = [
    "Date", "Hospital", "Department", "Doctor", "Patient",
    "AgeGroup", "Gender", "Visits", "TreatmentCost", "Revenue",
    "RecoveryRate", "SatisfactionScore"
]

# -------------------------
# Helper functions
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
    st.download_button("Download CSV", b, file_name=filename, mime="text/csv")

def to_currency(x):
    try:
        return "₹ " + f"{float(x):,.2f}"
    except:
        return x

# -------------------------
# Page header + overview
# -------------------------
st.markdown("<h1 style='margin-bottom:0.2rem'>Healthscope Insights</h1>", unsafe_allow_html=True)
st.markdown("Enterprise-ready healthcare analytics: Patient insights, hospital KPIs, and revenue analysis — actionable & visual.")

st.markdown("""
<style>
div[data-testid="stMarkdownContainer"] .card {
    background: rgba(255,255,255,0.07);
    padding: 18px 20px;
    border-radius: 14px;
    margin-bottom: 15px;
    border: 1px solid rgba(255,255,255,0.25);
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    backdrop-filter: blur(4px);
}

div[data-testid="stMarkdownContainer"] .metric-card {
    background: rgba(255,255,255,0.10);
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.30);
    font-weight: 600;
    font-size: 16px;
    transition: all 0.25s ease;
    box-shadow: 0 2px 10px rgba(0,0,0,0.18);
    backdrop-filter: blur(4px);
}

div[data-testid="stMarkdownContainer"] .metric-card:hover {
    background: rgba(255,255,255,0.20);
    border: 1px solid rgba(255,255,255,0.55);
    box-shadow: 0 0 18px rgba(255,255,255,0.4);
    transform: scale(1.04);
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Application"])

with tabs[0]:
    st.markdown("### Overview")
    st.markdown("""
    <div class='card'>
        This app provides hospital and patient analytics: track visits, treatment costs, revenue, recovery rates, and satisfaction scores. Identify trends and improve decision-making.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Purpose")
    st.markdown("""
    <div class='card'>
        • Analyze patient visits & hospital load<br>
        • Monitor department & doctor performance<br>
        • Understand revenue and treatment costs<br>
        • Track recovery & satisfaction KPIs<br>
        • Forecast hospital utilization and outcomes
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card' title='Total patient visits recorded'>Total Visits</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card' title='Total revenue generated'>Total Revenue</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card' title='Average recovery rate across patients'>Avg Recovery Rate</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card' title='Average patient satisfaction score'>Avg Satisfaction</div>", unsafe_allow_html=True)

with tabs[1]:
    st.header("Application")

    # -------------------------
    # Dataset input: default, upload, mapping
    # -------------------------
    st.markdown("### Step 1 — Load dataset")
    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)
    df = None

    if mode == "Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/healthcare/healthcare.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            df = ensure_datetime(df, "Date")
            st.dataframe(df.head())
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    elif mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.success("File uploaded")
            st.dataframe(df.head())

    else:  # Upload + mapping
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head())
            mapping = {}
            for req in REQUIRED_HEALTH_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + list(raw.columns))
            if st.button("Apply mapping"):
                missing = [k for k,v in mapping.items() if v=="-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v:k for k,v in mapping.items()})
                    st.success("Mapping applied")
                    st.dataframe(df.head())

    if df is None:
        st.stop()

    # -------------------------
    # Type conversions
    # -------------------------
    df = ensure_datetime(df, "Date")
    for col in ["Visits","TreatmentCost","Revenue","RecoveryRate","SatisfactionScore"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # -------------------------
    # Filters
    # -------------------------
    st.markdown("### Step 2 — Filters & preview")
    c1,c2,c3 = st.columns([2,2,1])
    hospitals_list = sorted(df["Hospital"].dropna().unique())
    departments_list = sorted(df["Department"].dropna().unique())

    with c1:
        sel_hospitals = st.multiselect("Hospital", options=hospitals_list, default=hospitals_list[:2])
    with c2:
        sel_departments = st.multiselect("Department", options=departments_list, default=departments_list[:2])
    with c3:
        date_range = st.date_input("Date range", value=(df["Date"].min(), df["Date"].max()))

    filt = df.copy()
    if sel_hospitals:
        filt = filt[filt["Hospital"].isin(sel_hospitals)]
    if sel_departments:
        filt = filt[filt["Department"].isin(sel_departments)]
    if date_range:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filt = filt[(filt["Date"] >= start) & (filt["Date"] <= end)]

    st.markdown("Filtered preview")
    st.dataframe(filt.head(5))
    download_df(filt.head(5), "filtered_healthcare_preview.csv")

    # -------------------------
    # KPIs
    # -------------------------
    st.markdown("### Key Metrics")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Visits", int(filt["Visits"].sum()))
    k2.metric("Total Revenue", to_currency(filt["Revenue"].sum()))
    k3.metric("Avg Recovery Rate", f"{filt['RecoveryRate'].mean():.2f}%")
    k4.metric("Avg Satisfaction", f"{filt['SatisfactionScore'].mean():.2f}/5")

    # -------------------------
    # Charts
    # -------------------------
    st.markdown("### Revenue & Visits by Hospital")
    agg = filt.groupby("Hospital").agg({"Visits":"sum","Revenue":"sum"}).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["Hospital"], y=agg["Visits"], name="Visits", text=agg["Visits"], textposition="outside"))
    fig.add_trace(go.Bar(x=agg["Hospital"], y=agg["Revenue"], name="Revenue", text=agg["Revenue"].apply(lambda x: f"₹{x:,}"), textposition="outside"))
    fig.update_layout(barmode='group', xaxis_title="Hospital", yaxis_title="Count / Revenue", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Visits Trend Over Time")
    ts = filt.groupby("Date")["Visits"].sum().reset_index().sort_values("Date")
    fig2 = px.line(ts, x="Date", y="Visits", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    # -------------------------
    # Simple ML: Predict Revenue from Visits + Dept + Hospital
    # -------------------------
    if len(filt) >= 50:
        st.markdown("### ML: Predict Revenue")
        X = filt[["Visits","Hospital","Department"]].copy()
        y = filt["Revenue"].astype(float)
        cat_cols = ["Hospital","Department"]
        num_cols = ["Visits"]
        preprocessor = ColumnTransformer(transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols)
        ], remainder="drop")
        X_t = preprocessor.fit_transform(X)
        X_train,X_test,y_train,y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=150, random_state=42)
        with st.spinner("Training RandomForest for Revenue prediction..."):
            rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        rmse = math.sqrt(np.mean((y_test - preds)**2))
        r2 = rf.score(X_test, y_test)
        st.write(f"Revenue Prediction — RMSE: {rmse:.2f}, R²: {r2:.3f}")

