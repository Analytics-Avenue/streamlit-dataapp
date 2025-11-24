import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import math
import warnings
warnings.filterwarnings("ignore")

# -------------------------

hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

st.set_page_config(page_title="Patient Visit Analytics & Hospital Performance", layout="wide")
# Hide sidebar
st.markdown("""<style>[data-testid="stSidebarNav"]{display:none;}</style>""", unsafe_allow_html=True)


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
# Helper utils
# -------------------------
REQUIRED_HEALTH_COLS = [
    "Date", "Department", "Doctor", "Patient_ID", "Age", "Gender",
    "Visit_Type", "Revenue", "Admission"
]

def ensure_datetime(df, col="Date"):
    try:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    except:
        pass
    return df

def to_currency(x):
    try:
        return "₹ " + f"{float(x):,.2f}"
    except:
        return x

def download_df(df, filename):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button("Download CSV", b, file_name=filename, mime="text/csv")

# -------------------------
# Page header + CSS
# -------------------------
st.markdown("<h1 style='margin-bottom:0.2rem'>Patient Visit Analytics & Hospital Performance</h1>", unsafe_allow_html=True)
st.markdown("Track patient visits, revenue, admissions, and departmental performance. Forecast patient load and optimize hospital resources.")

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

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["Overview", "Application"])

with tabs[0]:
    st.markdown("### Overview")
    st.markdown("""
    <div class='card'>
        This app analyzes hospital patient visits, revenue, admissions, and departmental performance.
        It forecasts patient load, identifies high-performing doctors, and provides actionable insights for hospital resource planning.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Total Patients</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Total Revenue</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Admission Rate</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Avg Revenue per Patient</div>", unsafe_allow_html=True)

with tabs[1]:
    st.header("Application")

    # -------------------------
    # Dataset input
    # -------------------------
    st.markdown("### Step 1 — Load dataset")
    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)
    df = None

    if mode == "Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/healthcare/healthcare%202.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            st.success("Default dataset loaded")
            st.dataframe(df.head())
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    elif mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            st.success("File uploaded.")
            st.dataframe(df.head())

    else:  # Upload + mapping
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head())
            st.markdown("Map your columns to required fields (only required fields shown).")
            mapping = {}
            for req in REQUIRED_HEALTH_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + list(raw.columns))
            if st.button("Apply mapping"):
                missing = [k for k,v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v:k for k,v in mapping.items()})
                    st.success("Mapping applied.")
                    st.dataframe(df.head())
                    sample_small = df.head(5).to_csv(index=False)
                    st.download_button("Download mapped sample (5 rows)", sample_small, "mapped_sample_5rows.csv", "text/csv")

    # -------------------------
    # Validate dataset
    # -------------------------
    if df is None:
        st.stop()

    df = df.copy()
    df = ensure_datetime(df, "Date")
    for col in ["Revenue", "Admission"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["Admission_Rate"] = np.where(df["Admission"]>0, 1, 0)
    df["Avg_Revenue_per_Patient"] = np.where(df["Revenue"]>0, df["Revenue"], 0)

    # -------------------------
    # Filters
    # -------------------------
    st.markdown("### Step 2 — Filters & preview")
    c1,c2,c3 = st.columns([2,2,1])
    departments = sorted(df["Department"].dropna().unique().tolist())
    doctors = sorted(df["Doctor"].dropna().unique().tolist())

    with c1:
        sel_departments = st.multiselect("Department", options=departments, default=departments[:3])
    with c2:
        sel_doctors = st.multiselect("Doctor", options=doctors, default=doctors[:3])
    with c3:
        date_range = st.date_input("Date range", value=(df["Date"].min().date(), df["Date"].max().date()))

    filt = df.copy()
    if sel_departments:
        filt = filt[filt["Department"].isin(sel_departments)]
    if sel_doctors:
        filt = filt[filt["Doctor"].isin(sel_doctors)]
    if date_range:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filt = filt[(filt["Date"] >= start) & (filt["Date"] <= end)]

    st.dataframe(filt.head(5), use_container_width=True)
    download_df(filt.head(5), "filtered_preview.csv")

    # -------------------------
    # KPIs
    # -------------------------
    st.markdown("### Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Patients", f"{filt['Patient_ID'].nunique():,}")
    k2.metric("Total Revenue", to_currency(filt['Revenue'].sum()))
    k3.metric("Admission Rate", f"{filt['Admission_Rate'].mean()*100:.2f}%")
    k4.metric("Avg Revenue per Patient", to_currency(filt['Revenue'].sum()/max(1,filt['Patient_ID'].nunique())))

    # -------------------------
    # Charts
    # -------------------------
    st.markdown("### Patient Visits over Time")
    ts = filt.groupby("Date")["Patient_ID"].nunique().reset_index().sort_values("Date")
    ts["MA_7"] = ts["Patient_ID"].rolling(7, min_periods=1).mean()
    fig = px.line(ts, x="Date", y=["Patient_ID","MA_7"], labels={"value":"Patients","variable":"Series"}, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Revenue by Department")
    dept_rev = filt.groupby("Department").agg({"Revenue":"sum"}).reset_index().sort_values("Revenue", ascending=False)
    fig2 = px.bar(dept_rev, x="Department", y="Revenue", text="Revenue", template="plotly_white")
    fig2.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Admission Rate by Doctor")
    doc_adm = filt.groupby("Doctor").agg({"Admission_Rate":"mean"}).reset_index().sort_values("Admission_Rate", ascending=False)
    fig3 = px.bar(doc_adm, x="Doctor", y="Admission_Rate", text="Admission_Rate", template="plotly_white")
    fig3.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    st.plotly_chart(fig3, use_container_width=True)

    # -------------------------
    # Automated insights
    # -------------------------
    st.markdown("### Automated Insights")
    insights = []
    best_dept = dept_rev.iloc[0]["Department"] if not dept_rev.empty else None
    if best_dept:
        insights.append(f"Top revenue-generating department: {best_dept}")
    best_doc = doc_adm.iloc[0]["Doctor"] if not doc_adm.empty else None
    if best_doc:
        insights.append(f"Doctor with highest admission rate: {best_doc}")
    for i,ins in enumerate(insights):
        st.markdown(f"**Insight {i+1}:** {ins}")

