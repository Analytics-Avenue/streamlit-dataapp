import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings("ignore")

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="Patient Visit Analytics & Hospital Performance", layout="wide")

# Hide sidebar
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display:none;}
section[data-testid="stSidebar"] {display:none;}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# ADVANCED CSS (Neon glow + hover + blur + fade)
# ==========================================================
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Fade-in animation for entire page */
.block-container {
    animation: fadeIn 0.6s ease-in-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Main title */
.big-header {
    font-size:42px;
    font-weight:900;
    color:#000;
    margin-bottom:10px;
}

/* Card styling */
.card {
    background:white;
    padding:20px;
    border-radius:15px;
    border:1px solid #e5e5e5;
    box-shadow:0 8px 25px rgba(0,0,0,0.1);
    transition:0.25s ease;
}
.card:hover {
    transform: translateY(-4px);
    box-shadow:0 18px 35px rgba(0,0,0,0.18);
}

/* KPI cards with glow */
.metric-card {
    background:#eef4ff;
    padding:18px;
    text-align:center;
    border-radius:14px;
    font-size:18px;
    font-weight:700;
    color:#064b86;
    box-shadow:0 4px 15px rgba(6,75,134,0.25);
    transition:0.25s;
}
.metric-card:hover {
    transform:translateY(-6px) scale(1.03);
    box-shadow:0 10px 30px rgba(6,75,134,0.35);
}

/* Variable boxes */
.variable-box {
    padding:16px;
    background:#eef4ff;
    border-radius:12px;
    text-align:center;
    font-size:17px;
    color:#064b86;
    margin-bottom:10px;
    font-weight:600;
    transition:0.25s ease;
    box-shadow:0 3px 10px rgba(6,75,134,0.15);
}
.variable-box:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 20px rgba(6,75,134,0.22);
}

/* Section titles */
.section-title {
    font-size:28px;
    font-weight:800;
    margin-top:25px;
    margin-bottom:10px;
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
    <div>
        <div style="color:#064b86; font-size:34px; font-weight:700;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:34px; font-weight:700;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='big-header'>Patient Visit Analytics & Hospital Performance</div>", unsafe_allow_html=True)
st.write("Track patient traffic, revenue, admissions, and forecast hospital workload using ML.")

# ==========================================================
# Required columns
# ==========================================================
REQUIRED_HEALTH_COLS = [
    "Date", "Department", "Doctor", "Patient_ID",
    "Age", "Gender", "Visit_Type", "Revenue", "Admission"
]

def ensure_datetime(df, col):
    try:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    except:
        pass
    return df

def download(df, filename):
    buf = BytesIO()
    buf.write(df.to_csv(index=False).encode())
    buf.seek(0)
    st.download_button("Download CSV", buf, filename, "text/csv")

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
# TAB 1 — OVERVIEW
# ==========================================================
with tab1:

    st.markdown("<div class='section-title'>Overview</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        Analyze hospital visits, doctor performance, admission rates, revenue trends, and patient load.
        AI-powered forecasting helps hospitals optimize staffing, scheduling, and resources.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>KPIs</div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Total Patients</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Total Revenue</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Admission Rate</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Revenue per Patient</div>", unsafe_allow_html=True)


# ==========================================================
# TAB 2 — IMPORTANT ATTRIBUTES
# ==========================================================
with tab2:

    st.markdown("<div class='section-title'>Required Column Data Dictionary</div>", unsafe_allow_html=True)

    data_dict = {
        "Date": "Recorded date of patient visit or admission.",
        "Department": "Hospital department handling the patient (Cardiology, Ortho, etc).",
        "Doctor": "Consulting or attending doctor name.",
        "Patient_ID": "Unique identifier for the patient.",
        "Age": "Patient age in years.",
        "Gender": "Patient gender (Male / Female / Others).",
        "Visit_Type": "OP / IP / Follow-up / Emergency visit type.",
        "Revenue": "Total bill amount from the visit.",
        "Admission": "Indicates if patient was admitted (1) or not (0)."
    }

    req_df = pd.DataFrame(
        [{"Column": k, "Description": v} for k, v in data_dict.items()]
    )

    st.dataframe(req_df, use_container_width=True)

    # Independent vs Dependent
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Independent Variables</div>", unsafe_allow_html=True)
        indep = ["Department", "Doctor", "Age", "Gender", "Visit_Type"]
        for v in indep:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-title'>Dependent Variables</div>", unsafe_allow_html=True)
        dep = ["Revenue", "Admission"]
        for v in dep:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# ==========================================================
# TAB 3 — APPLICATION
# ==========================================================
with tab3:

    st.markdown("<div class='section-title'>Step 1 — Load dataset</div>", unsafe_allow_html=True)

    mode = st.radio("Choose Option:", ["Default Dataset", "Upload CSV", "Upload + Column Mapping"], horizontal=True)
    df = None

    if mode == "Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/healthcare/healthcare%202.csv"
        df = pd.read_csv(URL)
        st.success("Dataset loaded")
        st.dataframe(df.head())

    elif mode == "Upload CSV":
        file = st.file_uploader("Upload file", type=["csv"])
        if file:
            df = pd.read_csv(file)
            st.success("Uploaded")
            st.dataframe(df.head())

    else:
        file = st.file_uploader("Upload file", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.write("Preview:", raw.head())

            mapping = {}
            for col in REQUIRED_HEALTH_COLS:
                mapping[col] = st.selectbox(f"Map → {col}", ["-- Select --"] + list(raw.columns))

            if st.button("Apply Mapping"):
                missing = [m for m,v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Missing mappings: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v:k for k,v in mapping.items()})
                    st.success("Mapping applied")
                    st.dataframe(df.head())

    if df is None:
        st.stop()

    df = ensure_datetime(df, "Date")

    for c in ["Revenue", "Admission"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["Admission_Rate"] = df["Admission"].apply(lambda x: 1 if x > 0 else 0)

    # ==========================================================
    # FILTERS
    # ==========================================================
    st.markdown("<div class='section-title'>Step 2 — Filters</div>", unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    dept = d1.multiselect("Department", df["Department"].unique())
    doc = d2.multiselect("Doctor", df["Doctor"].unique())
    date_range = d3.date_input("Date Range", (df["Date"].min(), df["Date"].max()))

    filt = df.copy()
    if dept:
        filt = filt[filt["Department"].isin(dept)]
    if doc:
        filt = filt[filt["Doctor"].isin(doc)]
    if date_range:
        s, e = pd.to_datetime(date_range)
        filt = filt[(filt["Date"] >= s) & (filt["Date"] <= e)]

    st.dataframe(filt.head())

    # ==========================================================
    # KPIs
    # ==========================================================
    st.markdown("<div class='section-title'>Key Metrics</div>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Total Patients", f"{filt['Patient_ID'].nunique():,}")
    k2.metric("Total Revenue", to_currency(filt['Revenue'].sum()))
    k3.metric("Admission Rate", f"{(filt['Admission_Rate'].mean()*100):.2f}%")
    k4.metric("Revenue per Patient", to_currency(filt['Revenue'].sum() / max(1, filt["Patient_ID"].nunique())))

    # ==========================================================
    # CHARTS
    # ==========================================================
    st.markdown("<div class='section-title'>Patient Visits Over Time</div>", unsafe_allow_html=True)

    ts = filt.groupby("Date")["Patient_ID"].nunique().reset_index()
    ts["MA7"] = ts["Patient_ID"].rolling(7, min_periods=1).mean()

    fig = px.line(ts, x="Date", y=["Patient_ID", "MA7"], labels={"value":"Patients"}, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Revenue by Department
    st.markdown("<div class='section-title'>Revenue by Department</div>", unsafe_allow_html=True)
    dept_rev = filt.groupby("Department")["Revenue"].sum().reset_index()
    fig2 = px.bar(dept_rev, x="Department", y="Revenue", text="Revenue", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    # Admission by Doctor
    st.markdown("<div class='section-title'>Admission Rate by Doctor</div>", unsafe_allow_html=True)
    doc_rate = filt.groupby("Doctor")["Admission_Rate"].mean().reset_index()
    fig3 = px.bar(doc_rate, x="Doctor", y="Admission_Rate", text="Admission_Rate", template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

    # ==========================================================
    # ML — Simple Admission Prediction
    # ==========================================================
    st.markdown("<div class='section-title'>ML Model: Predict Admission</div>", unsafe_allow_html=True)

    if len(filt) > 60:

        X = filt[["Department", "Doctor", "Age", "Gender"]]
        y = filt["Admission_Rate"]

        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Department","Doctor","Gender"]),
            ("num", StandardScaler(), ["Age"])
        ])

        X_trans = pre.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.25, random_state=42)
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)
        st.write(f"Prediction Accuracy: **{score*100:.2f}%**")

    # ==========================================================
    # AUTOMATED INSIGHTS TABLE
    # ==========================================================
    st.markdown("<div class='section-title'>Automated Insights</div>", unsafe_allow_html=True)

    insights = []

    if not dept_rev.empty:
        best_dept = dept_rev.sort_values("Revenue", ascending=False).iloc[0]["Department"]
        insights.append(["Top Revenue Department", best_dept])

    if not doc_rate.empty:
        top_doc = doc_rate.sort_values("Admission_Rate", ascending=False).iloc[0]["Doctor"]
        insights.append(["Best Admission Rate Doctor", top_doc])

    rev_trend = "Increasing" if filt.sort_values("Date")["Revenue"].diff().mean() > 0 else "Decreasing"
    insights.append(["Revenue Trend", rev_trend])

    insight_df = pd.DataFrame(insights, columns=["Insight", "Value"])
    st.dataframe(insight_df, use_container_width=True)

    download(insight_df, "automated_insights.csv")
