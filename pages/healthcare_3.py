import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="PatientFlow Navigator", layout="wide")

# ------------------------- HIDE SIDEBAR -------------------
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# ADVANCED NEON / GRID CSS THEME
# ==========================================================
st.markdown("""
<style>
/* Global font & background */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Futuristic background */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    background: radial-gradient(circle at top left, #07111f 0, #02030a 40%, #000000 100%);
    color: #f5f5f5;
    animation: fadeIn 0.6s ease;
    border-radius: 18px;
}

/* Neon grid overlay (subtle) */
.block-container::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    opacity: 0.6;
    pointer-events: none;
    z-index: -1;
}

/* Main title */
.big-header {
    font-size: 40px;
    font-weight: 900;
    background: linear-gradient(90deg, #4fd1ff, #8b5cf6, #f97316);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 4px;
}

/* Subtitle */
.app-subtitle {
    font-size: 15px;
    color: #b9c2d0;
    margin-bottom: 18px;
}

/* Neon logo wrapper */
.neon-logo-wrap {
    padding: 10px 16px;
    border-radius: 18px;
    border: 1px solid rgba(79,209,255,0.35);
    box-shadow: 0 0 22px rgba(79,209,255,0.25);
    background: radial-gradient(circle at top left, rgba(79,209,255,0.12), transparent 55%);
}

/* Generic card */
.card {
    background: radial-gradient(circle at top left, rgba(79,209,255,0.12), rgba(18,22,40,0.95));
    border-radius: 16px;
    padding: 20px 20px;
    border: 1px solid rgba(148,163,184,0.35);
    box-shadow: 0 14px 35px rgba(15,23,42,0.88);
    backdrop-filter: blur(16px);
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}

/* Card glow border animation */
.card::before {
    content: "";
    position: absolute;
    inset: -2px;
    border-radius: 18px;
    border: 1px solid transparent;
    background: linear-gradient(120deg, rgba(56,189,248,0.0), rgba(129,140,248,0.45), rgba(248,250,252,0.0)) border-box;
    mask: 
      linear-gradient(#000 0 0) padding-box, 
      linear-gradient(#000 0 0) border-box;
    mask-composite: exclude;
    opacity: 0;
    transition: opacity 0.35s ease;
}
.card:hover::before { opacity: 1; }

/* Section title */
.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #e5e7eb;
    margin-top: 18px;
    margin-bottom: 10px;
    position: relative;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.section-title::after {
    content: "";
    position: absolute;
    left: 0;
    bottom: -6px;
    width: 0%;
    height: 2px;
    background: linear-gradient(90deg, #22d3ee, #8b5cf6);
    box-shadow: 0 0 12px rgba(34,211,238,0.9);
    transition: width 0.3s ease;
}
.section-title:hover::after { width: 38%; }

/* KPI cards */
.metric-card {
    background: radial-gradient(circle at top left, rgba(56,189,248,0.22), rgba(15,23,42,0.95));
    border-radius: 16px;
    padding: 18px 14px;
    text-align: center;
    border: 1px solid rgba(56,189,248,0.55);
    box-shadow: 0 10px 30px rgba(8,47,73,0.9);
    color: #e0f2fe;
    font-weight: 600;
    font-size: 15px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    position: relative;
    overflow: hidden;
    transition: all 0.25s ease;
}
.metric-card::after {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at top, rgba(34,211,238,0.25), transparent 55%);
    opacity: 0;
    transition: opacity 0.25s ease;
}
.metric-card:hover {
    transform: translateY(-4px) scale(1.01);
    box-shadow: 0 15px 40px rgba(8,47,73,0.95);
}
.metric-card:hover::after { opacity: 1; }

/* KPI value style (if used inside card) */
.metric-value {
    display: block;
    font-size: 22px;
    font-weight: 800;
    margin-top: 6px;
    color: #f9fafb;
}

/* Variable / attribute boxes */
.variable-box {
    padding: 14px 14px;
    border-radius: 14px;
    border: 1px solid rgba(148,163,184,0.5);
    background: linear-gradient(120deg, rgba(15,23,42,0.95), rgba(30,64,175,0.85));
    box-shadow: 0 8px 26px rgba(15,23,42,0.95);
    color: #e0f2fe;
    font-size: 14px;
    font-weight: 500;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.variable-label {
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-size: 11px;
    color: #a5b4fc;
}
.variable-name {
    font-size: 15px;
    font-weight: 600;
}

/* Required table styling */
.required-table th {
    background: #020617 !important;
    color: #e5e7eb !important;
    font-size: 13px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.required-table td {
    background: #020617 !important;
    color: #cbd5f5 !important;
    font-size: 13px !important;
    border-bottom: 1px solid #1f2937 !important;
}
.required-table tr:hover td {
    background: #0b1120 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background: #020617;
    color: #9ca3af;
    border-radius: 999px;
    padding: 8px 18px;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    border: 1px solid rgba(148,163,184,0.6);
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(90deg, #22d3ee, #8b5cf6);
    color: #020617;
    border-color: transparent;
    box-shadow: 0 0 18px rgba(59,130,246,0.8);
}

/* Buttons */
.stButton>button,
.stDownloadButton>button {
    background: radial-gradient(circle at top left, #22d3ee, #2563eb) !important;
    color: #0b1120 !important;
    border-radius: 999px !important;
    border: none !important;
    padding: 6px 18px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    box-shadow: 0 10px 30px rgba(15,23,42,0.85);
    transition: all 0.25s ease !important;
}
.stButton>button:hover,
.stDownloadButton>button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 18px 40px rgba(30,64,175,0.95);
}

/* Dataframe tweaks */
.dataframe td, .dataframe th {
    color: #e5e7eb !important;
}

/* Fade animation */
@keyframes fadeIn {
  from { opacity:0; transform: translateY(8px); }
  to { opacity:1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# COMPANY HEADER
# ==========================================================
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

st.markdown(f"""
<div class="neon-logo-wrap" style="display:flex; align-items:center; margin-bottom:18px;">
    <img src="{logo_url}" width="54" style="margin-right:12px; border-radius:12px; box-shadow:0 0 22px rgba(56,189,248,0.7);">
    <div style="line-height:1;">
        <div style="color:#e5e7eb; font-size:28px; font-weight:700;">Analytics Avenue &</div>
        <div style="color:#bfdbfe; font-size:28px; font-weight:700;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='big-header'>PatientFlow Navigator</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='app-subtitle'>Patient journeys, risk, readmissions & cost forecasting inside a single neon-lit control tower.</div>",
    unsafe_allow_html=True
)

# ==========================================================
# REQUIRED COLUMNS & UTILITIES
# ==========================================================
REQUIRED_HEALTH_COLS = [
    "Patient_ID", "Age", "Gender", "Disease", "Symptoms", "Treatment",
    "Admission_Date", "Discharge_Date", "Treatment_Cost", "Readmission",
    "Department", "Visit_Date", "Outcome", "Risk_Level", "Risk_Score"
]

REQUIRED_DICT = {
    "Patient_ID": "Unique identifier for the patient.",
    "Age": "Patient age in years at the time of visit.",
    "Gender": "Gender of the patient (M/F/Other).",
    "Disease": "Primary diagnosis or disease category.",
    "Symptoms": "Key symptoms recorded for the visit.",
    "Treatment": "Primary treatment / procedure administered.",
    "Admission_Date": "Date when the patient was admitted.",
    "Discharge_Date": "Date when the patient was discharged.",
    "Treatment_Cost": "Total cost incurred for the treatment.",
    "Readmission": "Whether the patient was readmitted (Yes/No).",
    "Department": "Hospital department handling the case.",
    "Visit_Date": "OP/IP visit date for the encounter.",
    "Outcome": "Final patient outcome (Recovered / Ongoing / Deceased / etc.).",
    "Risk_Level": "Qualitative risk band (Low/Medium/High).",
    "Risk_Score": "Numeric risk score assigned by triage or risk engine."
}

def auto_map_health_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    cols = [c.strip() for c in df.columns]
    mapping_candidates = {
        "Patient_ID": ["patient_id", "id", "patientid"],
        "Age": ["age"],
        "Gender": ["gender", "sex"],
        "Disease": ["disease", "diagnosis", "condition"],
        "Symptoms": ["symptoms", "symptom", "symptom_list"],
        "Treatment": ["treatment", "treatment_type", "procedure"],
        "Admission_Date": ["admission", "admission_date", "admit_date"],
        "Discharge_Date": ["discharge", "discharge_date"],
        "Treatment_Cost": ["cost", "treatment_cost", "bill", "billing_amount"],
        "Readmission": ["readmission", "readmit", "rehospitalization"],
        "Department": ["department", "dept"],
        "Visit_Date": ["visit_date", "visitdate", "op_date"],
        "Outcome": ["outcome", "treatment_outcome"],
        "Risk_Level": ["risk_level", "risklevel"],
        "Risk_Score": ["risk_score", "riskscore"]
    }
    for req, candidates in mapping_candidates.items():
        for c in cols:
            low = c.lower().strip()
            for cand in candidates:
                if cand.lower() in low or low in cand.lower():
                    rename[c] = req
                    break
            if c in rename:
                break
    if rename:
        df = df.rename(columns=rename)
    return df

def download_df(df, filename, label="Download CSV"):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")

def to_currency(val):
    try:
        return f"₹ {float(val):,.2f}"
    except Exception:
        return str(val)

# ==========================================================
# TABS
# ==========================================================
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# ==========================================================
# TAB 1 – OVERVIEW
# ==========================================================
with tab1:
    st.markdown("<div class='section-title'>What this workspace does</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <b>PatientFlow Navigator</b> is a hospital analytics cockpit for patient journeys. It stitches together
        visits, risk scores, treatments, readmissions and costs so clinical & admin teams can see:
        <ul>
            <li>Which patients are most likely to be <b>readmitted</b></li>
            <li>Which profiles tend to drive <b>high treatment cost</b></li>
            <li>How risk and stay length interact with <b>outcomes</b></li>
            <li>Which departments may be over- or under-utilized</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-title'>Purpose</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            • Build a single patient-level view of visits, treatment, cost & risk<br>
            • Predict <b>readmissions</b> using demographic + clinical signals<br>
            • Estimate <b>treatment cost</b> from risk, department & stay length<br>
            • Segment patients by risk bands to prioritize interventions<br>
            • Give operations a realistic view of patient pressure & cost drivers
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='section-title'>Business impact</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            • Lower readmission and complication rates<br>
            • Better <b>bed planning</b> via length-of-stay patterns<br>
            • More precise <b>package pricing</b> by risk profile<br>
            • Targeted care pathways for high-risk, high-cost cohorts<br>
            • Stronger evidence for <b>payer negotiations</b> and audits
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>High-level KPIs (once data is loaded)</div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Total Patients</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Avg Age</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Avg Length of Stay</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Readmission Rate</div>", unsafe_allow_html=True)

# ==========================================================
# TAB 2 – IMPORTANT ATTRIBUTES (DATA DICTIONARY)
# ==========================================================
with tab2:
    st.markdown("<div class='section-title'>Required Columns – Data Dictionary</div>", unsafe_allow_html=True)

    dict_df = pd.DataFrame(
        [{"Column": k, "Description": v} for k, v in REQUIRED_DICT.items()]
    )
    st.dataframe(
        dict_df.style.set_table_attributes('class="required-table"'),
        use_container_width=True
    )

    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("<div class='section-title'>Independent Variables</div>", unsafe_allow_html=True)
        independents = [
            "Age",
            "Gender",
            "Disease",
            "Symptoms",
            "Treatment",
            "Department",
            "Risk_Level",
            "Risk_Score",
            "Admission_Date",
            "Discharge_Date",
            "Visit_Date"
        ]
        for v in independents:
            st.markdown(f"""
            <div class="variable-box">
                <span class="variable-name">{v}</span>
                <span class="variable-label">INPUT FEATURE</span>
            </div>
            """, unsafe_allow_html=True)

    with c_right:
        st.markdown("<div class='section-title'>Dependent Variables</div>", unsafe_allow_html=True)
        dependents = [
            "Readmission",
            "Length_of_Stay (derived)",
            "Treatment_Cost",
            "Outcome"
        ]
        for v in dependents:
            st.markdown(f"""
            <div class="variable-box">
                <span class="variable-name">{v}</span>
                <span class="variable-label">TARGET / KPI</span>
            </div>
            """, unsafe_allow_html=True)

# ==========================================================
# TAB 3 – APPLICATION
# ==========================================================
with tab3:
    st.markdown("<div class='section-title'>Step 1 – Load dataset</div>", unsafe_allow_html=True)

    mode = st.radio(
        "Dataset option:",
        ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"],
        horizontal=True
    )

    df = None

    # ------------------------- Default dataset -------------------------
    if mode == "Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/healthcare/healthcare_3.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            df = auto_map_health_columns(df)
            st.success("Default dataset loaded.")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    # ------------------------- Upload CSV -------------------------
    elif mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            df = auto_map_health_columns(df)
            st.success("File uploaded and auto-mapped.")
            st.dataframe(df.head(), use_container_width=True)

    # ------------------------- Upload + Mapping -------------------------
    else:
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded:
            raw = pd.read_csv(uploaded)
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head(), use_container_width=True)
            mapping = {}
            for req in REQUIRED_HEALTH_COLS:
                mapping[req] = st.selectbox(
                    f"Map → {req}",
                    options=["-- Select --"] + list(raw.columns)
                )
            if st.button("Apply mapping"):
                missing = [k for k, v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v: k for k, v in mapping.items()})
                    st.success("Mapping applied.")
                    st.dataframe(df.head(), use_container_width=True)

    if df is None:
        st.stop()

    # Keep original in session if you ever want to reuse
    st.session_state.df = df.copy()

    # ==========================================================
    # BASIC PREP
    # ==========================================================
    df = df.copy()

    # Date conversions
    for dcol in ["Visit_Date", "Admission_Date", "Discharge_Date"]:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

    # Make sure Risk_Score and Treatment_Cost numeric
    if "Risk_Score" in df.columns:
        df["Risk_Score"] = pd.to_numeric(df["Risk_Score"], errors="coerce")
    if "Treatment_Cost" in df.columns:
        df["Treatment_Cost"] = pd.to_numeric(df["Treatment_Cost"], errors="coerce")

    # Derived: Length_of_Stay
    if "Admission_Date" in df.columns and "Discharge_Date" in df.columns:
        df["Length_of_Stay"] = (df["Discharge_Date"] - df["Admission_Date"]).dt.days
        df["Length_of_Stay"] = df["Length_of_Stay"].fillna(0).clip(lower=0)
    else:
        df["Length_of_Stay"] = 0

    # Readmission flag
    if "Readmission" in df.columns:
        df["Readmission_Flag"] = df["Readmission"].apply(
            lambda x: 1 if str(x).lower() in ["yes", "1", "true", "y"] else 0
        )
    else:
        df["Readmission_Flag"] = 0

    # ==========================================================
    # FILTERS
    # ==========================================================
    st.markdown("<div class='section-title'>Step 2 – Filters & preview</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 2, 1])

    departments = df["Department"].dropna().unique().tolist() if "Department" in df.columns else []
    genders = df["Gender"].dropna().unique().tolist() if "Gender" in df.columns else []

    with c1:
        sel_dept = st.multiselect(
            "Department",
            options=departments,
            default=departments[:3] if len(departments) >= 3 else departments
        )
    with c2:
        sel_gender = st.multiselect(
            "Gender",
            options=genders,
            default=genders
        )
    with c3:
        if "Visit_Date" in df.columns and df["Visit_Date"].notna().any():
            min_dt = df["Visit_Date"].min().date()
            max_dt = df["Visit_Date"].max().date()
            date_range = st.date_input(
                "Visit Date Range",
                value=(min_dt, max_dt)
            )
        else:
            date_range = None

    filt = df.copy()
    if sel_dept:
        filt = filt[filt["Department"].isin(sel_dept)]
    if sel_gender:
        filt = filt[filt["Gender"].isin(sel_gender)]
    if date_range is not None:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        if "Visit_Date" in filt.columns:
            filt = filt[(filt["Visit_Date"] >= start) & (filt["Visit_Date"] <= end)]

    st.markdown("#### Filtered sample (top 10 rows)")
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(10), "filtered_healthcare_preview.csv", label="Download Preview CSV")

    # ==========================================================
    # KPIs
    # ==========================================================
    st.markdown("<div class='section-title'>Key Metrics</div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    total_patients = filt["Patient_ID"].nunique() if "Patient_ID" in filt.columns else len(filt)
    total_revenue = filt["Treatment_Cost"].sum() if "Treatment_Cost" in filt.columns else 0
    avg_los = filt["Length_of_Stay"].mean() if "Length_of_Stay" in filt.columns else 0
    readmit_rate = filt["Readmission_Flag"].mean() * 100 if "Readmission_Flag" in filt.columns else 0
    avg_age = filt["Age"].mean() if "Age" in filt.columns else np.nan

    k1.markdown(f"<div class='metric-card'>TOTAL PATIENTS<span class='metric-value'>{total_patients:,}</span></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='metric-card'>TOTAL TREATMENT COST<span class='metric-value'>{to_currency(total_revenue)}</span></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='metric-card'>AVG LENGTH OF STAY<span class='metric-value'>{avg_los:.1f} days</span></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='metric-card'>READMISSION RATE<span class='metric-value'>{readmit_rate:.1f}%</span></div>", unsafe_allow_html=True)

    # ==========================================================
    # CHARTS
    # ==========================================================
    st.markdown("<div class='section-title'>Descriptive dashboards</div>", unsafe_allow_html=True)

    # Visits / patients over time
    if "Visit_Date" in filt.columns and "Patient_ID" in filt.columns:
        ts = filt.groupby("Visit_Date")["Patient_ID"].nunique().reset_index().sort_values("Visit_Date")
        ts["MA_7"] = ts["Patient_ID"].rolling(7, min_periods=1).mean()
        fig_ts = px.line(
            ts, x="Visit_Date", y=["Patient_ID", "MA_7"],
            labels={"value": "Patients", "variable": "Series"},
            template="plotly_dark"
        )
        fig_ts.update_layout(title="Daily unique patients (with 7-day moving average)")
        st.plotly_chart(fig_ts, use_container_width=True)

    # Revenue by Department
    if "Department" in filt.columns and "Treatment_Cost" in filt.columns:
        dept_rev = filt.groupby("Department")["Treatment_Cost"].sum().reset_index().sort_values("Treatment_Cost", ascending=False)
        fig_dept = px.bar(
            dept_rev, x="Department", y="Treatment_Cost",
            text="Treatment_Cost",
            template="plotly_dark"
        )
        fig_dept.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
        fig_dept.update_layout(title="Total treatment cost by Department")
        st.plotly_chart(fig_dept, use_container_width=True)
    else:
        dept_rev = pd.DataFrame()

    # Risk vs Cost scatter
    if "Risk_Score" in filt.columns and "Treatment_Cost" in filt.columns:
        fig_risk = px.scatter(
            filt,
            x="Risk_Score",
            y="Treatment_Cost",
            color="Department" if "Department" in filt.columns else None,
            hover_data=["Patient_ID"] if "Patient_ID" in filt.columns else None,
            template="plotly_dark",
            title="Risk Score vs Treatment Cost"
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    # ==========================================================
    # PREDICTIVE ANALYTICS
    # ==========================================================
    st.markdown("<div class='section-title'>Predictive Analytics</div>", unsafe_allow_html=True)

    # ------------------------- Readmission Prediction ------------------
    st.markdown("#### Readmission Prediction (Classification)")

    req_clf_cols = ["Age", "Gender", "Department", "Risk_Score", "Length_of_Stay", "Readmission_Flag"]
    if all(col in filt.columns for col in req_clf_cols):
        clf_df = filt[req_clf_cols].dropna().copy()

        if len(clf_df) > 40 and clf_df["Readmission_Flag"].nunique() > 1:
            le_gender = LabelEncoder()
            le_dept = LabelEncoder()
            clf_df["Gender"] = le_gender.fit_transform(clf_df["Gender"].astype(str))
            clf_df["Department"] = le_dept.fit_transform(clf_df["Department"].astype(str))

            X_clf = clf_df.drop("Readmission_Flag", axis=1)
            y_clf = clf_df["Readmission_Flag"]

            X_train, X_test, y_train, y_test = train_test_split(
                X_clf, y_clf, test_size=0.2, random_state=42
            )
            clf_model = RandomForestClassifier(n_estimators=120, random_state=42)
            with st.spinner("Training Readmission model..."):
                clf_model.fit(X_train, y_train)

            y_pred = clf_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.markdown(f"**Model Accuracy:** {acc*100:.2f}%")

            # Predict for new patient
            st.markdown("##### Predict readmission for a new patient profile")
            c_a, c_b, c_c, c_d, c_e = st.columns(5)
            with c_a:
                age = st.number_input("Age", min_value=0, max_value=120, value=40)
            with c_b:
                gender = st.selectbox("Gender", clf_df["Gender"].unique(), format_func=lambda x: le_gender.inverse_transform([x])[0])
            with c_c:
                dept = st.selectbox("Department", clf_df["Department"].unique(), format_func=lambda x: le_dept.inverse_transform([x])[0])
            with c_d:
                risk_score = st.number_input("Risk Score", min_value=0.0, max_value=10.0, value=3.0)
            with c_e:
                los = st.number_input("Length of Stay (days)", min_value=0, max_value=60, value=4)

            if st.button("Predict Readmission"):
                x_new = pd.DataFrame([[age, gender, dept, risk_score, los]], columns=X_clf.columns)
                pred = clf_model.predict(x_new)[0]
                prob = clf_model.predict_proba(x_new)[0][1]
                st.success(f"Predicted Readmission: {'Yes' if pred == 1 else 'No'} (prob: {prob*100:.1f}%)")

            # Feature importance
            fi_df = pd.DataFrame({
                "Feature": X_clf.columns,
                "Importance": clf_model.feature_importances_
            }).sort_values("Importance", ascending=False)
            fig_fi = px.bar(
                fi_df, x="Feature", y="Importance",
                title="Readmission – Feature Importance (RandomForest)",
                template="plotly_dark",
                text="Importance"
            )
            fig_fi.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Not enough balanced rows to train a proper readmission model (need > 40 rows & both classes).")
    else:
        st.warning("Required columns for readmission model not present in dataset.")

    # ------------------------- Treatment Cost Prediction ------------------
    st.markdown("#### Treatment Cost Prediction (Regression)")

    req_reg_cols = ["Age", "Gender", "Department", "Risk_Score", "Length_of_Stay", "Treatment_Cost"]
    if all(col in filt.columns for col in req_reg_cols):
        reg_df = filt[req_reg_cols].dropna().copy()

        if len(reg_df) > 40:
            le_gender_r = LabelEncoder()
            le_dept_r = LabelEncoder()
            reg_df["Gender"] = le_gender_r.fit_transform(reg_df["Gender"].astype(str))
            reg_df["Department"] = le_dept_r.fit_transform(reg_df["Department"].astype(str))

            X_reg = reg_df.drop("Treatment_Cost", axis=1)
            y_reg = reg_df["Treatment_Cost"]

            X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
                X_reg, y_reg, test_size=0.2, random_state=42
            )
            reg_model = RandomForestRegressor(n_estimators=140, random_state=42)
            with st.spinner("Training Treatment Cost model..."):
                reg_model.fit(X_train_r, y_train_r)

            y_pred_r = reg_model.predict(X_test_r)
            rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
            st.markdown(f"**Model RMSE:** {rmse:,.2f}")

            st.markdown("##### Predict Treatment Cost for new patient")
            c1r, c2r, c3r, c4r, c5r = st.columns(5)
            with c1r:
                age_r = st.number_input("Age (Cost)", min_value=0, max_value=120, value=40)
            with c2r:
                gender_r = st.selectbox("Gender (Cost)", reg_df["Gender"].unique(), key="gender_r")
            with c3r:
                dept_r = st.selectbox("Department (Cost)", reg_df["Department"].unique(), key="dept_r")
            with c4r:
                risk_score_r = st.number_input("Risk Score (Cost)", min_value=0.0, max_value=10.0, value=3.0, key="risk_r")
            with c5r:
                los_r = st.number_input("Length of Stay (Cost)", min_value=0, max_value=60, value=4, key="los_r")

            if st.button("Predict Treatment Cost", key="predict_cost"):
                x_new_r = pd.DataFrame(
                    [[age_r, gender_r, dept_r, risk_score_r, los_r]],
                    columns=X_reg.columns
                )
                pred_cost = reg_model.predict(x_new_r)[0]
                st.success(f"Predicted Treatment Cost: {to_currency(pred_cost)}")

            # Feature importance
            fi_reg = pd.DataFrame({
                "Feature": X_reg.columns,
                "Importance": reg_model.feature_importances_
            }).sort_values("Importance", ascending=False)
            fig_reg = px.bar(
                fi_reg, x="Feature", y="Importance",
                title="Treatment Cost – Feature Importance (RandomForest)",
                template="plotly_dark", text="Importance"
            )
            fig_reg.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            st.plotly_chart(fig_reg, use_container_width=True)

            # SHAP summary
            try:
                explainer_r = shap.TreeExplainer(reg_model)
                shap_values_r = explainer_r.shap_values(X_test_r)

                fig_shap_r, ax2 = plt.subplots(figsize=(8, 5))
                shap.summary_plot(shap_values_r, X_test_r, plot_type="bar", show=False)
                ax2.set_title("Treatment Cost – SHAP Feature Importance", fontsize=13)
                st.pyplot(fig_shap_r)
                plt.close(fig_shap_r)
            except Exception as e:
                st.info(f"SHAP summary could not be generated: {e}")
        else:
            st.info("Not enough data to train treatment cost model (need > 40 rows).")
    else:
        st.warning("Required columns for treatment cost regression are not available.")

    # ==========================================================
    # AUTOMATED INSIGHTS
    # ==========================================================
    st.markdown("<div class='section-title'>Automated tabular insights</div>", unsafe_allow_html=True)

    insight_rows = []

    # Top dept by revenue
    if not dept_rev.empty:
        best_dept = dept_rev.iloc[0]
        insight_rows.append({
            "Insight": "Top revenue-generating department",
            "Entity": best_dept["Department"],
            "Metric": f"{to_currency(best_dept['Treatment_Cost'])}",
            "Suggested_Action": "Use this department profile as a benchmark for patient mix & pricing."
        })

    # Highest readmission department
    if "Department" in filt.columns and "Readmission_Flag" in filt.columns:
        dep_read = filt.groupby("Department")["Readmission_Flag"].mean().reset_index()
        dep_read = dep_read.sort_values("Readmission_Flag", ascending=False)
        if not dep_read.empty:
            worst = dep_read.iloc[0]
            insight_rows.append({
                "Insight": "Department with highest readmission rate",
                "Entity": worst["Department"],
                "Metric": f"{worst['Readmission_Flag']*100:.1f}%",
                "Suggested_Action": "Investigate discharge criteria, follow-up protocols & care pathways."
            })

    # Highest risk cohort (avg Risk_Score)
    if "Risk_Score" in filt.columns and "Disease" in filt.columns:
        risk_by_dis = filt.groupby("Disease")["Risk_Score"].mean().reset_index().sort_values("Risk_Score", ascending=False)
        if not risk_by_dis.empty:
            dri = risk_by_dis.iloc[0]
            insight_rows.append({
                "Insight": "Highest average risk disease cohort",
                "Entity": dri["Disease"],
                "Metric": f"{dri['Risk_Score']:.2f}",
                "Suggested_Action": "Flag this diagnosis for tighter monitoring and early escalation."
            })

    # Cost-heavy risk band
    if "Risk_Level" in filt.columns and "Treatment_Cost" in filt.columns:
        cost_by_band = filt.groupby("Risk_Level")["Treatment_Cost"].mean().reset_index().sort_values("Treatment_Cost", ascending=False)
        if not cost_by_band.empty:
            rb = cost_by_band.iloc[0]
            insight_rows.append({
                "Insight": "Most expensive risk band on average",
                "Entity": rb["Risk_Level"],
                "Metric": to_currency(rb["Treatment_Cost"]),
                "Suggested_Action": "Design special bundles/caps for this risk band with insurers."
            })

    if not insight_rows:
        st.info("No insights could be generated for the current filtered dataset.")
    else:
        insight_df = pd.DataFrame(insight_rows)
        st.dataframe(insight_df, use_container_width=True)
        download_df(insight_df, "patientflow_insights.csv", label="Download Insights CSV")
