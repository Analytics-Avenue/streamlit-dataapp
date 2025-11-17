import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# --- Page Setup ---
st.set_page_config(page_title="Healthcare Analytics & Insights", layout="wide")
st.markdown("<h1 style='margin-bottom:0.2rem'>Healthcare Analytics & Insights</h1>", unsafe_allow_html=True)
st.markdown("Enterprise-capable healthcare analytics: patient trends, treatment efficiency, readmission prediction, and cost forecasting.")

# -------------------------
# Dataset & Utilities
# -------------------------
REQUIRED_HEALTH_COLS = [
    "Patient_ID", "Age", "Gender", "Disease", "Symptoms", "Treatment", 
    "Admission_Date", "Discharge_Date", "Treatment_Cost", "Readmission", 
    "Department", "Visit_Date", "Outcome", "Risk_Level", "Risk_Score"
]

def auto_map_health_columns(df):
    rename = {}
    cols = [c.strip() for c in df.columns]
    mapping_candidates = {
        "Patient_ID":["patient_id","id","PatientID"],
        "Age":["age","Age"],
        "Gender":["gender","Gender","sex"],
        "Disease":["disease","Diagnosis","Condition"],
        "Symptoms":["symptoms","Symptom","Symptom_List"],
        "Treatment":["treatment","Treatment_Type"],
        "Admission_Date":["admission","admission_date","Admit_Date"],
        "Discharge_Date":["discharge","discharge_date","Discharge_Date"],
        "Treatment_Cost":["cost","Treatment_Cost","Bill"],
        "Readmission":["readmission","Readmit","Rehospitalization"],
        "Department":["department","Dept"],
        "Visit_Date":["visit_date","VisitDate"],
        "Outcome":["outcome","Treatment_Outcome"],
        "Risk_Level":["risk_level","RiskLevel"],
        "Risk_Score":["risk_score","RiskScore"]
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

def download_df(df, filename):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button("Download CSV", b, file_name=filename, mime="text/csv")

def to_currency(val):
    return f"₹ {val:,.2f}"

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["Overview", "Application", "Predictions"])

# -------------------------
# Overview Tab
# -------------------------
with tabs[0]:
    st.markdown("## Overview")
    st.markdown("""
    **Purpose:** Track patient trends, treatment efficiency, readmissions, and cost forecasting.  
    **Business Impact:** Helps hospitals reduce readmission rates, optimize costs, and improve patient outcomes.  
    **Capabilities:** Patient-level KPIs, treatment outcome analytics, predictive modeling, risk assessment.
    """)

    # Load default dataset only for computing KPIs
    DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/healthcare/healthcare_3.csv"
    try:
        df_default = pd.read_csv(DEFAULT_URL)
        df_default = auto_map_health_columns(df_default)
        
        # Derived metrics
        df_default["Length_of_Stay"] = (pd.to_datetime(df_default["Discharge_Date"]) - pd.to_datetime(df_default["Admission_Date"])).dt.days.fillna(0)
        df_default["Readmission_Flag"] = df_default["Readmission"].apply(lambda x: 1 if str(x).lower() in ["yes","1","true"] else 0)

        # KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Patients", f"{df_default['Patient_ID'].nunique():,}")
        k2.metric("Average Age", f"{df_default['Age'].mean():.1f}")
        k3.metric("Avg Length of Stay (days)", f"{df_default['Length_of_Stay'].mean():.1f}")
        k4.metric("Total Treatment Cost", to_currency(df_default['Treatment_Cost'].sum()))
        k5.metric("Readmission Rate", f"{df_default['Readmission_Flag'].mean()*100:.2f}%")
        
        # Example chart: Treatment Outcomes
        if "Outcome" in df_default.columns:
            fig_outcome = px.pie(df_default, names="Outcome", title="Treatment Outcomes Overview")
            st.plotly_chart(fig_outcome, use_container_width=True)
    except Exception as e:
        st.warning("Unable to compute KPIs from default dataset: " + str(e))

# -------------------------
# Application Tab
# -------------------------
with tabs[1]:
    st.markdown("### Step 1 — Load dataset")
    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)
    df = None
    
    if mode == "Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/healthcare/healthcare_3.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            df = auto_map_health_columns(df)
            st.success("Default dataset loaded")
            st.dataframe(df.head())
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()
    
    elif mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            df = auto_map_health_columns(df)
            st.success("File uploaded and mapped")
            st.dataframe(df.head())
    
    else:  # Upload + mapping
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded:
            raw = pd.read_csv(uploaded)
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head())
            mapping = {}
            for req in REQUIRED_HEALTH_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + list(raw.columns))
            if st.button("Apply mapping"):
                missing = [k for k,v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v:k for k,v in mapping.items()})
                    st.success("Mapping applied")
                    st.dataframe(df.head())
    
    if df is None:
        st.stop()
    
    # Save in session_state for Overview access
    st.session_state.df = df
    
    # -------------------------
    # Filters
    # -------------------------
    st.markdown("### Filters")
    if "Department" in df.columns:
        sel_dept = st.multiselect("Department", df["Department"].unique(), default=df["Department"].unique())
    else:
        sel_dept = []
    if "Gender" in df.columns:
        sel_gender = st.multiselect("Gender", df["Gender"].unique(), default=df["Gender"].unique())
    else:
        sel_gender = []
    if "Visit_Date" in df.columns:
        date_range = st.date_input("Visit Date Range", value=(df["Visit_Date"].min(), df["Visit_Date"].max()))
    else:
        date_range = None
    
    filt = df.copy()
    if sel_dept: filt = filt[filt["Department"].isin(sel_dept)]
    if sel_gender: filt = filt[filt["Gender"].isin(sel_gender)]
    if date_range is not None:
        filt = filt[(pd.to_datetime(filt["Visit_Date"]) >= pd.to_datetime(date_range[0])) & 
                    (pd.to_datetime(filt["Visit_Date"]) <= pd.to_datetime(date_range[1]))]
    
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(10), "filtered_healthcare_preview.csv")

# -------------------------
# Predictions Tab
# -------------------------
with tabs[2]:
    st.markdown("### Predictive Analytics")
    st.info("Add your readmission and cost prediction models here (requires dataset loaded in Application tab).")
