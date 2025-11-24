# app_predictive_maintenance_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
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

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(page_title="Production Downtime & Machine Failure Analytics", layout="wide")

# Hide sidebar
st.markdown("""<style>[data-testid="stSidebarNav"]{display:none;}</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Card Glow CSS
# ---------------------------------------------------------
st.markdown("""
<style>
.card {
    padding: 20px;
    border-radius: 12px;
    background: white;
    border: 1px solid #d9d9d9;
    transition: 0.3s;
    box-shadow: 0px 2px 4px rgba(0,0,0,0.1);
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0px 4px 18px rgba(6,75,134,0.35);
    border-color: #064b86;
}
.kpi {
    padding: 30px;
    border-radius: 14px;
    background: white;
    border: 1px solid #ccc;
    font-size: 26px;
    font-weight: bold;
    color: #064b86;
    text-align: center;
    transition: 0.3s;
}
.kpi:hover {
    transform: translateY(-4px);
    box-shadow: 0px 4px 15px rgba(6,75,134,0.30);
}
.small { font-size:13px; color:#666; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Header + Logo
# ---------------------------------------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

st.markdown(f"""
<div style="display:flex; align-items:center;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:34px; font-weight:bold;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:34px; font-weight:bold;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper for CSV download
# ---------------------------------------------------------
def download_df(df, filename):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode())
    b.seek(0)
    st.download_button("Download CSV", b, file_name=filename, mime="text/csv")

# ---------------------------------------------------------
# Safe CSV loader (handles duplicate column names)
# ---------------------------------------------------------
def read_csv_safe(url_or_file):
    """
    Read CSV from URL or file-like. If duplicate columns exist, make them unique
    by appending suffixes: col, col__dup1, col__dup2...
    """
    # read without modifying columns first
    df = pd.read_csv(url_or_file)
    cols = list(df.columns)
    if len(cols) != len(set(cols)):
        # make unique
        new_cols = []
        seen = {}
        for c in cols:
            base = str(c).strip()
            if base not in seen:
                seen[base] = 0
                new_cols.append(base)
            else:
                seen[base] += 1
                new_cols.append(f"{base}__dup{seen[base]}")
        df.columns = new_cols
    # strip whitespace in column names
    df.columns = [str(c).strip() for c in df.columns]
    return df

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tabs = st.tabs(["Overview", "Application"])

# ---------------------------------------------------------
# Overview Tab
# ---------------------------------------------------------
with tabs[0]:
    st.markdown("## Overview")

    st.markdown("""
    <div class='card'>
        This application monitors machine behavior, detects early failure patterns,
        predicts breakdown likelihood, and provides actionable insights to reduce downtime.
    </div>
    """, unsafe_allow_html=True)

    # Capabilities / Impact
    c1, c2 = st.columns(2)

    c1.markdown("### Capabilities")
    c1.markdown("""
    <div class='card'>
        • Sensor anomaly detection (Temperature, Vibration, RPM, Load)<br>
        • Predictive maintenance using ML<br>
        • Failure probability scoring per machine<br>
        • Trend visualizations of machine health<br>
        • Multi-filter exploration by Machine Type, ID, Date Range
    </div>
    """, unsafe_allow_html=True)

    c2.markdown("### Business Impact")
    c2.markdown("""
    <div class='card'>
        • Reduced unplanned downtime<br>
        • Increased machine lifespan<br>
        • Optimized preventive maintenance<br>
        • Avoid costly breakdowns<br>
        • Real-time monitoring
    </div>
    """, unsafe_allow_html=True)

    # KPI CARDS (label-only, big)
    st.markdown("## KPIs")
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Machines Tracked</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Avg Temperature</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Avg Vibration</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Failure Events</div>", unsafe_allow_html=True)

    st.markdown("### Who Should Use This App?")
    st.markdown("""
    <div class='card'>
        • Plant Managers<br>
        • Maintenance Engineers<br>
        • Operations Teams<br>
        • Data Analysts
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Application Tab
# ---------------------------------------------------------
with tabs[1]:
    st.header("Application")
    st.markdown("### Choose Dataset Option")

    mode = st.radio("Select:", [
        "Default dataset (GitHub URL)",
        "Upload CSV",
        "Upload CSV + Manual Column Mapping"
    ])

    df = None

    REQUIRED_COLS = [
        "Timestamp","Machine_ID","Machine_Type","Temperature","Vibration",
        "RPM","Load","Run_Hours","Temp_Anomaly","Vib_Anomaly",
        "Load_Anomaly","RPM_Anomaly","Failure_Flag"
    ]

    # ---------------------- DEFAULT DATASET ----------------------
    if mode == "Default dataset (GitHub URL)":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/manufacturing/machine_failure_data.csv"
        try:
            df = read_csv_safe(DEFAULT_URL)
            st.success("Default dataset loaded successfully.")
            st.dataframe(df.head())
        except Exception as e:
            st.error("Failed to load default dataset. Check DEFAULT_URL or network. Error: " + str(e))
            st.stop()

    # ---------------------- UPLOAD CSV ----------------------
    elif mode == "Upload CSV":
        file = st.file_uploader("Upload CSV", type=["csv"], key="upload_simple")
        if file:
            try:
                df = read_csv_safe(file)
                st.success("File uploaded.")
                st.dataframe(df.head())
            except Exception as e:
                st.error("Failed to read uploaded CSV: " + str(e))
                st.stop()
        else:
            st.stop()

    # ---------------------- UPLOAD + MAP ----------------------
    else:
        file = st.file_uploader("Upload CSV for mapping", type=["csv"], key="upload_map")
        if file:
            try:
                raw = read_csv_safe(file)
            except Exception as e:
                st.error("Failed to read uploaded CSV: " + str(e))
                st.stop()

            st.write("Preview:")
            st.dataframe(raw.head())

            mapping = {}
            cols_list = list(raw.columns)
            for col in REQUIRED_COLS:
                mapping[col] = st.selectbox(f"Map → {col}", ["-- Select --"] + cols_list, key=f"map_{col}")

            if st.button("Apply Mapping", key="apply_map"):
                miss = [m for m in mapping if mapping[m] == "-- Select --"]
                if miss:
                    st.error("Map all columns: " + ", ".join(miss))
                else:
                    # rename selected columns (only those mapped)
                    rename_map = {mapping[k]: k for k in mapping}
                    df = raw.rename(columns=rename_map)
                    st.success("Mapping applied.")
                    st.dataframe(df.head())
        else:
            st.stop()

    if df is None:
        st.stop()

    # ---------- Basic cleanup & safety ----------
    # trim column whitespace
    df.columns = [str(c).strip() for c in df.columns]

    # If there are duplicate logical columns (e.g., Temperature and Temperature__dup1),
    # prefer the first occurrence and drop exact duplicates after merging similar names.
    def prefer_column(df, base):
        # returns first matching column name for base (exact or starting with base)
        for c in df.columns:
            if c == base:
                return c
        for c in df.columns:
            if c.startswith(base + "__dup"):
                return c
        return None

    # Try to align common columns to canonical names where possible
    canonical_map = {}
    for base in ["Timestamp","Machine_ID","Machine_Type","Temperature","Vibration","RPM","Load","Run_Hours","Failure_Flag"]:
        col_found = prefer_column(df, base)
        if col_found and col_found != base:
            canonical_map[col_found] = base
    if canonical_map:
        df = df.rename(columns=canonical_map)

    # Ensure Timestamp is datetime if present
    if "Timestamp" in df.columns:
        try:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        except:
            # fallback: try common formats
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Ensure numeric columns exist and are numeric
    numeric_cols = ["Temperature","Vibration","RPM","Load","Run_Hours","Failure_Flag"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # If Failure_Flag missing, create default 0
    if "Failure_Flag" not in df.columns:
        df["Failure_Flag"] = 0

    # Replace NaNs in numeric cols with sensible defaults (median or 0)
    for c in ["Temperature","Vibration","RPM","Load","Run_Hours"]:
        if c in df.columns:
            if df[c].notna().any():
                df[c] = df[c].fillna(df[c].median())
            else:
                df[c] = df[c].fillna(0)

    # ---------------------- Filters ----------------------
    st.markdown("### Filters")

    # safe guards for missing columns
    machine_ids = df["Machine_ID"].unique().tolist() if "Machine_ID" in df.columns else []
    machine_types = df["Machine_Type"].unique().tolist() if "Machine_Type" in df.columns else []

    m1, m2, m3 = st.columns(3)
    sel_id = m1.multiselect("Machine ID", machine_ids, default=machine_ids if machine_ids else [])
    sel_type = m2.multiselect("Machine Type", machine_types, default=machine_types if machine_types else [])

    df_f = df.copy()
    if sel_id:
        if "Machine_ID" in df_f.columns:
            df_f = df_f[df_f["Machine_ID"].isin(sel_id)]
    if sel_type:
        if "Machine_Type" in df_f.columns:
            df_f = df_f[df_f["Machine_Type"].isin(sel_type)]

    st.dataframe(df_f.head(5), use_container_width=True)
    download_df(df_f, "filtered_machines.csv")

    # ---------------------- Dynamic KPIs ----------------------
    st.markdown("### Key Metrics (Dynamic)")
    k1,k2,k3,k4 = st.columns(4)
    # Use safe computations (if column missing, show N/A)
    machines_count = int(df_f["Machine_ID"].nunique()) if "Machine_ID" in df_f.columns else 0
    avg_temp = f"{df_f['Temperature'].mean():.2f}" if "Temperature" in df_f.columns and df_f['Temperature'].notna().any() else "N/A"
    avg_vib = f"{df_f['Vibration'].mean():.2f}" if "Vibration" in df_f.columns and df_f['Vibration'].notna().any() else "N/A"
    failures = int(df_f["Failure_Flag"].sum()) if "Failure_Flag" in df_f.columns else 0

    k1.metric("Machines", machines_count)
    k2.metric("Avg Temp", avg_temp)
    k3.metric("Avg Vibration", avg_vib)
    k4.metric("Failures", failures)

    # ---------------------- Charts ----------------------
    st.markdown("### Charts")
    # Safe plotting: check required columns
    if "Timestamp" in df_f.columns and "Temperature" in df_f.columns:
        fig1 = px.line(df_f.sort_values("Timestamp"), x="Timestamp", y="Temperature", color="Machine_ID", title="Temperature Trend")
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Timestamp or Temperature column missing — skipping time-series chart.")

    if "Machine_Type" in df_f.columns and "Vibration" in df_f.columns:
        fig2 = px.box(df_f, x="Machine_Type", y="Vibration", title="Vibration Distribution by Machine Type")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Machine_Type or Vibration column missing — skipping box chart.")

    # ---------------------- ML: Failure Prediction ----------------------
    st.markdown("### Machine Learning: Failure Prediction")
    # Require a minimum number of rows and columns
    ML_COLS = [c for c in ["Temperature","Vibration","RPM","Load","Run_Hours"] if c in df_f.columns]
    if len(df_f) >= 50 and len(ML_COLS) >= 2 and "Failure_Flag" in df_f.columns:
        X = df_f[ML_COLS].fillna(0)
        y = df_f["Failure_Flag"].astype(int).fillna(0)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X.values)

        Xtr, Xte, ytr, yte = train_test_split(Xs, y.values, test_size=0.2, random_state=42, stratify=y.values if len(np.unique(y.values))>1 else None)

        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(Xtr, ytr)

        preds = model.predict_proba(Xte)[:, 1]

        results = pd.DataFrame({
            "Actual": yte,
            "Predicted_Prob": preds
        })

        st.dataframe(results.head(10))
        download_df(results, "ml_predictions.csv")
    else:
        st.info("Not enough data or features for ML model. Need >=50 rows, at least 2 numeric features among Temperature/Vibration/RPM/Load/Run_Hours, and Failure_Flag column.")

    # ---------------------- Automated Insights ----------------------
    st.markdown("### Automated Insights")
    insights = []

    if "Machine_ID" in df_f.columns and "Failure_Flag" in df_f.columns:
        risk_series = df_f.groupby("Machine_ID")["Failure_Flag"].mean().sort_values(ascending=False)
        if not risk_series.empty:
            highest = risk_series.index[0]
            insights.append({
                "Insight":"Highest Failure-Prone Machine",
                "Machine_ID": highest,
                "Failure Rate": round(float(risk_series.iloc[0]),4)
            })
    # Add basic sensor summary
    if "Temperature" in df_f.columns:
        insights.append({"Insight":"Avg Temperature", "Value": round(float(df_f["Temperature"].mean()),2)})
    if "Vibration" in df_f.columns:
        insights.append({"Insight":"Avg Vibration", "Value": round(float(df_f["Vibration"].mean()),2)})

    ins_df = pd.DataFrame(insights)
    if ins_df.empty:
        st.info("No automated insights could be generated for the current filter.")
    else:
        st.dataframe(ins_df, use_container_width=True)
        download_df(ins_df, "automated_insights.csv")
