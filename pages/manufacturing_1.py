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

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(page_title="Production Downtime & Machine Failure Analytics", layout="wide")

# Hide sidebar navigation
st.markdown("""<style>[data-testid="stSidebarNav"]{display:none;}</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Logo + Header
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
# Helper: CSV download
# ---------------------------------------------------------
def download_df(df, filename):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode())
    b.seek(0)
    st.download_button("Download CSV", b, file_name=filename, mime="text/csv")


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tabs = st.tabs(["Overview", "Application"])


# ---------------------------------------------------------
# OVERVIEW PAGE
# ---------------------------------------------------------
with tabs[0]:

    st.markdown("## Overview")

    st.markdown("""
    <div style='padding:15px; border-radius:10px; background:#f7f7f7; border:1px solid #ddd;'>
        This application monitors machine behavior, detects early failure patterns,
        predicts breakdown likelihood, and provides actionable insights to reduce downtime.
    </div>
    """, unsafe_allow_html=True)

    # GRID: Capabilities (left) | Impact (right)
    c1, c2 = st.columns(2)

    c1.markdown("### Capabilities")
    c1.markdown("""
    <div style='padding:15px; border-radius:10px; background:#fafafa; border:1px solid #ccc;'>
        • Sensor anomaly detection (Temperature, Vibration, RPM, Load)<br>
        • Predictive maintenance using ML<br>
        • Failure probability scoring per machine<br>
        • Trend visualizations of machine health<br>
        • Multi-filter exploration by Machine Type, ID, Date Range
    </div>
    """, unsafe_allow_html=True)

    c2.markdown("### Business Impact")
    c2.markdown("""
    <div style='padding:15px; border-radius:10px; background:#fafafa; border:1px solid #ccc;'>
        • Reduced unplanned downtime<br>
        • Increased machine lifespan & performance<br>
        • Optimized preventive maintenance scheduling<br>
        • Avoid high-cost emergency repairs<br>
        • Real-time risk monitoring of production assets
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    st.markdown("### Key Metrics")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Machines Tracked", "Dynamic")
    k2.metric("Avg Temperature", "Dynamic")
    k3.metric("Avg Vibration", "Dynamic")
    k4.metric("Failure Events", "Dynamic")

    # Who should use
    st.markdown("### Who Should Use This App?")
    st.markdown("""
    <div style='padding:15px; border-radius:10px; background:#fafafa; border:1px solid #ccc;'>
        • Plant Managers monitoring machine health<br>
        • Maintenance Engineers planning predictive service<br>
        • Operations Teams avoiding production shocks<br>
        • Data Analysts analyzing large-scale sensor data
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# APPLICATION PAGE
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
            df = pd.read_csv(DEFAULT_URL)
            st.success("Default dataset loaded successfully.")
            st.dataframe(df.head())
        except Exception as e:
            st.error("Failed to load default dataset. Error: " + str(e))
            st.stop()

    # ---------------------- UPLOAD CSV ----------------------
    elif mode == "Upload CSV":
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)
            st.success("File uploaded.")
            st.dataframe(df.head())

    # ---------------------- UPLOAD + MAP ----------------------
    else:
        file = st.file_uploader("Upload CSV for mapping", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.write("Preview:")
            st.dataframe(raw.head())

            mapping = {}
            for col in REQUIRED_COLS:
                mapping[col] = st.selectbox(f"Map → {col}", ["-- Select --"] + list(raw.columns))

            if st.button("Apply Mapping"):
                miss = [m for m in mapping if mapping[m] == "-- Select --"]
                if miss:
                    st.error("Map all columns: " + ", ".join(miss))
                else:
                    df = raw.rename(columns={mapping[k]:k for k in mapping})
                    st.success("Mapping applied.")
                    st.dataframe(df.head())

    if df is None:
        st.stop()

    # ---------------------------------------------------------
    # Filters
    # ---------------------------------------------------------
    st.markdown("### Filters")

    m1, m2, m3 = st.columns(3)

    machine_ids = df["Machine_ID"].unique().tolist()
    machine_types = df["Machine_Type"].unique().tolist()

    sel_id = m1.multiselect("Machine ID", machine_ids, default=machine_ids)
    sel_type = m2.multiselect("Machine Type", machine_types, default=machine_types)

    df_f = df[df["Machine_ID"].isin(sel_id) & df["Machine_Type"].isin(sel_type)]

    st.dataframe(df_f.head(5))
    download_df(df_f, "filtered_machines.csv")

    # KPIs
    st.markdown("### Key Metrics (Dynamic)")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Machines", df_f["Machine_ID"].nunique())
    k2.metric("Avg Temp", f"{df_f['Temperature'].mean():.2f}")
    k3.metric("Avg Vibration", f"{df_f['Vibration'].mean():.2f}")
    k4.metric("Failures", df_f["Failure_Flag"].sum())

    # ---------------------------------------------------------
    # Charts
    # ---------------------------------------------------------
    st.markdown("### Charts")

    fig1 = px.line(df_f, x="Timestamp", y="Temperature", color="Machine_ID", title="Temperature Trend")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.box(df_f, x="Machine_Type", y="Vibration", title="Vibration Distribution by Machine Type")
    st.plotly_chart(fig2, use_container_width=True)

    # ---------------------------------------------------------
    # ML MODEL
    # ---------------------------------------------------------
    st.markdown("### Machine Learning: Failure Prediction")

    if len(df_f) > 50:

        ML_COLS = ["Temperature","Vibration","RPM","Load","Run_Hours"]
        X = df_f[ML_COLS]
        y = df_f["Failure_Flag"]

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(Xtr, ytr)

        preds = model.predict_proba(Xte)[:,1]

        results = pd.DataFrame({
            "Actual": yte.values,
            "Predicted_Prob": preds
        })

        st.dataframe(results.head(10))
        download_df(results, "ml_predictions.csv")

    else:
        st.info("Not enough data for ML model.")

    # ---------------------------------------------------------
    # Automated Insights
    # ---------------------------------------------------------
    st.markdown("### Automated Insights")

    insights = []

    # Highest Risk Machine
    risk = df_f.groupby("Machine_ID")["Failure_Flag"].mean().sort_values(ascending=False)
    highest = risk.index[0]
    insights.append({"Insight":"Highest Failure-Prone Machine", "Machine_ID":highest, "Failure Rate":risk.iloc[0]})

    ins_df = pd.DataFrame(insights)
    st.dataframe(ins_df)
    download_df(ins_df, "automated_insights.csv")
