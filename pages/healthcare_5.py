import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import math
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(page_title="HospitalOps — Dashboard", layout="wide")

# Logo
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display:flex;align-items:center;">
<img src="{logo_url}" width="60" style="margin-right:10px;">
<div style="line-height:1;">
<div style="color:#064b86;font-size:36px;font-weight:bold;margin:0;padding:0;">Analytics Avenue &</div>
<div style="color:#064b86;font-size:36px;font-weight:bold;margin:0;padding:0;">Advanced Analytics</div>
</div></div>
""", unsafe_allow_html=True)

st.title("HospitalOps Dashboard")

# -------------------------------------------------------------------
# CSS
# -------------------------------------------------------------------
st.markdown("""
<style>
*{color:#000 !important;}
.glow-card{
    background:#fff;padding:16px;border-radius:12px;
    box-shadow:0 10px 30px rgba(0,120,255,0.12),0 0 12px rgba(0,120,255,0.06) inset;
    border:1px solid rgba(0,120,255,0.18);margin-bottom:12px;transition:0.3s;
}
.glow-card:hover{box-shadow:0 0 25px #0078ff;transform:translateY(-3px);}
.metric-card {
    background: #ffffff;
    padding: 14px;
    border-radius: 10px;
    text-align: center;
    font-weight: 600;
    font-size: 16px;
    border: 1px solid rgba(0,0,0,0.1);
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
    transition: 0.25s ease-in-out;
}
.metric-card:hover {
    box-shadow: 0 0 18px rgba(0, 120, 255, 0.45);
    transform: translateY(-3px) scale(1.02);
    border-color: rgba(0,120,255,0.35);
}
[data-testid="stSidebarNav"]{display:none;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Globals
# -------------------------------------------------------------------
DEFAULT_DATA_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/healthcare/healthcare_syn.csv"

def remove_dup(df):
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = df.columns.str.strip()
    return df

def download_df(df, filename):
    bio = BytesIO()
    bio.write(df.to_csv(index=False).encode("utf-8"))
    bio.seek(0)
    st.download_button("Download CSV", bio, file_name=filename, mime="text/csv")

for key in ["hospital_master", "reg_pipe"]:
    st.session_state.setdefault(key, None)

# -------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------
tab1, tab2 = st.tabs(["Overview", "Application"])

# -------------------------------------------------------------------
# TAB 1 – OVERVIEW
# -------------------------------------------------------------------
with tab1:

    st.markdown('<div class="glow-card"><h2>About this Application</h2>'
                '<p>HospitalOps provides complete operational intelligence for hospital resource planning, equipment management, staffing, and patient flow analytics.</p>'
                '</div>', unsafe_allow_html=True)

    st.markdown('<div class="glow-card"><h3>Purpose</h3>'
                '<p>Monitor KPIs, understand shortages, benchmark hospital risk, and build ML-driven forecasts.</p>'
                '</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="glow-card"><h3>Capabilities</h3>
        <ul>
            <li>Dataset ingestion</li>
            <li>Hospital KPI monitoring</li>
            <li>Segmentation & clustering</li>
            <li>ML regression predictions</li>
            <li>Automated Insights</li>
            <li>Interactive filters</li>
        </ul></div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="glow-card"><h3>Business Impact</h3>
        <ul>
            <li>Improved hospital readiness</li>
            <li>Optimized equipment procurement</li>
            <li>Operational efficiency</li>
            <li>Bottleneck reduction</li>
            <li>Predictive capability for planning</li>
        </ul></div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="glow-card"><h3>Intended Users</h3>'
                '<p>Hospital administrators, operations teams, strategy, procurement & finance.</p>'
                '</div>', unsafe_allow_html=True)

    st.markdown("### KPIs")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown("<div class='metric-card'>High Patient Load</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Avg Beds Availability</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Equipment Shortage Score</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Patients per Staff</div>", unsafe_allow_html=True)
    k5.markdown("<div class='metric-card'>Overall Risk Score</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------
# TAB 2 – APPLICATION
# -------------------------------------------------------------------
with tab2:

    st.header("Application")

    mode = st.radio("Dataset Mode", ["Default (URL)", "Upload CSV"], horizontal=True)

    # ----------------------------------------------------------
    # DATA LOADING
    # ----------------------------------------------------------
    if mode == "Default (URL)":
        df = pd.read_csv(DEFAULT_DATA_URL)
        df = remove_dup(df)
        st.session_state["hospital_master"] = df
        st.success("Default dataset loaded.")
        st.dataframe(df.head())

    else:
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)
            df = remove_dup(df)
            st.session_state["hospital_master"] = df
            st.success("Uploaded dataset loaded.")
            st.dataframe(df.head())

    df = st.session_state.get("hospital_master")
    if df is None:
        st.stop()

    # CLEAN COLUMNS
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(" ", "_")

    # ----------------------------------------------------------
    # FILTERS
    # ----------------------------------------------------------
    st.subheader("Filters")

    col_l, col_t = st.columns(2)

    with col_l:
        locs = ["All"] + sorted(df_clean["location"].dropna().unique().tolist())
        floc = st.selectbox("Location", locs)
    with col_t:
        types = ["All"] + sorted(df_clean["hospital_type"].dropna().unique().tolist())
        ftype = st.selectbox("Hospital Type", types)

    df_f = df_clean.copy()
    if floc != "All":
        df_f = df_f[df_f["location"] == floc]
    if ftype != "All":
        df_f = df_f[df_f["hospital_type"] == ftype]

    st.write("Filtered Preview:", df_f.head())

    # ----------------------------------------------------------
    # CHARTS
    # ----------------------------------------------------------
    st.subheader("Charts & Visualizations")

    num_cols = df_f.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_f.select_dtypes(exclude=[np.number]).columns.tolist()

    if num_cols:
        xcol = st.selectbox("Histogram Column", num_cols)
        st.plotly_chart(px.histogram(df_f, x=xcol, nbins=30), use_container_width=True)

    # SCATTER
    if len(num_cols) >= 2:
        x = st.selectbox("X axis", num_cols, key="sx2")
        y = st.selectbox("Y axis", num_cols, key="sy2")
        st.plotly_chart(px.scatter(df_f, x=x, y=y,
                                   color=cat_cols[0] if cat_cols else None),
                         use_container_width=True)

    # ----------------------------------------------------------
    # ML PREDICTION
    # ----------------------------------------------------------
    st.subheader("ML Prediction (Regression)")

    numeric_cols = df_f.select_dtypes(include=[np.number]).columns.tolist()
    target = st.selectbox("Select Target Column", numeric_cols)

    features = [c for c in numeric_cols if c != target]

    X = df_f[features]
    y = df_f[target]

    test_size = st.slider("Test Size", 0.1, 0.4, 0.2)

    if st.button("Train Model"):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=150, random_state=42))
        ])

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
        pipe.fit(Xtr, ytr)

        preds = pipe.predict(Xte)
        rmse = math.sqrt(np.mean((yte - preds) ** 2))
        r2 = np.corrcoef(yte, preds)[0, 1] ** 2

        st.success(f"Model trained. RMSE = {rmse:.2f} | R² = {r2:.2f}")

        out = Xte.copy()
        out["Actual"] = yte.values
        out["Predicted"] = preds

        st.dataframe(out)
        download_df(out, "ml_predictions.csv")

    # ------------------------------------------------------
    # Automated Insights (Robust)
    # ------------------------------------------------------
    st.subheader("Automated Insights")
    
    df_clean = df.copy()
    
    # Normalize column names
    df_clean.columns = (
        df_clean.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    
    insights = []
    
    # Helper function
    def safe_top(df, col, n=3):
        return df.dropna(subset=[col]).nlargest(n, col) if col in df.columns else pd.DataFrame()
    
    # 1. Monthly Patients
    top_patients = safe_top(df_clean, "monthly_patients")
    if not top_patients.empty and "hospital_name" in df_clean.columns:
        for _, r in top_patients.iterrows():
            insights.append(
                f"🏥 **{r['hospital_name']}** handles **{int(r['monthly_patients'])} patients per month**, among the highest."
            )
    
    # 2. Equipment Shortage Score
    top_equipment = safe_top(df_clean, "equipment_shortage_score")
    if not top_equipment.empty and "hospital_name" in df_clean.columns:
        for _, r in top_equipment.iterrows():
            insights.append(
                f"⚠️ **{r['hospital_name']}** has a high equipment shortage score of **{r['equipment_shortage_score']}**."
            )
    
    # 3. Patients Per Staff
    top_load = safe_top(df_clean, "patients_per_staff")
    if not top_load.empty and "hospital_name" in df_clean.columns:
        for _, r in top_load.iterrows():
            insights.append(
                f"👥 **{r['hospital_name']}** has a heavy load with **{r['patients_per_staff']:.2f} patients per staff member**."
            )
    
    # Display insights
    if insights:
        df_ins = pd.DataFrame({"Insights": insights})
        st.dataframe(df_ins)
        download_df(df_ins, "automated_insights.csv")
    else:
        st.warning("No insights generated. Some required columns may be missing.")
