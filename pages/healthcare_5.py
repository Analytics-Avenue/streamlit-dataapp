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
.glow-card p,.glow-card h2,h3{text-align:left !important;}

.kpi-row{
    display:flex;gap:14px;flex-wrap:nowrap;margin-bottom:16px;
}
.kpi{
    background:#fff;padding:12px 16px;border-radius:10px;
    box-shadow:0 6px 18px rgba(0,0,0,0.04);border:1px solid rgba(0,0,0,0.04);
    width:220px;transition:0.3s;text-align:center;
}
.kpi:hover{box-shadow:0 0 20px #0078ff;transform:translateY(-2px);}

[data-testid="stSidebarNav"]{display:none;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
DEFAULT_DATA_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/healthcare/healthcare_syn.csv"

def remove_duplicate_columns(df):
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = df.columns.str.strip()
    return df

def download_df(df, filename="export.csv"):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button("Download CSV", b, file_name=filename, mime="text/csv")

def safe_onehot_kwargs():
    return {"handle_unknown": "ignore", "sparse_output": False}

for key in ["hospital_master", "reg_pipe"]:
    st.session_state.setdefault(key, None)

# -------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------
tab1, tab2 = st.tabs(["Overview", "Application"])

# -------------------------------------------------------------------
# TAB 1: OVERVIEW
# -------------------------------------------------------------------
with tab1:

    st.markdown('<div class="glow-card"><h2>About this Application</h2>'
                '<p>HospitalOps provides full operational visibility into hospital operations, supporting patient flow, equipment planning, and strategic decisions.</p>'
                '</div>', unsafe_allow_html=True)

    st.markdown('<div class="glow-card"><h3>Purpose</h3>'
                '<p>Optimize beds, track equipment usage, and enable rapid, data-driven hospital decisions.</p>'
                '</div>', unsafe_allow_html=True)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("""
            <div class="glow-card"><h3>Capabilities</h3>
            <ul style="padding-left:16px;">
                <li>Dataset ingestion</li>
                <li>Hospital KPIs monitoring</li>
                <li>Segmentation & clustering</li>
                <li>ML predictions</li>
                <li>Model explainability</li>
                <li>Automated insights</li>
            </ul></div>
        """, unsafe_allow_html=True)

    with cols[1]:
        st.markdown("""
            <div class="glow-card"><h3>Business Impact</h3>
            <ul style="padding-left:16px;">
                <li>Improved bed allocation</li>
                <li>Targeted procurement</li>
                <li>Lower operational bottlenecks</li>
                <li>Pathway for hospital expansion</li>
                <li>Data-driven efficiency</li>
            </ul></div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="glow-card"><h3>Intended Users</h3>'
                '<p>Hospital administrators, operations teams, finance, and procurement departments.</p>'
                '</div>', unsafe_allow_html=True)
    # -------------------------
    # -------------------------
    # KPI SECTION WITH HOVER-GLOW STYLE
    # -------------------------
    
    import streamlit as st
    
    # KPI Card CSS
    st.markdown("""
    <style>
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
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### KPIs")
    
    # Create 5 KPI columns
    c1, c2, c3, c4, c5 = st.columns(5)
    
    # Static placeholders – swap with your real computed values as needed
    c1.markdown("<div class='metric-card'>High Patient Load</div>", unsafe_allow_html=True)
    c2.markdown("<div class='metric-card'>Avg Beds Availability</div>", unsafe_allow_html=True)
    c3.markdown("<div class='metric-card'>Equipment Shortage Score</div>", unsafe_allow_html=True)
    c4.markdown("<div class='metric-card'>Patients per Staff</div>", unsafe_allow_html=True)
    c5.markdown("<div class='metric-card'>Facility Deficit Score</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------
# TAB 2: APPLICATION
# -------------------------------------------------------------------
with tab2:

    st.header("Application")

    mode = st.radio("Dataset mode", ["Default (URL)", "Upload CSV", "Upload + Mapping"], horizontal=True)

    # ------------------------------------------------------
    # Dataset
    # ------------------------------------------------------
    if mode == "Default (URL)":
        try:
            df = pd.read_csv(DEFAULT_DATA_URL)
            df = remove_duplicate_columns(df)
            st.session_state["hospital_master"] = df
            st.success("Default dataset loaded.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Failed to load: {e}")

    elif mode == "Upload CSV":
        st.markdown("#### Download Sample CSV for Reference")
        try:
            sample_df = pd.read_csv(DEFAULT_DATA_URL).head(5)
            st.download_button(
                "Download Sample CSV",
                sample_df.to_csv(index=False),
                "sample_dataset.csv",
                "text/csv"
            )
        except:
            st.info("Sample CSV unavailable.")

        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)
            st.session_state["hospital_master"] = df
            st.success("Uploaded dataset loaded.")
            st.dataframe(df.head())

    else:
        up = st.file_uploader("Upload CSV for mapping", type=["csv"])
        if up:
            raw = pd.read_csv(up)
            raw.columns = raw.columns.str.strip()
            raw = raw.loc[:, ~raw.columns.duplicated()].copy()

            canonical = [
                "Hospital_Name", "Location", "Hospital_Type", "Monthly_Patients",
                "Staff_Count", "Beds_Count", "Ventilators_Count",
                "Equipment Shortage Score", "Facility Deficit Score", "Overall Risk Score"
            ]

            mapping = {}
            for c in canonical:
                mapping[c] = st.selectbox(f"Map to {c}", ["-- None --"] + list(raw.columns))

            if st.button("Apply mapping"):
                mapped = pd.DataFrame()
                for c in canonical:
                    sel = mapping[c]
                    mapped[c] = raw[sel] if sel != "-- None --" else np.nan

                st.session_state["hospital_master"] = mapped
                st.success("Mapping applied.")
                st.dataframe(mapped.head())

    df = st.session_state.get("hospital_master")
    if df is None:
        st.stop()

    # ------------------------------------------------------
    # Charts
    # ------------------------------------------------------
    st.subheader("Charts & Visualizations")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    st.write("Preview:", df.head())

    if num_cols:
        xcol = st.selectbox("Histogram column", num_cols)
        st.plotly_chart(px.histogram(df, x=xcol, nbins=30), use_container_width=True)

    if len(num_cols) >= 2:
        x = st.selectbox("X axis", num_cols, key="sx")
        y = st.selectbox("Y axis", num_cols, key="sy")
        st.plotly_chart(px.scatter(df, x=x, y=y, color=cat_cols[0] if cat_cols else None),
                        use_container_width=True)

    # ------------------------------------------------------
    # ML Prediction
    # ------------------------------------------------------
    st.subheader("ML Prediction (Regression)")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col = st.selectbox("Select target column", numeric_cols)

    features = [c for c in numeric_cols if c != target_col]
    X = df[features]
    y = df[target_col]

    test_size = st.slider("Test size", 0.1, 0.4, 0.2)

    if st.button("Train ML Regressor"):
        numc = X.select_dtypes(include=[np.number]).columns.tolist()
        catc = X.select_dtypes(exclude=[np.number]).columns.tolist()

        transformers = []
        if numc: transformers.append(("num", StandardScaler(), numc))
        if catc: transformers.append(("cat", OneHotEncoder(**safe_onehot_kwargs()), catc))

        preproc = ColumnTransformer(transformers=transformers, remainder="drop")

        pipe = Pipeline([
            ("preproc", preproc),
            ("model", RandomForestRegressor(n_estimators=150, random_state=42))
        ])

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
        pipe.fit(Xtr, ytr)

        st.session_state["reg_pipe"] = pipe

        y_pred = pipe.predict(Xte)
        rmse = math.sqrt(np.mean((yte - y_pred) ** 2))
        r2 = np.corrcoef(yte, y_pred)[0, 1] ** 2

        st.success(f"Trained ML model for {target_col} — RMSE: {rmse:.2f} | R²: {r2:.2f}")

        out = Xte.copy()
        out[f"Actual_{target_col}"] = yte.values
        out[f"Predicted_{target_col}"] = y_pred

        st.dataframe(out)
        download_df(out, f"{target_col}_prediction.csv")

        # ------------------------------------------------------
        # Automated Insights
        # ------------------------------------------------------
        st.subheader("Automated Insights")

        insights = []
        df_clean = df.copy()
        df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(" ", "_")

        if "monthly_patients" in df_clean.columns:
            top_p = df_clean.nlargest(3, "monthly_patients")[["hospital_name", "monthly_patients"]]
            for _, r in top_p.iterrows():
                insights.append(f"{r['hospital_name']} handles {r['monthly_patients']} patients/month.")

        if "equipment_shortage_score" in df_clean.columns:
            top_e = df_clean.nlargest(3, "equipment_shortage_score")[["hospital_name", "equipment_shortage_score"]]
            for _, r in top_e.iterrows():
                insights.append(f"{r['hospital_name']} shows high equipment shortage score: {r['equipment_shortage_score']}")

        if "patients_per_staff" in df_clean.columns:
            top_r = df_clean.nlargest(3, "patients_per_staff")[["hospital_name", "patients_per_staff"]]
            for _, r in top_r.iterrows():
                insights.append(f"{r['hospital_name']} has high patient load per staff: {r['patients_per_staff']:.2f}")

        if insights:
            df_ins = pd.DataFrame({"Insights": insights})
            st.dataframe(df_ins)
            download_df(df_ins, "automated_insights.csv")
        else:
            st.info("No insights could be generated from the dataset.")
