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

# -------------------------
# Page Config & Logo
# -------------------------
st.set_page_config(page_title="HospitalOps — Full Dashboard", layout="wide")
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display:flex;align-items:center;">
<img src="{logo_url}" width="60" style="margin-right:10px;">
<div style="line-height:1;">
<div style="color:#064b86;font-size:36px;font-weight:bold;margin:0;padding:0;">Analytics Avenue &</div>
<div style="color:#064b86;font-size:36px;font-weight:bold;margin:0;padding:0;">Advanced Analytics</div>
</div></div>
""", unsafe_allow_html=True)
st.title("HospitalOps")

# -------------------------
# -------------------------
# CSS for cards and KPIs
# -------------------------
st.markdown("""
<style>
*{color:#000 !important;}
.glow-card{background:#fff;padding:16px;border-radius:12px;box-shadow:0 10px 30px rgba(0,120,255,0.12),0 0 12px rgba(0,120,255,0.06) inset;border:1px solid rgba(0,120,255,0.18);margin-bottom:12px;transition:0.3s;}
.glow-card:hover{box-shadow:0 0 25px #0078ff;transform:translateY(-3px);}
.glow-card p,.glow-card h2,h3{text-align:left !important;}
.kpi-row{display:flex;gap:14px;flex-wrap:nowrap;margin-bottom:16px;}
.kpi{background:#fff;padding:12px 16px;border-radius:10px;box-shadow:0 6px 18px rgba(0,0,0,0.04);border:1px solid rgba(0,0,0,0.04);width:220px;transition:0.3s;text-align:center;}
.kpi:hover{box-shadow:0 0 20px #0078ff;transform:translateY(-2px);}
[data-testid="stSidebarNav"]{display:none;}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Utilities
# -------------------------
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
    return {"handle_unknown":"ignore","sparse_output":False}

# -------------------------
# Session placeholders
# -------------------------
for key in ["hospital_master","reg_pipe"]:
    if key not in st.session_state:
        st.session_state[key] = None

# -------------------------
# -------------------------
# About Application
# -------------------------
st.markdown('<div class="glow-card"><h2>About this Application</h2>'
            '<p>HospitalOps provides full operational visibility into hospital operations: patient trends, equipment management, risk scoring, and dashboards to support smarter operational decisions.</p>'
            '</div>', unsafe_allow_html=True)

st.markdown('<div class="glow-card"><h3>Purpose</h3>'
            '<p>Enable hospitals to optimize bed allocation, track equipment usage, identify high-risk areas, and facilitate rapid, data-driven decisions for operational and administrative teams.</p>'
            '</div>', unsafe_allow_html=True)

# -------------------------
# Capabilities & Business Impact Side by Side
# -------------------------
cols = st.columns(2)
with cols[0]:
    st.markdown('<div class="glow-card"><h3>Capabilities</h3>'
                '<ul style="padding-left:16px;">'
                '<li>Dataset ingestion (default, upload, mapping)</li>'
                '<li>Key Performance Indicators (KPIs) tracking</li>'
                '<li>Clustering & segmentation analysis</li>'
                '<li>ML predictions (e.g., Equipment Shortage, Patients Per Staff)</li>'
                '<li>Model explainability & interpretability</li>'
                '<li>Automated insights generation</li>'
                '</ul></div>', unsafe_allow_html=True)
with cols[1]:
    st.markdown('<div class="glow-card"><h3>Business Impact</h3>'
                '<ul style="padding-left:16px;">'
                '<li>Faster triage and patient management</li>'
                '<li>Targeted equipment procurement</li>'
                '<li>Reduced bed shortages & improved allocation</li>'
                '<li>Data-driven expansion and strategic planning</li>'
                '<li>Enhanced operational efficiency across departments</li>'
                '</ul></div>', unsafe_allow_html=True)

# -------------------------
# Intended Users
# -------------------------
st.markdown('<div class="glow-card"><h3>Intended Users</h3>'
            '<p>Hospital managers, operational analysts, administrative staff, and finance/procurement teams seeking actionable insights to streamline operations.</p>'
            '</div>', unsafe_allow_html=True)

# -------------------------
# KPIs Row (5 in single line)
# -------------------------
st.markdown('<div class="kpi-row" style="flex-wrap:nowrap;">', unsafe_allow_html=True)
kpi_titles = ["High Patient Load", "Avg Beds Occupancy", "Equipment Shortage Score", "Patients per Staff", "Facility Deficit Score"]
for title in kpi_titles:
    st.markdown(f'''
        <div class="kpi">
            <div style="font-size:12px;color:#333">{title}</div>
            <div style="font-size:20px;font-weight:700">--</div>
        </div>
    ''', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)



# -------------------------
# Dataset Setup
# -------------------------
st.header("Application")
mode = st.radio("Dataset mode", ["Default (URL)","Upload CSV","Upload + Mapping"], horizontal=True)

if mode=="Default (URL)":
    try:
        df = pd.read_csv(DEFAULT_DATA_URL)
        df = remove_duplicate_columns(df)
        st.session_state["hospital_master"] = df
        st.success("Default dataset loaded.")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Failed to load: {e}")

elif mode=="Upload CSV":
    st.markdown("#### Download Sample CSV for Reference")
    try:
        sample_df = pd.read_csv(DEFAULT_DATA_URL).head(5)
        sample_csv = sample_df.to_csv(index=False)
        st.download_button("Download Sample CSV", sample_csv, "sample_dataset.csv", "text/csv")
    except Exception as e:
        st.info(f"Sample CSV unavailable: {e}")

    file = st.file_uploader("Upload your dataset", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.session_state["hospital_master"] = df
        st.success("Uploaded dataset loaded.")
        st.dataframe(df.head())

else:
    up = st.file_uploader("Upload CSV for mapping",type=["csv"])
    if up is not None:
        raw = pd.read_csv(up)
        raw = raw.loc[:,~raw.columns.duplicated()].copy()
        raw.columns=raw.columns.str.strip()
        canonical=["Hospital_Name","Location","Hospital_Type","Monthly_Patients","Staff_Count",
                   "Beds_Count","Ventilators_Count","Equipment Shortage Score","Facility Deficit Score","Overall Risk Score"]
        mapping={}
        for c in canonical:
            mapping[c] = st.selectbox(f"Map to {c}", ["-- None --"]+list(raw.columns))
        if st.button("Apply mapping"):
            mapped=pd.DataFrame()
            for c in canonical:
                sel=mapping[c]
                mapped[c]=raw[sel] if sel!="-- None --" else np.nan
            st.session_state["hospital_master"]=mapped
            st.success("Mapping applied.")
            st.dataframe(mapped.head())

df = st.session_state.get("hospital_master")
if df is None: st.stop()

# -------------------------
# Charts
# -------------------------
st.subheader("Charts & Visualizations")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
st.write("Preview of dataset:", df.head())

if num_cols:
    xcol = st.selectbox("Histogram column", num_cols)
    fig = px.histogram(df, x=xcol, nbins=30, title=f"Distribution of {xcol}")
    st.plotly_chart(fig, use_container_width=True)

if len(num_cols)>=2:
    x = st.selectbox("X axis", num_cols, key="scatter_x")
    y = st.selectbox("Y axis", num_cols, key="scatter_y")
    fig = px.scatter(df, x=x, y=y, color=cat_cols[0] if cat_cols else None)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# ML Prediction
# -------------------------
st.subheader("ML Prediction (Regression)")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
target_col = st.selectbox("Select target column for prediction", numeric_cols)

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
    pipe = Pipeline([("preproc", preproc), ("model", RandomForestRegressor(n_estimators=150, random_state=42))])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
    pipe.fit(X_tr, y_tr)
    st.session_state["reg_pipe"] = pipe
    y_pred = pipe.predict(X_te)
    rmse = math.sqrt(np.mean((y_te-y_pred)**2))
    r2 = np.corrcoef(y_te,y_pred)[0,1]**2
    st.success(f"ML model trained for predicting {target_col} — RMSE: {rmse:.2f} | R²: {r2:.2f}")
    df_result = X_te.copy()
    df_result[f"Actual_{target_col}"] = y_te.values
    df_result[f"Predicted_{target_col}"] = y_pred
    st.dataframe(df_result)
    download_df(df_result, f"{target_col}_prediction.csv")

    # -------------------------
    st.subheader("Automated Insights")
    insights = []
    
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(" ", "_")
    
    # 1️⃣ Top hospitals by monthly patients
    if "monthly_patients" in df_clean.columns:
        top_patients = df_clean.nlargest(3, "monthly_patients")[["hospital_name", "monthly_patients"]]
        for _, r in top_patients.iterrows():
            insights.append(f"High patient load: {r['hospital_name']} → {r['monthly_patients']} patients/month")
    
    # 2️⃣ Hospitals with high equipment shortage
    if "equipment_shortage_score" in df_clean.columns:
        top_shortage = df_clean.nlargest(3, "equipment_shortage_score")[["hospital_name", "equipment_shortage_score"]]
        for _, r in top_shortage.iterrows():
            insights.append(f"Equipment shortage alert: {r['hospital_name']} → Score {r['equipment_shortage_score']}")
    
    # 3️⃣ Hospitals with highest patients per staff
    if "patients_per_staff" in df_clean.columns:
        top_ratio = df_clean.nlargest(3, "patients_per_staff")[["hospital_name", "patients_per_staff"]]
        for _, r in top_ratio.iterrows():
            insights.append(f"High patients per staff: {r['hospital_name']} → {r['patients_per_staff']:.2f}")
    
    if insights:
        st.dataframe(pd.DataFrame({"Insights": insights}))
        download_df(pd.DataFrame({"Insights": insights}), "automated_insights.csv")
    else:
        st.info("No insights could be generated from the dataset.")
