import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import math
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="HospitalOps — Full Dashboard", layout="wide")
st.title("🏥 HospitalOps — Full Dashboard")

# -------------------------
# Logo + Company Name
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display:flex;align-items:center;">
<img src="{logo_url}" width="60" style="margin-right:10px;">
<div style="line-height:1;">
<div style="color:#064b86;font-size:36px;font-weight:bold;margin:0;padding:0;">Analytics Avenue &</div>
<div style="color:#064b86;font-size:36px;font-weight:bold;margin:0;padding:0;">Advanced Analytics</div>
</div></div>
""", unsafe_allow_html=True)

# -------------------------
# CSS (hover glow, left-align text)
# -------------------------
st.markdown("""
<style>
*{color:#000 !important;}
.glow-card{background:#fff;padding:16px;border-radius:12px;box-shadow:0 10px 30px rgba(0,120,255,0.12),0 0 12px rgba(0,120,255,0.06) inset;border:1px solid rgba(0,120,255,0.18);margin-bottom:12px;transition:0.3s;}
.glow-card:hover{box-shadow:0 0 25px #0078ff;transform:translateY(-3px);}
.glow-card p,.glow-card h2,h3{text-align:left !important;}
.kpi-row{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:16px;}
.kpi{background:#fff;padding:12px 16px;border-radius:10px;box-shadow:0 6px 18px rgba(0,0,0,0.04);border:1px solid rgba(0,0,0,0.04);width:220px;transition:0.3s;}
.kpi:hover{box-shadow:0 0 20px #0078ff;transform:translateY(-2px);}
.stTabs [role="tab"]{font-size:16px;padding:8px 12px;}
[data-testid="stSidebarNav"]{display:none;}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Utilities
# -------------------------
DEFAULT_DATA_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/healthcare/healthcare_syn.csv"

def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = df.columns.str.strip()
    return df

def download_df(df: pd.DataFrame, filename: str="export.csv"):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button("Download CSV", b, file_name=filename, mime="text/csv")

def safe_onehot_kwargs():
    return {"handle_unknown":"ignore","sparse_output":False}

# -------------------------
# Session placeholders
# -------------------------
for key in ["hospital_master","equipment_master","patient_risk","reg_pipe","automated_insights"]:
    if key not in st.session_state:
        st.session_state[key] = None

# -------------------------
# Tabs
# -------------------------
tab_overview, tab_app = st.tabs(["Overview","Application"])

# -------------------------
# Overview
# -------------------------
with tab_overview:
    st.markdown('<div class="glow-card"><h2>About this Application</h2><p>HospitalOps gives full operational visibility: patient trends, equipment, risk scoring, dashboards and decision support.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="glow-card"><h3>Purpose</h3><p>Improve allocation of beds and equipment, identify high-risk patients, enable rapid operational decisions.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="glow-card"><h3>Capabilities</h3><p>Dataset ingestion (default/upload/mapping), KPIs, clustering, ML predictions (regression), model explainability, automated insights.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="glow-card"><h3>Business Impact</h3><p>Faster triage, targeted procurement, reduced bed shortages, data-driven expansion and planning.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="glow-card"><h3>Intended Users</h3><p>Hospital managers, operational analysts, administrators, finance and procurement teams.</p></div>',unsafe_allow_html=True)

    # KPIs
    st.markdown('<div class="kpi-row">',unsafe_allow_html=True)
    kpis=[("Hospitals","120"),("High-Risk Hospitals","18"),("Avg Bed Occupancy","72%"),("Ventilators","1240"),("Avg Staff/Hospital","68")]
    for title,value in kpis:
        st.markdown(f'<div class="kpi"><div style="font-size:12px;color:#333">{title}</div><div style="font-size:20px;font-weight:700">{value}</div></div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

# -------------------------
# Application Tab
# -------------------------
with tab_app:
    st.header("Application")
    tab_ds, tab_charts, tab_ml, tab_insights = st.tabs(["Dataset Setup","Charts","ML Revenue Prediction","Automated Insights"])

    # -------------------------
    # Dataset Setup
    # -------------------------
    with tab_ds:
        st.subheader("Dataset Setup")
        mode = st.radio("Dataset mode", ["Default (URL)","Upload CSV","Upload + Mapping"],horizontal=True)

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
                canonical=["Hospital_Name","Location","Hospital_Type","Monthly_Patients","Staff_Count","Beds_Count","Ventilators_Count","Revenue"]
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

    # -------------------------
    # Charts
    # -------------------------
    with tab_charts:
        st.subheader("Charts")
        df = st.session_state.get("hospital_master")
        if df is None: st.info("Load dataset first."); st.stop()
        num_cols=df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols=df.select_dtypes(exclude=[np.number]).columns.tolist()
        st.write("Preview",df.head())

        if num_cols:
            xcol=st.selectbox("Histogram column",num_cols)
            fig=px.histogram(df,x=xcol,nbins=30,title=f"Distribution of {xcol}")
            st.plotly_chart(fig,use_container_width=True)
        if len(num_cols)>=2:
            x=st.selectbox("X axis",num_cols,key="scatter_x")
            y=st.selectbox("Y axis",num_cols,key="scatter_y")
            fig=px.scatter(df,x=x,y=y,color=cat_cols[0] if cat_cols else None)
            st.plotly_chart(fig,use_container_width=True)

    # -------------------------
    # ML Revenue Prediction
    # -------------------------
    with tab_ml:
        st.subheader("ML Revenue Prediction (Regression)")
        df = st.session_state.get("hospital_master")
        if df is None or "Revenue" not in df.columns: st.info("Ensure 'Revenue' numeric column exists."); st.stop()
        features=[c for c in df.columns if c not in ["Revenue","Hospital_Name"]]
        X=df[features]
        y=df["Revenue"]
        test_size=st.slider("Test size",0.1,0.4,0.2)
        if st.button("Train Revenue Regressor"):
            ct=[]
            numc=X.select_dtypes(include=[np.number]).columns.tolist()
            catc=X.select_dtypes(exclude=[np.number]).columns.tolist()
            if numc: ct.append(("num",StandardScaler(),numc))
            if catc: ct.append(("cat",OneHotEncoder(**safe_onehot_kwargs()),catc))
            preproc=ColumnTransformer(transformers=ct,remainder="drop")
            pipe=Pipeline([("pre",preproc),("model",RandomForestRegressor(n_estimators=150,random_state=42))])
            X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=test_size,random_state=42)
            pipe.fit(X_tr,y_tr)
            st.session_state["reg_pipe"]=pipe
            y_pred=pipe.predict(X_te)
            st.success(f"Regression trained — RMSE: {math.sqrt(np.mean((y_te-y_pred)**2)):.2f} | R2: {np.corrcoef(y_te,y_pred)[0,1]**2:.2f}")
            df_result=X_te.copy()
            df_result["Actual_Revenue"]=y_te.values
            df_result["Predicted_Revenue"]=y_pred
            st.dataframe(df_result)
            download_df(df_result,"revenue_prediction.csv")

    # -------------------------
    # Automated Insights
    # -------------------------
    with tab_insights:
        st.subheader("Automated Insights")
        df = st.session_state.get("hospital_master")
        if df is None: st.info("Load dataset first."); st.stop()
        insights=[]
        if "Revenue" in df.columns:
            top5=df.nlargest(5,"Revenue")[["Hospital_Name","Revenue"]]
            for _,r in top5.iterrows():
                insights.append(f"Top hospital by revenue: {r['Hospital_Name']} → {r['Revenue']:.2f}")
        st.dataframe(pd.DataFrame({"Insights":insights}))
        download_df(pd.DataFrame({"Insights":insights}),"automated_insights.csv")
