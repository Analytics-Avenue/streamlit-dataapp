import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from io import BytesIO

st.set_page_config(page_title="Real Estate Investment Opportunity Analyzer", layout="wide")

# --------------------------
# Hide default sidebar
# --------------------------
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
.card, .metric-card {background:#fff;border-radius:15px;padding:20px;margin-bottom:15px;
box-shadow:0 4px 20px rgba(0,0,0,0.08);transition: all 0.25s ease;text-align:left;}
.card:hover, .metric-card:hover {box-shadow:0 0 18px rgba(0,0,0,0.4); transform:scale(1.03);}
.metric-card {text-align:center;font-weight:600;}
.big-header {font-size:40px; font-weight:900; background: linear-gradient(90deg,#FF6B6B,#FFD93D);
-webkit-background-clip:text; -webkit-text-fill-color:transparent;}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Company Logo + Name
# --------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display:flex;align-items:center;">
<img src="{logo_url}" width="60" style="margin-right:10px;">
<div style="line-height:1;">
<div style="color:#064b86;font-size:36px;font-weight:bold;margin:0;">Analytics Avenue &</div>
<div style="color:#064b86;font-size:36px;font-weight:bold;margin:0;">Advanced Analytics</div>
</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='big-header'>Real Estate Investment Opportunity Analyzer</div>", unsafe_allow_html=True)

# --------------------------
# Tabs
# --------------------------
tab1, tab2 = st.tabs(["Overview","Application"])

# --------------------------
# Generic Overview Tab
# --------------------------
with tab1:
    st.markdown("### Overview")
    st.markdown("<div class='card'>Generic info about the app: Helps investors and developers identify high-return real estate opportunities using data-driven insights.</div>", unsafe_allow_html=True)

    st.markdown("### Capabilities")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("<div class='card'><b>Technical</b><br>• Conversion-adjusted ROI<br>• Interactive maps<br>• City & property segmentation<br>• Agent performance dashboards</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><b>Business</b><br>• Investment prioritization<br>• Market opportunity mapping<br>• Portfolio optimization<br>• Transparent scoring</div>", unsafe_allow_html=True)

    st.markdown("### Business Impact")
    st.markdown("<div class='card'>Provides actionable insights for strategic investments, maximizes ROI, identifies hotspots, and reduces decision risk.</div>", unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>High ROI Properties</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Top Performing Agents</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Hotspot Cities</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Average Conversion Rate</div>", unsafe_allow_html=True)

    st.markdown("### Use of App")
    st.markdown("<div class='card'>Monitor agent & property performance, evaluate lead conversion efficiency, segment markets, forecast revenue streams.</div>", unsafe_allow_html=True)

    st.markdown("### Who Should Use")
    st.markdown("<div class='card'>Real estate agents & managers, investors & portfolio managers, market analysts, property marketing teams.</div>", unsafe_allow_html=True)

# --------------------------
# Application Tab
# --------------------------
with tab2:
    st.markdown("### Step 1: Load Dataset")
    df = None
    mode = st.radio("Select Option:", ["Default Dataset","Upload CSV","Upload CSV + Column Mapping"], horizontal=True)

    # Default dataset
    if mode=="Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        try:
            df = pd.read_csv(URL)
            st.success("Default dataset loaded successfully.")
            st.dataframe(df.head(), use_container_width=True)
        except: st.error("Could not load default dataset.")

    # Upload CSV
    if mode=="Upload CSV":
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file: df = pd.read_csv(file)

    # Upload + Column Mapping
    REQUIRED_COLS = ["City","Property_Type","Price","Area_sqft","Agent_Name","Conversion_Probability","Latitude","Longitude"]
    if mode=="Upload CSV + Column Mapping":
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            raw=pd.read_csv(file)
            st.write(raw.head())
            mapping={}
            for col in REQUIRED_COLS:
                mapping[col]=st.selectbox(f"Map your column to: {col}", options=["-- Select --"]+list(raw.columns))
            if st.button("Apply Mapping"):
                missing=[c for c,v in mapping.items() if v=="-- Select --"]
                if missing: st.error(f"Map all columns: {missing}")
                else:
                    df=raw.rename(columns=mapping)
                    st.success("Mapping applied")
                    st.dataframe(df.head())

    if df is None: st.stop()
    df = df.dropna()

    # --------------------------
    # Filters
    # --------------------------
    f1,f2,f3 = st.columns(3)
    city = f1.multiselect("City", df["City"].unique())
    ptype = f2.multiselect("Property Type", df["Property_Type"].unique())
    agent = f3.multiselect("Agent Name", df["Agent_Name"].unique())
    filt = df.copy()
    if city: filt = filt[filt["City"].isin(city)]
    if ptype: filt = filt[filt["Property_Type"].isin(ptype)]
    if agent: filt = filt[filt["Agent_Name"].isin(agent)]
    st.dataframe(filt.head(), use_container_width=True)

    # --------------------------
    # ML Revenue Predictions
    # --------------------------
    st.markdown("### ML Revenue Predictions (Actual vs Predicted + Features)")
    X = filt[["City","Property_Type","Area_sqft","Conversion_Probability"]]
    y = filt["Price"]

    transformer = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["City","Property_Type"]),
        ("num","passthrough",["Area_sqft","Conversion_Probability"])
    ])
    X_trans = transformer.fit_transform(X)
    X_train,X_test,y_train,y_test = train_test_split(X_trans,y,test_size=0.2,random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train,y_train)

    y_pred = model.predict(X_trans)
    results = X.copy()
    results["Actual_Revenue"] = y
    results["Predicted_Revenue"] = y_pred
    st.dataframe(results,use_container_width=True)
    buf = BytesIO()
    results.to_csv(buf,index=False)
    st.download_button("Download ML Revenue Predictions", buf.getvalue(), "ml_revenue_predictions.csv", "text/csv")

    # --------------------------
    # Automated Insights
    # --------------------------
    st.markdown("### Automated Insights")
    insights = filt.groupby(["City","Property_Type"]).agg({"Price":["mean","max","min"]}).reset_index()
    insights.columns = ["City","Property_Type","Avg_Revenue","Max_Revenue","Min_Revenue"]
    st.dataframe(insights,use_container_width=True)
    buf2 = BytesIO()
    insights.to_csv(buf2,index=False)
    st.download_button("Download Automated Insights", buf2.getvalue(), "automated_insights.csv", "text/csv")
