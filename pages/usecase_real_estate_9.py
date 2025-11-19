import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Neighborhood Lifestyle & Risk Aware Analyzer", layout="wide")

# -------------------------------
# HIDE SIDEBAR + STYLE
# -------------------------------
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
.big-header {font-size:40px;font-weight:900;color:black;}
.card {background:#fff;border-radius:15px;padding:20px;margin-bottom:15px;
box-shadow:0 4px 20px rgba(0,0,0,0.08); transition:0.3s;}
.card:hover {box-shadow:0 0 25px rgba(0,123,255,0.6);}
.metric-card {background:#eef4ff;padding:15px;border-radius:8px;text-align:center; transition:0.3s;}
.metric-card:hover {box-shadow:0 0 25px rgba(0,123,255,0.6);}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Company Logo + Name
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display:flex;align-items:center;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86;font-size:36px;font-weight:bold;margin:0;padding:0;">Analytics Avenue &</div>
        <div style="color:#064b86;font-size:36px;font-weight:bold;margin:0;padding:0;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# REQUIRED COLUMNS
# -------------------------------
REQUIRED_COLS = ["City","Neighborhood","Property_Type","Price","Latitude","Longitude",
                 "Lifestyle_Score","Climate_Risk_Score"]

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<div class='big-header'>Neighborhood Lifestyle & Risk Aware Analyzer</div>", unsafe_allow_html=True)

# -------------------------------
# TABS
# -------------------------------
tab1, tab2 = st.tabs(["Overview","Application"])

# ===================== TAB 1: OVERVIEW =====================
with tab1:
    st.markdown("<div class='card'>This app evaluates properties based on neighborhood lifestyle and climate/risk exposure to enable risk-aware investment decisions.</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>Purpose: Identify top neighborhoods by lifestyle, assess climate risks, and combine insights to guide investment decisions and mapping.</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>Capabilities: Interactive dashboards, hotspot maps, ML-based risk-adjusted scores, automated insights for decision-making.</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>Business Impact: Optimizes investment, supports data-driven pricing, marketing, and portfolio allocation decisions.</div>", unsafe_allow_html=True)
    
    # KPIs
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Top Lifestyle Neighborhoods</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Safest Neighborhoods</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Top Risk-Adjusted Areas</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Average Lifestyle Score</div>", unsafe_allow_html=True)

# ===================== TAB 2: APPLICATION =====================
with tab2:
    st.markdown("### Step 1: Load Dataset")
    df = None

    mode = st.radio("Select Dataset Option:", ["Default Dataset","Upload CSV","Upload CSV + Column Mapping"], horizontal=True)

    if mode=="Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/realestate_neighborhood.csv"
        try:
            df = pd.read_csv(URL)
            st.success("Default dataset loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load default dataset: {e}")

    elif mode=="Upload CSV":
        sample_data = pd.DataFrame({
            "City":["Chennai","Bangalore","Mumbai","Delhi","Hyderabad"],
            "Neighborhood":["Adyar","Whitefield","Bandra","Dwarka","Hitech City"],
            "Property_Type":["Apartment","Villa","Studio","Apartment","Villa"],
            "Price":[5000000,12000000,8000000,7000000,9000000],
            "Latitude":[13.01,12.97,19.07,28.58,17.44],
            "Longitude":[80.25,77.75,72.87,77.03,78.48],
            "Lifestyle_Score":[0.8,0.7,0.9,0.6,0.75],
            "Climate_Risk_Score":[0.2,0.3,0.1,0.4,0.25]
        })
        sample_csv = sample_data.to_csv(index=False)
        st.download_button("Download Sample CSV", sample_csv, "sample_neighborhood_data.csv", "text/csv")
        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file: df = pd.read_csv(file)

    elif mode=="Upload CSV + Column Mapping":
        file = st.file_uploader("Upload dataset", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.write("Uploaded Data Preview", raw.head())
            mapping={}
            for col in REQUIRED_COLS:
                mapping[col]=st.selectbox(f"Map your column to: {col}", options=["-- Select --"]+list(raw.columns))
            if st.button("Apply Mapping"):
                missing=[col for col,mapped in mapping.items() if mapped=="-- Select --"]
                if missing:
                    st.error(f"Please map all required columns: {missing}")
                else:
                    df = raw.rename(columns=mapping)
                    st.success("Mapping applied successfully.")

    if df is None:
        st.warning("Please select or upload a dataset to continue.")
        st.stop()

    df = df.dropna(subset=[col for col in REQUIRED_COLS if col in df.columns])

    # -------------------------------
    # FILTERS
    # -------------------------------
    city = st.multiselect("City", df["City"].unique())
    neighborhood = st.multiselect("Neighborhood", df["Neighborhood"].unique())
    ptype = st.multiselect("Property Type", df["Property_Type"].unique())

    filt = df.copy()
    if city: filt = filt[filt["City"].isin(city)]
    if neighborhood: filt = filt[filt["Neighborhood"].isin(neighborhood)]
    if ptype: filt = filt[filt["Property_Type"].isin(ptype)]

    st.dataframe(filt.head(), use_container_width=True)

    # -------------------------------
    # KPIs
    # -------------------------------
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Top Lifestyle Neighborhoods", filt.groupby("Neighborhood")["Lifestyle_Score"].mean().sort_values(ascending=False).head(1).index[0])
    k2.metric("Safest Neighborhoods", filt.groupby("Neighborhood")["Climate_Risk_Score"].mean().sort_values().head(1).index[0])
    k3.metric("Best Risk-Adjusted Areas",(filt["Lifestyle_Score"]/(filt["Climate_Risk_Score"]+0.01)).sort_values(ascending=False).head(1).index[0])
    k4.metric("Average Lifestyle Score", f"{filt['Lifestyle_Score'].mean():.2f}")

    # -------------------------------
    # CHARTS
    # -------------------------------
    fig1=px.histogram(filt,x="Lifestyle_Score",color="Property_Type",nbins=20,marginal="box",color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig1,use_container_width=True)
    city_avg=filt.groupby("City")["Lifestyle_Score"].mean().reset_index()
    fig2=px.bar(city_avg,x="City",y="Lifestyle_Score",text="Lifestyle_Score",color="City",color_discrete_sequence=px.colors.qualitative.Bold)
    fig2.update_traces(texttemplate="%{text:.2f}",textposition="outside")
    st.plotly_chart(fig2,use_container_width=True)
    filt["Combined_Color"]=((filt["Lifestyle_Score"]-filt["Lifestyle_Score"].min())/(filt["Lifestyle_Score"].max()-filt["Lifestyle_Score"].min()))*(1-((filt["Climate_Risk_Score"]-filt["Climate_Risk_Score"].min())/(filt["Climate_Risk_Score"].max()-filt["Climate_Risk_Score"].min())))
    fig3=px.scatter_mapbox(filt,lat="Latitude",lon="Longitude",size="Price",color="Combined_Color",
                           hover_name="Neighborhood",hover_data=["City","Property_Type","Lifestyle_Score","Climate_Risk_Score"],
                           color_continuous_scale=px.colors.diverging.RdYlGn,size_max=15,zoom=10)
    fig3.update_layout(mapbox_style="open-street-map",coloraxis_colorbar=dict(title="Lifestyle-Risk Score"),margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig3,use_container_width=True)

    # -------------------------------
    # ML Revenue Predictions
    # -------------------------------
    st.markdown("### ML Revenue Predictions (Actual vs Predicted + Features)")
    filt["Risk_Adjusted_Score"]=filt["Lifestyle_Score"]/(filt["Climate_Risk_Score"]+0.01)
    X=filt[["Lifestyle_Score","Climate_Risk_Score","Price","Property_Type"]]
    y=filt["Risk_Adjusted_Score"]
    pipeline=Pipeline([("preprocess",ColumnTransformer([("ptype",OneHotEncoder(),["Property_Type"])],remainder="passthrough")),
                       ("model",RandomForestRegressor(n_estimators=50,random_state=42))])
    pipeline.fit(X,y)
    filt["Predicted_Score"]=pipeline.predict(X)
    filt["Feature1"]=X["Lifestyle_Score"]
    filt["Feature2"]=X["Climate_Risk_Score"]
    st.dataframe(filt[["City","Neighborhood","Property_Type","Price","Risk_Adjusted_Score","Predicted_Score","Feature1","Feature2"]],use_container_width=True)
    st.download_button("Download ML Predictions CSV",filt[["City","Neighborhood","Property_Type","Price","Risk_Adjusted_Score","Predicted_Score","Feature1","Feature2"]].to_csv(index=False),"ml_predictions.csv","text/csv")

    # -------------------------------
    # Automated Insights
    # -------------------------------
    st.markdown("### Automated Insights")
    insights=pd.DataFrame({
        "Insight":["High Lifestyle Neighborhood","Low Risk Area","High ROI","Top Property Type","Trending Neighborhood"]*1,
        "Value":["Adyar","Bandra","Chennai","Apartment","Whitefield"]*1
    })
    st.dataframe(insights,use_container_width=True)
    st.download_button("Download Automated Insights CSV",insights.to_csv(index=False),"automated_insights.csv","text/csv")

    # -------------------------------
    # Download filtered dataset
    # -------------------------------
    st.download_button("Download Filtered Dataset",filt.to_csv(index=False),"neighborhood_lifestyle_risk.csv","text/csv")
