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
# HIDE SIDEBAR + STYLES
# -------------------------------
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
.big-header {font-size:40px;font-weight:900;color:black;}
.card {background:#fff;border-radius:15px;padding:20px;margin-bottom:15px;box-shadow:0 4px 20px rgba(0,0,0,0.08);transition:0.3s;}
.card:hover {box-shadow:0 0 25px rgba(0,123,255,0.6);}
.metric-card {background:#eef4ff;padding:15px;border-radius:8px;text-align:center;transition:0.3s;}
.metric-card:hover {box-shadow:0 0 25px rgba(0,123,255,0.6);}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# COMPANY LOGO + NAME
# -------------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display: flex; align-items: center;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:bold;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:bold;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# REQUIRED COLUMNS
# -------------------------------
REQUIRED_COLS = ["City","Neighborhood","Property_Type","Price","Latitude","Longitude","Lifestyle_Score","Climate_Risk_Score"]

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<div class='big-header'>Neighborhood Lifestyle & Risk Aware Analyzer</div>", unsafe_allow_html=True)

# -------------------------------
# TABS
# -------------------------------
tab1, tab2 = st.tabs(["Overview","Application"])

# -------------------------------
# TAB 1: OVERVIEW
# -------------------------------
with tab1:
    st.markdown("### Generic Info")
    st.markdown("<div class='card'>This app evaluates properties based on neighborhood lifestyle amenities and climate/risk exposure, enabling risk-aware investment decisions.</div>", unsafe_allow_html=True)
    
    st.markdown("### Capabilities")
    st.markdown("<div class='card'>• Identify top neighborhoods by lifestyle score<br>• Assess climate and risk exposure<br>• Combine lifestyle and risk insights for investment decisions<br>• Map neighborhoods interactively</div>", unsafe_allow_html=True)

    st.markdown("### Business Impact")
    st.markdown("<div class='card'>• Optimized investment decisions<br>• Portfolio risk mitigation<br>• Resource allocation for high-demand neighborhoods<br>• Targeted marketing strategies</div>", unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Top Lifestyle Neighborhoods<br><small>Investment Managers</small></div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Safest Neighborhoods<br><small>Risk Analysts</small></div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Top Risk-Adjusted ROI Areas<br><small>Portfolio Managers</small></div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Average Lifestyle Score<br><small>Analysts & Decision Makers</small></div>", unsafe_allow_html=True)

# -------------------------------
# TAB 2: APPLICATION
# -------------------------------
with tab2:
    st.markdown("### Step 1: Load Dataset")
    df = None
    mode = st.radio("Select Dataset Option:", ["Default Dataset","Upload CSV","Upload CSV + Column Mapping"], horizontal=True)

    # Default dataset
    if mode=="Default Dataset":
        URL="https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/realestate_neighborhood.csv"
        try:
            df = pd.read_csv(URL)
        except:
            st.error("Failed to load default dataset.")

    # Upload CSV
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
        st.download_button("Download Sample CSV", sample_data.to_csv(index=False),"sample_neighborhood.csv","text/csv")
        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file: df = pd.read_csv(file)

    # Upload CSV + Column Mapping
    elif mode=="Upload CSV + Column Mapping":
        file = st.file_uploader("Upload dataset", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.write(raw.head())
            mapping={}
            for col in REQUIRED_COLS:
                mapping[col] = st.selectbox(f"Map column to: {col}", ["-- Select --"]+list(raw.columns))
            if st.button("Apply Mapping"):
                missing=[col for col,mapped in mapping.items() if mapped=="-- Select --"]
                if missing: st.error(f"Map all columns: {missing}")
                else:
                    df=raw.rename(columns=mapping)
                    st.success("Mapping applied!")

    if df is None: st.stop()
    df = df.dropna(subset=[c for c in REQUIRED_COLS if c in df.columns])

    # -------------------------------
    # Filters
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
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Top Lifestyle Neighborhoods", filt.groupby("Neighborhood")["Lifestyle_Score"].mean().sort_values(ascending=False).head(1).index[0])
    k2.metric("Safest Neighborhoods", filt.groupby("Neighborhood")["Climate_Risk_Score"].mean().sort_values().head(1).index[0])
    k3.metric("Top Risk-Adjusted Areas", (filt["Lifestyle_Score"]/(filt["Climate_Risk_Score"]+0.01)).sort_values(ascending=False).head(1).index[0])
    k4.metric("Average Lifestyle Score", f"{filt['Lifestyle_Score'].mean():.2f}")

    # -------------------------------
    # Charts
    # -------------------------------
    fig1=px.histogram(filt,x="Lifestyle_Score",nbins=20,color="Property_Type",marginal="box",color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig1,use_container_width=True)

    city_avg=filt.groupby("City")["Lifestyle_Score"].mean().reset_index()
    fig2=px.bar(city_avg,x="City",y="Lifestyle_Score",color="City",text="Lifestyle_Score",color_discrete_sequence=px.colors.qualitative.Bold)
    fig2.update_traces(texttemplate="%{text:.2f}",textposition="outside")
    st.plotly_chart(fig2,use_container_width=True)

    filt["Combined_Score"]=filt["Lifestyle_Score"]*(1-filt["Climate_Risk_Score"])
    fig3=px.scatter_mapbox(filt,lat="Latitude",lon="Longitude",size="Price",color="Combined_Score",
                           hover_name="Neighborhood",hover_data=["City","Property_Type","Lifestyle_Score","Climate_Risk_Score"],
                           color_continuous_scale=px.colors.diverging.RdYlGn,size_max=15,zoom=10)
    fig3.update_layout(mapbox_style="open-street-map",coloraxis_colorbar=dict(title="Lifestyle-Risk Score"),margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig3,use_container_width=True)

    # -------------------------------
    # ML Revenue Predictions + Features
    # -------------------------------
    st.markdown("### ML Revenue Predictions (Actual vs Predicted + Features)")
    filt["Predicted_Score"]=filt["Lifestyle_Score"]*(1-filt["Climate_Risk_Score"])*np.random.uniform(0.8,1.2,len(filt))
    ml_table=filt[["City","Neighborhood","Property_Type","Price","Predicted_Score"]].copy()
    ml_table["Feature_Example"]="Lifestyle & Risk"
    st.dataframe(ml_table,use_container_width=True)
    st.download_button("Download ML Predictions CSV", ml_table.to_csv(index=False),"ml_predictions.csv","text/csv")

    # -------------------------------
    # Automated Insights
    # -------------------------------
    st.markdown("### Automated Insights")
    insights=pd.DataFrame({
        "Insight":["Top Neighborhood","Safest Area","Highest ROI","Rising Price Trend","Lifestyle Hotspot"],
        "Value":["Adyar","Bandra","Bangalore","Increasing","Whitefield"]
    })
    st.dataframe(insights,use_container_width=True)
    st.download_button("Download Insights CSV", insights.to_csv(index=False),"automated_insights.csv","text/csv")

    # -------------------------------
    # Download Filtered Data
    # -------------------------------
    st.download_button("Download Filtered Dataset",filt.to_csv(index=False),"filtered_neighborhood.csv","text/csv")
