import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from io import BytesIO

st.set_page_config(page_title="Real Estate Investment Opportunity Analyzer", layout="wide")

# ----------------------------
# Hide Sidebar
# ----------------------------
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
.card, .metric-card, .hover-card {
    background:#fff;
    border-radius:15px;
    padding:20px;
    margin-bottom:15px;
    box-shadow:0 4px 20px rgba(0,0,0,0.08);
    text-align:left;
    transition: all 0.25s ease;
}
.card:hover, .metric-card:hover, .hover-card:hover {
    box-shadow:0 0 18px rgba(0,0,0,0.4);
    transform:scale(1.03);
}
.metric-card {background:#eef4ff; font-weight:600; text-align:center;}
.big-header {font-size:40px; font-weight:900; background: linear-gradient(90deg,#FF6B6B,#FFD93D);
 -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Company Logo + Name
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display: flex; align-items: center;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0; padding:0;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0; padding:0;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------
# Required Columns
# --------------------------
REQUIRED_COLS = [
    "City", "Property_Type", "Price", "Area_sqft", "Agent_Name",
    "Conversion_Probability", "Latitude", "Longitude"
]

# --------------------------
# Header
# --------------------------
st.markdown("<div class='big-header'>Real Estate Investment Opportunity Analyzer</div>", unsafe_allow_html=True)

# ==========================================================
# Tabs
# ==========================================================
tab1, tab2 = st.tabs(["Overview", "Application"])

# ==========================================================
# TAB 1 - OVERVIEW
# ==========================================================
with tab1:
    st.markdown("<div class='hover-card'>This application helps investors identify high-return real estate opportunities by analyzing city-level, agent-level, and property-type performance metrics.</div>", unsafe_allow_html=True)
    st.markdown("<div class='hover-card'>It highlights properties with strong conversion potential, enabling data-driven investment decisions across regions and segments.</div>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("<div class='hover-card'><b>Technical Capabilities</b><br>• Conversion-adjusted ROI<br>• Interactive maps with hover<br>• City and property segmentation<br>• Agent performance dashboards</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='hover-card'><b>Business Impact</b><br>• Investment prioritization<br>• Transparent property scoring<br>• Market opportunity mapping<br>• Optimized portfolio allocation</div>", unsafe_allow_html=True)

    k1,k2,k3,k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>High ROI Properties</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Top Performing Agents</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Hotspot Cities</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Average Conversion Rate</div>", unsafe_allow_html=True)

    st.markdown("<div class='hover-card'>Use: Monitor properties, analyze conversion, segment markets, and forecast revenue.</div>", unsafe_allow_html=True)
    st.markdown("<div class='hover-card'>Users: Investors, portfolio managers, real estate agents, and market analysts.</div>", unsafe_allow_html=True)

# ==========================================================
# TAB 2 - APPLICATION
# ==========================================================
with tab2:
    st.markdown("### Step 1: Load Dataset")
    df=None
    mode = st.radio("Select Option:", ["Default Dataset","Upload CSV","Upload CSV + Column Mapping"], horizontal=True)

    if mode=="Default Dataset":
        URL="https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        df = pd.read_csv(URL)
        st.success("Default dataset loaded")
        st.dataframe(df.head(), use_container_width=True)

    if mode=="Upload CSV":
        st.markdown("#### Download Sample CSV")
        URL="https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        sample_df=pd.read_csv(URL).head(5)
        st.download_button("Download Sample CSV", sample_df.to_csv(index=False), "sample.csv","text/csv")
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file: df=pd.read_csv(file)

    if mode=="Upload CSV + Column Mapping":
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            raw=pd.read_csv(file)
            st.write(raw.head())
            mapping={}
            for col in REQUIRED_COLS:
                mapping[col]=st.selectbox(f"Map column to {col}", options=["-- Select --"]+list(raw.columns))
            if st.button("Apply Mapping"):
                missing=[c for c,v in mapping.items() if v=="-- Select --"]
                if missing: st.error(f"Map all columns: {missing}")
                else:
                    df=raw.rename(columns=mapping)
                    st.success("Mapping applied")
                    st.dataframe(df.head())

    if df is None: st.stop()
    df=df.dropna()
    if not all(col in df.columns for col in REQUIRED_COLS):
        st.error("Missing required columns"); st.stop()

    # ==========================================================
    # Filters
    # ==========================================================
    st.markdown("### Step 2: Dashboard Filters")
    f1,f2,f3 = st.columns(3)
    city = f1.multiselect("City", df["City"].unique())
    ptype = f2.multiselect("Property Type", df["Property_Type"].unique())
    agent = f3.multiselect("Agent Name", df["Agent_Name"].unique())

    filt=df.copy()
    if city: filt=filt[filt["City"].isin(city)]
    if ptype: filt=filt[filt["Property_Type"].isin(ptype)]
    if agent: filt=filt[filt["Agent_Name"].isin(agent)]

    st.dataframe(filt.head(), use_container_width=True)

    # ==========================================================
    # KPIs
    # ==========================================================
    st.markdown("### Key Metrics")
    filt["Expected_ROI"]=filt["Price"]*filt["Conversion_Probability"]
    k1.metric("High ROI Properties", len(filt[filt["Conversion_Probability"]>0.7]))
    k2.metric("Top Performing Agents", filt.groupby("Agent_Name")["Conversion_Probability"].mean().idxmax())
    k3.metric("Hotspot Cities", filt.groupby("City")["Expected_ROI"].mean().idxmax())
    k4.metric("Average Conversion Rate", f"{filt['Conversion_Probability'].mean():.2f}")

    # ==========================================================
    # Charts
    # ==========================================================
    st.markdown("### ROI by City")
    city_roi=filt.groupby("City")["Expected_ROI"].mean().reset_index()
    fig1=px.bar(city_roi,x="City",y="Expected_ROI",color="City",text="Expected_ROI",color_discrete_sequence=px.colors.qualitative.Bold)
    fig1.update_traces(texttemplate="₹ %{text:,.0f}",textposition="outside")
    st.plotly_chart(fig1,use_container_width=True)

    st.markdown("### ROI by Property Type")
    ptype_roi=filt.groupby("Property_Type")["Expected_ROI"].mean().reset_index()
    fig2=px.bar(ptype_roi,x="Property_Type",y="Expected_ROI",color="Property_Type",text="Expected_ROI",color_discrete_sequence=px.colors.qualitative.Vivid)
    fig2.update_traces(texttemplate="₹ %{text:,.0f}",textposition="outside")
    st.plotly_chart(fig2,use_container_width=True)

    # ==========================================================
    # Hotspot Map
    # ==========================================================
    filt["Conversion_Normalized"]=(filt["Conversion_Probability"]-filt["Conversion_Probability"].min())/(filt["Conversion_Probability"].max()-filt["Conversion_Probability"].min()+1e-9)
    fig3=px.scatter_mapbox(filt,lat="Latitude",lon="Longitude",size="Expected_ROI",color="Conversion_Normalized",
                           hover_name="Property_Type",hover_data={"City":True,"Price":True,"Agent_Name":True,"Conversion_Probability":True,"Expected_ROI":True,"Latitude":False,"Longitude":False,"Conversion_Normalized":False},
                           color_continuous_scale=px.colors.sequential.Viridis,size_max=20,zoom=10)
    fig3.update_layout(mapbox_style="open-street-map",coloraxis_colorbar=dict(title="Conversion Probability"),margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig3,use_container_width=True)

    # ==========================================================
    # Top Investment Properties
    # ==========================================================
    st.markdown("### Top Investment Properties")
    top_inv=filt.sort_values("Expected_ROI",ascending=False).head(10)
    st.dataframe(top_inv[["City","Property_Type","Price","Conversion_Probability","Expected_ROI","Agent_Name"]])
    st.download_button("Download Top Investment Properties", top_inv.to_csv(index=False),"top_investments.csv","text/csv")

    # ==========================================================
    # ML Revenue Predictions
    # ==========================================================
    st.markdown("### ML Revenue Predictions (Actual vs Predicted)")
    X=filt[["City","Property_Type","Conversion_Probability","Area_sqft"]]
    y=filt["Price"]
    ct=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),["City","Property_Type"])], remainder="passthrough")
    X_tr=ct.fit_transform(X)
    X_train,X_test,y_train,y_test=train_test_split(X_tr,y,test_size=0.2,random_state=42)
    model=RandomForestRegressor(n_estimators=300,random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_tr)
    ml_results=X.copy()
    ml_results["Actual_Price"]=y
    ml_results["Predicted_Price"]=y_pred
    st.dataframe(ml_results,use_container_width=True)
    buf=BytesIO()
    ml_results.to_csv(buf,index=False)
    st.download_button("Download ML Revenue Predictions",buf.getvalue(),"ml_revenue_predictions.csv","text/csv")

    # ==========================================================
    # Automated Insights
    # ==========================================================
    st.markdown("### Automated Insights")
    insights=filt.groupby(["City","Property_Type"]).agg({"Price":["mean","max","min"],"Conversion_Probability":"mean"}).reset_index()
    insights.columns=["City","Property_Type","Avg_Price","Max_Price","Min_Price","Avg_Conversion"]
    st.dataframe(insights,use_container_width=True)
    buf2=BytesIO()
    insights.to_csv(buf2,index=False)
    st.download_button("Download Automated Insights",buf2.getvalue(),"automated_insights.csv","text/csv")
