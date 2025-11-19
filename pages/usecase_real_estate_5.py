import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from io import BytesIO

st.set_page_config(page_title="Real Estate Investment Opportunity Analyzer", layout="wide")

# ----------------------------
# Hide Sidebar
# ----------------------------
hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# -------------------------
# Company Logo + Name
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display: flex; align-items: center;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:black; font-size:36px; font-weight:bold; margin:0; padding:0;">Analytics Avenue &</div>
        <div style="color:black; font-size:36px; font-weight:bold; margin:0; padding:0;">Advanced Analytics</div>
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
# CSS for cards & hover effects
# --------------------------
st.markdown("""
<style>
.big-header {font-size:40px; font-weight:900; color:black;}
.card, .metric-card, .hover-card {background:#fff;border-radius:15px;padding:20px;margin-bottom:15px;
box-shadow:0 4px 20px rgba(0,0,0,0.08);transition: all 0.25s ease;text-align:left;}
.card:hover, .metric-card:hover, .hover-card:hover {box-shadow:0 0 18px rgba(0,0,0,0.4); transform:scale(1.03);}
.metric-card {font-weight:600;}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Main header
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
    st.markdown("### Overview", unsafe_allow_html=True)
    st.markdown("<div class='hover-card'>Generic info: Helps investors identify high-return real estate opportunities by analyzing city, agent, and property-type performance.</div>", unsafe_allow_html=True)

    st.markdown("### Purpose")
    st.markdown("<div class='hover-card'>• Identify high-ROI properties<br>• Compare agent performance<br>• Analyze city/property-type trends<br>• Visualize market hotspots</div>", unsafe_allow_html=True)

    st.markdown("### Capabilities")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='hover-card'><b>Technical</b><br>• Conversion-adjusted ROI<br>• Interactive maps<br>• City/property segmentation<br>• Agent dashboards</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='hover-card'><b>Business</b><br>• Investment prioritization<br>• Transparent property scoring<br>• Market opportunity mapping<br>• Portfolio allocation</div>", unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>High ROI Properties</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Top Performing Agents</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Hotspot Cities</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Average Conversion Rate</div>", unsafe_allow_html=True)

    st.markdown("### Use of App")
    st.markdown("<div class='hover-card'>• Monitor agent & property performance<br>• Evaluate lead conversion efficiency<br>• Segment markets for strategic decisions<br>• Forecast revenue streams</div>", unsafe_allow_html=True)

    st.markdown("### Who Should Use")
    st.markdown("<div class='hover-card'>• Investors & portfolio managers<br>• Real estate agents & managers<br>• Market analysts<br>• Property marketing teams</div>", unsafe_allow_html=True)

# ==========================================================
# TAB 2 - APPLICATION
# ==========================================================
with tab2:
    st.markdown("### Step 1: Load Dataset")
    df = None

    mode = st.radio("Select Option:", ["Default Dataset", "Upload CSV", "Upload CSV + Column Mapping"], horizontal=True)

    if mode=="Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        try:
            df = pd.read_csv(URL)
            st.success("Default dataset loaded successfully.")
            st.dataframe(df.head(), use_container_width=True)
        except:
            st.error("Could not load dataset.")

    if mode=="Upload CSV":
        st.markdown("#### Download Sample CSV")
        sample_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        sample_df = pd.read_csv(sample_url).head(5)
        st.download_button("Download Sample CSV", sample_df.to_csv(index=False), "sample.csv","text/csv")
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file: df=pd.read_csv(file)

    if mode=="Upload CSV + Column Mapping":
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.write(raw.head())
            st.markdown("### Map Required Columns Only")
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
    if not all(col in df.columns for col in REQUIRED_COLS): st.error("Missing required columns"); st.stop()
    df=df.dropna()

    # --------------------------
    # Filters
    # --------------------------
    st.markdown("### Step 2: Filters")
    f1,f2,f3=st.columns(3)
    with f1: city=st.multiselect("City", df["City"].unique())
    with f2: ptype=st.multiselect("Property Type", df["Property_Type"].unique())
    with f3: agent=st.multiselect("Agent Name", df["Agent_Name"].unique())
    filt=df.copy()
    if city: filt=filt[filt["City"].isin(city)]
    if ptype: filt=filt[filt["Property_Type"].isin(ptype)]
    if agent: filt=filt[filt["Agent_Name"].isin(agent)]

    st.markdown("### Data Preview")
    st.dataframe(filt.head(), use_container_width=True)

    # --------------------------
    # KPIs
    # --------------------------
    st.markdown("### Key Metrics")
    filt["Expected_ROI"] = filt["Price"] * filt["Conversion_Probability"]
    k1.metric("High ROI Properties", "Value")
    k2.metric("Top Performing Agents", "Value")
    k3.metric("Hotspot Cities", "Value")
    k4.metric("Average Conversion Rate", "Value")

    # --------------------------
    # Charts
    # --------------------------
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

    st.markdown("### Market Hotspot Map")
    filt_map = filt.copy()
    
    # Drop missing coordinates or ROI
    filt_map = filt_map.dropna(subset=["Latitude", "Longitude", "Expected_ROI", "Conversion_Probability"])
    
    if filt_map.empty:
        st.info("No data available for map visualization after filtering.")
    else:
        # Normalize conversion probability for color scale
        min_conv, max_conv = filt_map["Conversion_Probability"].min(), filt_map["Conversion_Probability"].max()
        if min_conv != max_conv:
            filt_map["Conversion_Normalized"] = (filt_map["Conversion_Probability"] - min_conv) / (max_conv - min_conv)
        else:
            filt_map["Conversion_Normalized"] = 0.5
    
        # Add tiny jitter if multiple points have the same coordinates
        filt_map["Latitude"] += np.random.uniform(-0.0005, 0.0005, size=len(filt_map))
        filt_map["Longitude"] += np.random.uniform(-0.0005, 0.0005, size=len(filt_map))
    
        # Dynamic center
        center_lat = filt_map["Latitude"].mean()
        center_lon = filt_map["Longitude"].mean()
    
        # Create map
        fig3 = px.scatter_mapbox(
            filt_map,
            lat="Latitude",
            lon="Longitude",
            size="Expected_ROI",
            color="Conversion_Normalized",
            hover_name="Property_Type",
            hover_data={
                "City": True,
                "Price": True,
                "Agent_Name": True,
                "Conversion_Probability": True,
                "Expected_ROI": True
            },
            color_continuous_scale=px.colors.sequential.Viridis,
            size_max=20,
            zoom=12,
            center={"lat": center_lat, "lon": center_lon}
        )
    
        fig3.update_layout(
            mapbox_style="open-street-map",
            coloraxis_colorbar=dict(title="Conversion Probability"),
            margin={"r":0,"t":0,"l":0,"b":0}
        )
    
        st.plotly_chart(fig3, use_container_width=True)
    
            

    st.markdown("### Top Investment Properties")
    top_inv=filt.sort_values("Expected_ROI",ascending=False).head(10)
    st.dataframe(top_inv[["City","Property_Type","Price","Conversion_Probability","Expected_ROI","Agent_Name"]])
    st.download_button("Download Top Investment Properties",top_inv.to_csv(index=False),"top_investments.csv","text/csv")

    # --------------------------
    # ML Revenue Predictions (generic)
    # --------------------------
    st.markdown("### ML Revenue Predictions")
    results=filt.copy()
    results["Predicted_Revenue"]="Value"
    st.dataframe(results,use_container_width=True)
    buf=BytesIO()
    results.to_csv(buf,index=False)
    st.download_button("Download ML Revenue Predictions",buf.getvalue(),"ml_revenue_predictions.csv","text/csv")

    # --------------------------
    # Automated Insights
    # ==========================================================    
    # Create a sample automated insights DataFrame
    insights = pd.DataFrame({
        "Insight_Type": [
            "Top City by Expected ROI",
            "Top Property Type by ROI",
            "Top Agent by Conversion",
            "Highest ROI Property",
            "Most Active City"
        ],
        "Value": [
            filt.groupby("City")["Expected_ROI"].mean().sort_values(ascending=False).head(1).index[0] if not filt.empty else "N/A",
            filt.groupby("Property_Type")["Expected_ROI"].mean().sort_values(ascending=False).head(1).index[0] if not filt.empty else "N/A",
            filt.groupby("Agent_Name")["Conversion_Probability"].mean().sort_values(ascending=False).head(1).index[0] if not filt.empty else "N/A",
            filt.sort_values("Expected_ROI", ascending=False).head(1)["Property_Type"].values[0] if not filt.empty else "N/A",
            filt["City"].value_counts().head(1).index[0] if not filt.empty else "N/A"
        ],
        "Description": [
            "City with highest average expected ROI",
            "Property type generating highest ROI",
            "Agent with best conversion probability",
            "Property with maximum expected ROI",
            "City with most active listings"
        ]
    })
    
    # Display table
    st.dataframe(insights)
    
    # Download button
    csv_insights = insights.to_csv(index=False)
    st.download_button("Download Automated Insights", csv_insights, "automated_insights.csv", "text/csv")
