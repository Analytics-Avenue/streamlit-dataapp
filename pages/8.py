import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Real Estate Buyer Sentiment Analyzer", layout="wide")

# ----------------------------
# CSS & Header
# ----------------------------
hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

st.markdown("""
<style>
.big-header {
    font-size: 40px; font-weight: 900;
    background: linear-gradient(90deg,#FF6B6B,#FFD93D);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.card {background:#fff;border-radius:15px;padding:20px;margin-bottom:15px;
box-shadow:0 4px 20px rgba(0,0,0,0.08);}
.metric-card {background:#eef4ff;padding:15px;border-radius:8px;text-align:center;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-header'>Real Estate Buyer Sentiment Analyzer</div>", unsafe_allow_html=True)

# ==========================================================
# TABS
# ==========================================================
tab1, tab2 = st.tabs(["Overview", "Application"])

# ==========================================================
# TAB 1 - OVERVIEW
# ==========================================================
with tab1:
    st.markdown("### Overview")
    st.markdown("""
    <div class='card'>
    This application analyzes buyer sentiment across cities, localities, and property types.
    Investors and developers can identify high-demand areas and customer preferences using data-driven insights.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Purpose")
    st.markdown("""
    <div class='card'>
    • Understand buyer preferences and demand<br>
    • Identify hotspots and under-served areas<br>
    • Optimize property offerings and marketing strategy<br>
    • Improve lead conversion and investment decisions
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Capabilities")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='card'>
        <b>Technical</b><br>
        • Sentiment analysis<br>
        • Hotspot mapping<br>
        • Interactive dashboards<br>
        • Agent performance & scoring
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='card'>
        <b>Business</b><br>
        • Customer-driven property strategy<br>
        • Investment hotspot detection<br>
        • Market segmentation<br>
        • Optimized portfolio planning
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Top Localities by Sentiment</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Highest ROI Potential</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Top Performing Agents</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Avg Buyer Sentiment</div>", unsafe_allow_html=True)

# ==========================================================
# TAB 2 - APPLICATION
# ==========================================================
with tab2:
    st.markdown("### Step 1: Load Dataset")
    df = None

    mode = st.radio(
        "Select Option:",
        ["Default Dataset", "Upload CSV", "Upload CSV + Column Mapping"],
        horizontal=True
    )

    # ----------------------------
    # DEFAULT DATASET
    # ----------------------------
    if mode == "Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        try:
            df = pd.read_csv(URL)
            st.success("Default dataset loaded successfully.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Could not load dataset: {e}")

    # ----------------------------
    # UPLOAD CSV
    # ----------------------------
    if mode == "Upload CSV":
        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)
            st.success("Dataset uploaded.")
            st.dataframe(df.head())

    # ----------------------------
    # UPLOAD + COLUMN MAPPING
    # ----------------------------
    if mode == "Upload CSV + Column Mapping":
        file = st.file_uploader("Upload dataset", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.write("Uploaded Data", raw.head())
            st.markdown("### Map Required Columns")

            REQUIRED_COLS = ["City","Locality","Property_Type","Price","Buyer_Sentiment","Agent_Name","Latitude","Longitude"]

            mapping = {}
            for col in REQUIRED_COLS:
                mapping[col] = st.selectbox(
                    f"Map your column to: {col}",
                    options=["-- Select --"] + list(raw.columns)
                )

            if st.button("Apply Mapping"):
                missing = [col for col, mapped in mapping.items() if mapped == "-- Select --"]
                if missing:
                    st.error(f"Please map all required columns: {missing}")
                else:
                    df = raw.rename(columns=mapping)
                    st.success("Mapping applied successfully.")
                    st.dataframe(df.head())

    # ----------------------------
    # PROCEED ONLY IF DATA LOADED
    # ----------------------------
    if df is None:
        st.stop()

    df = df.dropna(subset=["Latitude","Longitude","Price","Buyer_Sentiment"])

    # ==========================================================
    # Filters
    # ==========================================================
    st.markdown("### Step 2: Dashboard Filters")
    f1, f2, f3 = st.columns(3)
    with f1:
        city = st.multiselect("City", df["City"].unique())
    with f2:
        locality = st.multiselect("Locality", df["Locality"].unique())
    with f3:
        ptype = st.multiselect("Property Type", df["Property_Type"].unique())

    filt = df.copy()
    if city:
        filt = filt[filt["City"].isin(city)]
    if locality:
        filt = filt[filt["Locality"].isin(locality)]
    if ptype:
        filt = filt[filt["Property_Type"].isin(ptype)]

    st.markdown("### Data Preview")
    st.dataframe(filt.head(), use_container_width=True)

    # ==========================================================
    # KPIs
    # ==========================================================
    st.markdown("### Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Top Localities by Sentiment", filt.groupby("Locality")["Buyer_Sentiment"].mean().sort_values(ascending=False).head(1).index[0])
    k2.metric("Highest ROI Potential", (filt["Price"] * filt["Buyer_Sentiment"]).max())
    k3.metric("Top Performing Agents", filt.groupby("Agent_Name")["Buyer_Sentiment"].mean().sort_values(ascending=False).head(1).index[0])
    k4.metric("Avg Buyer Sentiment", f"{filt['Buyer_Sentiment'].mean():.2f}")

    # ==========================================================
    # CHART DROPDOWN FUNCTION
    # ==========================================================
    def chart_dropdown(title, purpose, tip, chart_func):
        choice = st.selectbox(f"{title} (Click to view Purpose & Tip)", ["Show Chart"])
        if choice == "Show Chart":
            with st.expander(f"{title} - Purpose & Quick Tip"):
                st.markdown(f"**Purpose:** {purpose}")
                st.markdown(f"**Quick Tip:** {tip}")
            chart_func()

    # ==========================================================
    # Chart 1: ROI by Locality
    # ==========================================================
    def chart1():
        locality_roi = (filt["Price"] * filt["Buyer_Sentiment"]).groupby(filt["Locality"]).mean().reset_index(name="Expected_ROI")
        fig = px.bar(locality_roi, x="Locality", y="Expected_ROI", color="Locality", text="Expected_ROI", color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_traces(texttemplate="₹ %{text:,.0f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    chart_dropdown(
        "ROI by Locality",
        "Shows the average expected ROI in each locality based on price and buyer sentiment.",
        "Focus on top localities for strategic investments.",
        chart1
    )

    # ==========================================================
    # Chart 2: Buyer Sentiment by Property Type
    # ==========================================================
    def chart2():
        ptype_sentiment = filt.groupby("Property_Type")["Buyer_Sentiment"].mean().reset_index()
        fig = px.bar(ptype_sentiment, x="Property_Type", y="Buyer_Sentiment", text="Buyer_Sentiment", color="Property_Type", color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    chart_dropdown(
        "Buyer Sentiment by Property Type",
        "Analyzes average buyer sentiment across property types.",
        "Target property types with highest positive sentiment.",
        chart2
    )

    # ==========================================================
    # Chart 3: Buyer Sentiment Hotspot Map
    # ==========================================================
    def chart3():
        filt["Sentiment_Normalized"] = (filt["Buyer_Sentiment"] - filt["Buyer_Sentiment"].min()) / (
            filt["Buyer_Sentiment"].max() - filt["Buyer_Sentiment"].min()
        )
        filt["Expected_ROI"] = filt["Price"] * filt["Buyer_Sentiment"]
        fig = px.scatter_mapbox(
            filt, lat="Latitude", lon="Longitude", size="Expected_ROI", color="Sentiment_Normalized",
            hover_name="Locality", hover_data=["City","Property_Type","Price","Buyer_Sentiment"],
            color_continuous_scale=px.colors.sequential.Viridis, size_max=15, zoom=10
        )
        fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

    chart_dropdown(
        "Buyer Sentiment Hotspot Map",
        "Displays geospatial hotspots for properties with high buyer sentiment.",
        "Look for clusters with strong positive sentiment for potential investment.",
        chart3
    )

    # ==========================================================
    # Top Properties Table
    # ==========================================================
    st.markdown("### Top Properties by Expected ROI")
    top_inv = filt.sort_values("Expected_ROI", ascending=False).head(10)
    st.dataframe(top_inv[["City","Locality","Property_Type","Price","Buyer_Sentiment","Expected_ROI","Agent_Name"]])
    csv = top_inv.to_csv(index=False)
    st.download_button("Download Top Properties", csv, "top_properties.csv", "text/csv")
