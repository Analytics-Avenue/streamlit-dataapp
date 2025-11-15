import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Real Estate Buyer Sentiment Analyzer", layout="wide")

hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# ----------------------------------------------------------
# REQUIRED COLUMNS ONLY FOR THIS APPLICATION
# ----------------------------------------------------------
REQUIRED_COLS = [
    "City", "Locality", "Property_Type", "Price", "Latitude", "Longitude", "Buyer_Sentiment"
]

# ----------------------------------------------------------
# PAGE SETTINGS + CSS
# ----------------------------------------------------------
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

# ----------------------------------------------------------
# MAIN HEADER
# ----------------------------------------------------------
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
    This application analyzes buyer sentiment trends across cities and property types, 
    identifies hot spots, and quantifies demand patterns in real estate markets. 
    Investors, developers, and agencies can leverage this to optimize sales strategy 
    and focus on high-demand localities.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Purpose")
    st.markdown("""
    <div class='card'>
    • Evaluate buyer preferences and sentiment trends<br>
    • Identify high-demand neighborhoods<br>
    • Visualize property-type popularity<br>
    • Support data-driven marketing & sales
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Capabilities")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='card'>
        <b>Technical</b><br>
        • Sentiment scoring dashboards<br>
        • Hotspot mapping<br>
        • City and property segmentation<br>
        • Interactive visual analytics
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='card'>
        <b>Business</b><br>
        • Optimize sales focus<br>
        • Prioritize high-demand areas<br>
        • Targeted property marketing<br>
        • Enhance buyer engagement
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>High Sentiment Areas</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Top Localities</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Popular Property Types</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Average Buyer Sentiment</div>", unsafe_allow_html=True)

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

    # 1. DEFAULT DATASET
    if mode == "Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        try:
            df = pd.read_csv(URL)
            st.success("Default dataset loaded successfully.")
            st.write(df.head())
        except Exception as e:
            st.error(f"Could not load dataset: {e}")

    # 2. UPLOAD CSV
    if mode == "Upload CSV":
        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)
            st.success("Dataset uploaded.")
            st.dataframe(df.head())

    # 3. UPLOAD + COLUMN MAPPING
    if mode == "Upload CSV + Column Mapping":
        file = st.file_uploader("Upload dataset", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.write("Uploaded Data", raw.head())
            st.markdown("### Map Required Columns Only")
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

    # ----------------------------------------------------------
    # PROCEED ONLY IF DATA LOADED
    # ----------------------------------------------------------
    if df is None:
        st.stop()

    # Use mapped columns dynamically
    LAT_COL = "Latitude"
    LON_COL = "Longitude"
    PRICE_COL = "Price"
    SENT_COL = "Buyer_Sentiment"
    LOCAL_COL = "Locality"

    if mode == "Upload CSV + Column Mapping" and 'mapping' in locals():
        LAT_COL = mapping["Latitude"]
        LON_COL = mapping["Longitude"]
        PRICE_COL = mapping["Price"]
        SENT_COL = mapping["Buyer_Sentiment"]
        LOCAL_COL = mapping["Locality"]

    # Drop rows with missing values in required columns
    df = df.dropna(subset=[LAT_COL, LON_COL, PRICE_COL, SENT_COL])

    # ==========================================================
    # FILTERS
    # ==========================================================
    st.markdown("### Step 2: Filters")
    f1, f2, f3 = st.columns(3)
    with f1:
        city = st.multiselect("City", df["City"].unique())
    with f2:
        ptype = st.multiselect("Property Type", df["Property_Type"].unique())
    with f3:
        locality = st.multiselect("Locality", df[LOCAL_COL].unique())

    filt = df.copy()
    if city:
        filt = filt[filt["City"].isin(city)]
    if ptype:
        filt = filt[filt["Property_Type"].isin(ptype)]
    if locality:
        filt = filt[filt[LOCAL_COL].isin(locality)]

    st.markdown("### Data Preview")
    st.dataframe(filt.head(), use_container_width=True)

    # ==========================================================
    # KPIs
    # ==========================================================
    st.markdown("### Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("High Sentiment Properties", len(filt[filt[SENT_COL] > 0.7]))
    k2.metric("Top Locality", filt.groupby(LOCAL_COL)[SENT_COL].mean().sort_values(ascending=False).head(1).index[0])
    k3.metric("Popular Property Type", filt.groupby("Property_Type")[SENT_COL].mean().sort_values(ascending=False).head(1).index[0])
    k4.metric("Average Buyer Sentiment", f"{filt[SENT_COL].mean():.2f}")

    # ==========================================================
    # CHARTS FUNCTION WITH DROPDOWN FOR PURPOSE/TIP
    # ==========================================================
    def chart_dropdown(title, purpose, tip, chart_func):
        st.markdown(f"### {title}")
        with st.expander("Purpose & Quick Tip"):
            st.markdown(f"**Purpose:** {purpose}")
            st.markdown(f"**Quick Tip:** {tip}")
        chart_func()

    # ----------------------------
    # Chart 1: Buyer Sentiment by City
    # ----------------------------
    def chart1():
        city_sent = filt.groupby("City")[SENT_COL].mean().reset_index()
        fig = px.bar(city_sent, x="City", y=SENT_COL, text=SENT_COL,
                     color="City", color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    chart_dropdown(
        "Buyer Sentiment by City",
        "Evaluate sentiment trends across cities to understand market preference.",
        "High sentiment cities indicate better buyer engagement.",
        chart1
    )

    # ----------------------------
    # Chart 2: Buyer Sentiment by Property Type
    # ----------------------------
    def chart2():
        type_sent = filt.groupby("Property_Type")[SENT_COL].mean().reset_index()
        fig = px.bar(type_sent, x="Property_Type", y=SENT_COL, text=SENT_COL,
                     color="Property_Type", color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    chart_dropdown(
        "Buyer Sentiment by Property Type",
        "Identify which property types are most preferred by buyers.",
        "Property types with higher sentiment scores are easier to sell.",
        chart2
    )

    # ----------------------------
    # Chart 3: Hotspot Map
    # ----------------------------
    filt["Sentiment_Normalized"] = (filt[SENT_COL] - filt[SENT_COL].min()) / (
        filt[SENT_COL].max() - filt[SENT_COL].min()
    )
    filt["Expected_ROI"] = filt[PRICE_COL] * filt[SENT_COL]

    def chart3():
        fig = px.scatter_mapbox(
            filt, lat=LAT_COL, lon=LON_COL, size="Expected_ROI", color="Sentiment_Normalized",
            hover_name=LOCAL_COL, hover_data=["City","Property_Type",PRICE_COL,SENT_COL],
            color_continuous_scale=px.colors.sequential.Viridis, size_max=15, zoom=10
        )
        fig.update_layout(mapbox_style="open-street-map",
                          coloraxis_colorbar=dict(title="Buyer Sentiment"),
                          margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

    chart_dropdown(
        "Buyer Sentiment Hotspot Map",
        "Visualize neighborhoods with high buyer sentiment and potential ROI.",
        "Normalized sentiment values show top-demand areas.",
        chart3
    )

    # ==========================================================
    # Top Properties by Sentiment
    # ==========================================================
    st.markdown("### Top 10 High Sentiment Properties")
    top_props = filt.sort_values("Expected_ROI", ascending=False).head(10)
    st.dataframe(top_props[["City","Locality","Property_Type",PRICE_COL,SENT_COL,"Expected_ROI"]])
    csv = top_props.to_csv(index=False)
    st.download_button("Download Top Properties", csv, "top_properties.csv", "text/csv")
