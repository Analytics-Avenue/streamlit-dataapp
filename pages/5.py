import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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

# ----------------------------------------------------------
# REQUIRED COLUMNS ONLY FOR THIS APPLICATION
# ----------------------------------------------------------
REQUIRED_COLS = [
    "City", "Property_Type", "Price", "Area_sqft", "Agent_Name",
    "Conversion_Probability", "Latitude", "Longitude"
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
st.markdown("<div class='big-header'>Real Estate Investment Opportunity Analyzer</div>", unsafe_allow_html=True)

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
    This application identifies high-return real estate opportunities by analyzing city-level, agent-level, 
    and property-type performance. Ideal for investors and developers to make strategic investment decisions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Purpose")
    st.markdown("""
    <div class='card'>
    • Highlight high-ROI properties<br>
    • Analyze agent performance<br>
    • Identify market hotspots<br>
    • Support data-driven investment decisions
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Capabilities")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='card'>
        <b>Technical</b><br>
        • Conversion-adjusted ROI<br>
        • Interactive maps<br>
        • City and property segmentation<br>
        • Agent performance dashboards
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='card'>
        <b>Business</b><br>
        • Investment prioritization<br>
        • Transparent property scoring<br>
        • Market opportunity mapping<br>
        • Optimized portfolio allocation
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>High ROI Properties</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Top Performing Agents</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Hotspot Cities</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Average Conversion Rate</div>", unsafe_allow_html=True)

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

    if not all(col in df.columns for col in REQUIRED_COLS):
        st.error("Dataset missing required columns.")
        st.write("Required columns:", REQUIRED_COLS)
        st.stop()

    df = df.dropna()

    # ==========================================================
    # FILTERS
    # ==========================================================
    st.markdown("### Step 2: Dashboard Filters")
    f1, f2, f3 = st.columns(3)
    with f1:
        city = st.multiselect("City", df["City"].unique())
    with f2:
        ptype = st.multiselect("Property Type", df["Property_Type"].unique())
    with f3:
        agent = st.multiselect("Agent Name", df["Agent_Name"].unique())

    filt = df.copy()
    if city:
        filt = filt[filt["City"].isin(city)]
    if ptype:
        filt = filt[filt["Property_Type"].isin(ptype)]
    if agent:
        filt = filt[filt["Agent_Name"].isin(agent)]

    st.markdown("### Data Preview")
    st.dataframe(filt.head(), use_container_width=True)

    # ==========================================================
    # KPIs
    # ==========================================================
    st.markdown("### Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    filt["Expected_ROI"] = filt["Price"] * filt["Conversion_Probability"]
    
    k1.metric("High ROI Properties", len(filt[filt["Conversion_Probability"] > 0.7]))
    k2.metric(
        "Top Performing Agents", 
        filt.groupby("Agent_Name")["Conversion_Probability"].mean().sort_values(ascending=False).head(1).index[0]
    )
    k3.metric(
        "Hotspot Cities", 
        filt.groupby("City")["Price"].mean().sort_values(ascending=False).head(1).index[0]
    )
    k4.metric("Average Conversion Rate", f"{filt['Conversion_Probability'].mean():.2f}")

    # ==========================================================
    # Charts
    # ==========================================================
    st.markdown("### ROI by City")
    city_roi = filt.groupby("City")["Expected_ROI"].mean().reset_index()
    fig1 = px.bar(city_roi, x="City", y="Expected_ROI", color="City", text="Expected_ROI",
                  color_discrete_sequence=px.colors.qualitative.Bold)
    fig1.update_traces(texttemplate="₹ %{text:,.0f}", textposition="outside")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### ROI by Property Type")
    ptype_roi = filt.groupby("Property_Type")["Expected_ROI"].mean().reset_index()
    fig2 = px.bar(ptype_roi, x="Property_Type", y="Expected_ROI", color="Property_Type", text="Expected_ROI",
                  color_discrete_sequence=px.colors.qualitative.Vivid)
    fig2.update_traces(texttemplate="₹ %{text:,.0f}", textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    # ==========================================================
    # Hotspot Map
    # ==========================================================
    # Normalize Conversion_Probability
    if filt["Conversion_Probability"].nunique() > 1:
        filt["Conversion_Normalized"] = (filt["Conversion_Probability"] - filt["Conversion_Probability"].min()) / (
            filt["Conversion_Probability"].max() - filt["Conversion_Probability"].min()
        )
    else:
        filt["Conversion_Normalized"] = 0.5

    fig3 = px.scatter_mapbox(
        filt,
        lat="Latitude",
        lon="Longitude",
        size="Expected_ROI",
        color="Conversion_Normalized",
        hover_name="Property_Type",
        hover_data=["City", "Price", "Agent_Name", "Conversion_Probability", "Expected_ROI"],
        color_continuous_scale=px.colors.sequential.Viridis,
        size_max=20,
        zoom=10
    )
    fig3.update_layout(
        mapbox_style="open-street-map",
        coloraxis_colorbar=dict(title="Conversion Probability"),
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ==========================================================
    # Top Investment Properties
    # ==========================================================
    st.markdown("### Top Investment Properties")
    top_inv = filt.sort_values("Expected_ROI", ascending=False).head(10)
    st.dataframe(top_inv[["City","Property_Type","Price","Conversion_Probability","Expected_ROI","Agent_Name"]])

    csv = top_inv.to_csv(index=False)
    st.download_button("Download Top Investment Properties", csv, "top_investments.csv", "text/csv")
