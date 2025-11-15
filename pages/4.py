import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Real Estate Agent & Market Insights", layout="wide")

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
    "City",
    "Property_Type",
    "Agent_Name",
    "Price",
    "Lead_Score",
    "Conversion_Probability",
    "Days_On_Market"
]

# ----------------------------------------------------------
# PAGE SETTINGS + CSS
# ----------------------------------------------------------
st.markdown("""
<style>
.big-header {
    font-size: 40px; font-weight: 900;
    background: linear-gradient(90deg,#0A5EB0,#2E82FF);
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
st.markdown("<div class='big-header'>Real Estate Agent & Market Insights</div>", unsafe_allow_html=True)

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
    This platform provides insights into agent performance, lead conversion, and market segmentation.
    It allows analytics-driven decision making for real estate teams and investors.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Purpose")
    st.markdown("""
    <div class='card'>
    • Evaluate agent performance<br>
    • Track lead-to-sale conversion<br>
    • Understand market segments and pricing trends<br>
    • Identify high-demand property areas
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Capabilities")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='card'>
        <b>Analytics</b><br>
        • Agent performance dashboard<br>
        • Lead conversion metrics<br>
        • Market segmentation with clustering<br>
        • Pricing trends visualization
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='card'>
        <b>Business</b><br>
        • Incentive planning for agents<br>
        • Target high-conversion segments<br>
        • Optimize marketing spend<br>
        • Strategic portfolio allocation
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Avg Lead Score</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Avg Conversion %</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Avg Days on Market</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Top Performing Agent</div>", unsafe_allow_html=True)

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

    # --------------------------------------------------------------
    # 1. DEFAULT DATASET
    # --------------------------------------------------------------
    if mode == "Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        try:
            df = pd.read_csv(URL)
            st.success("Default dataset loaded successfully.")
            st.write(df.head())
        except Exception as e:
            st.error(f"Could not load dataset: {e}")

    # --------------------------------------------------------------
    # 2. UPLOAD CSV
    # --------------------------------------------------------------
    if mode == "Upload CSV":
        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)
            st.success("Dataset uploaded.")
            st.dataframe(df.head())

    # --------------------------------------------------------------
    # 3. UPLOAD + COLUMN MAPPING (ONLY REQUIRED COLS)
    # --------------------------------------------------------------
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

    # --------------------------------------------------------------
    if df is None:
        st.stop()

    # Ensure required columns exist
    if not all(col in df.columns for col in REQUIRED_COLS):
        st.error("Dataset missing required columns.")
        st.write("Required columns:", REQUIRED_COLS)
        st.stop()

    df = df.dropna(subset=REQUIRED_COLS)

    # ==========================================================
    # FILTERS
    # ==========================================================
    st.markdown("### Step 2: Filters")
    f1, f2, f3 = st.columns(3)

    with f1:
        city_filter = st.multiselect("City", df["City"].unique())
    with f2:
        ptype_filter = st.multiselect("Property Type", df["Property_Type"].unique())
    with f3:
        agent_filter = st.multiselect("Agent Name", df["Agent_Name"].unique())

    filt = df.copy()
    if city_filter:
        filt = filt[filt["City"].isin(city_filter)]
    if ptype_filter:
        filt = filt[filt["Property_Type"].isin(ptype_filter)]
    if agent_filter:
        filt = filt[filt["Agent_Name"].isin(agent_filter)]

    st.markdown("### Filtered Data Preview")
    st.dataframe(filt.head(), use_container_width=True)

    # ==========================================================
    # KPIs
    # ==========================================================
    st.markdown("### Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Lead Score", f"{filt['Lead_Score'].mean():.2f}")
    k2.metric("Avg Conversion %", f"{filt['Conversion_Probability'].mean()*100:.2f}%")
    k3.metric("Avg Days on Market", f"{filt['Days_On_Market'].mean():.1f}")
    top_agent = filt.groupby("Agent_Name")["Conversion_Probability"].mean().idxmax()
    k4.metric("Top Performing Agent", top_agent)

    # ==========================================================
    # CHARTS
    # ==========================================================
    st.markdown("### Agent-wise Conversion")
    agent_conv = filt.groupby("Agent_Name")["Conversion_Probability"].mean().reset_index()
    fig1 = px.bar(agent_conv, x="Agent_Name", y="Conversion_Probability",
                  color="Conversion_Probability", text="Conversion_Probability",
                  color_continuous_scale=px.colors.sequential.Viridis)
    fig1.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### City-wise Average Price")
    city_price = filt.groupby("City")["Price"].mean().reset_index()
    fig2 = px.bar(city_price, x="City", y="Price", color="City", text="Price",
                  color_discrete_sequence=px.colors.qualitative.Bold)
    fig2.update_traces(texttemplate="₹ %{text:,.0f}", textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Market Segmentation (KMeans)")
    seg_features = filt[["Price", "Days_On_Market", "Conversion_Probability"]].copy()
    scaler = StandardScaler()
    seg_scaled = scaler.fit_transform(seg_features)

    kmeans = KMeans(n_clusters=4, random_state=42)
    filt["Segment"] = kmeans.fit_predict(seg_scaled)

    fig3 = px.scatter(filt, x="Price", y="Conversion_Probability", color="Segment",
                      hover_data=["Agent_Name", "City", "Property_Type"],
                      color_discrete_sequence=px.colors.qualitative.D3)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Download Filtered Data")
    csv = filt.to_csv(index=False)
    st.download_button("Download Filtered Data", csv, "agent_market_insights.csv", "text/csv")
