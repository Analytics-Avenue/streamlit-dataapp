import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Real Estate Market Buzz & Activity Dashboard", layout="wide")

# ---------------------------
# Hide Sidebar (optional)
# ---------------------------
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
    "City", "Locality", "Property_Type", "Price", "Area_sqft",
    "Latitude", "Longitude", "Listing_Date"
]

# ----------------------------------------------------------
# CSS for headers and cards
# ----------------------------------------------------------
st.markdown("""
<style>
.big-header {
    font-size: 40px; font-weight: 900;
    background: linear-gradient(90deg,#FF7F50,#1E90FF);
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
st.markdown("<div class='big-header'>Real Estate Market Buzz & Activity Dashboard</div>", unsafe_allow_html=True)

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
    This application identifies the hottest real estate neighborhoods by analyzing listing activity, 
    property types, and simulated buyer sentiment. Ideal for investors and developers to spot 
    emerging opportunities and understand market buzz.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Purpose")
    st.markdown("""
    <div class='card'>
    • Spot high-activity neighborhoods quickly<br>
    • Understand buyer sentiment per locality<br>
    • Identify trending property types<br>
    • Inform strategic investments
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Capabilities")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='card'>
        <b>Technical</b><br>
        • Heatmaps & scatter maps<br>
        • Trend detection on property types<br>
        • Sentiment scoring per locality<br>
        • Interactive charts with hover info
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='card'>
        <b>Business</b><br>
        • Market buzz insights<br>
        • Identify emerging zones<br>
        • Investment prioritization<br>
        • Portfolio allocation support
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Most Active Localities</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Top Trending Property Type</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Average Price per SqFt</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Avg Buyer Sentiment</div>", unsafe_allow_html=True)

# ==========================================================
# TAB 2 - APPLICATION
# ==========================================================
with tab2:
    st.markdown("### Step 1: Load Dataset")
    df = None

    mode = st.radio(
        "Select Option:",
        ["Default Dataset", "Upload CSV"],
        horizontal=True
    )

    # Default Dataset
    if mode == "Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        try:
            df = pd.read_csv(URL)
            st.success("Default dataset loaded successfully.")
        except Exception as e:
            st.error(f"Could not load dataset: {e}")

    if mode == "Upload CSV":
        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)
            st.success("Dataset uploaded.")

    if df is None:
        st.stop()

    if not all(col in df.columns for col in REQUIRED_COLS):
        st.error(f"Dataset must contain columns: {REQUIRED_COLS}")
        st.stop()

    df = df.dropna()
    df["Listing_Date"] = pd.to_datetime(df["Listing_Date"])

    # ==========================================================
    # Filters
    # ==========================================================
    st.markdown("### Step 2: Dashboard Filters")
    f1, f2 = st.columns(2)
    with f1:
        city = st.multiselect("City", df["City"].unique())
    with f2:
        ptype = st.multiselect("Property Type", df["Property_Type"].unique())

    filt = df.copy()
    if city:
        filt = filt[filt["City"].isin(city)]
    if ptype:
        filt = filt[filt["Property_Type"].isin(ptype)]

    st.markdown("### Data Preview")
    st.dataframe(filt.head(), use_container_width=True)

    # ==========================================================
    # Step 3: Compute Metrics
    # ==========================================================
    # Simulate buyer sentiment as random for demo (0 to 1)
    np.random.seed(42)
    filt["Buyer_Sentiment"] = np.random.rand(len(filt))

    st.markdown("### Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    most_active_locality = filt.groupby("Locality")["Listing_Date"].count().idxmax()
    top_property_type = filt["Property_Type"].mode()[0]
    avg_price_sqft = (filt["Price"] / filt["Area_sqft"]).mean()
    avg_sentiment = filt["Buyer_Sentiment"].mean()

    k1.metric("Most Active Locality", most_active_locality)
    k2.metric("Top Property Type", top_property_type)
    k3.metric("Avg Price per SqFt", f"₹ {avg_price_sqft:,.0f}")
    k4.metric("Avg Buyer Sentiment", f"{avg_sentiment:.2f}")

    # ==========================================================
    # Charts
    # ==========================================================
    # Purpose & Quick Tip Dropdowns
    def chart_dropdown(title, purpose, quick_tip, chart_func):
        with st.expander(f"{title}"):
            st.markdown(f"**Purpose:** {purpose}")
            st.markdown(f"**Quick Tip:** {quick_tip}")
            chart_func()

    # Chart 1: Listings Heatmap by Locality
    def chart1():
        heatmap_data = filt.groupby("Locality").size().reset_index(name="Listings_Count")
        fig = px.bar(heatmap_data, x="Locality", y="Listings_Count", color="Listings_Count",
                     color_continuous_scale=px.colors.sequential.Oranges, text="Listings_Count")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    chart_dropdown(
        "Listings Activity by Locality",
        "Shows which neighborhoods have the highest listing activity over time.",
        "High listing counts indicate emerging hotspots for investment.",
        chart1
    )

    # Chart 2: Avg Price per SqFt by Property Type
    def chart2():
        price_data = filt.groupby("Property_Type")["Price"].mean().reset_index()
        fig = px.bar(price_data, x="Property_Type", y="Price", color="Price",
                     color_continuous_scale=px.colors.sequential.Viridis, text="Price")
        fig.update_traces(texttemplate="₹ %{text:,.0f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    chart_dropdown(
        "Average Price per Property Type",
        "Displays average property price per type for market evaluation.",
        "Compare property types to identify premium vs. affordable segments.",
        chart2
    )

    # Chart 3: Buyer Sentiment Map
    def chart3():
        filt["Sentiment_Normalized"] = (filt["Buyer_Sentiment"] - filt["Buyer_Sentiment"].min()) / (
            filt["Buyer_Sentiment"].max() - filt["Buyer_Sentiment"].min())
        fig = px.scatter_mapbox(
            filt, lat="Latitude", lon="Longitude", size="Price", color="Sentiment_Normalized",
            hover_name="Locality", hover_data=["City","Property_Type","Price","Buyer_Sentiment"],
            color_continuous_scale=px.colors.sequential.RdYlGn, size_max=15, zoom=10
        )
        fig.update_layout(mapbox_style="open-street-map",
                          coloraxis_colorbar=dict(title="Buyer Sentiment"),
                          margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

    chart_dropdown(
        "Buyer Sentiment Hotspot Map",
        "Visualizes areas with higher buyer interest/sentiment for investments.",
        "Green areas indicate positive sentiment, red areas negative.",
        chart3
    )

    # Chart 4: Top Localities by Price and Sentiment
    def chart4():
        top_loc = filt.groupby("Locality").agg(
            Avg_Price=("Price", "mean"),
            Avg_Sentiment=("Buyer_Sentiment", "mean")
        ).sort_values("Avg_Sentiment", ascending=False).head(10).reset_index()
        fig = px.scatter(top_loc, x="Avg_Price", y="Avg_Sentiment", size="Avg_Price", color="Avg_Sentiment",
                         hover_name="Locality", color_continuous_scale=px.colors.sequential.Plasma,
                         size_max=20)
        st.plotly_chart(fig, use_container_width=True)

    chart_dropdown(
        "Top Localities by Price & Sentiment",
        "Identifies neighborhoods with high price and positive buyer sentiment.",
        "High price + high sentiment = premium investment opportunity.",
        chart4
    )

    # ==========================================================
    # Download filtered dataset
    # ==========================================================
    st.markdown("### Download Filtered Dataset")
    csv = filt.to_csv(index=False)
    st.download_button("Download Filtered Data", csv, "filtered_market_buzz.csv", "text/csv")
