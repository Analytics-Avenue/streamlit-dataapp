import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Real Estate Buyer Sentiment Analyzer", layout="wide")

# --------------------------
# Hide default sidebar
# --------------------------
hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# --------------------------
# Required Columns
# --------------------------
REQUIRED_COLS = [
    "City", "Property_Type", "Price", "Latitude", "Longitude", "Buyer_Sentiment"
]

# --------------------------
# PAGE SETTINGS + CSS
# --------------------------
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

# --------------------------
# HEADER
# --------------------------
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
    This application provides insights into buyer sentiment across properties, cities, and property types.
    Investors, developers, and real estate agencies can identify regions with high buyer positivity and prioritize 
    marketing and sales strategies accordingly.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Purpose")
    st.markdown("""
    <div class='card'>
    • Analyze buyer sentiment distribution across cities and property types.<br>
    • Highlight properties and regions with high positive sentiment.<br>
    • Support investment and marketing prioritization.<br>
    • Understand market perception trends over time.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Capabilities")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='card'>
        <b>Technical</b><br>
        • Interactive sentiment heatmaps<br>
        • City & property type segmentation<br>
        • Filterable dashboards<br>
        • Buyer sentiment scoring
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='card'>
        <b>Business</b><br>
        • Identify high-opportunity regions<br>
        • Optimize property marketing<br>
        • Prioritize sales outreach<br>
        • Data-driven investor guidance
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>High Sentiment Properties</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Top Cities by Positivity</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Top Property Types</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Avg Sentiment Score</div>", unsafe_allow_html=True)

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
            # Generate fake Buyer_Sentiment column
            np.random.seed(42)
            df["Buyer_Sentiment"] = np.random.choice(["Positive", "Neutral", "Negative"], size=len(df))
            st.success("Default dataset loaded successfully.")
            st.write(df.head())
        except Exception as e:
            st.error(f"Could not load dataset: {e}")

    # 2. UPLOAD CSV
    if mode == "Upload CSV":
        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)
            if "Buyer_Sentiment" not in df.columns:
                np.random.seed(42)
                df["Buyer_Sentiment"] = np.random.choice(["Positive", "Neutral", "Negative"], size=len(df))
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
                    # Ensure Buyer_Sentiment exists
                    if "Buyer_Sentiment" not in df.columns:
                        np.random.seed(42)
                        df["Buyer_Sentiment"] = np.random.choice(["Positive", "Neutral", "Negative"], size=len(df))
                    st.success("Mapping applied successfully.")
                    st.dataframe(df.head())

    if df is None:
        st.stop()

    # Drop rows with essential missing values
    LAT_COL, LON_COL, PRICE_COL, SENT_COL = "Latitude", "Longitude", "Price", "Buyer_Sentiment"
    df = df.dropna(subset=[LAT_COL, LON_COL, PRICE_COL, SENT_COL])

    # Map sentiment to numeric for calculations
    sentiment_map = {"Positive": 0.9, "Neutral": 0.5, "Negative": 0.2}
    df["Sentiment_Score"] = df[SENT_COL].map(sentiment_map)

    # ==========================================================
    # FILTERS
    # ==========================================================
    st.markdown("### Step 2: Filters")
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
    # KPIs
    # ==========================================================
    st.markdown("### Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("High Sentiment Properties", len(filt[filt["Sentiment_Score"] > 0.7]))
    k2.metric("Top Cities by Positivity", filt.groupby("City")["Sentiment_Score"].mean().sort_values(ascending=False).head(1).index[0])
    k3.metric("Top Property Types", filt.groupby("Property_Type")["Sentiment_Score"].mean().sort_values(ascending=False).head(1).index[0])
    k4.metric("Average Sentiment Score", f"{filt['Sentiment_Score'].mean():.2f}")

    # ==========================================================
    # CHARTS
    # ==========================================================
    def chart_dropdown(title, purpose, quick_tip, chart_func):
        with st.expander(f"{title} ▼", expanded=True):
            st.markdown(f"**Purpose:** {purpose}")
            st.markdown(f"**Quick Tip:** {quick_tip}")
            chart_func()

    # ---------------- Chart 1: Sentiment Distribution ----------------
    def chart1():
        dist = filt["Buyer_Sentiment"].value_counts().reset_index()
        dist.columns = ["Sentiment", "Count"]
        fig = px.pie(dist, names="Sentiment", values="Count", color="Sentiment",
                     color_discrete_map={"Positive":"green", "Neutral":"orange", "Negative":"red"})
        st.plotly_chart(fig, use_container_width=True)

    chart_dropdown(
        "Buyer Sentiment Distribution",
        "Understand the overall sentiment trend across all properties.",
        "Positive sentiment often indicates higher buyer interest.",
        chart1
    )

    # ---------------- Chart 2: City-wise Avg Sentiment ----------------
    def chart2():
        city_sent = filt.groupby("City")["Sentiment_Score"].mean().reset_index()
        fig = px.bar(city_sent, x="City", y="Sentiment_Score", text="Sentiment_Score",
                     color="Sentiment_Score", color_continuous_scale=px.colors.sequential.Teal)
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    chart_dropdown(
        "City-wise Avg Sentiment Score",
        "Compare buyer sentiment across different cities.",
        "Target marketing in cities with high positive sentiment.",
        chart2
    )

    # ---------------- Chart 3: Buyer Sentiment Hotspot Map ----------------
    def chart3():
        fig = px.scatter_mapbox(
            filt,
            lat=LAT_COL,
            lon=LON_COL,
            size=PRICE_COL,
            color="Sentiment_Score",
            hover_name="Property_Type",
            hover_data=["City", "Price", "Buyer_Sentiment"],
            color_continuous_scale=px.colors.sequential.RdYlGn,
            size_max=15,
            zoom=10
        )
        fig.update_layout(mapbox_style="open-street-map",
                          coloraxis_colorbar=dict(title="Sentiment Score"),
                          margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

    chart_dropdown(
        "Buyer Sentiment Hotspot Map",
        "Visualize geographic regions with high buyer positivity.",
        "Focus investments or marketing in hotspots with high sentiment.",
        chart3
    )

    # -------------------------
    # Top Properties by Sentiment
    # -------------------------
    st.markdown("### Top Properties by Sentiment Score")
    top_props = filt.sort_values("Sentiment_Score", ascending=False).head(10)
    st.dataframe(top_props[["City","Property_Type","Price","Buyer_Sentiment","Sentiment_Score"]])
    csv = top_props.to_csv(index=False)
    st.download_button("Download Top Properties", csv, "top_properties_sentiment.csv", "text/csv")
