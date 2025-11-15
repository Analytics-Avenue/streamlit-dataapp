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
    "City", "Property_Type", "Price", "Latitude", "Longitude", "Buyer_Sentiment"
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
    This application measures buyer sentiment across properties and locations, helping 
    investors and agents understand demand trends, identify high-interest areas, and 
    optimize property marketing strategies.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Purpose")
    st.markdown("""
    <div class='card'>
    • Monitor buyer sentiment for properties<br>
    • Identify city-wise and property-type trends<br>
    • Highlight potential investment hotspots<br>
    • Support data-driven marketing & pricing decisions
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Capabilities")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='card'>
        <b>Technical</b><br>
        • Sentiment scoring<br>
        • Interactive sentiment hotspot maps<br>
        • City & property segmentation dashboards
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='card'>
        <b>Business</b><br>
        • Identify high-demand properties<br>
        • Track agent engagement effectiveness<br>
        • Optimize marketing and sales campaigns
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>High Sentiment Properties</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Top Cities by Sentiment</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Top Property Types</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Average Sentiment Score</div>", unsafe_allow_html=True)

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

    # ----------------------------------------------------------
    # DEFAULT DATASET
    # ----------------------------------------------------------
    if mode == "Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        try:
            df = pd.read_csv(URL)
            st.success("Default dataset loaded successfully.")
        except Exception as e:
            st.error(f"Could not load dataset: {e}")

    # ----------------------------------------------------------
    # UPLOAD CSV
    # ----------------------------------------------------------
    if mode == "Upload CSV":
        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)
            st.success("Dataset uploaded.")

    # ----------------------------------------------------------
    # UPLOAD + COLUMN MAPPING
    # ----------------------------------------------------------
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
    # CHECK DATA LOADED
    # ----------------------------------------------------------
    if df is None:
        st.stop()

    if not all(col in df.columns for col in REQUIRED_COLS):
        st.warning("Some columns missing. Faker dataset will be generated.")
        # Generate faker data
        n = 200
        df = pd.DataFrame({
            "City": np.random.choice(["Chennai", "Bangalore", "Mumbai", "Delhi"], n),
            "Property_Type": np.random.choice(["Apartment", "Villa", "Studio"], n),
            "Price": np.random.randint(3000000, 50000000, n),
            "Latitude": np.random.uniform(12, 28, n),
            "Longitude": np.random.uniform(77, 77.5, n),
            "Buyer_Sentiment": np.round(np.random.uniform(0, 1, n), 2)
        })

    df = df.dropna(subset=REQUIRED_COLS)

    # ==========================================================
    # FILTERS
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
    # KPIs
    # ==========================================================
    st.markdown("### Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("High Sentiment Properties", len(filt[filt["Buyer_Sentiment"] > 0.7]))
    k2.metric("Top Cities by Sentiment", filt.groupby("City")["Buyer_Sentiment"].mean().sort_values(ascending=False).head(1).index[0])
    k3.metric("Top Property Types", filt.groupby("Property_Type")["Buyer_Sentiment"].mean().sort_values(ascending=False).head(1).index[0])
    k4.metric("Average Sentiment Score", f"{filt['Buyer_Sentiment'].mean():.2f}")

    # ==========================================================
    # Charts with Purpose + Quick Tip dropdown
    # ==========================================================
    def chart_dropdown(title, purpose, tip, chart_func):
        with st.expander(title):
            st.markdown(f"**Purpose:** {purpose}")
            st.markdown(f"**Quick Tip:** {tip}")
            chart_func()

    # ----------------------------
    # Chart 1 – Sentiment Distribution
    # ----------------------------
    def chart1():
        fig = px.histogram(
            filt,
            x="Buyer_Sentiment",
            nbins=20,
            color="Property_Type",
            marginal="box",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)

    chart_dropdown(
        "Buyer Sentiment Distribution by Property Type",
        "Understand the distribution of buyer sentiment scores across different property types.",
        "High sentiment properties usually indicate strong market interest.",
        chart1
    )

    # ----------------------------
    # Chart 2 – City-wise Average Sentiment
    # ----------------------------
    def chart2():
        city_avg = filt.groupby("City")["Buyer_Sentiment"].mean().reset_index()
        fig = px.bar(
            city_avg,
            x="City",
            y="Buyer_Sentiment",
            color="City",
            text="Buyer_Sentiment",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    chart_dropdown(
        "City-wise Average Buyer Sentiment",
        "Compare average sentiment across cities to identify high-interest markets.",
        "Focus on cities with sentiment above 0.7 for potential investment hotspots.",
        chart2
    )

    # ----------------------------
    # Chart 3 – Buyer Sentiment Hotspot Map
    # ----------------------------
    def chart3():
        filt["Sentiment_Norm"] = (filt["Buyer_Sentiment"] - filt["Buyer_Sentiment"].min()) / (
            filt["Buyer_Sentiment"].max() - filt["Buyer_Sentiment"].min()
        )
        fig = px.scatter_mapbox(
            filt,
            lat="Latitude",
            lon="Longitude",
            size="Price",
            color="Sentiment_Norm",
            hover_name="Property_Type",
            hover_data=["City", "Price", "Buyer_Sentiment"],
            color_continuous_scale=px.colors.diverging.RdYlGn,
            size_max=15,
            zoom=10
        )
        fig.update_layout(
            mapbox_style="open-street-map",
            coloraxis_colorbar=dict(title="Sentiment Score"),
            margin={"r":0,"t":0,"l":0,"b":0}
        )
        st.plotly_chart(fig, use_container_width=True)

    chart_dropdown(
        "Buyer Sentiment Hotspot Map",
        "Visualize sentiment geographically to identify property hotspots.",
        "Green areas indicate positive sentiment; red areas indicate negative sentiment.",
        chart3
    )

    # ----------------------------
    # Download filtered data
    # ----------------------------
    csv = filt.to_csv(index=False)
    st.download_button("Download Filtered Dataset", csv, "buyer_sentiment_filtered.csv", "text/csv")
