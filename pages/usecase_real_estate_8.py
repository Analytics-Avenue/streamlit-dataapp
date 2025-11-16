import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Real Estate Buyer Sentiment Analyzer", layout="wide")

# Hide sidebar
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
""", unsafe_allow_html=True)

REQUIRED_COLS = [
    "City", "Property_Type", "Price", "Latitude", "Longitude", "Buyer_Sentiment"
]

# Main header
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

# Tabs
tab1, tab2 = st.tabs(["Overview", "Application"])

with tab1:
    st.markdown("### Overview")
    st.markdown("<div class='card'>This app measures buyer sentiment across properties and locations for investment insights.</div>", unsafe_allow_html=True)
    st.markdown("### Purpose")
    st.markdown("<div class='card'>• Monitor buyer sentiment for properties<br>• Identify city-wise and property-type trends<br>• Highlight potential investment hotspots<br>• Support data-driven marketing & pricing decisions</div>", unsafe_allow_html=True)
    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>High Sentiment Properties</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Top Cities by Sentiment</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Top Property Types</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Average Sentiment Score</div>", unsafe_allow_html=True)

# ================= APPLICATION =================
with tab2:
    st.markdown("### Step 1: Load Dataset")
    df = None

    mode = st.radio(
        "Select Option:",
        ["Default Dataset", "Upload CSV", "Upload CSV + Column Mapping"],
        horizontal=True
    )

    # 1. Default Dataset
    if mode == "Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate.csv"
        try:
            df = pd.read_csv(URL)
        except Exception as e:
            st.error(f"Could not load default dataset: {e}")

    # 2. Upload CSV
    elif mode == "Upload CSV":
        st.markdown("#### Download Sample CSV for Reference")
        try:
            # Load default dataset
            sample_df = df.head(5)  # Take first 5 rows
            sample_csv = sample_df.to_csv(index=False)
            st.download_button("Download Sample CSV", sample_csv, "sample_dataset.csv", "text/csv")
        except Exception as e:
            st.info(f"Sample CSV unavailable: {e}")
    
        # Upload actual CSV
        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)

    # 3. Upload CSV + Column Mapping
    elif mode == "Upload CSV + Column Mapping":
        file = st.file_uploader("Upload dataset", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.write("Uploaded Data", raw.head())
            mapping = {}
            for col in REQUIRED_COLS:
                mapping[col] = st.selectbox(f"Map your column to: {col}", options=["-- Select --"] + list(raw.columns))
            if st.button("Apply Mapping"):
                missing = [col for col, mapped in mapping.items() if mapped == "-- Select --"]
                if missing:
                    st.error(f"Please map all required columns: {missing}")
                else:
                    df = raw.rename(columns=mapping)
                    st.success("Mapping applied successfully.")

    if df is None:
        st.stop()

    # Ensure required columns exist
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        st.error(f"The following required columns are missing in your dataset: {missing_cols}")
        st.stop()

    # Ensure numeric types for sentiment & price
    df["Buyer_Sentiment"] = pd.to_numeric(df["Buyer_Sentiment"], errors='coerce')
    df["Price"] = pd.to_numeric(df["Price"], errors='coerce')
    df = df.dropna(subset=REQUIRED_COLS)

    # ================= FILTERS =================
    st.markdown("### Step 2: Filters")
    city = st.multiselect("City", df["City"].unique())
    ptype = st.multiselect("Property Type", df["Property_Type"].unique())

    filt = df.copy()
    if city: filt = filt[filt["City"].isin(city)]
    if ptype: filt = filt[filt["Property_Type"].isin(ptype)]

    st.markdown("### Data Preview")
    st.dataframe(filt.head(), use_container_width=True)

    # ================= KPIs =================
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("High Sentiment Properties", len(filt[filt["Buyer_Sentiment"] > 0.7]))
    k2.metric("Top Cities by Sentiment", filt.groupby("City")["Buyer_Sentiment"].mean().sort_values(ascending=False).head(1).index[0])
    k3.metric("Top Property Types", filt.groupby("Property_Type")["Buyer_Sentiment"].mean().sort_values(ascending=False).head(1).index[0])
    k4.metric("Average Sentiment Score", f"{filt['Buyer_Sentiment'].mean():.2f}")

    # ================= PURPOSE & QUICK TIP =================
    with st.expander("Purpose & Quick Tip for Charts"):
        st.markdown("**Purpose:** Track buyer sentiment across cities and property types to guide marketing and investment decisions.")
        st.markdown("**Quick Tip:** Focus on properties and areas with sentiment >0.7 for high demand opportunities.")

    # ================= CHARTS =================
    st.markdown("### Buyer Sentiment Distribution by Property Type")
    fig1 = px.histogram(filt, x="Buyer_Sentiment", nbins=20, color="Property_Type", marginal="box", color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### City-wise Average Buyer Sentiment")
    city_avg = filt.groupby("City")["Buyer_Sentiment"].mean().reset_index()
    fig2 = px.bar(city_avg, x="City", y="Buyer_Sentiment", color="City", text="Buyer_Sentiment", color_discrete_sequence=px.colors.qualitative.Bold)
    fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Buyer Sentiment Hotspot Map")
    # Normalize sentiment
    sent_min, sent_max = filt["Buyer_Sentiment"].min(), filt["Buyer_Sentiment"].max()
    if sent_max - sent_min > 0:
        filt["Sentiment_Norm"] = (filt["Buyer_Sentiment"] - sent_min) / (sent_max - sent_min)
    else:
        filt["Sentiment_Norm"] = 0.5

    fig3 = px.scatter_mapbox(
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
    fig3.update_layout(mapbox_style="open-street-map", coloraxis_colorbar=dict(title="Sentiment Score"), margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig3, use_container_width=True)

    # ================= DOWNLOAD =================
    csv = filt.to_csv(index=False)
    st.download_button("Download Filtered Dataset", csv, "buyer_sentiment_filtered.csv", "text/csv")
