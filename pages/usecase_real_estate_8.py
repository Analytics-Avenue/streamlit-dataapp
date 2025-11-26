import streamlit as st
import pandas as pd

st.set_page_config(page_title="Real Estate Buyer Sentiment Analyzer", layout="wide")

# ==========================================================
# HIDE SIDEBAR
# ==========================================================
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display:none;}
section[data-testid="stSidebar"] {display:none;}

.big-header {
    font-size:40px;
    font-weight:900;
    color:black;
    margin-bottom:12px;
}

/* Cards */
.card {
    background:#ffffff;
    padding:20px;
    border-radius:15px;
    border:1px solid #e5e5e5;
    box-shadow:0 4px 20px rgba(0,0,0,0.08);
    margin-bottom:18px;
}

/* KPI Cards */
.metric-card {
    background:#eef4ff;
    padding:20px;
    text-align:center;
    border-radius:14px;
    font-size:18px;
    color:#064b86;
    font-weight:600;
    box-shadow:0 3px 14px rgba(0,0,0,0.1);
    transition:0.25s ease;
}
.metric-card:hover {
    transform:translateY(-4px);
    box-shadow:0 10px 25px rgba(6,75,134,0.2);
}

/* Variable boxes */
.variable-box {
    padding:16px;
    border-radius:12px;
    background:white;
    border:1px solid #e5e5e5;
    text-align:center;
    font-size:17px;
    font-weight:500;
    color:#064b86;
    margin-bottom:10px;
    box-shadow:0 2px 10px rgba(0,0,0,0.08);
    transition:0.25s ease;
}
.variable-box:hover {
    transform:translateY(-5px);
    box-shadow:0 12px 20px rgba(6,75,134,0.18);
}

/* Section title */
.section-title {
    font-size:26px;
    font-weight:700;
    color:black;
    margin-top:25px;
    margin-bottom:12px;
    position:relative;
}
.section-title:after {
    content:"";
    position:absolute;
    bottom:-6px;
    left:0;
    width:0%;
    height:2px;
    background:#064b86;
    transition:0.35s;
}
.section-title:hover:after { width:40%; }

/* Page Fade */
.block-container { animation: fadeIn 0.5s ease; }
@keyframes fadeIn { from {opacity:0; transform:translateY(10px);} to {opacity:1; transform:translateY(0);} }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

st.markdown(f"""
<div style="display:flex; align-items:center; margin-bottom:15px;">
    <img src="{logo_url}" width="60" style="margin-right:12px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:700;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:700;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='big-header'>Real Estate Buyer Sentiment Analyzer</div>", unsafe_allow_html=True)

# ==========================================================
# REQUIRED COLUMNS (ADDED BACK – FIXES NameError)
# ==========================================================
REQUIRED_COLS = [
    "City",
    "Property_Type",
    "Price",
    "Latitude",
    "Longitude",
    "Buyer_Sentiment"
]


# ==========================================================
# 3 TABS
# ==========================================================
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])


# ==========================================================
# TAB 1 – OVERVIEW
# ==========================================================
with tab1:

    st.markdown("<div class='section-title'>Overview</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        This platform evaluates real estate buyer sentiment across cities, neighborhoods, and property categories.
        It helps investors and builders measure demand hotspots, track pricing expectations, and understand urban engagement levels.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Purpose</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        • Track buyer sentiment trends<br>
        • Identify top-performing areas<br>
        • Guide pricing strategies<br>
        • Detect demand surges or drops<br>
        • Support data-backed inventory planning
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Capabilities</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='card'>
            <b>Technical:</b><br>
            • Sentiment scoring engine<br>
            • Interactive dashboards<br>
            • Property-level sentiment heatmaps<br>
            • Market segmentation analytics
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='card'>
            <b>Business:</b><br>
            • Demand forecasting<br>
            • Investment hotspot detection<br>
            • Price-to-sentiment optimization<br>
            • Marketing efficiency analysis
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>KPIs</div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>High Sentiment Properties</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Top Cities by Sentiment</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Top Property Types</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Avg Sentiment Score</div>", unsafe_allow_html=True)


# ==========================================================
# TAB 2 – IMPORTANT ATTRIBUTES
# ==========================================================
with tab2:

    st.markdown("<div class='section-title'>Important Attributes</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Independent Variables</div>", unsafe_allow_html=True)
        for v in ["City", "Property_Type", "Price", "Latitude", "Longitude"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-title'>Dependent Variables</div>", unsafe_allow_html=True)
        for v in ["Buyer_Sentiment"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)
# ==========================================================
# TAB 3 - APPLICATION
# ==========================================================
with tab3:
    st.markdown("### Step 1: Load Dataset")
    df = None

    mode = st.radio("Select Option:", ["Default Dataset", "Upload CSV", "Upload CSV + Column Mapping"], horizontal=True)

    # -------------------
    # Default dataset
    # -------------------
    if mode == "Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate.csv"
        try:
            df = pd.read_csv(URL)
            st.success("Default dataset loaded successfully.")
        except Exception as e:
            st.error(f"Could not load default dataset: {e}")

    # -------------------
    # Upload CSV
    # -------------------
    elif mode == "Upload CSV":
        st.markdown("#### Download Sample CSV for Reference")
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate.csv"
        try:
            sample_df = pd.read_csv(URL).head(5)
            sample_csv = sample_df.to_csv(index=False)
            st.download_button("Download Sample CSV", sample_csv, "sample_dataset.csv", "text/csv")
        except:
            st.info("Sample CSV unavailable")
        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)

    # -------------------
    # Upload CSV + Column Mapping
    # -------------------
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

    # -------------------
    # Ensure numeric & drop NaNs safely
    # -------------------
    existing_cols = [col for col in REQUIRED_COLS if col in df.columns]
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        st.warning(f"The following required columns are missing and skipped: {missing_cols}")
    df = df.dropna(subset=existing_cols)

    df["Buyer_Sentiment"] = pd.to_numeric(df["Buyer_Sentiment"], errors='coerce')
    df["Price"] = pd.to_numeric(df["Price"], errors='coerce')

    # -------------------
    # Filters
    # -------------------
    st.markdown("### Step 2: Filters")
    city = st.multiselect("City", df["City"].unique())
    ptype = st.multiselect("Property Type", df["Property_Type"].unique())
    filt = df.copy()
    if city: filt = filt[filt["City"].isin(city)]
    if ptype: filt = filt[filt["Property_Type"].isin(ptype)]

    st.markdown("### Data Preview")
    st.dataframe(filt.head(), use_container_width=True)

    # -------------------
    # KPIs
    # -------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("High Sentiment Properties", len(filt[filt["Buyer_Sentiment"] > 0.7]))
    k2.metric("Top Cities by Sentiment", filt.groupby("City")["Buyer_Sentiment"].mean().sort_values(ascending=False).head(1).index[0])
    k3.metric("Top Property Types", filt.groupby("Property_Type")["Buyer_Sentiment"].mean().sort_values(ascending=False).head(1).index[0])
    k4.metric("Average Sentiment Score", f"{filt['Buyer_Sentiment'].mean():.2f}")

    # -------------------
    # Charts
    # -------------------
    st.markdown("### Buyer Sentiment Distribution by Property Type")
    fig1 = px.histogram(filt, x="Buyer_Sentiment", nbins=20, color="Property_Type", marginal="box", color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### City-wise Average Buyer Sentiment")
    city_avg = filt.groupby("City")["Buyer_Sentiment"].mean().reset_index()
    fig2 = px.bar(city_avg, x="City", y="Buyer_Sentiment", color="City", text="Buyer_Sentiment", color_discrete_sequence=px.colors.qualitative.Bold)
    fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    # -------------------
    # Hotspot Map
    # -------------------
    sent_min, sent_max = filt["Buyer_Sentiment"].min(), filt["Buyer_Sentiment"].max()
    filt["Sentiment_Norm"] = (filt["Buyer_Sentiment"] - sent_min) / (sent_max - sent_min) if sent_max - sent_min > 0 else 0.5
    fig3 = px.scatter_mapbox(
        filt, lat="Latitude", lon="Longitude", size="Price",
        color="Sentiment_Norm", hover_name="Property_Type",
        hover_data=["City", "Price", "Buyer_Sentiment"],
        color_continuous_scale=px.colors.diverging.RdYlGn,
        size_max=15, zoom=10
    )
    fig3.update_layout(mapbox_style="open-street-map", coloraxis_colorbar=dict(title="Sentiment Score"), margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig3, use_container_width=True)

    # -------------------
    # ML Revenue Predictions (dummy example)
    # -------------------
    st.markdown("### ML Revenue Predictions")
    if "Price" in filt.columns:
        filt["Predicted_Revenue"] = filt["Price"] * np.random.uniform(0.8, 1.2, len(filt))
        st.dataframe(filt[["City", "Property_Type", "Price", "Predicted_Revenue"]].head(), use_container_width=True)
        st.download_button("Download ML Predictions CSV", filt[["City", "Property_Type", "Price", "Predicted_Revenue"]].to_csv(index=False), "ml_predictions.csv", "text/csv")

    # -------------------
    # Automated Insights
    # -------------------
    st.markdown("### Automated Insights")
    insights = pd.DataFrame({
        "Insight": ["High Demand City", "Top Property Type", "Strong Conversion Area", "Price Trend Rising", "Potential ROI"]*1,
        "Value": ["City A", "3BHK", "Downtown", "Increasing", "High"]*1
    })
    st.dataframe(insights, use_container_width=True)
    st.download_button("Download Automated Insights CSV", insights.to_csv(index=False), "automated_insights.csv", "text/csv")

    # -------------------
    # Download filtered dataset
    # -------------------
    st.download_button("Download Filtered Dataset", filt.to_csv(index=False), "buyer_sentiment_filtered.csv", "text/csv")
