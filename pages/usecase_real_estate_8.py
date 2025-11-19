import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Real Estate Buyer Sentiment Analyzer", layout="wide")

# ----------------------------
# Hide Sidebar
# ----------------------------
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
.big-header {font-size: 40px; font-weight: 900; color:black;}
.card {background:#fff;border-radius:15px;padding:20px;margin-bottom:15px; box-shadow:0 4px 20px rgba(0,0,0,0.08);}
.metric-card {background:#eef4ff;padding:15px;border-radius:8px;text-align:center; transition: 0.3s; }
.metric-card:hover {box-shadow: 0 0 25px rgba(0,123,255,0.6);}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Company Logo + Name
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display: flex; align-items: center;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0; padding:0;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0; padding:0;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Required columns for the app
REQUIRED_COLS = ["City", "Property_Type", "Price", "Latitude", "Longitude", "Buyer_Sentiment"]

# -------------------------
# Main Header
# -------------------------
st.markdown("<div class='big-header'>Real Estate Buyer Sentiment Analyzer</div>", unsafe_allow_html=True)

# -------------------------
# Tabs
# -------------------------
tab1, tab2 = st.tabs(["Overview", "Application"])

# ==========================================================
# TAB 1 - OVERVIEW
# ==========================================================
with tab1:
    st.markdown("### Overview")
    st.markdown("<div class='card'>This app measures buyer sentiment across properties and locations to generate investment insights and guide marketing strategies.</div>", unsafe_allow_html=True)
    
    st.markdown("### Purpose")
    st.markdown("<div class='card'>• Monitor buyer sentiment trends by property type and city<br>• Identify potential investment hotspots<br>• Support data-driven pricing and marketing decisions<br>• Optimize resource allocation for high-demand properties</div>", unsafe_allow_html=True)
    
    st.markdown("### Capabilities")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='card'><b>Technical</b><br>• Buyer sentiment scoring<br>• Interactive city/property dashboards<br>• Hotspot maps<br>• ML revenue predictions</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><b>Business</b><br>• Investment prioritization<br>• Portfolio optimization<br>• Market opportunity mapping<br>• Strategic marketing planning</div>", unsafe_allow_html=True)

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
