import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Marketing Campaign Performance Analyzer", layout="wide")

# -------------------------------
# HIDE SIDEBAR
# -------------------------------
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
.big-header {font-size: 40px; font-weight: 900;
background: linear-gradient(90deg,#FF6B6B,#FFD93D);
-webkit-background-clip:text; -webkit-text-fill-color:transparent;}
.card {background:#fff;border-radius:15px;padding:20px;margin-bottom:15px;
box-shadow:0 4px 20px rgba(0,0,0,0.08);}
.metric-card {background:#eef4ff;padding:15px;border-radius:8px;text-align:center;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# REQUIRED COLUMNS
# -------------------------------
REQUIRED_COLS = [
    "Campaign", "Channel", "Date", "Impressions", "Clicks", "Leads", "Conversions", "Spend"
]

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<div class='big-header'>Marketing Campaign Performance Analyzer</div>", unsafe_allow_html=True)

# -------------------------------
# TABS
# -------------------------------
tab1, tab2 = st.tabs(["Overview", "Application"])

# -------------------------------
# TAB 1: OVERVIEW
# -------------------------------
with tab1:
    st.markdown("### Overview")
    st.markdown("<div class='card'>This app tracks marketing campaign performance across multiple channels, analyzing lead generation, conversion, and ROI metrics.</div>", unsafe_allow_html=True)
    
    st.markdown("### Purpose")
    st.markdown("<div class='card'>• Evaluate campaign efficiency by channel<br>• Track impressions, clicks, and conversions<br>• Calculate ROI per campaign<br>• Identify high-performing campaigns for scaling</div>", unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Total Impressions</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Total Clicks</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Total Leads</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Average Conversion Rate</div>", unsafe_allow_html=True)

# -------------------------------
# TAB 2: APPLICATION
# -------------------------------
with tab2:
    st.markdown("### Step 1: Load Dataset")
    df = None

    mode = st.radio(
        "Select Dataset Option:",
        ["Default Dataset", "Upload CSV", "Upload CSV + Column Mapping"],
        horizontal=True
    )

    # -------------------------------
    # DEFAULT DATASET
    # -------------------------------
    if mode == "Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/Marketing_Analytics.csv"
        try:
            df = pd.read_csv(URL)
        except Exception as e:
            st.error(f"Failed to load default dataset: {e}")

    # -------------------------------
    # UPLOAD CSV
    # -------------------------------
    elif mode == "Upload CSV":
        st.markdown("#### Download Sample CSV for Reference")
        try:
            sample_df = pd.read_csv(URL).head(5)
            sample_csv = sample_df.to_csv(index=False)
            st.download_button("Download Sample CSV", sample_csv, "sample_marketing.csv", "text/csv")
        except Exception as e:
            st.info(f"Sample CSV unavailable: {e}")
        
        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)

    # -------------------------------
    # UPLOAD CSV + COLUMN MAPPING
    # -------------------------------
    elif mode == "Upload CSV + Column Mapping":
        st.markdown("#### Download Sample CSV for Reference")
        try:
            sample_df = pd.read_csv(URL).head(5)
            sample_csv = sample_df.to_csv(index=False)
            st.download_button("Download Sample CSV", sample_csv, "sample_marketing.csv", "text/csv")
        except Exception as e:
            st.info(f"Sample CSV unavailable: {e}")

        file = st.file_uploader("Upload dataset", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.write("Uploaded Data Preview", raw.head())
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

    # -------------------------------
    # VALIDATION
    # -------------------------------
    if df is None:
        st.warning("Please select or upload a dataset to continue.")
        st.stop()

    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        st.error(f"The following required columns are missing: {missing_cols}")
        st.stop()

    df = df.dropna(subset=REQUIRED_COLS)

    # -------------------------------
    # FILTERS
    # -------------------------------
    channels = st.multiselect("Channel", df["Channel"].unique())
    campaigns = st.multiselect("Campaign", df["Campaign"].unique())

    filt = df.copy()
    if channels: filt = filt[filt["Channel"].isin(channels)]
    if campaigns: filt = filt[filt["Campaign"].isin(campaigns)]

    st.markdown("### Data Preview")
    st.dataframe(filt.head(), use_container_width=True)

    # -------------------------------
    # KPIs
    # -------------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Impressions", filt["Impressions"].sum())
    k2.metric("Total Clicks", filt["Clicks"].sum())
    k3.metric("Total Leads", filt["Leads"].sum())
    conv_rate = filt["Conversions"].sum() / (filt["Clicks"].sum() + 1e-6)
    k4.metric("Average Conversion Rate", f"{conv_rate*100:.2f}%")

    # -------------------------------
    # PURPOSE & QUICK TIP
    # -------------------------------
    with st.expander("Purpose & Quick Tip"):
        st.markdown("**Purpose:** Evaluate marketing campaigns by impressions, clicks, leads, and conversions to optimize ROI.")
        st.markdown("**Quick Tip:** Focus on campaigns with high conversion rates and low spend for efficiency.")

    # -------------------------------
    # CHARTS
    # -------------------------------
    st.markdown("### Clicks vs Impressions by Campaign")
    fig1 = px.scatter(filt, x="Impressions", y="Clicks", color="Campaign", size="Leads", hover_data=["Channel","Conversions"])
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### Conversion Rate by Channel")
    channel_df = filt.groupby("Channel")[["Clicks","Conversions"]].sum().reset_index()
    channel_df["Conversion_Rate"] = channel_df["Conversions"] / (channel_df["Clicks"] + 1e-6)
    fig2 = px.bar(channel_df, x="Channel", y="Conversion_Rate", color="Channel", text="Conversion_Rate", color_discrete_sequence=px.colors.qualitative.Bold)
    fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Spend vs Leads")
    fig3 = px.scatter(filt, x="Spend", y="Leads", color="Channel", size="Conversions", hover_data=["Campaign"])
    st.plotly_chart(fig3, use_container_width=True)

    # -------------------------------
    # DOWNLOAD
    # -------------------------------
    csv = filt.to_csv(index=False)
    st.download_button("Download Filtered Dataset", csv, "marketing_filtered.csv", "text/csv")
