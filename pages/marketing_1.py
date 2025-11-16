import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO

st.set_page_config(page_title="Marketing Analytics Dashboard", layout="wide")

# -------------------------------
# STYLING
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

st.markdown("<div class='big-header'>Marketing Analytics Dashboard</div>", unsafe_allow_html=True)

# -------------------------------
# REQUIRED COLUMNS
# -------------------------------
REQUIRED_COLS = ['Campaign','Channel','Date','Impressions','Clicks','Leads','Conversions','Spend']

# -------------------------------
# TABS
# -------------------------------
tab1, tab2 = st.tabs(["Overview","Application"])

# -------------------------------
# TAB 1: OVERVIEW
# -------------------------------
with tab1:
    st.markdown("### Overview")
    st.markdown("<div class='card'>This app tracks marketing campaign performance across channels, helping measure engagement, conversions, and ROI.</div>", unsafe_allow_html=True)
    st.markdown("### Purpose")
    st.markdown("<div class='card'>• Measure campaign effectiveness<br>• Identify top-performing channels<br>• Track conversions and ROI<br>• Support budget allocation and optimization</div>", unsafe_allow_html=True)
    st.markdown("### KPIs")
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Total Impressions</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Total Clicks</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Total Leads</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Total Spend</div>", unsafe_allow_html=True)

# -------------------------------
# TAB 2: APPLICATION
# -------------------------------
with tab2:
    st.markdown("### Step 1: Load Dataset")
    df = None

    mode = st.radio("Select Dataset Option:",
                    ["Default Dataset","Upload CSV","Upload CSV + Column Mapping"],
                    horizontal=True)

    # -------------------------------
    # DEFAULT DATASET
    # -------------------------------
    if mode == "Default Dataset":
    URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/Marketing_Analytics.csv"
        try:
            df = pd.read_csv(URL)
            df.columns = df.columns.str.strip()  # remove extra spaces
            # Auto-map columns
            fb_map = {
                "Campaign": "Campaign name",
                "Channel": "Page Name",
                "Date": "Day",
                "Impressions": "Impressions",
                "Clicks": "Link clicks",
                "Leads": "Results",
                "Conversions": "Results",
                "Spend": "Amount spent (INR)"
            }
            if all(v in df.columns for v in fb_map.values()):
                df = df.rename(columns={v:k for k,v in fb_map.items()})
            st.success("Default dataset loaded and auto-mapped successfully.")
        except Exception as e:
            st.error(f"Failed to load default dataset: {e}")


    # -------------------------------
    # UPLOAD CSV
    # -------------------------------
    elif mode == "Upload CSV":
        file = st.file_uploader("Upload your CSV file", type=["csv"])
        if file:
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            st.success("File uploaded successfully.")

            # Auto-map Facebook Ads dataset if applicable
            fb_map = {
                "Campaign": "Campaign name",
                "Channel": "Page Name",
                "Date": "Day",
                "Impressions": "Impressions",
                "Clicks": "Link clicks",
                "Leads": "Results",
                "Conversions": "Results",
                "Spend": "Amount spent (INR)"
            }
            if all(v in df.columns for v in fb_map.values()):
                df = df.rename(columns={v:k for k,v in fb_map.items()})
                st.info("Auto-mapped Facebook Ads dataset successfully.")

            # Provide sample CSV download
            sample_csv = df.head(5).to_csv(index=False)
            st.download_button("Download Sample CSV (5 rows)", sample_csv, "sample_marketing.csv", "text/csv")

    # -------------------------------
    # UPLOAD CSV + COLUMN MAPPING
    # -------------------------------
    elif mode == "Upload CSV + Column Mapping":
        file = st.file_uploader("Upload your CSV file", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            raw.columns = raw.columns.str.strip()
            st.write("Uploaded Data Preview", raw.head())
            mapping = {}
            for col in REQUIRED_COLS:
                mapping[col] = st.selectbox(f"Map your column to: {col}", options=["-- Select --"] + list(raw.columns))
            if st.button("Apply Mapping"):
                missing = [col for col, mapped in mapping.items() if mapped=="-- Select --"]
                if missing:
                    st.error(f"Please map all required columns: {missing}")
                else:
                    df = raw.rename(columns=mapping)
                    st.success("Column mapping applied successfully.")

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
    campaign = st.multiselect("Campaign", df["Campaign"].unique())
    channel = st.multiselect("Channel", df["Channel"].unique())

    filt = df.copy()
    if campaign: filt = filt[filt["Campaign"].isin(campaign)]
    if channel: filt = filt[filt["Channel"].isin(channel)]

    st.markdown("### Data Preview")
    st.dataframe(filt.head(), use_container_width=True)

    # -------------------------------
    # KPIs
    # -------------------------------
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Impressions", int(filt["Impressions"].sum()))
    k2.metric("Total Clicks", int(filt["Clicks"].sum()))
    k3.metric("Total Leads", int(filt["Leads"].sum()))
    k4.metric("Total Spend (INR)", round(filt["Spend"].sum(),2))

    # -------------------------------
    # CHARTS
    # -------------------------------
    st.markdown("### Campaign-wise Clicks")
    fig1 = px.bar(filt, x="Campaign", y="Clicks", color="Campaign", text="Clicks",
                  color_discrete_sequence=px.colors.qualitative.Bold)
    fig1.update_traces(texttemplate="%{text}", textposition="outside")
    st.plotly_chart(fig1,use_container_width=True)

    st.markdown("### Channel-wise Leads")
    fig2 = px.pie(filt, names="Channel", values="Leads",
                  color="Channel", color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig2,use_container_width=True)

    st.markdown("### Spend vs Conversions")
    fig3 = px.scatter(filt, x="Spend", y="Conversions", size="Impressions", color="Channel",
                      hover_data=["Campaign"], size_max=30, color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig3,use_container_width=True)

    # -------------------------------
    # DOWNLOAD FILTERED DATA
    # -------------------------------
    csv = filt.to_csv(index=False)
    st.download_button("Download Filtered Dataset", csv, "marketing_filtered.csv", "text/csv")
