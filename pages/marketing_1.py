import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------------------------------
# Base Config
# ---------------------------------------------------------
st.set_page_config(page_title="Marketing Campaign Performance Analyzer", layout="wide")

# Hide sidebar navigation
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Company Header
# ---------------------------------------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

st.markdown(f"""
<div style="display: flex; align-items: center;">
    <img src="{logo_url}" width="60" style="margin-right:12px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:900;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:900;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# ADVANCED GLOBAL CSS
# ---------------------------------------------------------
st.markdown("""
<style>

/* ------------------------------------------------ */
/* GLOBAL FONT + COLOR SYSTEM */
/* ------------------------------------------------ */
body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 16px !important;
    color:#000 !important;
}

.big-header {
    font-size: 38px !important;
    font-weight: 800 !important;
    color:#000 !important;
    margin-bottom:8px;
}

/* ------------------------------------------------ */
/* SECTION TITLES with ANIMATED UNDERLINE */
/* ------------------------------------------------ */
.section-title {
    font-size: 24px !important;
    font-weight: 700 !important;
    margin-top:30px;
    margin-bottom:12px;
    color:#000 !important;
    position:relative;
}
.section-title:after {
    content:"";
    position:absolute;
    bottom:-4px;
    left:0;
    height:3px;
    width:0%;
    background:#064b86;
    transition:width 0.35s ease;
}
.section-title:hover:after {
    width:45%;
}

/* ------------------------------------------------ */
/* MODERN CARD EFFECT */
/* ------------------------------------------------ */
.card {
    background:#fff;
    padding:22px;
    border-radius:14px;
    border:1px solid #d9d9d9;
    font-size:16px !important;
    color:#064b86 !important;
    font-weight:500;
    box-shadow:0 3px 14px rgba(0,0,0,0.10);
    transition: all 0.25s ease;
}
.card:hover {
    transform:translateY(-5px);
    box-shadow:0 12px 26px rgba(0,0,0,0.20);
    border-color:#064b86;
}

/* ------------------------------------------------ */
/* KPI CARDS */
/* ------------------------------------------------ */
.kpi {
    background:#ffffff;
    padding:26px;
    border-radius:14px;
    border:1px solid #d9d9d9;
    font-size:20px !important;
    font-weight:700 !important;
    text-align:center;
    color:#064b86 !important;
    box-shadow:0 3px 14px rgba(0,0,0,0.10);
    transition:0.25s ease;
}
.kpi:hover {
    transform:translateY(-4px);
    box-shadow:0 12px 26px rgba(0,0,0,0.20);
    border-color:#064b86;
}

.small { font-size:13px; color:#5b5b5b !important; }

/* ------------------------------------------------ */
/* VARIABLE CARDS (Independent vs Dependent) */
/* ------------------------------------------------ */
.variable-box {
    padding:18px;
    border-radius:14px;
    background:white;
    border:1px solid #dcdcdc;
    box-shadow:0 3px 12px rgba(0,0,0,0.12);
    transition:0.25s ease;
    text-align:center;
    font-size:17px !important;
    font-weight:600 !important;
    color:#064b86 !important;
    margin-bottom:14px;
}
.variable-box:hover {
    transform:translateY(-5px);
    box-shadow:0 12px 26px rgba(0,0,0,0.18);
    border-color:#064b86;
}

/* ------------------------------------------------ */
/* TABLE STYLING */
/* ------------------------------------------------ */
.dataframe th {
    background:#000 !important;
    color:#fff !important;
    padding:10px !important;
    font-size:15px !important;
}
.dataframe td {
    font-size:15px !important;
    color:#000 !important;
    padding:9px !important;
    border-bottom:1px solid #e6e6e6 !important;
}
.dataframe tbody tr:hover {
    background:#f1f6fa !important;
}

/* ------------------------------------------------ */
/* BUTTONS */
/* ------------------------------------------------ */
.stButton>button,
.stDownloadButton>button {
    background:#064b86 !important;
    color:white !important;
    border:none;
    padding:9px 20px;
    border-radius:8px !important;
    font-size:15px !important;
    font-weight:600 !important;
    transition:0.25s ease;
}
.stButton>button:hover,
.stDownloadButton>button:hover {
    transform:translateY(-3px);
    background:#0a6eb3 !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-header'>Marketing Campaign Performance Analyzer</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# REQUIRED COLUMNS + AUTO MAPPER
# ---------------------------------------------------------
REQUIRED_COLS = [
    'Campaign','Channel','Date','Impressions','Clicks','Leads','Conversions','Spend'
]

FB_MAP = {
    "Campaign": ["Campaign name", "campaign_name"],
    "Channel": ["Page Name", "page_name"],
    "Date": ["Date", "Day"],
    "Impressions": ["Impressions", "impressions"],
    "Clicks": ["Link clicks", "clicks"],
    "Leads": ["Results", "leads"],
    "Conversions": ["Conversions", "Website conversions"],
    "Spend": ["Amount spent (INR)", "Spend"]
}

def auto_map_columns(df):
    rename_dict = {}
    for req, arr in FB_MAP.items():
        for col in df.columns:
            if col.strip() in arr:
                rename_dict[col] = req
                break
    return df.rename(columns=rename_dict)

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# ---------------------------------------------------------
# TAB 1 – OVERVIEW
# ---------------------------------------------------------
with tab1:
    st.markdown("### Overview")
    st.markdown("""
    <div class='card'>
    <b>Purpose:</b> Modern marketing runs on data, not guesswork. This analyzer helps you understand how every campaign, channel, and ad interaction contributes to your outcomes. 
    It brings all performance signals into one place so you can quickly identify what's working, what’s wasting money, and where to focus your efforts for maximum ROI. 
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown("#### Capabilities")
        st.markdown("""
        <div class='card'>
        • Unify performance across channels like Facebook, Google, Instagram, and Email<br>
        • Understand full-funnel behavior from impressions to conversions<br>
        • Identify winning creatives, audiences, and high-value customer actions<br>
        • Compare campaign ROI with cost-efficient benchmarking<br>
        • Instantly spot spikes, drops, or unusual campaign behavior<br>
        • Export ready-to-use dashboards and datasets for deeper BI work<br>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("#### Business Impact")
        st.markdown("""
        <div class='card'>
        • Reduce wasted ad spend by pinpointing ineffective campaigns<br>
        • Improve lead quality by understanding channel-to-outcome patterns<br>
        • Strengthen planning with reliable performance forecasting<br>
        • Allocate budgets with confidence based on actual cost-per-result trends<br>
        • Improve conversion efficiency with actionable funnel insights<br>
        • Build a scalable marketing measurement system that grows with your business<br>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Total Impressions</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Total Clicks</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Total Leads</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Total Spend</div>", unsafe_allow_html=True)

    st.markdown("### Who Should Use This & Why")
    st.markdown("""
    <div class='card'>
    <b>Who is this for?</b> Marketing teams, founders, growth managers, data analysts, campaign specialists, and anyone responsible for scaling revenue efficiently.<br><br>
    <b>Why it matters:</b> With increasing ad costs and fragmented platforms, businesses need a single source of truth to decide where to invest, what to optimize, and how to grow sustainably.
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# TAB 2 – IMPORTANT ATTRIBUTES
# ---------------------------------------------------------
with tab2:

    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    data_dict = {
        "Campaign": "Marketing campaign name.",
        "Channel": "Source platform (FB, Google, Email).",
        "Date": "Date of activity.",
        "Impressions": "Times ad was shown.",
        "Clicks": "Total ad clicks.",
        "Leads": "Interested users.",
        "Conversions": "Completed actions.",
        "Spend": "Amount spent."
    }

    df_dict = pd.DataFrame(
        [{"Attribute": k, "Description": v} for k, v in data_dict.items()]
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.dataframe(df_dict, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    indep = ["Campaign","Channel","Date","Impressions","Clicks","Spend"]
    dep = ["Leads","Conversions"]

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        for v in indep:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        for v in dep:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 3 – APPLICATION
# ---------------------------------------------------------
with tab3:

    st.markdown("### Step 1: Load Dataset")

    df = None
    mode = st.radio("Select Dataset Option:", ["Default Dataset","Upload CSV","Upload CSV + Column Mapping"], horizontal=True)

    if mode == "Default Dataset":
        try:
            URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/marketing.csv"
            df = pd.read_csv(URL)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            for col in REQUIRED_COLS:
                if col not in df.columns:
                    df[col] = 0
            st.success("Default dataset loaded!")
        except:
            st.error("Failed to load dataset.")

    elif mode == "Upload CSV":
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            for col in REQUIRED_COLS:
                if col not in df.columns:
                    df[col] = 0
            st.success("File uploaded successfully.")

    else:
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            raw.columns = raw.columns.str.strip()
            st.write("Preview:", raw.head())

            mapping = {}
            for col in REQUIRED_COLS:
                mapping[col] = st.selectbox(f"Map → {col}", ["-- Select --"] + list(raw.columns))

            if st.button("Apply Mapping"):
                missing = [k for k,v in mapping.items() if v=="-- Select --"]
                if missing:
                    st.error(f"Missing mappings: {missing}")
                else:
                    df = raw.rename(columns=mapping)
                    st.success("Column mapping applied!")

    if df is None:
        st.warning("Please load a dataset.")
        st.stop()

    # Validate
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    df = df.dropna(subset=["Campaign","Channel"])

    # Filters
    campaign = st.multiselect("Campaign", df["Campaign"].unique())
    channel = st.multiselect("Channel", df["Channel"].unique())

    filt = df.copy()
    if campaign:
        filt = filt[filt["Campaign"].isin(campaign)]
    if channel:
        filt = filt[filt["Channel"].isin(channel)]

    st.markdown("### Data Preview")
    st.dataframe(filt.head(), use_container_width=True)

    # KPIs
    def inr(x): return f"₹{x:,.2f}"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Impressions", f"{filt['Impressions'].sum():,}")
    k2.metric("Total Clicks", f"{filt['Clicks'].sum():,}")
    k3.metric("Total Leads", f"{filt['Leads'].sum():,}")
    k4.metric("Total Spend", inr(filt['Spend'].sum()))

    # Charts
    st.markdown("### Campaign-wise Clicks")
    st.plotly_chart(px.bar(filt, x="Campaign", y="Clicks", color="Campaign", text="Clicks"))

    st.markdown("### Channel-wise Leads")
    st.plotly_chart(px.pie(filt, names="Channel", values="Leads"))

    st.markdown("### Spend vs Conversions")
    st.plotly_chart(px.scatter(
        filt, x="Spend", y="Conversions",
        size="Impressions", color="Channel", hover_data=["Campaign"]
    ))

    st.download_button("Download Filtered Dataset", filt.to_csv(index=False), "marketing_filtered.csv")

