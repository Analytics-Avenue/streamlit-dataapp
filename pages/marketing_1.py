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
<div style="display: flex; align-items: center; margin-bottom:20px;">
    <img src="{logo_url}" width="60" style="margin-right:14px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:900;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:900;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# ADVANCED GLOBAL CSS (FINAL CLEAN VERSION)
# ---------------------------------------------------------
st.markdown("""
<style>

* { font-family: 'Inter', sans-serif; }

/* GLOBAL TEXT */
body, [class*="css"] {
    color:#000 !important;
    font-size:17px;
}

/* MAIN HEADER */
.big-header {
    font-size: 36px !important;
    font-weight: 700 !important;
    color:#000 !important;
    margin-bottom:12px;
    animation: fadeIn 1s ease;
}

/* Fade-in Animation */
@keyframes fadeIn {
    from {opacity:0; transform:translateY(10px);}
    to {opacity:1; transform:translateY(0);}
}

/* SECTION TITLES (used in ALL tabs) */
.section-title {
    font-size: 24px !important;
    font-weight: 600 !important;
    margin-top:32px;
    margin-bottom:14px;
    color:#000 !important;
    position:relative;
    animation: fadeIn 0.8s ease;
}

.section-title:after {
    content:"";
    position:absolute;
    bottom:-5px;
    left:0;
    height:2px;
    width:0%;
    background:#064b86;
    transition: width 0.35s ease;
}
.section-title:hover:after { width:40%; }

/* CARD (Pure black text except KPIs/Variables) */
.card {
    background:#ffffff;
    padding:24px;
    border-radius:14px;
    border:1px solid #e6e6e6;
    font-size:16.5px;
    color:#000 !important;
    font-weight:500;
    box-shadow:0 3px 14px rgba(0,0,0,0.08);
    transition: all 0.25s ease;
    animation: fadeIn 1s ease;
}
.card:hover {
    transform:translateY(-4px);
    box-shadow:0 12px 25px rgba(6,75,134,0.18);
    border-color:#064b86;
}

/* KPI CARDS — blue text only */
.kpi {
    background:white;
    padding:22px;
    border-radius:14px;
    border:1px solid #e2e2e2;
    font-size:20px !important;
    font-weight:600 !important;
    text-align:center;
    color:#064b86 !important;
    box-shadow:0 3px 14px rgba(0,0,0,0.07);
    transition:0.25s ease;
    animation: fadeIn 0.9s ease;
}
.kpi:hover {
    transform:translateY(-4px);
    box-shadow:0 13px 26px rgba(6,75,134,0.20);
    border-color:#064b86;
}

/* VARIABLE BOXES — blue text only */
.variable-box {
    padding:20px;
    border-radius:14px;
    background:white;
    border:1px solid #e5e5e5;
    box-shadow:0 2px 10px rgba(0,0,0,0.10);
    transition:0.25s ease;
    text-align:center;
    font-size:17.5px !important;
    font-weight:500 !important;
    color:#064b86 !important;
    margin-bottom:14px;
    animation: fadeIn 1s ease;
}
.variable-box:hover {
    transform:translateY(-5px);
    box-shadow:0 12px 22px rgba(6,75,134,0.18);
    border-color:#064b86;
}

/* TABLE */
.dataframe th {
    background:#064b86 !important;
    color:#fff !important;
    padding:11px !important;
    font-size:15.5px !important;
}
.dataframe td {
    font-size:15.5px !important;
    color:#000 !important;
    padding:9px !important;
    border-bottom:1px solid #efefef !important;
}
.dataframe tbody tr:hover { background:#f4f9ff !important; }

/* BUTTONS */
.stButton>button,
.stDownloadButton>button {
    background:#064b86 !important;
    color:white !important;
    border:none;
    padding:10px 22px;
    border-radius:8px !important;
    font-size:15.5px !important;
    font-weight:600 !important;
    transition:0.25s ease;
}
.stButton>button:hover,
.stDownloadButton>button:hover {
    transform:translateY(-3px);
    background:#0a6eb3 !important;
}

/* PAGE FADE-IN */
.block-container { animation: fadeIn 0.5s ease; }

</style>
""", unsafe_allow_html=True)


st.markdown("<div class='big-header'>Marketing Campaign Performance Analyzer</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Required Columns / Mapper
# ---------------------------------------------------------
REQUIRED_COLS = [
    "Campaign","Channel","Date","Impressions","Clicks","Leads","Conversions","Spend"
]

FB_MAP = {
    "Campaign": ["Campaign name","campaign_name"],
    "Channel": ["Page Name","page_name"],
    "Date": ["Date","Day"],
    "Impressions": ["Impressions","impressions"],
    "Clicks": ["Link clicks","clicks"],
    "Leads": ["Results","leads"],
    "Conversions": ["Conversions","Website conversions"],
    "Spend": ["Amount spent (INR)","Spend"]
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
# Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# ---------------------------------------------------------
# TAB 1 – OVERVIEW
# ---------------------------------------------------------
with tab1:

    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
    <b>Purpose:</b><br><br>
    This analyzer helps simplify data-driven marketing performance tracking.
    It brings together all major metrics—impressions, clicks, leads, conversions, and spend—
    allowing you to identify what drives ROI and what drains your budget.
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class='card'>
        • Full-funnel analytics from impression to conversion<br>
        • Creative, geo & audience performance<br>
        • Cost efficiency benchmarks<br>
        • Campaign anomaly detection<br>
        • Funnel drop-off insights<br>
        • Dataset export for BI dashboards
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class='card'>
        • Reduce wasted ad spend<br>
        • Improve marketing ROI<br>
        • Build predictable campaign performance<br>
        • Strengthen budget allocation decisions<br>
        • Improve conversion quality<br>
        • Build scalable marketing analytics systems
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Key Performance Indicators</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Total Impressions</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Total Clicks</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Total Leads</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Total Spend</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Who Should Use This & Why</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
    Made for: marketing teams, founders, performance marketers & analysts.<br><br>
    Why it matters:<br>
    It consolidates fragmented signals across platforms and helps identify what truly drives revenue growth.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2 – IMPORTANT ATTRIBUTES
# ---------------------------------------------------------
with tab2:

    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    data_dict = {
        "Campaign": "Name of the marketing campaign.",
        "Channel": "Source of traffic (Facebook, Google, etc.)",
        "Date": "Date of activity.",
        "Impressions": "Total ad views.",
        "Clicks": "Total user clicks.",
        "Leads": "Interested users.",
        "Conversions": "Completed expected action.",
        "Spend": "Total advertising spend."
    }

    df_dict = pd.DataFrame([{"Attribute": k, "Description": v} for k, v in data_dict.items()])

    st.dataframe(df_dict, use_container_width=True)

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

    st.markdown('<div class="section-title">Step 1: Load Dataset</div>', unsafe_allow_html=True)

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

    # Validate required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    df = df.dropna(subset=["Campaign","Channel"])

    st.markdown('<div class="section-title">Filters</div>', unsafe_allow_html=True)

    campaign = st.multiselect("Campaign", df["Campaign"].unique())
    channel = st.multiselect("Channel", df["Channel"].unique())

    filt = df.copy()
    if campaign:
        filt = filt[filt["Campaign"].isin(campaign)]
    if channel:
        filt = filt[filt["Channel"].isin(channel)]

    st.markdown('<div class="section-title">Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(filt.head(), use_container_width=True)

    # KPIs
    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)

    def inr(x): return f"₹{x:,.2f}"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Impressions", f"{int(filt['Impressions'].sum()):,}")
    k2.metric("Total Clicks", f"{int(filt['Clicks'].sum()):,}")
    k3.metric("Total Leads", f"{int(filt['Leads'].sum()):,}")
    k4.metric("Total Spend", inr(filt['Spend'].sum()))

    # Charts
    st.markdown('<div class="section-title">Campaign-wise Clicks</div>', unsafe_allow_html=True)
    st.plotly_chart(px.bar(filt, x="Campaign", y="Clicks", color="Campaign", text="Clicks"))

    st.markdown('<div class="section-title">Channel-wise Leads</div>', unsafe_allow_html=True)
    st.plotly_chart(px.pie(filt, names="Channel", values="Leads"))

    st.markdown('<div class="section-title">Spend vs Conversions</div>', unsafe_allow_html=True)
    st.plotly_chart(px.scatter(
        filt, x="Spend", y="Conversions",
        size="Impressions", color="Channel", hover_data=["Campaign"]
    ))

    st.download_button("Download Filtered Dataset", filt.to_csv(index=False), "marketing_filtered.csv")

