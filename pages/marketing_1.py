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
# Global CSS
# ---------------------------------------------------------
st.markdown("""
<style>

.big-header {
    font-size: 36px;
    font-weight: 900;
    color: black;
}

/* Global font */
body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Standard card */
.card {
    padding:20px;
    border-radius:12px;
    background:#ffffff;
    border:1px solid #e6e6e6;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 8px 30px rgba(6,75,134,0.18);
    border-color:#064b86;
}

/* KPI card */
.kpi {
    padding:28px;
    border-radius:12px;
    background:#ffffff;
    border:1px solid #e6e6e6;
    text-align:center;
    font-weight:700;
    color:#064b86;
    font-size:20px;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.kpi:hover {
    transform: translateY(-6px);
    box-shadow: 0 8px 30px rgba(6,75,134,0.18);
    border-color:#064b86;
}

.small { color:#666; font-size:13px; }

/* REAL VARIABLE CARDS */
.variable-box {
    padding: 22px;
    background: white;
    border-radius: 14px;
    border: 1px solid #e0e6ed;
    box-shadow: 0 4px 14px rgba(6, 75, 134, 0.15);
    transition: 0.25s ease;
    text-align: center;
    font-size: 18px;
    font-weight: 700;
    color:#064b86;
    margin-bottom: 12px;
}
.variable-box:hover {
    transform: translateY(-6px);
    box-shadow: 0 10px 28px rgba(6, 75, 134, 0.25);
    border-color:#064b86;
}

/* Table styling */
.dataframe th {
    background: #064b86 !important;
    color: white !important;
    padding: 10px !important;
}
.dataframe td {
    padding: 8px !important;
    border-bottom: 1px solid #e6e6e6 !important;
}
.dataframe tbody tr:hover {
    background: #f3faff !important;
}

.section-title {
    font-size: 26px;
    font-weight: 800;
    margin-top: 20px;
    margin-bottom: 10px;
    color:#064b86;
}

</style>
""", unsafe_allow_html=True)


st.markdown("<div class='big-header'>Marketing Campaign Performance Analyzer</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Required Columns
# ---------------------------------------------------------
REQUIRED_COLS = [
    'Campaign','Channel','Date','Impressions',
    'Clicks','Leads','Conversions','Spend'
]

FB_MAP = {
    "Campaign": ["Campaign name", "campaign_name"],
    "Channel": ["Page Name", "page_name", "Channel"],
    "Date": ["Date", "Day"],
    "Impressions": ["Impressions", "impressions"],
    "Clicks": ["Link clicks", "clicks"],
    "Leads": ["Results", "leads"],
    "Conversions": ["Conversions", "Website conversions"],
    "Spend": ["Amount spent (INR)", "Spend"]
}

def auto_map_columns(df):
    rename_dict = {}
    for req, possible in FB_MAP.items():
        for col in df.columns:
            if col.strip() in possible:
                rename_dict[col] = req
                break
    return df.rename(columns=rename_dict)

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# ----------------------------------------------------------
# TAB 1 – Overview (Matching your Route Optimization Layout)
# ----------------------------------------------------------
with tab1:

    st.markdown("""
    <div style="text-align:left;">
      <h2 style="margin:0; padding:0;">Marketing Campaign Performance Analyzer</h2>
      <p style="margin-top:4px; color:#555">Track performance, optimize spend, and unlock smarter marketing decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Overview")
    st.markdown("""
    <div class='card'>
      <b>Purpose</b>: Optimize campaigns, reduce wasted spend, improve conversions, and understand what drives ROI across channels.
    </div>
    """, unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### Capabilities")
        st.markdown("""
        <div class='card'>
        • Track performance across channels<br>
        • Optimize ROI & reduce wasted spend<br>
        • Conversion funnel analytics<br>
        • Multi-channel cost efficiency scoring<br>
        • Dashboard-ready datasets<br>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown("#### Business Impact")
        st.markdown("""
        <div class='card'>
        • Boost conversions & reduce CPL<br>
        • Allocate budget intelligently<br>
        • Identify high-performing creatives/channels<br>
        • Improve forecasting & planning<br>
        • Strengthen overall marketing efficiency<br>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Total Impressions<br><span class='small'>(Dataset driven)</span></div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Total Clicks<br><span class='small'>(Dataset driven)</span></div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Total Leads<br><span class='small'>(Dataset driven)</span></div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Total Spend<br><span class='small'>(Dataset driven)</span></div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# TAB 2 – IMPORTANT ATTRIBUTES (NOW WITH REAL CARDS)
# ----------------------------------------------------------
with tab2:

    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    data_dict = {
        "Campaign": "Name of the marketing campaign.",
        "Channel": "Marketing channel (Facebook, Google, etc.)",
        "Date": "Date of activity.",
        "Impressions": "Number of times ad was shown.",
        "Clicks": "Number of ad clicks.",
        "Leads": "Users expressing interest.",
        "Conversions": "Completed desired actions.",
        "Spend": "Amount spent."
    }

    df_dict = pd.DataFrame(
        [{"Attribute": k, "Description": v} for k, v in data_dict.items()]
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.dataframe(df_dict, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # REAL variable cards
    indep = ["Campaign", "Channel", "Date", "Impressions", "Clicks", "Spend"]
    dep = ["Leads", "Conversions"]

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        for v in indep:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        for v in dep:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# TAB 3 – APPLICATION
# ----------------------------------------------------------
with tab3:

    st.markdown("### Step 1: Load Dataset")
    df = None

    mode = st.radio(
        "Select Dataset Option:",
        ["Default Dataset", "Upload CSV", "Upload CSV + Column Mapping"],
        horizontal=True,
    )

    if mode == "Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/marketing.csv"
        try:
            df = pd.read_csv(URL)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            for col in REQUIRED_COLS:
                if col not in df.columns:
                    df[col] = 0
            st.success("Default dataset loaded!")
        except:
            st.error("Failed to load default dataset.")

    elif mode == "Upload CSV":
        file = st.file_uploader("Upload your CSV", type=["csv"])
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
                missing = [k for k, v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error(f"Map all required columns: {missing}")
                else:
                    df = raw.rename(columns=mapping)
                    st.success("Mapping completed.")

    if df is None:
        st.warning("Please load a dataset.")
        st.stop()

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    df = df.dropna(subset=["Campaign", "Channel"])

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
    k4.metric("Total Spend", inr(filt["Spend"].sum()))

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
