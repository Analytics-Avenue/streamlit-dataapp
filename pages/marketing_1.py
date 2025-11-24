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

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

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

/* Standard Card */
.card {
    padding:20px;
    border-radius:12px;
    background:#ffffff;
    border:1px solid #e6e6e6;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    text-align:left;
}
.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 8px 30px rgba(6,75,134,0.18);
    border-color:#064b86;
}

/* KPI Card */
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

.variable-card-list {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    transition: 0.3s ease;
    font-size: 16px;
}
.variable-card-list:hover {
    transform: translateY(-4px);
    box-shadow: 0 0 18px rgba(6, 75, 134, 0.45);
    border-color: #064b86;
}
.variable-card-list ul {
    margin: 0;
    padding-left: 18px;
}
.variable-card-list li {
    padding-bottom: 6px;
}

/* Variable Glow Cards */
.variable-card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    transition: 0.3s ease;
}
.variable-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 0 18px rgba(6, 75, 134, 0.45);
    border-color: #064b86;
}

/* DataFrame Styling */
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

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Title
# ---------------------------------------------------------
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
# Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# ----------------------------------------------------------
# TAB 1 — NEW OVERVIEW LAYOUT (MATCHING ROUTE OPTIMIZATION STYLE)
# ----------------------------------------------------------
with tab1:

    st.markdown("""
    <div style="text-align:left;">
      <h2 style="margin:0; padding:0;">Marketing Campaign Performance Analyzer</h2>
      <p style="margin-top:4px; color:#555">Track performance, optimize spend, and unlock smarter marketing decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    # Purpose
    st.markdown("### Overview")
    st.markdown("""
    <div class='card'>
      <b>Purpose</b>: Optimize campaigns, reduce wasted spend, improve conversions, and understand what drives ROI across channels.
    </div>
    """, unsafe_allow_html=True)

    # Two-column layout
    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### Capabilities")
        st.markdown("""
        <div class='card'>
        • Track campaign & channel-level performance<br>
        • Optimize ROI & reduce wasted spend<br>
        • Conversion funnel analytics<br>
        • Multi-channel cost efficiency scoring<br>
        • Automated dashboard-ready insights<br>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown("#### Business Impact")
        st.markdown("""
        <div class='card'>
        • Boost conversions & reduce CPL<br>
        • Allocate budget more intelligently<br>
        • Identify high-performing creatives<br>
        • Improve forecasting & planning<br>
        • Strengthen overall marketing effectiveness<br>
        </div>
        """, unsafe_allow_html=True)

    # KPI row
    st.markdown("#### KPIs")
    k1, k2, k3, k4 = st.columns(4)

    k1.markdown("<div class='kpi'>Total Impressions</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Total Clicks</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Total Leads</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Total Spend</div>", unsafe_allow_html=True)

    # Who should use this
    st.markdown("### Who should use this & How")
    st.markdown("""
    <div class='card'>
      <b>Who</b>: Marketing analysts, growth teams, brand managers, founders.<br><br>
      <b>How</b>:  
      1) Load your dataset.  
      2) Filter by campaign or channel.  
      3) Review KPIs & charts.  
      4) Use insights to reallocate spend smartly.  
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------
# TAB 2 — IMPORTANT ATTRIBUTES
# ----------------------------------------------------------
with tab2:

    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    data_dict = {
        "Campaign": "Name of the marketing campaign.",
        "Channel": "Marketing channel (Facebook, Google, Instagram, etc.)",
        "Date": "Activity date.",
        "Impressions": "Number of ad views.",
        "Clicks": "Number of clicks.",
        "Leads": "Interested users.",
        "Conversions": "Completed actions (sale/signup).",
        "Spend": "Amount spent."
    }

    df_dict = pd.DataFrame(
        [{"Attribute": k, "Description": v} for k, v in data_dict.items()]
    )

    st.markdown('<div class="dict-card">', unsafe_allow_html=True)
    st.dataframe(df_dict, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Variables LEFT & RIGHT (CARD VERSION — no tables)
    
    indep = ["Campaign", "Channel", "Date", "Impressions", "Clicks", "Spend"]
    dep = ["Leads", "Conversions"]
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        st.markdown('<div class="variable-card-list">', unsafe_allow_html=True)
        st.markdown("<ul>" + "".join([f"<li>{v}</li>" for v in indep]) + "</ul>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with c2:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        st.markdown('<div class="variable-card-list">', unsafe_allow_html=True)
        st.markdown("<ul>" + "".join([f"<li>{v}</li>" for v in dep]) + "</ul>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# TAB 3 — APPLICATION
# ----------------------------------------------------------
with tab3:

    st.markdown("### Step 1: Load Dataset")
    df = None

    mode = st.radio(
        "Select Dataset Option:",
        ["Default Dataset", "Upload CSV", "Upload CSV + Column Mapping"],
        horizontal=True,
    )

    # Default data
    if mode == "Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/marketing.csv"
        try:
            df = pd.read_csv(URL)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            for col in REQUIRED_COLS:
                if col not in df.columns:
                    df[col] = 0
            st.success("Default dataset loaded.")
        except:
            st.error("Failed to load default dataset.")

    # Upload CSV
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

    # Upload + Map
    else:
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            raw.columns = raw.columns.str.strip()
            st.write("Uploaded Data Preview:", raw.head())

            mapping = {}
            for col in REQUIRED_COLS:
                mapping[col] = st.selectbox(f"Map to: {col}", ["-- Select --"] + list(raw.columns))

            if st.button("Apply Mapping"):
                missing = [k for k, v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error(f"Map all required columns: {missing}")
                else:
                    df = raw.rename(columns=mapping)
                    st.success("Column mapping applied.")

    # Validation
    if df is None:
        st.warning("Load or upload a dataset.")
        st.stop()

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
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
