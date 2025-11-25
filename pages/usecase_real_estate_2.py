import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

# =======================================================================================
# PAGE CONFIG
# =======================================================================================
st.set_page_config(page_title="Real Estate Demand Forecasting Lab", layout="wide", initial_sidebar_state="collapsed")

# Hide sidebar
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display:none;}
section[data-testid="stSidebar"] {display:none;}
</style>
""", unsafe_allow_html=True)

# =======================================================================================
# GLOBAL CONSTANTS
# =======================================================================================
BLUE = "#064b86"
BLACK = "#000000"
BASE_FONT_SIZE_PX = 17

REQUIRED_COLS = ["City", "Listing_Date", "Property_Type", "Price"]

AUTO_MAPS = {
    "City": ["city","location","area"],
    "Listing_Date": ["listing_date","date","posted_on","created_at"],
    "Property_Type": ["property_type","type","category"],
    "Price": ["price","amount","cost"]
}

# =======================================================================================
# HELPERS
# =======================================================================================
def auto_map_columns(df):
    rename = {}
    for req, patterns in AUTO_MAPS.items():
        for col in df.columns:
            c = col.lower().strip()
            for p in patterns:
                p = p.lower().strip()
                if p == c or p in c or c in p:
                    rename[col] = req
                    break
            if col in rename:
                break
    return df.rename(columns=rename)

def ensure_datetime(df, col):
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        except:
            pass
    return df

def download_df(df, filename, label="Download CSV"):
    if df is None or df.empty:
        return st.info("No data available.")
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label=label, data=b, file_name=filename, mime="text/csv")

def render_required_table(df):
    df2 = df.reset_index(drop=True)
    styled = df2.style.set_table_attributes('class="required-table"')
    html = styled.to_html().replace("<th></th>","").replace("<td></td>","")
    st.write(html, unsafe_allow_html=True)

# =======================================================================================
# GLOBAL CSS (Marketing Lab Standard)
# =======================================================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {{
    font-family: 'Inter', sans-serif;
    color: {BLACK};
    font-size: {BASE_FONT_SIZE_PX}px;
}}

.section-title {{
    font-size: 22px;
    font-weight: 600;
    margin: 16px 0 12px 0;
    color: {BLACK};
    position: relative;
    display: inline-block;
}}
.section-title:hover::after {{
    content:"";
    position:absolute;
    left:0;
    bottom:-6px;
    width:40%;
    height:3px;
    background:{BLUE};
    border-radius:2px;
}}

.card {{
    background:#ffffff;
    color:{BLACK};
    border-radius:12px;
    border:1px solid #e6e6e6;
    padding:18px;
    box-shadow:0 4px 14px rgba(0,0,0,0.06);
    transition:0.25s ease;
}}
.card:hover {{
    transform: translateY(-5px);
    border-color:{BLUE};
    box-shadow:0 10px 24px rgba(0,0,0,0.12);
}}

.kpi-card {{
    background:#ffffff;
    color:{BLUE};
    border-radius:12px;
    border:1px solid #e6e6e6;
    padding:16px;
    text-align:center;
    font-weight:600;
    box-shadow:0 4px 12px rgba(0,0,0,0.06);
}}
.kpi-card:hover {{
    transform:translateY(-5px);
    box-shadow:0 12px 26px rgba(6,75,134,0.2);
}}

.variable-box {{
    background:white;
    color:{BLUE};
    border-radius:12px;
    padding:14px;
    border:1px solid #e5e5e5;
    box-shadow:0 4px 12px rgba(0,0,0,0.06);
    text-align:center;
    margin-bottom:12px;
}}

.required-table {{
    border-collapse:collapse;
    width:100%;
}}
.required-table thead th {{
    border-bottom:2px solid #000;
    padding:10px;
}}
.required-table tbody td {{
    padding:10px;
    border-bottom:1px solid #eee;
}}
.required-table tbody tr:hover {{
    background:#f8f8f8;
}}

.stButton>button, .stDownloadButton>button {{
    background:{BLUE} !important;
    color:white !important;
    border:none;
    padding:10px 20px;
    border-radius:8px;
    font-weight:600;
}}
.stButton>button:hover, .stDownloadButton>button:hover {{
    transform:translateY(-3px);
    background:#0a6eb3 !important;
}}
</style>
""", unsafe_allow_html=True)

# =======================================================================================
# HEADER BLOCK
# =======================================================================================
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

st.markdown(f"""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
    <img src="{logo_url}" width="60">
    <div>
        <div style="font-size:32px; font-weight:700; color:{BLUE};">Analytics Avenue & Advanced Analytics</div>
        <div style="font-size:14px; color:{BLACK}; opacity:0.7;">Real Estate Demand Forecasting Lab</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =======================================================================================
# TABS
# =======================================================================================
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# =======================================================================================
# TAB 1: Overview
# =======================================================================================
with tab1:

    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        A complete forecasting engine for real estate demand analysis.  
        Tracks market cycles, price trends, inventory behavior, and predicts demand for the upcoming months.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        • Time-series demand tracking<br>
        • Property-type segmentation<br>
        • Price sensitivity analysis<br>
        • ML-based forecasting (6-month linear regression)<br>
        • Automated insights for cities & property types
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        • Predict market cycles early<br>
        • Prevent over/under inventory allocation<br>
        • Optimize pricing strategy<br>
        • Support strategic investment & expansion
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">High-Level KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi-card'>Avg Monthly Sales</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi-card'>Demand Growth</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi-card'>Top Cities</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi-card'>Price Sensitivity</div>", unsafe_allow_html=True)

# =======================================================================================
# TAB 2: IMPORTANT ATTRIBUTES
# =======================================================================================
with tab2:

    st.markdown('<div class="section-title">Required Column Dictionary</div>', unsafe_allow_html=True)

    dict_df = pd.DataFrame([
        {"Column": "City", "Description": "City of the property listing."},
        {"Column": "Listing_Date", "Description": "Date when property was listed."},
        {"Column": "Property_Type", "Description": "Type of property (Villa, Plot, Apartment, etc)."},
        {"Column": "Price", "Description": "Listed property price."},
    ])

    render_required_table(dict_df)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        for v in ["City","Listing_Date","Property_Type","Price"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        for v in ["Demand","Trend","Forecast"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# =======================================================================================
# TAB 3: APPLICATION
# =======================================================================================
with tab3:

    st.markdown('<div class="section-title">Step 1 — Load Dataset</div>', unsafe_allow_html=True)
    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)
    df = None

    # DEFAULT DATASET
    if mode == "Default dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        try:
            df = pd.read_csv(URL)
            df = auto_map_columns(df)
            st.success("Default dataset loaded.")
            render_required_table(df.head(3))
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    # UPLOAD CSV — AUTO DETECT
    elif mode == "Upload CSV":
        st.markdown("#### Sample CSV (optional)", unsafe_allow_html=True)
        SAMPLE_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        try:
            smp = pd.read_csv(SAMPLE_URL).head(5)
            st.download_button("Download Sample CSV", smp.to_csv(index=False), "sample_realestate.csv")
        except:
            pass

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            df = auto_map_columns(df)
            st.success("File uploaded & auto-mapped.")
            render_required_table(df.head(5))

    # MANUAL MAPPING
    else:
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded:
            raw = pd.read_csv(uploaded)
            st.markdown("Preview:", unsafe_allow_html=True)
            render_required_table(raw.head(5))

            mapping = {}
            options = ["-- Select --"] + list(raw.columns)

            for req in REQUIRED_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=options, key=f"map_{req}")

            if st.button("Apply Mapping"):
                missing = [k for k, v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Map all required fields: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v: k for k, v in mapping.items()})
                    st.success("Mapping applied.")
                    render_required_table(df.head(5))

    if df is None:
        st.stop()

    # CLEANING
    df = ensure_datetime(df, "Listing_Date")
    df = df.dropna(subset=["Listing_Date"])
    df["Date"] = df["Listing_Date"]
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)

    # FILTERS
    st.markdown('<div class="section-title">Step 2 — Filters</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    cities = sorted(df["City"].dropna().unique())
    types = sorted(df["Property_Type"].dropna().unique())

    with c1:
        sel_city = st.multiselect("City", options=cities, default=cities[:5])
    with c2:
        sel_type = st.multiselect("Property Type", options=types, default=types[:5])
    with c3:
        try:
            min_d = df["Date"].min().date()
            max_d = df["Date"].max().date()
            date_range = st.date_input("Date range", value=(min_d, max_d))
        except:
            date_range = st.date_input("Date range")

    filt = df.copy()
    if sel_city:
        filt = filt[filt["City"].isin(sel_city)]
    if sel_type:
        filt = filt[filt["Property_Type"].isin(sel_type)]

    try:
        if len(date_range) == 2:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            filt = filt[(filt["Date"] >= start) & (filt["Date"] <= end)]
    except:
        pass

    st.markdown("Filtered Preview", unsafe_allow_html=True)

    download_df(filt, "filtered_real_estate_data.csv", label="Download filtered dataset")

    # DEMAND TREND
    st.markdown('<div class="section-title">Monthly Demand Trend</div>', unsafe_allow_html=True)
    filt["Month"] = filt["Date"].dt.to_period("M").astype(str)
    trend = filt.groupby("Month").size().reset_index(name="Demand")
    if not trend.empty:
        fig = px.line(trend, x="Month", y="Demand", markers=True, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # PROPERTY TYPE DEMAND
    st.markdown('<div class="section-title">Demand by Property Type</div>', unsafe_allow_html=True)
    prop = filt.groupby("Property_Type").size().reset_index(name="Count")
    if not prop.empty:
        fig2 = px.bar(prop, x="Property_Type", y="Count", text="Count", template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

    # FORECASTING (6 months)
    st.markdown('<div class="section-title">6-Month Demand Forecast</div>', unsafe_allow_html=True)
    if len(trend) >= 3:
        trend["Index"] = np.arange(len(trend))
        model = LinearRegression().fit(trend[["Index"]], trend["Demand"])
        future_idx = np.arange(len(trend), len(trend)+6)
        preds = model.predict(future_idx.reshape(-1,1))

        fdf = pd.DataFrame({"Month":[f"Future {i+1}" for i in range(6)], "Forecast":preds})
        render_required_table(fdf)
        download_df(fdf, "demand_forecast.csv", label="Download forecast")

    # AUTOMATED INSIGHTS
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)
    try:
        insights = filt.groupby(["City","Property_Type"]).agg({
            "Price":["mean","max","min"]
        }).reset_index()
        insights.columns = ["City","Property_Type","Avg_Price","Max_Price","Min_Price"]
        render_required_table(insights)
        download_df(insights, "automated_insights.csv", label="Download insights")
    except:
        st.info("Not enough data for insights.")

# END OF FILE
