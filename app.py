import streamlit as st
import os

# -------------------------
# Streamlit setup
# -------------------------
st.set_page_config(page_title="Data Analytics Solutions", layout="wide")
st.markdown("""<style>[data-testid="stSidebarNav"]{display:none;}</style>""", unsafe_allow_html=True)

# -------------------------
# Company Logo + Name
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display: flex; align-items: center; margin-bottom:20px;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# GLOBAL CSS
# -------------------------
st.markdown("""
<style>

.card-box {
    border: 1px solid #c9d7f0;
    border-radius: 12px;
    padding: 15px;
    min-height: 530px;
    background: #ffffff;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    transition: all 0.25s ease-in-out;
    margin-bottom: 25px;
}

.card-box:hover {
    transform: translateY(-6px);
    box-shadow: 0 4px 18px rgba(0,0,0,0.18);
    border-color: #8bb3ff;
}

.card-box img {
    border-radius: 8px;
    outline: 1px solid #dde6ff;
    transition: 0.25s;
}

.card-box:hover img {
    outline-color: #8bb3ff;
    box-shadow: 0 0 8px rgba(140,170,255,0.4);
}

/* Tool Buttons */
.tool-btn {
    background: #eef4ff;
    border-radius: 6px;
    padding: 5px 9px;
    font-size: 12px;
    border: 1px solid #c6d7ff;
    text-align: center;
    font-weight: 600;
    margin: 3px;
    display: inline-block;
    transition: 0.2s;
}
.tool-btn:hover {
    background: #d9e7ff;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Directories
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# -------------------------
# Sector Data
# -------------------------
sector_overview = {
    "Marketing Analytics": """Marketing Analytics helps businesses understand customer behavior, optimize campaign performance, and maximize ROI.  
It brings together digital data, ad metrics, and sales signals to uncover real-time insights.  
It enables smarter decision-making through forecasting, segmentation, and attribution modelling.""",

    "Real Estate Analytics": """Real Estate Analytics supports valuation, forecasting, and project feasibility analysis.  
It combines pricing data, market demand, and geospatial insights to evaluate opportunities.  
It helps developers, investors, and buyers make smarter, data-backed housing decisions.""",

    "Health Care Analytics": """Health Care Analytics improves patient outcomes, reduces operational delays, and optimizes hospital workflows.  
It analyzes EMR data, patient flow, and treatment patterns to enhance system efficiency.  
It empowers hospitals to allocate staff, predict demand, and strengthen patient care quality."""
}

# Popular & practical tools only
sector_tools = {
    "Marketing Analytics": [
        "Python", "SQL", "Power BI", "Excel",
        "Google Analytics", "Meta Ads Data", "Machine Learning",
        "Pandas", "A/B Testing", "Segmentation"
    ],

    "Real Estate Analytics": [
        "Python", "SQL", "Power BI",
        "Excel", "Regression Models", "Time Series",
        "QGIS", "GeoPandas", "Price Prediction"
    ],

    "Health Care Analytics": [
        "Python", "R", "SQL", "Excel",
        "Power BI", "Forecasting Models",
        "Classification Models", "NLP"
    ]
}

# Thumbnail paths
home_thumbs = {
    "Marketing Analytics": os.path.join(ASSETS_DIR, "marketing_thumb.jpg"),
    "Real Estate Analytics": os.path.join(ASSETS_DIR, "real_estate_thumb.jpg"),
    "Health Care Analytics": os.path.join(ASSETS_DIR, "healthcare_thumb.jpg"),
}

# -------------------------
# Session State
# -------------------------
if "sector" not in st.session_state:
    st.session_state["sector"] = None

# ============================================================
# HOME PAGE
# ============================================================

if st.session_state["sector"] is None:

    st.title("Data Analytics Solutions")
    st.write("Choose a sector to explore:")

    sectors = list(sector_overview.keys())
    cols = st.columns(3)

    for col, sector in zip(cols, sectors):
        with col:
            st.markdown("<div class='card-box'>", unsafe_allow_html=True)

            # Thumbnail
            thumb_path = home_thumbs.get(sector)
            if os.path.exists(thumb_path):
                st.image(thumb_path, use_container_width=True)

            # Title
            st.markdown(
                f"<h3 style='color:#064b86; margin-top:10px;'>{sector}</h3>",
                unsafe_allow_html=True
            )

            # Detailed overview
            st.markdown(
                f"<p style='font-size:14.5px; color:#444; text-align:justify;'>{sector_overview[sector]}</p>",
                unsafe_allow_html=True
            )

            # Tools header
            st.markdown("<b>Tools & Tech:</b><br>", unsafe_allow_html=True)

            # Tools grid multiline
            tools_html = "".join([f"<span class='tool-btn'>{t}</span>" for t in sector_tools[sector]])
            st.markdown(tools_html, unsafe_allow_html=True)

            # Explore button
            if st.button(f"Explore {sector}", key=f"explore_{sector}"):
                st.session_state["sector"] = sector

            st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# SECTOR PAGE (Coming Soon Page)
# ============================================================

else:
    sector_name = st.session_state["sector"]
    st.header(f"{sector_name} – Use Cases Coming Soon")

    if st.button("Back to Home"):
        st.session_state["sector"] = None
