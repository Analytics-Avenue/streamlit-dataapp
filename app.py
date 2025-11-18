import streamlit as st
import os

# -------------------------
# Page Config
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
    border-radius: 14px;
    padding: 15px;
    background: #ffffff;
    transition: 0.25s ease-in-out;
    min-height: 5px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

/* Hover */
.card-box:hover {
    transform: translateY(-6px);
    box-shadow: 0 6px 22px rgba(0,0,0,0.18);
    border-color: #7fa8ff;
    background: #f9fbff;
}

/* Thumbnail glow */
.card-box img {
    border-radius: 8px;
    outline: 1px solid #dce6ff;
    transition: 0.25s ease-in-out;
}
.card-box:hover img {
    outline-color: #7fa8ff;
    box-shadow: 0px 0px 10px rgba(130,160,255,0.5);
    transform: scale(1.02);
}

/* Tool button */
.tool-btn {
    background: #eef4ff;
    border-radius: 6px;
    padding: 5px 9px;
    font-size: 12px;
    border: 1px solid #c6d7ff;
    margin: 3px;
    display: inline-block;
    font-weight: 600;
    transition: 0.2s;
}
.tool-btn:hover {
    background: #d9e7ff;
    transform: scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Sector Overview
# -------------------------
sector_overview = {
    "Marketing Analytics":
    """Analyze customer journeys, optimize ad spends, improve campaign ROAS, and track funnel drop-offs.
Use segmentation, forecasting, and attribution insights to supercharge marketing performance.
Gain a unified 360-degree view of customers across channels for smarter decisions.""",

    "Real Estate Analytics":
    """Understand locality demand, analyze pricing trends, compare property attributes, and forecast ROI.
Leverage geospatial intelligence and ML models to identify high-growth micro-markets.
Support investment decisions using rental yield and accurate price prediction analytics.""",

    "Health Care Analytics":
    """Improve patient flow, predict OPD/ER volumes, enhance doctor allocation, and reduce waiting times.
Use forecasting, classification, and EMR/EHR data to optimize hospital operations.
Boost care quality with real-time monitoring and clinical performance analytics."""
}

# -------------------------
# Sector Tools
# -------------------------
sector_tools = {
    "Marketing Analytics": [
        "Python", "SQL", "Excel", "Power BI", "Tableau",
        "Google Analytics 4", "Pandas", "NumPy", "Scikit-Learn",
        "A/B Testing", "Attribution Models", "Segmentation Models"
    ],

    "Real Estate Analytics": [
        "Python", "SQL", "Excel", "Power BI", "Tableau",
        "QGIS", "GeoPandas", "Google Maps API",
        "Regression Models", "Time Series", "Clustering",
        "Price Prediction", "Rental Yield Models"
    ],

    "Health Care Analytics": [
        "Python", "R", "SQL", "Excel", "Power BI", "Tableau",
        "EMR/EHR Data", "Time Series Forecasting", "Classification Models",
        "NLP", "Patient Flow Forecasting"
    ]
}

# -------------------------
# RAW Thumbnail URLs
# -------------------------
home_thumbs = {
    "Marketing Analytics": "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/assets/marketing_thumb.jpg",
    "Real Estate Analytics": "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/assets/real_estate_thumb.jpg",
    "Health Care Analytics": "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/assets/healthcare_thumb.jpg",
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

    sector_list = list(sector_overview.keys())
    rows = [sector_list[i:i+3] for i in range(0, len(sector_list), 3)]

    for row in rows:
        cols = st.columns(3)
        for col, sector in zip(cols, row):
            with col:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)

                # Thumbnail
                st.image(home_thumbs[sector], use_container_width=True)

                # Title
                st.markdown(
                    f"<h3 style='color:#064b86; margin-top:12px;'>{sector}</h3>",
                    unsafe_allow_html=True
                )

                # Overview
                st.markdown(
                    f"<p style='font-size:14.5px; color:#444; text-align:justify;'>{sector_overview[sector]}</p>",
                    unsafe_allow_html=True
                )

                # Tools
                st.markdown("<b>Tools & Tech:</b><br>", unsafe_allow_html=True)
                tool_html = "".join([f"<span class='tool-btn'>{t}</span>" for t in sector_tools[sector]])
                st.markdown(tool_html, unsafe_allow_html=True)

                # Button
                if st.button(f"Explore {sector}", key=f"btn_{sector}"):
                    st.session_state["sector"] = sector

                st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# SECTOR PAGE
# ============================================================

else:
    sector = st.session_state["sector"]
    st.header(f"{sector} – Use Cases Coming Soon")

    if st.button("Back to Home"):
        st.session_state["sector"] = None
