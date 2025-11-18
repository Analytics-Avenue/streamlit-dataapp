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
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
}
.card-box:hover {
    transform: translateY(-6px);
    box-shadow: 0 6px 22px rgba(0,0,0,0.18);
    border-color: #7fa8ff;
    background: #f9fbff;
}
.card-box img {
    border-radius: 8px;
    outline: 1px solid #dce6ff;
    transition: 0.25s ease-in-out;
    max-height: 120px;
    object-fit: cover;
}
.card-box:hover img {
    outline-color: #7fa8ff;
    box-shadow: 0px 0px 10px rgba(130,160,255,0.5);
    transform: scale(1.02);
}
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
# Sector Use Cases
# -------------------------
sectors = {
    "Marketing Analytics": [
        {"name": "Marketing Campaign Performance Analyzer", "image": "marketing_thumb.jpg", "page": "marketing_1.py"},
        {"name": "Marketing Intelligence & Forecasting Lab", "image": "marketing_thumb.jpg", "page": "marketing_2.py"},
        {"name": "Click & Convertion Analytics", "image": "marketing_thumb.jpg", "page": "marketing_3.py"},
        {"name": "Marketing Performance Analysis", "image": "marketing_thumb.jpg", "page": "marketing_4.py"},
        {"name": "Content & SEO Performance Dashboard", "image": "marketing_thumb.jpg", "page": "marketing_5.py"},
        {"name": "Customer Retention & Churn Analysis", "image": "marketing_thumb.jpg", "page": "marketing_6.py"},
        {"name": "Customer Journey & Funnel Insights,", "image": "marketing_thumb.jpg", "page": "marketing_7.py"},
        {"name": "Google Ads Performance Analytics.", "image": "marketing_thumb.jpg", "page": "marketing_8.py"},
        {"name": "Email & WhatsApp Marketing Forecast Lab", "image": "marketing_thumb.jpg", "page": "marketing_9.py"},
    ],
    "Real Estate Analytics": [
        {"name": "Real Estate Intelligence Suite", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_1.py"},
        {"name": "Real Estate Demand Forecasting System", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_2.py"},
        {"name": "Price vs Property Features Analyzer", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_3.py"},
        {"name": "Agent & Market Insights Dashboard", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_4.py"},
        {"name": "Real Estate Investment Opportunity Analyzer", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_5.py"},
        {"name": "Tenant Risk & Market Trend Analyzer", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_6.py"},
        {"name": "Rental Yield & Investment Analyzer", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_7.py"},
        {"name": "Real Estate Buyer Sentiment Analyzer", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_8.py"},
        {"name": "Neighborhood Lifestyle & Risk Aware Analyzer", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_9.py"},
        {"name": "Real Estate Intelligence — Hybrid Dashboard (Property + CRM)", "image": "real_estate_thumb.jpg", "page": "realestate.py"},
    ],
    "Health Care Analytics": [
        {"name": "Healthscope Insights", "image": "real_estate_thumb.jpg", "page": "healthcare_1.py"},
        {"name": "Patient Visit Analytics & Hospital Performance", "image": "real_estate_thumb.jpg", "page": "healthcare_2.py"},
        {"name": "PatientFlow Navigator", "image": "real_estate_thumb.jpg", "page": "healthcare_3.py"},
        {"name": "Ambulance Ops & Routing Lab", "image": "real_estate_thumb.jpg", "page": "healthcare_4.py"},
        {"name": "Health Care Analytics1", "image": "real_estate_thumb.jpg", "page": "healthcare_5.py"},
        {"name": "Health Care Analytics2", "image": "real_estate_thumb.jpg", "page": "healthcare_6.py"},
    ],
}

# -------------------------
# RAW Thumbnail URLs
# -------------------------
thumb_urls = {
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
                st.image(thumb_urls[sector], use_container_width=True)

                # Title
                st.markdown(f"<h3 style='color:#064b86; margin-top:12px;'>{sector}</h3>", unsafe_allow_html=True)

                # Overview
                st.markdown(f"<p style='font-size:14.5px; color:#444; text-align:justify;'>{sector_overview[sector]}</p>", unsafe_allow_html=True)

                # Tools
                st.markdown("<b>Tools & Tech:</b><br>", unsafe_allow_html=True)
                tool_html = "".join([f"<span class='tool-btn'>{t}</span>" for t in sector_tools[sector]])
                st.markdown(tool_html, unsafe_allow_html=True)

                # Button
                if st.button(f"Explore {sector}", key=f"btn_{sector}"):
                    st.session_state["sector"] = sector

                st.markdown("</div>", unsafe_allow_html=True)




# ============================================================
# ============================================================
# SECTOR PAGE (Projects / Use Cases)
# ============================================================
else:
    sector_name = st.session_state["sector"]
    st.header(f"{sector_name} – Projects / Use Cases")

    usecases = sectors[sector_name]
    rows = [usecases[i:i+3] for i in range(0, len(usecases), 3)]

    for row in rows:
        cols = st.columns(3)
        for col, uc in zip(cols, row):
            with col:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)

                # Thumbnail image
                st.image(thumb_urls[sector_name], use_container_width=True)

                # Project title
                st.markdown(f"<h4 style='color:#064b86; margin-top:8px;'>{uc['name']}</h4>", unsafe_allow_html=True)

                # Construct deployed page URL
                # Streamlit converts pages to slugs: spaces -> _, special chars removed
                page_slug = uc['name'].replace(" ", "_").replace("&", "and").replace("-", "_")
                deployed_url = f"https://analytics-avenue.streamlit.app/{page_slug}"

                # Open button as HTML link (new tab)
                st.markdown(f"""
                    <a href="{deployed_url}" target="_blank" 
                       style="text-decoration:none;">
                       <div style="background:#eef4ff; color:#064b86; padding:6px 12px; border-radius:6px; text-align:center; font-weight:600; margin-top:5px;">
                           Open
                       </div>
                    </a>
                """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

    # Back button to home page
    if st.button("Back to Home"):
        st.session_state["sector"] = None
        st.experimental_rerun()



