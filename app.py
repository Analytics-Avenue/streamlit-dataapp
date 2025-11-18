import streamlit as st
import os

# ----------------------------------------
# Page Setup
# ----------------------------------------
st.set_page_config(page_title="Data Analytics Solutions", layout="wide")
st.markdown("""<style>[data-testid="stSidebarNav"]{display:none;}</style>""", unsafe_allow_html=True)

# ----------------------------------------
# Global CSS
# ----------------------------------------
st.markdown("""
<style>

.tool-btn {
    background: #e8f1ff;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
    border: 1px solid #bcd2ff;
    text-align: center;
    font-weight: 600;
    margin: 3px;
    display: inline-block;
    transition: 0.2s;
}
.tool-btn:hover {
    background: #d8e8ff;
    transform: scale(1.05);
}

.card-box {
    border: 1px solid #c9d7f0;
    border-radius: 12px;
    padding: 15px;
    background: #ffffff;
    min-height: 540px;
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
    outline: 1px solid #dce7ff;
    transition: 0.25s;
}

.card-box:hover img {
    outline-color: #8bb3ff;
    box-shadow: 0 0 8px rgba(140,170,255,0.5);
}

.sector-title {
    color: #064b86;
    font-size: 26px;
    font-weight: 700;
    margin-top: 10px;
}

.sector-overview {
    font-size: 14.5px;
    color: #444;
    text-align: justify;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# Logo and Company Header
# ----------------------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display: flex; align-items: center;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:bold;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:bold;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------
# Directories for thumbnails
# ----------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# ----------------------------------------
# Detailed Sector Overviews
# ----------------------------------------
sector_overview = {
    "Marketing Analytics": (
        "Marketing analytics transforms raw customer interactions into actionable business insight. "
        "It enables brands to understand patterns in campaign performance, optimize audience targeting, "
        "improve ROI visibility, and strengthen customer retention strategies. Modern analytics also "
        "combines automation, machine learning, and cross-channel attribution to deliver unified "
        "performance dashboards. When implemented well, it becomes the backbone of scalable growth. "
        "This helps businesses make informed decisions and win in competitive digital environments."
    ),

    "Real Estate Analytics": (
        "Real estate analytics provides a deep understanding of market movement, buyer preferences, "
        "and property valuation trends. It enables forecasting of price fluctuations, rental demand, "
        "and investment viability using structured models. With geospatial intelligence, investors "
        "can assess neighborhood growth and infrastructure impact. Developers benefit from sales "
        "performance insights and customer segmentation. Ultimately, analytics helps de-risk decisions "
        "and improve project profitability at every stage."
    ),

    "Health Care Analytics": (
        "Healthcare analytics enhances patient outcomes by optimizing hospital operations and predicting "
        "resource requirements. It uses structured and unstructured medical data to identify patterns "
        "in diseases, patient flow, and treatment efficiency. Strong analytics systems support "
        "emergency forecasting, readmission risk scoring, and cost reduction programs. Hospitals can "
        "minimize bottlenecks, reduce wait time, and deploy staff effectively. This creates a more "
        "efficient, patient-centric care ecosystem."
    )
}

# ----------------------------------------
# Tools (Filtered & Popular Only)
# ----------------------------------------
sector_tools = {
    "Marketing Analytics": [
        "Python", "SQL", "Power BI", "Tableau", "Google Analytics 4",
        "Google Ads", "Meta Ads", "CRM Data", "Pandas", "ML Models"
    ],
    "Real Estate Analytics": [
        "Python", "Power BI", "SQL", "Regression Models", "GeoPandas",
        "QGIS", "Time Series", "Forecasting", "ETL"
    ],
    "Health Care Analytics": [
        "Python", "R", "SQL", "Power BI", "EMR/EHR Data",
        "Forecasting Models", "ML Models", "NLP"
    ]
}

home_thumbs = {
    "Marketing Analytics": os.path.join(ASSETS_DIR, "marketing_thumb.jpg"),
    "Real Estate Analytics": os.path.join(ASSETS_DIR, "real_estate_thumb.jpg"),
    "Health Care Analytics": os.path.join(ASSETS_DIR, "healthcare_thumb.jpg"),
}

# ----------------------------------------
# Session State
# ----------------------------------------
if "sector" not in st.session_state:
    st.session_state["sector"] = None

# ----------------------------------------
# HOME PAGE
# ----------------------------------------
if st.session_state["sector"] is None:

    st.title("Data Analytics Solutions")
    st.write("Choose a sector to explore:")

    sectors = list(sector_overview.keys())
    rows = [sectors[i:i+3] for i in range(0, len(sectors), 3)]

    for row in rows:
        cols = st.columns(3)
        for col, sector in zip(cols, row):
            with col:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)

                thumb = home_thumbs.get(sector)
                if os.path.exists(thumb):
                    st.image(thumb, use_container_width=True)

                st.markdown(f"<div class='sector-title'>{sector}</div>", unsafe_allow_html=True)
                st.markdown(f"<p class='sector-overview'>{sector_overview[sector]}</p>", unsafe_allow_html=True)

                st.markdown("<b>Tools & Tech:</b><br>", unsafe_allow_html=True)
                tools = sector_tools[sector]
                tool_html = "".join([f"<span class='tool-btn'>{t}</span>" for t in tools])
                st.markdown(tool_html, unsafe_allow_html=True)

                if st.button(f"Explore {sector}", key=f"btn_{sector}"):
                    st.session_state["sector"] = sector

                st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------
# SECTOR PAGE
# ----------------------------------------
else:
    sector = st.session_state["sector"]
    st.header(f"{sector} – Use Cases Coming Soon")

    if st.button("Back to Home"):
        st.session_state["sector"] = None
