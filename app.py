import streamlit as st
import os

# -------------------------
# Company Logo + Name
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display: flex; align-items: center;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0; padding:0;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0; padding:0;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Data Analytics Solutions", layout="wide")

# Hide sidebar nav
st.markdown("""<style>[data-testid="stSidebarNav"]{display:none;}</style>""",
            unsafe_allow_html=True)

# --------------------------------
# PATHS
# --------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# --------------------------------
# CARD CSS (from your new UI)
# --------------------------------
st.markdown("""
<style>

.card {
    border: 1.5px solid #d4d4d4;
    border-radius: 10px;
    padding: 15px;
    background: white;
    transition: all 0.3s ease;
    box-shadow: 0px 0px 4px rgba(0,0,0,0.05);
    height: 100%;
}

.card:hover {
    transform: scale(1.02);
    box-shadow: 0px 4px 18px rgba(0,0,0,0.15);
}

.thumbnail {
    width: 100%;
    height: 180px;
    object-fit: cover;
    border-radius: 8px;
}

.title {
    font-size: 20px;
    font-weight: 700;
    margin-top: 10px;
}

.overview {
    font-size: 13.5px;
    text-align: justify;
    margin-top: 5px;
}

.tool-tag {
    display: inline-block;
    background: #eef2ff;
    padding: 5px 10px;
    border-radius: 6px;
    margin: 4px 4px 0 0;
    font-size: 12px;
    color: #333;
    border: 1px solid #cdd3ff;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------
# SECTOR OVERVIEW + TOOLS
# --------------------------------

sector_details = {
    "Marketing Analytics": {
        "overview": """Marketing analytics helps brands decode customer behavior, optimize campaigns, 
        manage budgets smartly, improve retention, and forecast demand. It unifies cross-channel data 
        to reveal how customers think and how businesses can grow ROI.""",
        "tools": [
            "Power BI", "Tableau", "Python", "SQL", "Excel",
            "Google Analytics", "BigQuery", "Meta Ads Manager", "Google Ads"
        ]
    },
    "Real Estate Analytics": {
        "overview": """Real estate analytics powers pricing intelligence, valuation models, rental forecasting, 
        buyer profiling, and investment scoring. It combines geography, market trends, and predictive modeling 
        for smarter property decisions.""",
        "tools": [
            "Power BI", "Python", "SQL", "Excel", "GIS Tools",
            "ArcGIS", "QGIS", "GeoPandas", "ML Models"
        ]
    },
    "Health Care Analytics": {
        "overview": """Healthcare analytics improves patient outcomes, predicts disease patterns, optimizes 
        hospital operations, and supports treatment planning. It merges EMR/EHR systems with predictive models 
        to elevate diagnosis accuracy.""",
        "tools": [
            "Power BI", "Python", "R", "SQL", "Excel",
            "FHIR APIs", "Healthcare Dashboards", "ML Models"
        ]
    }
}

# --------------------------------
# USE CASE FILE MAPPING
# --------------------------------
sectors = {
    "Marketing Analytics": [
        {"name": "Marketing Campaign Performance Analyzer", "image": "marketing_thumb.jpg", "page": "marketing_1.py"},
        {"name": "Marketing Intelligence & Forecasting Lab", "image": "marketing_thumb.jpg", "page": "marketing_2.py"},
        {"name": "Click & Convertion Analytics", "image": "marketing_thumb.jpg", "page": "marketing_3.py"},
    ],
    "Real Estate Analytics": [
        {"name": "Real Estate Intelligence Suite", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_1.py"},
        {"name": "Demand Forecasting System", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_2.py"},
        {"name": "Price vs Features Analyzer", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_3.py"},
    ],
    "Health Care Analytics": [
        {"name": "Healthscope Insights", "image": "healthcare_thumb.jpg", "page": "healthcare_1.py"},
        {"name": "Hospital Performance Dashboard", "image": "healthcare_thumb.jpg", "page": "healthcare_2.py"},
        {"name": "PatientFlow Navigator", "image": "healthcare_thumb.jpg", "page": "healthcare_3.py"},
    ]
}

# --------------------------------
# SESSION STATE
# --------------------------------
if "sector" not in st.session_state:
    st.session_state["sector"] = None

# --------------------------------
# SIDEBAR NAVIGATION
# --------------------------------
st.sidebar.title("Navigation")
sidebar_choice = st.sidebar.radio(
    "Go to:",
    ["Home"] + list(sectors.keys()),
)

if sidebar_choice == "Home":
    st.session_state["sector"] = None
else:
    st.session_state["sector"] = sidebar_choice

# --------------------------------
# HOME PAGE WITH NEW CARDS
# --------------------------------
if st.session_state["sector"] is None:

    st.title("Data Analytics Solutions")
    st.write("Choose a sector to explore:")

    cols = st.columns(3)

    for idx, sector_name in enumerate(sectors.keys()):
        with cols[idx]:

            thumb_path = os.path.join(ASSETS_DIR, sector_name.lower().replace(" ", "_") + "_thumb.jpg")
            overview = sector_details[sector_name]["overview"]
            tools = sector_details[sector_name]["tools"]

            st.markdown("<div class='card'>", unsafe_allow_html=True)

            if os.path.exists(thumb_path):
                st.markdown(f"<img class='thumbnail' src='file://{thumb_path}'>",
                            unsafe_allow_html=True)

            st.markdown(f"<div class='title'>{sector_name}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='overview'>{overview}</div>", unsafe_allow_html=True)

            tool_html = "".join([f"<span class='tool-tag'>{t}</span>" for t in tools])
            st.markdown(tool_html, unsafe_allow_html=True)

            if st.button(f"Explore {sector_name}", key=f"explore_{idx}"):
                st.session_state["sector"] = sector_name

            st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# SECTOR USE CASE PAGE
# --------------------------------
else:
    sector_name = st.session_state["sector"]
    st.header(f"{sector_name} Use Cases")

    usecases = sectors[sector_name]

    for i in range(0, len(usecases), 3):
        cols = st.columns(3)
        for j, uc in enumerate(usecases[i:i+3]):
            with cols[j]:

                thumb_path = os.path.join(ASSETS_DIR, uc["image"])
                if os.path.exists(thumb_path):
                    st.image(thumb_path, use_container_width=True)

                st.markdown(f"### {uc['name']}")
                st.write("Explore insights and dashboards.")

                if st.button(f"Go to {uc['name']}", key=uc["name"]):
                    st.switch_page(f"pages/{uc['page']}")
