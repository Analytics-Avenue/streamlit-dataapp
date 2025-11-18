import streamlit as st
import os

# -------------------------
# Company Logo + Name
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

# Custom CSS for glow, hover, cards
st.markdown("""
<style>

.sector-box {
    border: 2px solid #064b86;
    border-radius: 12px;
    padding: 15px;
    transition: all 0.3s ease-in-out;
    box-shadow: 0 0 10px rgba(6, 75, 134, 0.15);
    background: white;
}

.sector-box:hover {
    transform: translateY(-5px);
    box-shadow: 0 0 25px rgba(6, 75, 134, 0.6);
    border-color: #0a6cc4;
}

.tool-card {
    background: #f7faff;
    padding: 8px;
    border-radius: 6px;
    border: 1px solid #dce6f5;
    box-shadow: 0 0 5px rgba(6, 75, 134, 0.15);
    text-align: center;
    font-weight: 600;
    margin-bottom: 6px;
    font-size: 13px;
}

.tool-card:hover {
    background: #e9f1ff;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Streamlit Setup
# -------------------------
st.set_page_config(page_title="Data Analytics Solutions", layout="wide")

# Hide default sidebar navigation
st.markdown("""<style>[data-testid="stSidebarNav"]{display:none;}</style>""",
            unsafe_allow_html=True)

# Logo + Title
st.markdown(f"""
<div style="display: flex; align-items: center;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:bold;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:bold;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# -------------------------
# Sector + Usecase Data
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

home_thumbs = {
    "Marketing Analytics": os.path.join(ASSETS_DIR, "marketing_thumb.jpg"),
    "Real Estate Analytics": os.path.join(ASSETS_DIR, "real_estate_thumb.jpg"),
    "Health Care Analytics": os.path.join(ASSETS_DIR, "healthcare_thumb.jpg"),
}

# -------------------------
# Sector Overview Text
# -------------------------
sector_overview = {
    "Marketing Analytics": "Unlock demand, optimize campaigns, decode customer intent, and drive measurable brand outcomes.",
    "Real Estate Analytics": "Forecast prices, analyze micro-markets, evaluate ROI, and uncover investment-ready opportunities.",
    "Health Care Analytics": "Improve patient care, optimize hospital operations, and drive data-backed medical decisions."
}

# -------------------------
# Tools for Each Sector
# -------------------------
sector_tools = {
    "Marketing Analytics": ["SQL", "Python", "Power BI", "Machine Learning", "Forecasting Models", "Customer Segmentation", "GenAI"],
    "Real Estate Analytics": ["SQL", "Python", "Predictive Modeling", "GIS Mapping", "Clustering Models", "Power BI", "GenAI"],
    "Health Care Analytics": ["SQL", "Python", "Time-Series Models", "Operational Dashboards", "ML", "Power BI", "GenAI"]
}

# -------------------------
# Session State
# -------------------------
if "sector" not in st.session_state:
    st.session_state["sector"] = None

# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.title("Navigation")

sidebar_choice = st.sidebar.radio(
    "Go to:",
    ["Home"] + list(sectors.keys()),
    index=0 if st.session_state["sector"] is None else list(sectors.keys()).index(st.session_state["sector"]) + 1
)

if sidebar_choice == "Home":
    st.session_state["sector"] = None
else:
    st.session_state["sector"] = sidebar_choice

# -------------------------
# HOME PAGE
# -------------------------
if st.session_state["sector"] is None:
    st.title("Data Analytics Solutions")
    st.write("Pick a sector and explore:")

    cols = st.columns(len(sectors))

    for idx, (sector_name, usecases) in enumerate(sectors.items()):
        with cols[idx]:
            st.markdown('<div class="sector-box">', unsafe_allow_html=True)

            thumb_path = home_thumbs.get(sector_name)
            if os.path.exists(thumb_path):
                st.image(thumb_path, use_container_width=True)

            st.markdown(f"### {sector_name}")

            st.markdown(
                f"<p style='font-size:14px; color:#333;'>{sector_overview[sector_name]}</p>",
                unsafe_allow_html=True
            )

            st.markdown("**Tools & Tech:**")

            for tool in sector_tools[sector_name]:
                st.markdown(f"<div class='tool-card'>{tool}</div>", unsafe_allow_html=True)

            if st.button(f"Explore {sector_name}", key=f"btn_{sector_name}"):
                st.session_state["sector"] = sector_name

            st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# SECTOR PAGE
# -------------------------
else:
    sector_name = st.session_state["sector"]
    st.header(f"{sector_name} Use Cases")
    st.markdown("Select a use case to open its dashboard:")

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
