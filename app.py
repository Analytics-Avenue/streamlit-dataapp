import streamlit as st
import os

# -------------------------
# Streamlit page setup
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
# DIRECTORIES
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# -------------------------
# OVERVIEW TEXT (Add Yours)
# -------------------------
sector_overview = {
    "Marketing Analytics": "Get insights into consumer behavior, campaign ROI, brand performance, funnels, and retention.",
    "Real Estate Analytics": "Analyze pricing, investment opportunities, market trends, buyer sentiment, and locality insights.",
    "Health Care Analytics": "Improve patient outcomes, hospital performance, routing efficiency, and clinical decision making."
}

# -------------------------
# TOOLS FOR EACH SECTOR
# -------------------------
sector_tools = {
    "Marketing Analytics": ["Python", "SQL", "Power BI", "Tableau", "Machine Learning", "Forecasting", "Marketing Mix Modelling"],
    "Real Estate Analytics": ["Python", "SQL", "GIS", "Power BI", "ML Models", "Valuation Models", "Regression"],
    "Health Care Analytics": ["Python", "SQL", "Power BI", "Predictive Analytics", "Routing Algorithms", "NLP"]
}

# -------------------------
# SECTORS + UseCases
# -------------------------
sectors = {
    "Marketing Analytics": [
        {"name": "Marketing Campaign Performance Analyzer", "image": "marketing_thumb.jpg", "page": "marketing_1.py"},
        {"name": "Marketing Intelligence & Forecasting Lab", "image": "marketing_thumb.jpg", "page": "marketing_2.py"},
        {"name": "Click & Convertion Analytics", "image": "marketing_thumb.jpg", "page": "marketing_3.py"},
        {"name": "Marketing Performance Analysis", "image": "marketing_thumb.jpg", "page": "marketing_4.py"},
        {"name": "Content & SEO Performance Dashboard", "image": "marketing_thumb.jpg", "page": "marketing_5.py"},
        {"name": "Customer Retention & Churn Analysis", "image": "marketing_thumb.jpg", "page": "marketing_6.py"},
        {"name": "Customer Journey & Funnel Insights", "image": "marketing_thumb.jpg", "page": "marketing_7.py"},
        {"name": "Google Ads Performance Analytics", "image": "marketing_thumb.jpg", "page": "marketing_8.py"},
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
        {"name": "Real Estate Intelligence — Hybrid Dashboard", "image": "real_estate_thumb.jpg", "page": "realestate.py"},
    ],

    "Health Care Analytics": [
        {"name": "Healthscope Insights", "image": "healthcare_thumb.jpg", "page": "healthcare_1.py"},
        {"name": "Patient Visit Analytics & Hospital Performance", "image": "healthcare_thumb.jpg", "page": "healthcare_2.py"},
        {"name": "PatientFlow Navigator", "image": "healthcare_thumb.jpg", "page": "healthcare_3.py"},
        {"name": "Ambulance Ops & Routing Lab", "image": "healthcare_thumb.jpg", "page": "healthcare_4.py"},
        {"name": "Health Care Analytics 1", "image": "healthcare_thumb.jpg", "page": "healthcare_5.py"},
        {"name": "Health Care Analytics 2", "image": "healthcare_thumb.jpg", "page": "healthcare_6.py"},
    ],
}

home_thumbs = {
    "Marketing Analytics": os.path.join(ASSETS_DIR, "marketing_thumb.jpg"),
    "Real Estate Analytics": os.path.join(ASSETS_DIR, "real_estate_thumb.jpg"),
    "Health Care Analytics": os.path.join(ASSETS_DIR, "healthcare_thumb.jpg"),
}

# -------------------------
# SESSION STATE
# -------------------------
if "sector" not in st.session_state:
    st.session_state["sector"] = None

# -------------------------
# GLOBAL CSS (Card UI)
# -------------------------
st.markdown("""
<style>

.card {
    border: 2px solid #064b86;
    border-radius: 14px;
    padding: 15px;
    background: white;
    transition: 0.3s;
    box-shadow: 0 0 10px rgba(6, 75, 134, 0.15);
}

.card:hover {
    box-shadow: 0 0 25px rgba(6, 75, 134, 0.4);
    transform: translateY(-3px);
}

.tool-tag {
    display: inline-block;
    background: #e8f1ff;
    border: 1px solid #bcd4ff;
    padding: 4px 8px;
    margin: 3px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    color: #064b86;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# HOME PAGE (3 Card Grid)
# -------------------------
if st.session_state["sector"] is None:

    st.title("Data Analytics Solutions")
    st.write("Choose a sector to explore:")

    sector_list = list(sectors.keys())
    rows = [sector_list[i:i+3] for i in range(0, len(sector_list), 3)]

    for row in rows:
        cols = st.columns(3)
        for col, sector_name in zip(cols, row):
            with col:
                st.markdown("<div class='card'>", unsafe_allow_html=True)

                thumb = home_thumbs.get(sector_name)
                if os.path.exists(thumb):
                    st.image(thumb, use_container_width=True)

                st.markdown(f"<h3 style='color:#064b86;'>{sector_name}</h3>", unsafe_allow_html=True)

                st.markdown(f"<p>{sector_overview[sector_name]}</p>", unsafe_allow_html=True)

                st.markdown("<p><strong>Tools & Tech:</strong></p>", unsafe_allow_html=True)

                # Tools grid inside card
                for tool in sector_tools[sector_name]:
                    st.markdown(f"<span class='tool-tag'>{tool}</span>", unsafe_allow_html=True)

                if st.button(f"Explore {sector_name}", key=f"go_{sector_name}"):
                    st.session_state["sector"] = sector_name

                st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# SECTOR PAGE (Usecase Grid)
# -------------------------
else:
    sector_name = st.session_state["sector"]
    st.header(f"{sector_name} Use Cases")

    usecases = sectors[sector_name]

    for i in range(0, len(usecases), 3):
        cols = st.columns(3)

        for j, uc in enumerate(usecases[i:i+3]):
            with cols[j]:
                st.markdown("<div class='card'>", unsafe_allow_html=True)

                img = os.path.join(ASSETS_DIR, uc["image"])
                if os.path.exists(img):
                    st.image(img, use_container_width=True)

                st.markdown(f"### {uc['name']}")

                if st.button("Open", key=uc["name"]):
                    st.switch_page(f"pages/{uc['page']}")

                st.markdown("</div>", unsafe_allow_html=True)
