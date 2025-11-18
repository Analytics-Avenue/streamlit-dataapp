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


# --- Streamlit setup ---
st.set_page_config(page_title="Data Analytics Solutions", layout="wide")

# Hide default sidebar navigation (optional)
st.markdown("""<style>[data-testid="stSidebarNav"]{display:none;}</style>""", unsafe_allow_html=True)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# --- Hierarchy Data ---
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

# --- Home page thumbnails ---
home_thumbs = {
    "Marketing Analytics": os.path.join(ASSETS_DIR, "marketing_thumb.jpg"),
    "Real Estate Analytics": os.path.join(ASSETS_DIR, "real_estate_thumb.jpg"),
    "Health Care Analytics": os.path.join(ASSETS_DIR, "healthcare_thumb.jpg"),
}

# --- Session state ---
if "sector" not in st.session_state:
    st.session_state["sector"] = None

# --- Sidebar ---
st.sidebar.title("Navigation")
sidebar_choice = st.sidebar.radio(
    "Go to:",
    ["Home"] + list(sectors.keys()),
    index=0 if st.session_state["sector"] is None else list(sectors.keys()).index(st.session_state["sector"]) + 1
)

# Handle sidebar navigation
if sidebar_choice == "Home":
    st.session_state["sector"] = None
else:
    st.session_state["sector"] = sidebar_choice

# --- Home Page ---
if st.session_state["sector"] is None:
    st.title("Data Analytics Solutions")
    st.markdown("Welcome! Choose a sector to explore its use cases:")

    cols = st.columns(len(sectors))
    for idx, (sector_name, usecases) in enumerate(sectors.items()):
        with cols[idx]:
            thumb_path = home_thumbs.get(sector_name)
            if os.path.exists(thumb_path):
                st.image(thumb_path, use_container_width=True)
            st.markdown(f"### {sector_name}")
            st.write(f"{len(usecases)} use cases available.")
            if st.button(f"Explore {sector_name}", key=sector_name):
                st.session_state["sector"] = sector_name

# --- Sector Page ---
else:
    sector_name = st.session_state["sector"]
    st.header(f"{sector_name} Use Cases")
    st.markdown("Select a use case to go to its project page:")

    usecases = sectors[sector_name]

    # Grid: 3 columns
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
