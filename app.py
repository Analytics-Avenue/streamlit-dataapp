import streamlit as st
import os

# --- Streamlit setup ---
st.set_page_config(page_title="Data Analytics Solutions", layout="wide")

# Hide sidebar
st.markdown("""<style>[data-testid="stSidebarNav"]{display:none;}</style>""", unsafe_allow_html=True)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# --- Hierarchy Data ---
sectors = {
    "Marketing Analytics": [
        {"name": "Marketing Campaign Performance Analyzer", "image": "marketing_thumb.jpg", "page": "marketing_1.py"},
        {"name": "Customer Journey & Funnel Insights", "image": "marketing_thumb.jpg", "page": "marketing_2.py"},
        {"name": "Marketing Spend vs ROI Dashboard", "image": "marketing_thumb.jpg", "page": "marketing_3.py"},
        {"name": "Lead Scoring & Conversion Analytics", "image": "marketing_thumb.jpg", "page": "marketing_4.py"},
        {"name": "Social Media Engagement Tracker", "image": "marketing_thumb.jpg", "page": "marketing_5.py"},
        {"name": "Email Campaign Performance Dashboard", "image": "marketing_thumb.jpg", "page": "marketing_6.py"},
        {"name": "Content Performance & SEO Insights", "image": "marketing_thumb.jpg", "page": "marketing_7.py"},
        {"name": "Customer Retention & Churn Analysis", "image": "marketing_thumb.jpg", "page": "marketing_8.py"},
        {"name": "Market Segmentation & Persona Insights", "image": "marketing_thumb.jpg", "page": "marketing_9.py"},
        {"name": "Advertising Channel Performance Dashboard", "image": "marketing_thumb.jpg", "page": "marketing_10.py"},
    ],
    "Real Estate Analytics": [
        {"name": "Real Estate Intelligence Suite", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_1.py"},
        {"name": "Customer Segmentation & Buyer Persona Intelligence", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_2.py"},
        {"name": "Price vs Property Features Analyzer", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_3.py"},
        {"name": "Agent & Market Insights Dashboard", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_4.py"},
        {"name": "Property Demand vs Supply Analyzer", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_5.py"},
        {"name": "Tenant Risk & Market Trend Analyzer", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_6.py"},
        {"name": "Rental Yield & Investment Analyzer", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_7.py"},
        {"name": "Market Buzz & Activity Dashboard", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_8.py"},
    ],
}

# --- Home page thumbnails ---
home_thumbs = {
    "Marketing Analytics": os.path.join(ASSETS_DIR, "marketing_thumb.jpg"),
    "Real Estate Analytics": os.path.join(ASSETS_DIR, "real_estate_thumb.jpg"),
}

# --- Session state ---
if "sector" not in st.session_state:
    st.session_state["sector"] = None

# --- Home Page ---
if st.session_state["sector"] is None:
    st.title("Data Analytics Hub")
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
                st.experimental_rerun()  # only rerun on sector selection

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

                # Direct switch_page, works first click
                if st.button(f"Go to {uc['name']}", key=uc["name"]):
                    st.switch_page(f"pages/{uc['page']}")

    # Back button
    if st.button("⬅️ Back to Sectors"):
        st.session_state["sector"] = None
        st.experimental_rerun()
