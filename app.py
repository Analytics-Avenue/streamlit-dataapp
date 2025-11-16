import streamlit as st
import os

# --- Streamlit setup ---
st.set_page_config(
    page_title="Data Analytics Solutions",
    layout="wide",
)


# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
PAGES_DIR = os.path.join(BASE_DIR, "pages")


# --- Hierarchy Data with actual project names ---
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


# --- Direct thumbnail paths for home page ---
home_thumbs = {
    "Marketing Analytics": os.path.join(ASSETS_DIR, "real_estate_thumb.jpg"),
    "Real Estate Analytics": os.path.join(ASSETS_DIR, "real_estate_thumb.jpg"),
}

# --- Initialize session state ---
if "sector" not in st.session_state:
    st.session_state["sector"] = None

# --- Home Page ---
if st.session_state["sector"] is None:
    st.title("Data Analytics Solutions")
    st.markdown("Welcome! Choose a sector to explore its use cases.")

    cols = st.columns(len(sectors))
    for idx, (sector_name, usecases) in enumerate(sectors.items()):
        with cols[idx]:
            thumb_path = home_thumbs.get(sector_name)
            if os.path.exists(thumb_path):
                st.image(thumb_path, use_container_width=True)
            else:
                st.warning(f"Thumbnail not found for {sector_name}")
            
            st.markdown(f"### {sector_name}")
            st.write(f"Explore {len(usecases)} use cases in {sector_name}.")
            
            if st.button(f"Explore {sector_name}", key=sector_name):
                st.session_state["sector"] = sector_name
                st.experimental_rerun()

# --- Sector Page ---
else:
    sector_name = st.session_state["sector"]
    st.header(f"{sector_name} Use Cases")
    st.markdown("Select a use case to go to its project page.")
    
    usecases = sectors[sector_name]
    
    # Display use cases in grid: 3 columns
    for i in range(0, len(usecases), 3):
        cols = st.columns(3)
        for j, uc in enumerate(usecases[i:i+3]):
            with cols[j]:
                thumb_path = os.path.join(ASSETS_DIR, uc["image"])
                if os.path.exists(thumb_path):
                    st.image(thumb_path, use_container_width=True)
                else:
                    st.warning(f"Thumbnail not found for {uc['name']}")
                
                st.markdown(f"### {uc['name']}")
                st.write("Dive into the data, uncover insights, and visualize trends.")
                
                # Path-based button navigation
                page_path = f"pages/{uc['page']}"
                if st.button(f"Go to {uc['name']}", key=uc["name"]):
                    try:
                        st.switch_page(page_path)
                    except Exception:
                        st.error(f"⚠️ Could not link to {page_path}. Make sure it exists in /pages/ folder.")

    # Back button to Home
    if st.button("⬅️ Back to Sectors"):
        st.session_state["sector"] = None
        st.experimental_rerun()
