import streamlit as st
import os

# --- Streamlit setup ---
st.set_page_config(
    page_title="Data Analytics Hub",
    layout="wide",
)

# Hide the default Streamlit sidebar page selector
hide_default_format = """
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(hide_default_format, unsafe_allow_html=True)

st.title("Data Analytics Case Studies")
st.markdown("Welcome! Choose a project to explore its dashboard.")

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# --- Use cases and preview images ---
use_cases = {
    "Marketing Analytics": {
        "image": os.path.join(ASSETS_DIR, "real_estate_preview.jpg"),
        "page": "marketing.py"
    },
    "Real Estate Analytics": {
        "image": os.path.join(ASSETS_DIR, "real_estate_preview.jpg"),
        "page": "realestate.py"
    },
    "#1 Real Estate Intelligence Suite": {
        "image": os.path.join(ASSETS_DIR, "real_estate_preview.jpg"),
        "page": "1.py"
    },
    "#2 Real Estate Customer Segmentation & Buyer Persona Intelligence": {
        "image": os.path.join(ASSETS_DIR, "real_estate_preview.jpg"),
        "page": "2.py"
    },
    "#3 REAL ESTATE PRICE VS PROPERTY FEATURES ANALYZER": {
        "image": os.path.join(ASSETS_DIR, "real_estate_preview.jpg"),
        "page": "3.py"
    },
    "#4 Real Estate Agent & Market Insights": {
        "image": os.path.join(ASSETS_DIR, "real_estate_preview.jpg"),
        "page": "4.py"
    },
    "#5 Real Estate Agent & Market Insights": {
        "image": os.path.join(ASSETS_DIR, "real_estate_preview.jpg"),
        "page": "5.py"
    },
    "#6 Tenant Risk & Market Trend Analyzer": {
        "image": os.path.join(ASSETS_DIR, "real_estate_preview.jpg"),
        "page": "6.py"
    }
}

# --- Layout: 3 columns ---
cols = st.columns(3)

# --- Display each card ---
for i, (name, info) in enumerate(use_cases.items()):
    with cols[i % 3]:
        if os.path.exists(info["image"]):
            st.image(info["image"], use_container_width=True)
        else:
            st.warning(f"Preview not found for {name}")

        st.markdown(f"### {name}")
        st.write("Dive into the data, uncover insights, and visualize trends.")

        # --- Button navigation ---
        page_path = f"pages/{info['page']}"
        if st.button(f"Go to {name}", key=name):
            try:
                st.switch_page(page_path)
            except Exception:
                st.error(f"⚠️ Could not link to {page_path}. Make sure it exists in /pages/ folder.")
