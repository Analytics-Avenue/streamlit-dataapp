import streamlit as st
import os

# --- Streamlit setup ---
st.set_page_config(page_title="Data Analytics Hub", layout="wide")
st.title("Data Analytics Case Studies")
st.markdown("Welcome! Choose a project to explore its dashboard.")

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# --- Use cases and preview images ---
use_cases = {
    "Marketing Analytics": os.path.join(ASSETS_DIR, "marketing_preview.jpg"),
    "Healthcare Analytics": os.path.join(ASSETS_DIR, "healthcare_preview.jpg"),
    "Retail Analytics": os.path.join(ASSETS_DIR, "retail_preview.jpg")
}

# --- Layout: 3 columns ---
cols = st.columns(3)

# --- Display each card ---
for i, (name, image) in enumerate(use_cases.items()):
    with cols[i % 3]:
        if os.path.exists(image):
            st.image(image, use_container_width=True)
        else:
            st.warning(f"Preview not found for {name}")

        st.markdown(f"### {name}")
        st.write("Dive into the data, uncover insights, and visualize trends.")

        # --- Button navigation ---
        page_path = f"pages/{i+1}_{name.replace(' ', '_')}.py"
        if st.button(f"Go to {name}", key=name):
            try:
                st.switch_page(page_path)
            except Exception:
                st.error(f"⚠️ Could not link to {page_path}. Make sure it exists in /pages/ folder.")
