import streamlit as st
import os

# --- Basic Page Config ---
st.set_page_config(page_title="Data Analytics Hub", layout="wide")
st.title("Data Analytics Case Studies")
st.markdown("Welcome! Choose a project to explore its dashboard.")

# --- Folder Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
PAGES_DIR = os.path.join(BASE_DIR, "pages")

# --- Use Case List ---
use_cases = {
    "Marketing Analytics": "marketing_preview.jpg",
    "Healthcare Analytics": "healthcare_preview.jpg",
    "Real Estate Analytics": "real_estate_preview.jpg"
}

# --- Layout: 3 Columns ---
cols = st.columns(3)

for i, (name, image_file) in enumerate(use_cases.items()):
    image_path = os.path.join(ASSETS_DIR, image_file)
    page_file = f"{i+1}_{name.replace(' ', '_')}.py"
    page_path = f"pages/{page_file}"  # Must exactly match file name under /pages

    with cols[i % 3]:
        # Show preview image
        if os.path.exists(image_path):
            st.image(image_path, use_column_width=True)
        else:
            st.warning(f"⚠️ Missing image: {image_path}")

        st.markdown(f"### {name}")
        st.write("Dive into the data, uncover insights, and visualize trends.")

        # --- Page Navigation ---
        if os.path.exists(os.path.join(PAGES_DIR, page_file)):
            st.page_link(f"/{page_path}", label=f"Go to {name}", icon="📊")
        else:
            st.error(f"⚠️ Could not link to {page_file}. Make sure it exists in /pages/")
