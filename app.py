import streamlit as st
import os

st.set_page_config(page_title="Data Analytics Hub", layout="wide")
st.title("Data Analytics Case Studies")
st.markdown("Welcome! Choose a project to explore its dashboard.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

use_cases = {
    "Marketing Analytics": os.path.join(ASSETS_DIR, "marketing_preview.jpg"),
    "Healthcare Analytics": os.path.join(ASSETS_DIR, "healthcare_preview.jpg"),
    "Real Estate Analytics": os.path.join(ASSETS_DIR, "real_estate_preview.jpg")
}

cols = st.columns(3)

for i, (name, image_path) in enumerate(use_cases.items()):
    with cols[i % 3]:
        if os.path.exists(image_path):
            st.image(image_path, width="stretch")
        else:
            st.warning(f"⚠️ Missing image: {image_path}")
        st.markdown(f"### {name}")
        st.write("Dive into the data, uncover insights, and visualize trends.")

        # Page navigation (absolute path fix)
        page_file = f"{i+1}_{name.replace(' ', '_')}.py"
        page_path = f"/{page_file.lower()}"  # lowercase URL path to match Streamlit internal routing

        try:
            st.page_link(page_path, label=f"Go to {name}", icon="📊")
        except Exception:
            st.error(f"⚠️ Could not link to {page_file}. Make sure it exists in /pages/")
