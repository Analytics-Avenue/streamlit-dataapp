import os
import streamlit as st

st.set_page_config(page_title="Data Analytics Hub", layout="wide")

st.title("Data Analytics Case Studies")
st.markdown("Welcome! Choose a project to explore its dashboard.")

# --- File path setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

use_cases = {
    "Marketing Analytics": os.path.join(ASSETS_DIR, "marketing_preview.jpg"),
    "Healthcare Analytics": os.path.join(ASSETS_DIR, "healthcare_preview.jpg"),
    "Retail Analytics": os.path.join(ASSETS_DIR, "retail_preview.jpg")
}

cols = st.columns(3)

for i, (name, image_path) in enumerate(use_cases.items()):
    with cols[i % 3]:
        # Show image safely
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            st.warning(f"⚠️ Missing preview image for {name}")

        st.markdown(f"### {name}")
        st.write("Dive into the data, uncover insights, and visualize trends.")
        
        # Page navigation
        page_file = f"{i+1}_{name.replace(' ', '_')}.py"
        st.page_link(f"pages/{page_file}", label=f"Go to {name}", icon="📊")
