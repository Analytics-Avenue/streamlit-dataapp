import streamlit as st

st.set_page_config(page_title="Data Analytics Hub", layout="wide")

st.title("Marketing Analytics Case studies")
st.markdown("Welcome! Choose a project to explore its dashboard.")

# Use cases and their preview images
use_cases = {
    "Marketing Analytics": "assets/marketing_preview.jpg",
    "Healthcare Analytics": "assets/healthcare_preview.jpg",
    "Retail Analytics": "assets/retail_preview.jpg"
}

cols = st.columns(3)

for i, (name, image) in enumerate(use_cases.items()):
    with cols[i % 3]:
        st.image(image, use_column_width=True)
        st.markdown(f"### {name}")
        st.write("Dive into the data, uncover insights, and visualize trends.")
        page_name = f"{name.replace(' ', '_')}.py"
        st.page_link(f"pages/{i+1}_{page_name}", label=f"Go to {name}", icon="📈")
