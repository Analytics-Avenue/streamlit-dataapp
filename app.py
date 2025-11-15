import streamlit as st
import os
import re

st.set_page_config(page_title="Data Analytics Hub", layout="wide")

# Hide sidebar
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAGES_DIR = os.path.join(BASE_DIR, "pages")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Detect all pages dynamically
all_pages = [f for f in os.listdir(PAGES_DIR) if f.endswith(".py")]

# Auto-group by sector
sector_patterns = {
    "Marketing Analytics": r"usecase_marketing_\d+\.py",
    "Real Estate Analytics": r"usecase_real_estate_\d+\.py",
    "Customer Intelligence": r"usecase_customer_\d+\.py",
    "Sales & Revenue Analytics": r"usecase_sales_\d+\.py",
    "Operational Insights": r"usecase_ops_\d+\.py",
}

sectors = {}
for sector_name, pattern in sector_patterns.items():
    matched_pages = sorted([p for p in all_pages if re.match(pattern, p)])
    usecases = []
    for p in matched_pages:
        i = re.findall(r'\d+', p)[0]
        usecases.append({
            "name": f"{sector_name} Use Case {i}",
            "image": f"{sector_name.lower().replace(' ', '_')}_thumb.png",
            "page": p.replace(".py", "")
        })
    sectors[sector_name] = usecases

# Session state
if "sector" not in st.session_state:
    st.session_state["sector"] = None
if "go_to_page" not in st.session_state:
    st.session_state["go_to_page"] = None

# --- Handle page navigation ---
if st.session_state["go_to_page"]:
    page_name = st.session_state["go_to_page"]
    st.session_state["go_to_page"] = None
    st.switch_page(page_name)

# --- Home Page ---
if st.session_state["sector"] is None:
    st.title("Data Analytics Hub")
    st.markdown("Welcome! Choose a sector to explore its use cases.")

    cols = st.columns(5)
    for idx, (sector_name, usecases) in enumerate(sectors.items()):
        with cols[idx]:
            thumb_file = f"{sector_name.lower().replace(' ', '_')}_thumb.png"
            thumb_path = os.path.join(ASSETS_DIR, thumb_file)
            if os.path.exists(thumb_path):
                st.image(thumb_path, use_container_width=True)
            else:
                st.warning(f"Thumbnail not found for {sector_name}")

            st.markdown(f"### {sector_name}")
            st.write(f"Explore {len(usecases)} use cases in {sector_name}.")

            if st.button(f"Explore {sector_name}", key=sector_name):
                st.session_state["sector"] = sector_name
                st.experimental_rerun()  # safe here

# --- Sector Page ---
else:
    sector_name = st.session_state["sector"]
    st.header(f"{sector_name} Use Cases")
    st.markdown("Select a use case to go to its project page.")

    usecases = sectors[sector_name]

    # 3-column grid
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

                if st.button(f"Go to {uc['name']}", key=uc["name"]):
                    # ✅ Use session state to avoid rerun inside loop
                    st.session_state["go_to_page"] = uc["page"]
                    st.experimental_rerun()

    # Back button
    if st.button("⬅️ Back to Sectors"):
        st.session_state["sector"] = None
        st.experimental_rerun()
