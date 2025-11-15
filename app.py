import streamlit as st
import os

# --- Streamlit setup ---
st.set_page_config(
    page_title="Data Analytics Hub",
    layout="wide",
)

# Hide default sidebar
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
PAGES_DIR = os.path.join(BASE_DIR, "pages")

# --- Hierarchy Data ---
sectors = {
    "Marketing Analytics": [
        {"name": f"Marketing Use Case {i+1}", "image": "real_estate_preview.jpg", "page": f"marketing_{i+1}.py"} 
        for i in range(10)
    ],
    "Real Estate Analytics": [
        {"name": f"Real Estate Use Case {i+1}", "image": "real_estate_preview.jpg", "page": f"{i+1}.py"} 
        for i in range(10)
    ],
    "Customer Intelligence": [
        {"name": f"Customer Use Case {i+1}", "image": "real_estate_preview.jpg", "page": f"customer_{i+1}.py"} 
        for i in range(10)
    ],
    "Sales & Revenue Analytics": [
        {"name": f"Sales Use Case {i+1}", "image": "real_estate_preview.jpg", "page": f"sales_{i+1}.py"} 
        for i in range(10)
    ],
    "Operational Insights": [
        {"name": f"Operations Use Case {i+1}", "image": "real_estate_preview.jpg", "page": f"ops_{i+1}.py"} 
        for i in range(10)
    ],
}

# --- Session state ---
if "sector" not in st.session_state:
    st.session_state["sector"] = None

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
                st.experimental_rerun()

# --- Sector Page ---
else:
    sector_name = st.session_state["sector"]
    st.header(f"{sector_name} Use Cases")
    st.markdown("Select a use case to go to its project page.")
    
    usecases = sectors[sector_name]
    
    # Display use cases in 3-column grid
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
                
                # Remove .py extension for switch_page
                page_file = uc["page"]           # e.g., "8.py"
                page_name = page_file.replace(".py", "")  # remove extension
                
                if st.button(f"Go to {uc['name']}", key=uc["name"]):
                    try:
                        st.switch_page(page_name)  # just the name, no path
                    except Exception as e:
                        st.error(f"⚠️ Could not link to '{page_name}'. Make sure it exists in /pages/ folder.\nError: {e}")

    # Back button to Home
    if st.button("⬅️ Back to Sectors"):
        st.session_state["sector"] = None
        st.experimental_rerun()
