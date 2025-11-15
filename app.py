import streamlit as st
import os
import webbrowser

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

# --- Hierarchy Data ---
sectors = {
    "Marketing Analytics": [
        {"name": f"Marketing Use Case {i+1}", 
         "image": "marketing_thumb.png", 
         "url": f"https://github.com/Analytics-Avenue/streamlit-dataapp/blob/main/pages/{i+1}.py"} 
        for i in range(1, 9)  # adjust range if more pages
    ],
    "Real Estate Analytics": [
        {"name": f"Real Estate Use Case {i+1}", 
         "image": "real_estate_thumb.png", 
         "url": "https://github.com/Analytics-Avenue/streamlit-dataapp/blob/main/pages/{i+1}.py"}  # last URL
    ],
    "Customer Intelligence": [
        {"name": f"Customer Use Case {i+1}", 
         "image": "customer_thumb.png", 
         "url": "https://github.com/Analytics-Avenue/streamlit-dataapp/blob/main/pages/customer_{i+1}.py"} 
        for i in range(1, 10)
    ],
    "Sales & Revenue Analytics": [
        {"name": f"Sales Use Case {i+1}", 
         "image": "sales_thumb.png", 
         "url": "https://github.com/Analytics-Avenue/streamlit-dataapp/blob/main/pages/sales_{i+1}.py"} 
        for i in range(1, 10)
    ],
    "Operational Insights": [
        {"name": f"Operations Use Case {i+1}", 
         "image": "ops_thumb.png", 
         "url": "https://github.com/Analytics-Avenue/streamlit-dataapp/blob/main/pages/ops_{i+1}.py"} 
        for i in range(1, 10)
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
    st.markdown("Select a use case to open its project page on GitHub.")
    
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
                
                if st.button(f"Go to {uc['name']}", key=uc["name"]):
                    st.markdown(f"[Open Project]({uc['url']})", unsafe_allow_html=True)

    # Back button to Home
    if st.button("⬅️ Back to Sectors"):
        st.session_state["sector"] = None
        st.experimental_rerun()
