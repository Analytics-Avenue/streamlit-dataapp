import streamlit as st
import os

# -------------------------
# Company Logo + Name
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display: flex; align-items: center;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Streamlit setup ---
st.set_page_config(page_title="Data Analytics Solutions", layout="wide")
st.markdown("""<style>[data-testid="stSidebarNav"]{display:none;}</style>""", unsafe_allow_html=True)

# -------------------------
# GLOBAL CSS
# -------------------------
st.markdown("""
<style>
.card-box {
    border: 2px solid #064b86;
    border-radius: 15px;
    padding: 15px;
    margin-bottom: 25px;
    background: #ffffff;
    box-shadow: 0 0 10px rgba(6, 75, 134, 0.15);
    transition: 0.25s;
}
.card-box:hover {
    box-shadow: 0 0 22px rgba(6, 75, 134, 0.45);
    transform: translateY(-4px);
}

.tool-btn {
    background: #e8f1ff;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
    border: 1px solid #bcd2ff;
    text-align: center;
    font-weight: 600;
    margin: 3px;
    display: inline-block;
    transition: 0.2s;
}
.tool-btn:hover {
    background: #d8e8ff;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Directories
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# -------------------------
# Sector Data
# -------------------------
sector_overview = {
    "Marketing Analytics": "Unlock customer insights, forecast trends, and track real-time campaign performance.",
    "Real Estate Analytics": "Evaluate market trends, forecast pricing, and analyze property investment insights.",
    "Health Care Analytics": "Enhance patient experience, streamline hospital operations, and increase care efficiency."
}

sector_tools = {
    "Marketing Analytics": ["Python", "SQL", "Power BI", "Tableau", "ML Models", "Pandas", "NLP"],
    "Real Estate Analytics": ["Python", "GIS", "Regression", "Time Series", "Power BI", "GeoSpatial", "ETL"],
    "Health Care Analytics": ["Python", "R", "Power BI", "Forecasting", "ML Models", "NLP"]
}

sectors = {
    "Marketing Analytics": [],
    "Real Estate Analytics": [],
    "Health Care Analytics": [],
}

# thumbnails
home_thumbs = {
    "Marketing Analytics": os.path.join(ASSETS_DIR, "marketing_thumb.jpg"),
    "Real Estate Analytics": os.path.join(ASSETS_DIR, "real_estate_thumb.jpg"),
    "Health Care Analytics": os.path.join(ASSETS_DIR, "healthcare_thumb.jpg"),
}

# -------------------------
# Session State
# -------------------------
if "sector" not in st.session_state:
    st.session_state["sector"] = None

# ============================================================
# HOME PAGE (GRID OF 3 CARDS)
# ============================================================

if st.session_state["sector"] is None:

    st.title("Data Analytics Solutions")
    st.write("Choose a sector to explore:")

    sector_names = list(sector_overview.keys())

    # Split into rows of 3
    rows = [sector_names[i:i+3] for i in range(0, len(sector_names), 3)]

    for row in rows:
        cols = st.columns(3)

        for col, sector_name in zip(cols, row):
            with col:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)

                # Thumbnail
                thumb_path = home_thumbs.get(sector_name)
                if os.path.exists(thumb_path):
                    st.image(thumb_path, use_container_width=True)

                # Title
                st.markdown(
                    f"<h3 style='color:#064b86; margin-top:10px;'>{sector_name}</h3>",
                    unsafe_allow_html=True
                )

                # Overview
                st.markdown(
                    f"<p style='font-size:14.5px; color:#444;'>{sector_overview[sector_name]}</p>",
                    unsafe_allow_html=True
                )

                # Tools grid
                st.markdown("<b>Tools & Tech:</b><br>", unsafe_allow_html=True)

                tools = sector_tools[sector_name]
                tool_html = "".join([f"<span class='tool-btn'>{t}</span>" for t in tools])
                st.markdown(tool_html, unsafe_allow_html=True)

                # Explore button
                if st.button(f"Explore {sector_name}", key=f"explr_{sector_name}"):
                    st.session_state["sector"] = sector_name

                st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# SECTOR PAGE (YOU CAN ADD YOUR USE CASE CODE HERE)
# ============================================================

else:
    sector_name = st.session_state["sector"]
    st.header(f"{sector_name} – Use Cases Coming Soon")

    if st.button("Back to Home"):
        st.session_state["sector"] = None
