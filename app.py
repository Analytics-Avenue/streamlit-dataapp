import streamlit as st
import os

# Dummy data (replace with your actual dictionaries)
sectors = {
    "Retail Analytics": ["Usecase1", "Usecase2"],
    "Marketing Analytics": ["Usecase1", "Usecase2"],
    "Healthcare Analytics": ["Usecase1", "Usecase2"],
    "Finance Analytics": ["Usecase1", "Usecase2"],
    "Real Estate Analytics": ["Usecase1", "Usecase2"],
    "Supply Chain Analytics": ["Usecase1", "Usecase2"],
}

sector_overview = {
    "Retail Analytics": "Retail sector overview goes here.",
    "Marketing Analytics": "Marketing overview goes here.",
    "Healthcare Analytics": "Healthcare overview goes here.",
    "Finance Analytics": "Finance overview goes here.",
    "Real Estate Analytics": "Real Estate overview goes here.",
    "Supply Chain Analytics": "Supply Chain overview goes here."
}

sector_tools = {
    "Retail Analytics": ["Python", "SQL", "Power BI", "TensorFlow", "Excel", "Snowflake"],
    "Marketing Analytics": ["Python", "SQL", "Google Analytics", "Power BI", "dbt"],
    "Healthcare Analytics": ["R", "Python", "SQL", "Tableau"],
    "Finance Analytics": ["Python", "SQL", "Power BI", "SAS"],
    "Real Estate Analytics": ["Python", "SQL", "ArcGIS", "Power BI", "Excel"],
    "Supply Chain Analytics": ["Python", "SQL", "Power BI", "SAP BI"]
}

home_thumbs = {
    "Retail Analytics": "thumbs/retail.jpg",
    "Marketing Analytics": "thumbs/marketing.jpg",
    "Healthcare Analytics": "thumbs/healthcare.jpg",
    "Finance Analytics": "thumbs/finance.jpg",
    "Real Estate Analytics": "thumbs/realestate.jpg",
    "Supply Chain Analytics": "thumbs/supply.jpg"
}

# ===================== CSS ======================
st.markdown("""
<style>

.sector-card {
    border: 2px solid #064b86;
    border-radius: 14px;
    padding: 18px;
    background: #ffffff;
    box-shadow: 0 0 10px rgba(6, 75, 134, 0.15);
    transition: 0.25s;
    height: 100%;
}

.sector-card:hover {
    box-shadow: 0 0 25px rgba(6, 75, 134, 0.45);
    transform: translateY(-4px);
}

.tool-pill {
    background: #f4f8ff;
    border: 1px solid #c9dfff;
    padding: 6px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    margin: 4px;
    white-space: nowrap;
}

.tool-pill:hover {
    background: #e7efff;
}

</style>
""", unsafe_allow_html=True)

# ===================== HOME PAGE ======================
st.title("Data Analytics Solutions")
st.write("Choose a sector to explore insights:")

sector_list = list(sectors.keys())

# 3 cards per row
for i in range(0, len(sector_list), 3):
    row = st.columns(3)

    for idx, sector_name in enumerate(sector_list[i:i+3]):
        with row[idx]:
            with st.container():
                st.markdown('<div class="sector-card">', unsafe_allow_html=True)

                # Thumbnail
                thumb = home_thumbs.get(sector_name, None)
                if thumb and os.path.exists(thumb):
                    st.image(thumb, use_container_width=True)

                # Title
                st.markdown(
                    f"<h3 style='margin-top:10px; color:#064b86; text-align:center;'>{sector_name}</h3>",
                    unsafe_allow_html=True
                )

                # Overview
                st.markdown(
                    f"<p style='font-size:14px; color:#444; text-align:center;'>{sector_overview[sector_name]}</p>",
                    unsafe_allow_html=True
                )

                # Tools Section
                st.markdown("<h5 style='margin-top:10px;'>Tools & Tech</h5>", unsafe_allow_html=True)

                tool_list = sector_tools[sector_name]
                tools_html = "".join([f"<span class='tool-pill'>{tool}</span>" for tool in tool_list])
                st.markdown(tools_html, unsafe_allow_html=True)

                # Explore Button
                if st.button(f"Explore", key=f"btn_{sector_name}"):
                    st.session_state["sector"] = sector_name

                st.markdown("</div>", unsafe_allow_html=True)
