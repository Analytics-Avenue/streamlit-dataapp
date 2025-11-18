import streamlit as st
import os

st.set_page_config(page_title="Data Analytics Solutions", layout="wide")

# ==================================================================
#                           CSS STYLES
# ==================================================================

st.markdown("""
<style>

.sector-box {
    border: 2px solid #064b86;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 35px;
    transition: 0.3s;
    box-shadow: 0 0 10px rgba(6, 75, 134, 0.2);
    background: white;
}

.sector-box:hover {
    box-shadow: 0 0 25px rgba(6, 75, 134, 0.55);
    transform: translateY(-3px);
}

.tool-card {
    background: #f4f8ff;
    border: 1px solid #c9dfff;
    padding: 6px 0;
    border-radius: 6px;
    text-align: center;
    font-weight: 600;
    font-size: 12.5px;
    box-shadow: 0 0 5px rgba(180, 200, 255, 0.35);
    transition: 0.2s;
    margin-bottom: 8px;
}

.tool-card:hover {
    background: #e7efff;
    transform: scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# ==================================================================
#                        SECTOR DEFINITIONS
# ==================================================================

sectors = {
    "Marketing Analytics": [
        {"name": "Customer Segmentation", "image": "sections/marketing/customer_segmentation.png"},
        {"name": "Campaign Analysis", "image": "sections/marketing/campaign_analysis.png"},
        {"name": "Sales Forecasting", "image": "sections/marketing/sales_forecast.png"}
    ],
    "Real Estate Analytics": [
        {"name": "Market Price Prediction", "image": "sections/realestate/price_prediction.png"},
        {"name": "Rental Yield Analysis", "image": "sections/realestate/rental_yield.png"},
        {"name": "Location Scoring", "image": "sections/realestate/location_scoring.png"}
    ],
    "Health Care Analytics": [
        {"name": "Patient Flow Optimization", "image": "sections/healthcare/patient_flow.png"},
        {"name": "Hospital Operations Dashboard", "image": "sections/healthcare/hospital_ops.png"},
        {"name": "Medical Outcome Prediction", "image": "sections/healthcare/outcome_prediction.png"}
    ]
}

home_thumbs = {
    "Marketing Analytics": "sections/marketing/thumbnail.png",
    "Real Estate Analytics": "sections/realestate/thumbnail.png",
    "Health Care Analytics": "sections/healthcare/thumbnail.png"
}

sector_overview = {
    "Marketing Analytics": "Unlock demand, optimize campaigns, decode buying behavior, and fuel ROI-driven decisions.",
    "Real Estate Analytics": "Forecast prices, evaluate properties, detect investment patterns, and analyze city-wide housing trends.",
    "Health Care Analytics": "Improve patient care, streamline hospital operations, and support medical decisions with data-driven insights."
}

sector_tools = {
    "Marketing Analytics": ["SQL", "Python", "Power BI", "Machine Learning", "Forecasting", "GenAI", "Customer Segmentation"],
    "Real Estate Analytics": ["SQL", "Python", "GIS", "Predictive Modeling", "Power BI", "Clustering", "GenAI"],
    "Health Care Analytics": ["SQL", "Python", "Time-Series", "Dashboards", "ML", "GenAI", "Automation"]
}

# ==================================================================
#                      SESSION INIT
# ==================================================================

if "sector" not in st.session_state:
    st.session_state["sector"] = None

# ==================================================================
#                       HOME PAGE
# ==================================================================

if st.session_state["sector"] is None:

    st.title("Data Analytics Solutions")
    st.write("Pick a sector and explore:")

    for sector_name in sectors.keys():

        st.markdown('<div class="sector-box">', unsafe_allow_html=True)

        # Thumbnail
        thumb_path = home_thumbs.get(sector_name)
        if os.path.exists(thumb_path):
            st.image(thumb_path, use_container_width=True)

        # Title
        st.markdown(
            f"<h2 style='margin-top:10px; color:#064b86;'>{sector_name}</h2>",
            unsafe_allow_html=True
        )

        # Overview
        st.markdown(
            f"<p style='font-size:15px; color:#444;'>{sector_overview[sector_name]}</p>",
            unsafe_allow_html=True
        )

        # Tools header
        st.markdown("<h4 style='margin-top:15px;'>Tools & Tech</h4>",
                    unsafe_allow_html=True)

        # Tools grid (5 per row)
        tools = sector_tools[sector_name]
        rows = [tools[i:i+5] for i in range(0, len(tools), 5)]

        for row in rows:
            cols = st.columns(5)
            for col, tool in zip(cols, row):
                with col:
                    st.markdown(
                        f"<div class='tool-card'>{tool}</div>",
                        unsafe_allow_html=True
                    )

        # Explore button
        if st.button(f"Explore {sector_name}", key=f"btn_{sector_name}"):
            st.session_state["sector"] = sector_name

        st.markdown("</div>", unsafe_allow_html=True)

# ==================================================================
#                  INDIVIDUAL SECTOR PAGES
# ==================================================================

else:
    sector = st.session_state["sector"]

    st.title(f"{sector}")
    st.write("Choose a use case:")

    usecases = sectors[sector]
    cols = st.columns(3)

    for idx, usecase in enumerate(usecases):
        with cols[idx % 3]:

            st.markdown("""
            <div style="border: 2px solid #064b86; border-radius: 12px; padding: 15px;
            box-shadow: 0 0 10px rgba(6,75,134,0.2); margin-bottom: 25px;">
            """, unsafe_allow_html=True)

            if os.path.exists(usecase["image"]):
                st.image(usecase["image"], use_container_width=True)

            st.subheader(usecase["name"])

            if st.button(f"Get Solution for {usecase['name']}", key=f"usecase_{usecase['name']}"):
                st.write(f"Generating tailored solution for **{usecase['name']}**...")

            st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Back to Home"):
        st.session_state["sector"] = None
