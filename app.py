import streamlit as st
import os

# -------------------------------
# YOUR DATA (Replace these)
# -------------------------------

# Example thumbnails
home_thumbs = {
    "Marketing Analytics": "thumbs/marketing.jpg",
    "Retail Analytics": "thumbs/retail.jpg",
    "Healthcare Analytics": "thumbs/health.jpg",
}

# Example overview
sector_overview = {
    "Marketing Analytics": "Turn raw marketing data into insights that drive conversions.",
    "Retail Analytics": "Track your store operations, customer behavior, and revenue flows.",
    "Healthcare Analytics": "Leverage analytics for diagnosis, optimization and efficiency."
}

# Example tools
sector_tools = {
    "Marketing Analytics": ["Python", "SQL", "Tableau", "Power BI", "ML Models", "NLP", "Excel"],
    "Retail Analytics": ["Python", "SQL", "Power BI", "Tableau", "Forecasting", "Excel"],
    "Healthcare Analytics": ["Python", "R", "Tableau", "ML Models", "Deep Learning"]
}

# Example usecases (You already have these in your app)
sectors = {
    "Marketing Analytics": {
        "Campaign Analyzer": "app_marketing_campaign",
        "Churn Prediction": "app_churn"
    },
    "Retail Analytics": {
        "Inventory Forecaster": "app_inventory",
        "Sales Tracker": "app_sales"
    },
    "Healthcare Analytics": {
        "Patient Diagnosis ML": "app_patient",
        "Hospital Analytics": "app_hospital"
    }
}

# -------------------------------
# STYLES
# -------------------------------

st.markdown("""
<style>

.sector-card {
    border: 2px solid #064b86;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 25px;
    box-shadow: 0 0 14px rgba(6,75,134,0.18);
    transition: 0.25s;
    background: #ffffff;
}

.sector-card:hover {
    box-shadow: 0 0 25px rgba(6,75,134,0.45);
    transform: translateY(-4px);
}

.tool-chip {
    background: #f2f6ff;
    padding: 6px 10px;
    border-radius: 6px;
    border: 1px solid #d5e3ff;
    text-align: center;
    font-size: 12.5px;
    font-weight: 600;
    margin-bottom: 8px;
    box-shadow: 0 0 6px rgba(180,200,255,0.35);
    transition: 0.2s;
}

.tool-chip:hover {
    background: #e8eeff;
    transform: scale(1.05);
}

</style>
""", unsafe_allow_html=True)


# -------------------------------
# SESSION STATE
# -------------------------------

if "sector" not in st.session_state:
    st.session_state["sector"] = None


# -------------------------------
# HOME PAGE (3 CARDS/ROW)
# -------------------------------

if st.session_state["sector"] is None:

    st.title("Data Analytics Solutions")
    st.write("Choose a sector below")

    sector_list = list(sectors.keys())
    rows = [sector_list[i:i+3] for i in range(0, len(sector_list), 3)]

    for row in rows:
        cols = st.columns(3)

        for col, sector_name in zip(cols, row):
            with col:
                st.markdown("<div class='sector-card'>", unsafe_allow_html=True)

                # Thumbnail
                thumb = home_thumbs.get(sector_name)
                if thumb and os.path.exists(thumb):
                    st.image(thumb, use_container_width=True)

                # Title
                st.markdown(
                    f"<h3 style='margin-top:10px; color:#064b86;'>{sector_name}</h3>",
                    unsafe_allow_html=True
                )

                # Overview
                st.markdown(
                    f"<p style='font-size:14px; color:#333;'>{sector_overview[sector_name]}</p>",
                    unsafe_allow_html=True
                )

                # Tools heading
                st.markdown("<h5 style='margin-top:8px;'>Tools & Tech</h5>",
                            unsafe_allow_html=True)

                # Tools grid
                tools = sector_tools[sector_name]
                tool_rows = [tools[i:i+5] for i in range(0, len(tools), 5)]

                for tr in tool_rows:
                    tool_cols = st.columns(5)
                    for tc, tool in zip(tool_cols, tr):
                        with tc:
                            st.markdown(
                                f"<div class='tool-chip'>{tool}</div>",
                                unsafe_allow_html=True
                            )

                # Explore button
                if st.button(f"Explore {sector_name}", key=f"btn_{sector_name}"):
                    st.session_state["sector"] = sector_name

                st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------
# SECTOR PAGE
# -------------------------------

else:
    sector_name = st.session_state["sector"]

    st.title(sector_name)
    st.write("Select a use case:")

    usecases = sectors[sector_name]
    for uc_name, app_file in usecases.items():
        if st.button(uc_name):
            st.write(f"Launching {uc_name}... (this is where you load {app_file}.py)")

    if st.button("Back to Home"):
        st.session_state["sector"] = None
