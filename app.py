import streamlit as st

st.set_page_config(page_title="Sector Overview", layout="wide")

# ------------------ CSS ------------------
st.markdown("""
<style>

body {
    background-color: #f6f8fc;
}

.card-box {
    border: 1px solid #c9d7f0;
    border-radius: 14px;
    padding: 15px;
    background: #ffffff;
    min-height: 540px;
    transition: 0.25s ease-in-out;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    margin-bottom: 25px;
    position: relative;
}

.card-box:hover {
    transform: translateY(-6px);
    box-shadow: 0 6px 22px rgba(0,0,0,0.18);
    border-color: #7fa8ff;
    background: #f9fbff;
}

/* Thumbnail container */
.thumbnail-wrapper {
    position: relative;
    width: 100%;
}

/* Thumbnail image */
.thumb {
    width: 100%;
    border-radius: 10px;
    transition: 0.25s ease-in-out;
    position: relative;
    z-index: 2;
}

/* Overlay behind image */
.thumb-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    border-radius: 10px;
    background: rgba(120,150,255,0.25);
    box-shadow: 0 0 18px rgba(120,150,255,0.45);
    opacity: 0;
    transition: 0.25s ease-in-out;
    z-index: 1;
}

/* Hover activates overlay */
.thumbnail-wrapper:hover .thumb-overlay {
    opacity: 1;
}

/* Tool chip */
.tool-btn {
    background: #eef4ff;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 12px;
    border: 1px solid #c6d7ff;
    margin: 4px;
    display: inline-block;
    font-weight: 600;
    transition: 0.2s;
}

.tool-btn:hover {
    background: #d9e7ff;
    transform: scale(1.05);
}

/* Titles */
.card-title {
    font-size: 20px;
    font-weight: 700;
    margin-top: 12px;
}

.card-sub {
    font-size: 15px;
    font-weight: 600;
    margin-top: 18px;
}

</style>
""", unsafe_allow_html=True)


# ------------------ DATA ------------------

sector_overviews = {
    "Marketing Analytics": (
        "Marketing analytics helps businesses understand customer behavior, optimize ad spend, "
        "and measure performance across campaigns. It connects data from multiple channels to create "
        "clear insights for decision-making and growth."
    ),

    "Real Estate Analytics": (
        "Real estate analytics deals with property pricing, market demand, investment potential, "
        "and location intelligence. It blends geospatial insights, forecasting, and financial modeling "
        "to support developers, buyers, and investors."
    ),

    "Health Care Analytics": (
        "Health care analytics uses patient data, hospital operations, and treatment outcomes to improve "
        "care quality. It supports forecasting patient flow, identifying risks early, and simplifying "
        "clinical and administrative decisions."
    )
}

sector_tools = {
    "Marketing Analytics": [
        "Python", "SQL", "Excel", "Google Sheets",
        "Power BI", "Tableau", "Looker Studio",
        "Google Analytics", "Meta Ads", "HubSpot",
        "Pandas", "NumPy", "Scikit-Learn",
        "A/B Testing", "Segmentation Models"
    ],

    "Real Estate Analytics": [
        "Python", "SQL", "Excel",
        "Power BI", "Tableau",
        "GeoPandas", "Google Maps API",
        "Regression", "Clustering",
        "Price Prediction Models"
    ],

    "Health Care Analytics": [
        "Python", "SQL", "Excel",
        "Power BI", "Tableau",
        "Classification Models", "Time Series",
        "NLP", "Patient Flow Forecasting"
    ]
}

thumbnail_url = "https://via.placeholder.com/300x160.png?text=Thumbnail"


# ------------------ CARD RENDER FUNCTION ------------------

def render_sector_card(title, overview, tools):
    st.markdown(f"""
    <div class='card-box'>
        
        <!-- Thumbnail -->
        <div class='thumbnail-wrapper'>
            <img src='{thumbnail_url}' class='thumb'>
            <div class='thumb-overlay'></div>
        </div>

        <!-- Title -->
        <div class='card-title'>{title}</div>

        <!-- Overview -->
        <p style='font-size:14px; margin-top:8px;'>{overview}</p>

        <!-- Tools -->
        <div class='card-sub'>Tools & Tech</div>
        <div>
    """, unsafe_allow_html=True)

    # Add tools
    tool_html = "".join([f"<span class='tool-btn'>{t}</span>" for t in tools])
    st.markdown(tool_html, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)




# ------------------ LAYOUT ------------------

col1, col2, col3 = st.columns(3)

with col1:
    render_sector_card("Marketing Analytics", sector_overviews["Marketing Analytics"], sector_tools["Marketing Analytics"])

with col2:
    render_sector_card("Real Estate Analytics", sector_overviews["Real Estate Analytics"], sector_tools["Real Estate Analytics"])

with col3:
    render_sector_card("Health Care Analytics", sector_overviews["Health Care Analytics"], sector_tools["Health Care Analytics"])
