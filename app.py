import streamlit as st
import os

st.set_page_config(page_title="Data Analytics Solutions", layout="wide")

# Hide sidebar nav for clean UI
st.markdown("<style>[data-testid='stSidebarNav']{display:none;}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------
# PATH SETUP (FIXED)
# ------------------------------------------------------------
if "__file__" in globals():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    BASE_DIR = os.getcwd()

ASSETS_DIR = os.path.join(BASE_DIR, "assets")


def load_image(img):
    """Return a valid path or None."""
    path = os.path.join(ASSETS_DIR, img)
    if os.path.exists(path):
        return path
    return None


# ------------------------------------------------------------
# SECTOR DATA
# ------------------------------------------------------------

sector_details = {
    "Marketing Analytics": {
        "overview": """Marketing Analytics helps organizations decode customer behavior, optimize campaigns, 
        improve conversion funnels, and measure ROI. The focus is on predicting trends, automating insights, 
        and delivering smarter decision frameworks.""",

        "tools": ["Python", "Pandas", "NumPy", "Power BI", "Streamlit",
                  "Google Analytics", "TensorFlow", "Scikit-learn", "SQL", "Matplotlib"]
    },

    "Real Estate Analytics": {
        "overview": """Real Estate Analytics provides market forecasting, property valuation, price modeling, 
        agent performance insights, and customer segmentation. It enables investors and builders to make data-driven 
        strategic decisions with precision.""",

        "tools": ["Python", "PropTech APIs", "Power BI", "XGBoost", "GeoPandas",
                  "SQL", "Tableau", "Matplotlib", "Seaborn", "NumPy"]
    },

    "Health Care Analytics": {
        "overview": """Healthcare Analytics empowers hospitals with data-driven patient flow prediction, 
        performance dashboards, operational forecasting, and clinical insights to improve efficiency, safety, 
        and patient experience.""",

        "tools": ["Python", "Power BI", "Healthcare APIs", "SciPy", "Streamlit",
                  "SQL", "TensorFlow", "Matplotlib", "Pandas", "HL7 Standards"]
    }
}


# ------------------------------------------------------------
# USE CASES (Simplified)
# ------------------------------------------------------------

usecases = {
    "Marketing Analytics": [
        {"name": "Marketing Campaign Performance Analyzer", "image": "marketing_thumb.jpg"},
        {"name": "Marketing Intelligence & Forecasting Lab", "image": "marketing_thumb.jpg"},
        {"name": "Click & Conversion Analytics", "image": "marketing_thumb.jpg"},
    ],

    "Real Estate Analytics": [
        {"name": "Real Estate Intelligence Suite", "image": "real_estate_thumb.jpg"},
        {"name": "Demand Forecasting System", "image": "real_estate_thumb.jpg"},
        {"name": "Price vs Property Features Analyzer", "image": "real_estate_thumb.jpg"},
    ],

    "Health Care Analytics": [
        {"name": "Healthscope Insights", "image": "healthcare_thumb.jpg"},
        {"name": "Patient Flow Navigator", "image": "healthcare_thumb.jpg"},
        {"name": "Ambulance Ops Routing Lab", "image": "healthcare_thumb.jpg"},
    ],
}


# ------------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------------

st.title("Data Analytics Solutions")
st.write("Explore sectors below:")

cols = st.columns(3)

for idx, sector in enumerate(sector_details.keys()):
    with cols[idx]:
        img = load_image(f"{sector.lower().replace(' ', '_')}_thumb.jpg")
        if img:
            st.image(img, use_container_width=True)

        st.markdown(f"<h3 style='text-align:center;'>{sector}</h3>", unsafe_allow_html=True)

        if st.button(f"Explore {sector}", key=sector):
            st.session_state["sector"] = sector


# ------------------------------------------------------------
# SECTOR PAGE
# ------------------------------------------------------------
if "sector" in st.session_state:

    sector = st.session_state["sector"]
    data = sector_details[sector]

    st.header(f"{sector}")
    st.write("")

    # ------------------------------------------------------------
    # HOVER CARD CONTAINER
    # ------------------------------------------------------------
    st.markdown(
        """
        <style>
        .hover-card {
            background: rgba(255,255,255,0.8);
            border-radius: 12px;
            padding: 25px;
            border: 2px solid #d4d4d4;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .hover-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 20px 45px rgba(0,0,0,0.22);
        }
        .tool-tag {
            display: inline-block;
            background: #064b86;
            color: white;
            padding: 4px 10px;
            border-radius: 6px;
            margin: 4px;
            font-size: 13px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Card container
    st.markdown("<div class='hover-card'>", unsafe_allow_html=True)

    # Thumbnail
    img_path = load_image(f"{sector.lower().replace(' ', '_')}_thumb.jpg")
    if img_path:
        st.image(img_path, use_container_width=True)

    # Overview
    st.subheader("Overview")
    st.write(data["overview"])

    # Tools
    st.subheader("Tools & Technologies")

    tools = data["tools"]

    # First 5 tools in line 1
    row1 = tools[:5]
    # Remaining next row
    row2 = tools[5:]

    st.write("### Primary Tools")
    st.markdown("".join([f"<span class='tool-tag'>{t}</span>" for t in row1]), unsafe_allow_html=True)

    if row2:
        st.write("### Additional Tools")
        st.markdown("".join([f"<span class='tool-tag'>{t}</span>" for t in row2]), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.subheader("Use Cases")

    # Use Case Grid (3 columns)
    ucs = usecases[sector]

    for i in range(0, len(ucs), 3):
        row = st.columns(3)
        for j, uc in enumerate(ucs[i:i+3]):
            with row[j]:
                thumb = load_image(uc["image"])
                if thumb:
                    st.image(thumb, use_container_width=True)
                st.markdown(f"### {uc['name']}")
                st.button("Open", key=f"{uc['name']}_btn")


# END OF CODE
