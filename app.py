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

# --- Directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# --- Sector Data ---
sectors = {
    "Marketing Analytics": [
        {"name": "Marketing Campaign Performance Analyzer", "image": "marketing_thumb.jpg", "page": "marketing_1.py"},
        {"name": "Marketing Intelligence & Forecasting Lab", "image": "marketing_thumb.jpg", "page": "marketing_2.py"},
        {"name": "Click & Convertion Analytics", "image": "marketing_thumb.jpg", "page": "marketing_3.py"},
        {"name": "Marketing Performance Analysis", "image": "marketing_thumb.jpg", "page": "marketing_4.py"},
        {"name": "Content & SEO Performance Dashboard", "image": "marketing_thumb.jpg", "page": "marketing_5.py"},
        {"name": "Customer Retention & Churn Analysis", "image": "marketing_thumb.jpg", "page": "marketing_6.py"},
        {"name": "Customer Journey & Funnel Insights", "image": "marketing_thumb.jpg", "page": "marketing_7.py"},
        {"name": "Google Ads Performance Analytics", "image": "marketing_thumb.jpg", "page": "marketing_8.py"},
        {"name": "Email & WhatsApp Marketing Forecast Lab", "image": "marketing_thumb.jpg", "page": "marketing_9.py"},
    ],

    "Real Estate Analytics": [
        {"name": "Real Estate Intelligence Suite", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_1.py"},
        {"name": "Real Estate Demand Forecasting System", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_2.py"},
        {"name": "Price vs Property Features Analyzer", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_3.py"},
        {"name": "Agent & Market Insights Dashboard", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_4.py"},
        {"name": "Real Estate Investment Opportunity Analyzer", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_5.py"},
        {"name": "Tenant Risk & Market Trend Analyzer", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_6.py"},
        {"name": "Rental Yield & Investment Analyzer", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_7.py"},
        {"name": "Real Estate Buyer Sentiment Analyzer", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_8.py"},
        {"name": "Neighborhood Lifestyle & Risk Aware Analyzer", "image": "real_estate_thumb.jpg", "page": "usecase_real_estate_9.py"},
        {"name": "Real Estate Intelligence — Hybrid Dashboard", "image": "real_estate_thumb.jpg", "page": "realestate.py"},
    ],

    "Health Care Analytics": [
        {"name": "Healthscope Insights", "image": "healthcare_thumb.jpg", "page": "healthcare_1.py"},
        {"name": "Patient Visit Analytics & Hospital Performance", "image": "healthcare_thumb.jpg", "page": "healthcare_2.py"},
        {"name": "PatientFlow Navigator", "image": "healthcare_thumb.jpg", "page": "healthcare_3.py"},
        {"name": "Ambulance Ops & Routing Lab", "image": "healthcare_thumb.jpg", "page": "healthcare_4.py"},
        {"name": "Health Care Analytics 1", "image": "healthcare_thumb.jpg", "page": "healthcare_5.py"},
        {"name": "Health Care Analytics 2", "image": "healthcare_thumb.jpg", "page": "healthcare_6.py"},
    ],
}

# --- Thumbnails for home cards ---
home_thumbs = {
    "Marketing Analytics": os.path.join(ASSETS_DIR, "marketing_thumb.jpg"),
    "Real Estate Analytics": os.path.join(ASSETS_DIR, "real_estate_thumb.jpg"),
    "Health Care Analytics": os.path.join(ASSETS_DIR, "healthcare_thumb.jpg"),
}

# --- Session state ---
if "sector" not in st.session_state:
    st.session_state["sector"] = None

# ------------------------
# CLEAN CARD CSS
# ------------------------
st.markdown("""
<style>
.card-box {
    border: 2px solid #064b86;
    border-radius: 14px;
    padding: 18px;
    background: white;
    margin-bottom: 30px;
}
.card-box:hover {
    transform: translateY(-4px);
    transition: 0.2s;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# HOME PAGE (3 Cards Grid)
# -------------------------

if st.session_state["sector"] is None:

    st.title("Data Analytics Solutions")
    st.write("Choose a sector to explore:")

    sector_names = list(sectors.keys())

    # 3 card grid
    rows = [sector_names[i:i+3] for i in range(0, len(sector_names), 3)]

    for row in rows:
        cols = st.columns(3)

        for col, sector_name in zip(cols, row):
            with col:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)

                # thumbnail
                thumb_path = home_thumbs.get(sector_name)
                if os.path.exists(thumb_path):
                    st.image(thumb_path, use_container_width=True)

                # title
                st.markdown(
                    f"<h3 style='color:#064b86; margin-top:10px;'>{sector_name}</h3>",
                    unsafe_allow_html=True
                )

                # usecase count
                st.write(f"{len(sectors[sector_name])} use cases available.")

                # button
                if st.button(f"Explore {sector_name}", key=f"home_{sector_name}"):
                    st.session_state["sector"] = sector_name

                st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# SECTOR PAGE (Usecase grid)
# -------------------------

else:
    sector_name = st.session_state["sector"]
    st.header(f"{sector_name} Use Cases")

    usecases = sectors[sector_name]

    for i in range(0, len(usecases), 3):
        cols = st.columns(3)

        for j, uc in enumerate(usecases[i:i+3]):
            with cols[j]:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)

                img_path = os.path.join(ASSETS_DIR, uc["image"])
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)

                st.markdown(f"### {uc['name']}")

                if st.button(f"Open", key=uc["name"]):
                    st.switch_page(f"pages/{uc['page']}")

                st.markdown("</div>", unsafe_allow_html=True)
