import streamlit as st
import os

# -------------------------
# Company Logo + Name
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

# --- Streamlit setup ---
st.set_page_config(page_title="Data Analytics Solutions", layout="wide")

st.markdown(f"""
<div style="display:flex; align-items:center; margin-bottom:20px;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:bold;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:bold;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Hide Streamlit sidebar nav
st.markdown("""<style>[data-testid="stSidebarNav"]{display:none;}</style>""", unsafe_allow_html=True)


# --- CSS for glow, cards, layout ---
st.markdown("""
<style>

.sector-box {
    padding: 25px;
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e0e0e0;
    transition: 0.3s;
}

.sector-box:hover {
    border: 1px solid #4fa3ff;
    box-shadow: 0 0 20px rgba(79,163,255,0.45);
}

.tool-card {
    text-align: center;
    padding: 15px 10px;
    background: #f8f9fa;
    border-radius: 10px;
    border: 1px solid #e0e0e0;
    font-weight: 600;
    transition: 0.25s;
    font-size: 15px;
}

.tool-card:hover {
    background: #e9f3ff;
    border-color: #4fa3ff;
    box-shadow: 0 0 15px rgba(79,163,255,0.4);
    transform: translateY(-4px);
}

.tool-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-top: 18px;
}

</style>
""", unsafe_allow_html=True)


# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")


# --- Hierarchy Data ---
sectors = {
    "Marketing Analytics": [
        {"name": "Marketing Campaign Performance Analyzer", "image": "marketing_thumb.jpg", "page": "marketing_1.py"},
        {"name": "Marketing Intelligence & Forecasting Lab", "image": "marketing_thumb.jpg", "page": "marketing_2.py"},
        {"name": "Click & Convertion Analytics", "image": "marketing_thumb.jpg", "page": "marketing_3.py"},
        {"name": "Marketing Performance Analysis", "image": "marketing_thumb.jpg", "page": "marketing_4.py"},
        {"name": "Content & SEO Performance Dashboard", "image": "marketing_thumb.jpg", "page": "marketing_5.py"},
        {"name": "Customer Retention & Churn Analysis", "image": "marketing_thumb.jpg", "page": "marketing_6.py"},
        {"name": "Customer Journey & Funnel Insights,", "image": "marketing_thumb.jpg", "page": "marketing_7.py"},
        {"name": "Google Ads Performance Analytics.", "image": "marketing_thumb.jpg", "page": "marketing_8.py"},
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
        {"name": "Real Estate Intelligence — Hybrid Dashboard (Property + CRM)", "image": "real_estate_thumb.jpg", "page": "realestate.py"},
    ],
    "Health Care Analytics": [
        {"name": "Healthscope Insights", "image": "real_estate_thumb.jpg", "page": "healthcare_1.py"},
        {"name": "Patient Visit Analytics & Hospital Performance", "image": "real_estate_thumb.jpg", "page": "healthcare_2.py"},
        {"name": "PatientFlow Navigator", "image": "real_estate_thumb.jpg", "page": "healthcare_3.py"},
        {"name": "Ambulance Ops & Routing Lab", "image": "real_estate_thumb.jpg", "page": "healthcare_4.py"},
        {"name": "Health Care Analytics1", "image": "real_estate_thumb.jpg", "page": "healthcare_5.py"},
        {"name": "Health Care Analytics2", "image": "real_estate_thumb.jpg", "page": "healthcare_6.py"},
    ],
}

# --- Home page thumbnails ---
home_thumbs = {
    "Marketing Analytics": os.path.join(ASSETS_DIR, "marketing_thumb.jpg"),
    "Real Estate Analytics": os.path.join(ASSETS_DIR, "real_estate_thumb.jpg"),
    "Health Care Analytics": os.path.join(ASSETS_DIR, "healthcare_thumb.jpg"),
}

# Overview text for sectors
overview_text = {
    "Marketing Analytics": """
Marketing analytics transforms campaigns from guesswork into data-backed performance engines. 
It tracks customer behavior, conversions, funnels, budgets, retention and forecasting to make 
marketing teams more strategic & revenue-focused. This helps optimize spending, improve customer 
targeting, refine content and build long-term loyalty.  
""",

    "Real Estate Analytics": """
Real estate analytics powers pricing intelligence, demand forecasting, customer profiling, 
market insights and investment performance measurement. Developers, agents and investors make 
faster and sharper decisions by understanding micro-markets, locality scoring and buyer sentiment.  
""",

    "Health Care Analytics": """
Healthcare analytics improves patient care, hospital performance, treatment efficiency, 
resource planning, cost optimization and workflow automation. It enhances doctor productivity, 
reduces wait times and strengthens decision-making across the entire medical ecosystem.  
"""
}

# --- Session state ---
if "sector" not in st.session_state:
    st.session_state["sector"] = None

# --- Sidebar ---
st.sidebar.title("Navigation")
sidebar_choice = st.sidebar.radio(
    "Go to:",
    ["Home"] + list(sectors.keys()),
    index=0 if st.session_state["sector"] is None else list(sectors.keys()).index(st.session_state["sector"]) + 1
)

# Handle sidebar navigation
if sidebar_choice == "Home":
    st.session_state["sector"] = None
else:
    st.session_state["sector"] = sidebar_choice


# -------------------------
# HOME PAGE
# -------------------------
if st.session_state["sector"] is None:
    st.title("Data Analytics Solutions")
    st.markdown("Welcome! Choose a sector to explore its use cases:")

    cols = st.columns(len(sectors))

    for idx, (sector_name, usecases) in enumerate(sectors.items()):
        with cols[idx]:
            thumb = home_thumbs.get(sector_name)
            if os.path.exists(thumb):
                st.image(thumb, use_container_width=True)

            st.markdown(f"### {sector_name}")
            st.write(f"{len(usecases)} use cases available.")

            if st.button(f"Explore {sector_name}", key=sector_name):
                st.session_state["sector"] = sector_name


# -------------------------
# SECTOR PAGE
# -------------------------
else:
    sector_name = st.session_state["sector"]
    st.header(f"{sector_name} Use Cases")

    # Overview + Tools Section
    st.markdown(f"""
    <div class="sector-box">
        <h3 style="color:#064b86;">Overview</h3>
        <p style="font-size:16px; line-height:1.6;">
            {overview_text[sector_name]}
        </p>

        <h3 style="color:#064b86; margin-top:25px;">Tools & Technologies Used</h3>

        <div class="tool-grid">
            <div class="tool-card">SQL</div>
            <div class="tool-card">Python</div>
            <div class="tool-card">Pandas</div>
            <div class="tool-card">NumPy</div>
            <div class="tool-card">Power BI</div>
            <div class="tool-card">Machine Learning</div>
            <div class="tool-card">Deep Learning</div>
            <div class="tool-card">GenAI</div>
            <div class="tool-card">APIs</div>
            <div class="tool-card">Cloud Compute</div>
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Projects")

    usecases = sectors[sector_name]

    # 3-column grid
    for i in range(0, len(usecases), 3):
        cols = st.columns(3)
        for j, uc in enumerate(usecases[i:i+3]):
            with cols[j]:
                thumb = os.path.join(ASSETS_DIR, uc["image"])
                if os.path.exists(thumb):
                    st.image(thumb, use_container_width=True)

                st.markdown(f"### {uc['name']}")
                st.write("Explore insights and dashboards.")

                if st.button(f"Go to {uc['name']}", key=uc["name"]):
                    st.switch_page(f"pages/{uc['page']}")
