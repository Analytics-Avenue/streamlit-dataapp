# app.py
import streamlit as st
import os

# -------------------------
# Config / Assets
# -------------------------
st.set_page_config(page_title="Data Analytics Solutions", layout="wide")

logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# make sure assets dir exists (no crash)
if not os.path.isdir(ASSETS_DIR):
    # no big deal — we'll just use the logo_url instead of local thumbs
    os.makedirs(ASSETS_DIR, exist_ok=True)

# -------------------------
# CSS (glow + cards)
# -------------------------
st.markdown(
    """
<style>
.header-row { display:flex; align-items:center; margin-bottom:18px; }
.header-row img { margin-right:10px; }
.sector-box {
    padding: 20px;
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e6e6e6;
    transition: 0.3s;
}
.sector-box:hover {
    border: 1px solid #4fa3ff;
    box-shadow: 0 0 20px rgba(79,163,255,0.35);
}
.tool-card {
    text-align: center;
    padding: 12px 8px;
    background: #f8f9fa;
    border-radius: 10px;
    border: 1px solid #e0e0e0;
    font-weight: 600;
    transition: 0.2s;
    font-size: 14px;
}
.tool-card:hover {
    background: #e9f3ff;
    border-color: #4fa3ff;
    box-shadow: 0 0 12px rgba(79,163,255,0.28);
    transform: translateY(-4px);
}
.tool-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 14px; }
.sector-thumb { border-radius: 8px; }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Data (hierarchy)
# -------------------------
sectors = {
    "Marketing Analytics": [
        {"name": "Marketing Campaign Performance Analyzer", "image": "marketing_thumb.jpg", "page": "marketing_1.py"},
        {"name": "Marketing Intelligence & Forecasting Lab", "image": "marketing_thumb.jpg", "page": "marketing_2.py"},
        {"name": "Click & Conversion Analytics", "image": "marketing_thumb.jpg", "page": "marketing_3.py"},
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
        {"name": "Real Estate Intelligence — Hybrid Dashboard (Property + CRM)", "image": "real_estate_thumb.jpg", "page": "realestate.py"},
    ],
    "Health Care Analytics": [
        {"name": "Healthscope Insights", "image": "healthcare_thumb.jpg", "page": "healthcare_1.py"},
        {"name": "Patient Visit Analytics & Hospital Performance", "image": "healthcare_thumb.jpg", "page": "healthcare_2.py"},
        {"name": "PatientFlow Navigator", "image": "healthcare_thumb.jpg", "page": "healthcare_3.py"},
        {"name": "Ambulance Ops & Routing Lab", "image": "healthcare_thumb.jpg", "page": "healthcare_4.py"},
        {"name": "Health Care Analytics1", "image": "healthcare_thumb.jpg", "page": "healthcare_5.py"},
        {"name": "Health Care Analytics2", "image": "healthcare_thumb.jpg", "page": "healthcare_6.py"},
    ],
}

# Local thumbnails mapping (if local file exists)
home_thumbs = {
    "Marketing Analytics": os.path.join(ASSETS_DIR, "marketing_thumb.jpg"),
    "Real Estate Analytics": os.path.join(ASSETS_DIR, "real_estate_thumb.jpg"),
    "Health Care Analytics": os.path.join(ASSETS_DIR, "healthcare_thumb.jpg"),
}

overview_text = {
    "Marketing Analytics": (
        "Marketing analytics transforms campaigns from guesswork into data-backed performance engines. "
        "It tracks customer behavior, conversions, funnels, budgets, retention and forecasting to make "
        "marketing teams more strategic & revenue-focused. This helps optimize spending, improve customer "
        "targeting, refine content and build long-term loyalty."
    ),
    "Real Estate Analytics": (
        "Real estate analytics powers pricing intelligence, demand forecasting, customer profiling, "
        "market insights and investment performance measurement. Developers, agents and investors make "
        "faster and sharper decisions by understanding micro-markets, locality scoring and buyer sentiment."
    ),
    "Health Care Analytics": (
        "Healthcare analytics improves patient care, hospital performance, treatment efficiency, "
        "resource planning, cost optimization and workflow automation. It enhances doctor productivity, "
        "reduces wait times and strengthens decision-making across the entire medical ecosystem."
    ),
}

# -------------------------
# Session state + Sidebar
# -------------------------
if "sector" not in st.session_state:
    st.session_state["sector"] = None

st.sidebar.title("Navigation")
all_options = ["Home"] + list(sectors.keys())
# ensure index calculation is safe
try:
    default_index = all_options.index("Home") if st.session_state["sector"] is None else all_options.index(st.session_state["sector"])
except ValueError:
    default_index = 0

sidebar_choice = st.sidebar.radio("Go to:", all_options, index=default_index)

if sidebar_choice == "Home":
    st.session_state["sector"] = None
else:
    st.session_state["sector"] = sidebar_choice

# -------------------------
# Header (common)
# -------------------------
st.markdown(
    f"""
<div class="header-row">
    <img src="{logo_url}" width="60" />
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:30px; font-weight:700;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:30px; font-weight:700;">Advanced Analytics</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------
# HOME PAGE: show only MAIN SECTORS
# -------------------------
if st.session_state["sector"] is None:
    st.title("Data Analytics Solutions")
    st.markdown("Welcome — pick a sector to explore its use cases:")

    sector_names = list(sectors.keys())
    # show sectors in columns (responsive-ish)
    cols = st.columns(len(sector_names))
    for idx, name in enumerate(sector_names):
        with cols[idx]:
            thumb_path = home_thumbs.get(name)
            # if local thumb exists, use it; otherwise fallback to logo_url
            if thumb_path and os.path.exists(thumb_path):
                st.image(thumb_path, use_column_width="always", caption=name, output_format="auto")
            else:
                # fallback image
                st.image(logo_url, use_column_width="always", caption=name)
            st.markdown(f"### {name}")
            st.write(f"{len(sectors[name])} use cases available.")
            # clicking sets session state and re-renders to sector page
            if st.button(f"Explore {name}", key=f"explore_{name}"):
                st.session_state["sector"] = name
                # force rerun so sidebar reflects change
                st.experimental_rerun()


# -------------------------
# SECTOR PAGE: list projects for chosen sector
# -------------------------
else:
    sector_name = st.session_state["sector"]
    st.header(f"{sector_name} Use Cases")

    # Overview box
    desc = overview_text.get(sector_name, "Overview not available for this sector.")
    st.markdown(
        f"""
    <div class="sector-box">
        <h3 style="color:#064b86; margin-bottom:6px;">Overview</h3>
        <p style="font-size:15px; line-height:1.6; margin-top:0;">{desc}</p>

        <h4 style="color:#064b86; margin-top:18px;">Tools & Technologies Used</h4>
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
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("## Projects")

    usecases = sectors.get(sector_name, [])

    # show projects in 3-column rows
    for i in range(0, len(usecases), 3):
        row = usecases[i : i + 3]
        cols = st.columns(3)
        for j, uc in enumerate(row):
            with cols[j]:
                thumb = os.path.join(ASSETS_DIR, uc.get("image", ""))
                if uc.get("image") and os.path.exists(thumb):
                    st.image(thumb, use_column_width=True)
                else:
                    # small fallback image
                    st.image(logo_url, width=150)
                st.markdown(f"### {uc['name']}")
                st.write("Explore insights and dashboards.")

                # attempt to switch page if supported; otherwise show intended page path
                if st.button(f"Open {uc['name']}", key=f"open_{sector_name}_{j}"):
                    try:
                        # newer Streamlit has st.switch_page
                        st.switch_page(f"pages/{uc['page']}")
                    except Exception:
                        st.warning(
                            "Your Streamlit version doesn't support programmatic page switching. "
                            f"Intended page: pages/{uc['page']}. You can navigate manually or upgrade Streamlit."
                        )

# footer (small)
st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
