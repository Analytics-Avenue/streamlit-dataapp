import streamlit as st

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Data Analytics Solutions", layout="wide")
st.markdown("""<style>[data-testid="stSidebarNav"]{display:none;}</style>""", unsafe_allow_html=True)

# -------------------------
# Company Logo + Name
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display:flex; align-items:center; margin-bottom:25px;">
    <img src="{logo_url}" width="60" style="margin-right:15px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Global CSS (Polished Grid)
# -------------------------
st.markdown("""
<style>
.card-box {
    border:1px solid #c9d7f0;
    border-radius:14px;
    padding:15px;
    background:#fff;
    transition:0.25s ease-in-out;
    min-height:280px;
    box-shadow:0 2px 12px rgba(0,0,0,0.08);
    margin-bottom:25px;
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:flex-start;
}
.card-box:hover {
    transform:translateY(-8px);
    box-shadow:0 8px 28px rgba(0,0,0,0.18);
    border-color:#7fa8ff;
    background:#f9fbff;
}
.card-box img {
    border-radius:8px;
    outline:1px solid #dce6ff;
    transition:0.25s ease-in-out;
    max-height:120px;
    object-fit:cover;
    width:100%;
}
.card-box:hover img {
    outline-color:#7fa8ff;
    box-shadow:0px 0px 12px rgba(130,160,255,0.5);
    transform:scale(1.02);
}
.tool-btn {
    background:#eef4ff;
    border-radius:6px;
    padding:5px 9px;
    font-size:12px;
    border:1px solid #c6d7ff;
    margin:3px;
    display:inline-block;
    font-weight:600;
    transition:0.2s;
}
.tool-btn:hover {
    background:#d9e7ff;
    transform:scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Sector Data
# -------------------------
sector_overview = {
    "Marketing Analytics": "Analyze customer journeys, optimize ad spends, improve campaign ROAS, and track funnel drop-offs.",
    "Real Estate Analytics": "Understand locality demand, analyze pricing trends, compare property attributes, and forecast ROI.",
    "Health Care Analytics": "Improve patient flow, predict OPD/ER volumes, enhance doctor allocation, and reduce waiting times."
}

sector_tools = {
    "Marketing Analytics": ["Python","SQL","Excel","Power BI","Tableau","Google Analytics 4","Pandas","NumPy","Scikit-Learn","A/B Testing","Attribution Models","Segmentation Models"],
    "Real Estate Analytics": ["Python","SQL","Excel","Power BI","Tableau","QGIS","GeoPandas","Google Maps API","Regression Models","Time Series","Clustering","Price Prediction","Rental Yield Models"],
    "Health Care Analytics": ["Python","R","SQL","Excel","Power BI","Tableau","EMR/EHR Data","Time Series Forecasting","Classification Models","NLP","Patient Flow Forecasting"]
}

sectors = {
    "Marketing Analytics": [
        {"name":"Marketing Campaign Performance Analyzer","page":"marketing_1.py"},
        {"name":"Marketing Intelligence & Forecasting Lab","page":"marketing_2.py"},
        {"name":"Click & Convertion Analytics","page":"marketing_3.py"},
        {"name":"Marketing Performance Analysis","page":"marketing_4.py"},
        {"name":"Content & SEO Performance Dashboard","page":"marketing_5.py"},
        {"name":"Customer Retention & Churn Analysis","page":"marketing_6.py"},
        {"name":"Customer Journey & Funnel Insights","page":"marketing_7.py"},
        {"name":"Google Ads Performance Analytics","page":"marketing_8.py"},
        {"name":"Email & WhatsApp Marketing Forecast Lab","page":"marketing_9.py"},
    ],
    "Real Estate Analytics": [
        {"name":"Real Estate Intelligence Suite","page":"usecase_real_estate_1.py"},
        {"name":"Real Estate Demand Forecasting System","page":"usecase_real_estate_2.py"},
        {"name":"Price vs Property Features Analyzer","page":"usecase_real_estate_3.py"},
        {"name":"Agent & Market Insights Dashboard","page":"usecase_real_estate_4.py"},
        {"name":"Real Estate Investment Opportunity Analyzer","page":"usecase_real_estate_5.py"},
        {"name":"Tenant Risk & Market Trend Analyzer","page":"usecase_real_estate_6.py"},
        {"name":"Rental Yield & Investment Analyzer","page":"usecase_real_estate_7.py"},
        {"name":"Real Estate Buyer Sentiment Analyzer","page":"usecase_real_estate_8.py"},
        {"name":"Neighborhood Lifestyle & Risk Aware Analyzer","page":"usecase_real_estate_9.py"},
        {"name":"Real Estate Intelligence — Hybrid Dashboard (Property + CRM)","page":"realestate.py"},
    ],
    "Health Care Analytics": [
        {"name":"Healthscope Insights","page":"healthcare_1.py"},
        {"name":"Patient Visit Analytics & Hospital Performance","page":"healthcare_2.py"},
        {"name":"PatientFlow Navigator","page":"healthcare_3.py"},
        {"name":"Ambulance Ops & Routing Lab","page":"healthcare_4.py"},
        {"name":"Health Care Analytics1","page":"healthcare_5.py"},
        {"name":"Health Care Analytics2","page":"healthcare_6.py"},
    ]
}

thumb_urls = {
    "Marketing Analytics": "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/assets/marketing_thumb.jpg",
    "Real Estate Analytics": "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/assets/real_estate_thumb.jpg",
    "Health Care Analytics": "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/assets/healthcare_thumb.jpg",
}

# -------------------------
# Session State
# -------------------------
if "sector" not in st.session_state:
    st.session_state["sector"] = None

# -------------------------
# HOME PAGE
# -------------------------
if st.session_state["sector"] is None:
    st.title("Data Analytics Solutions")
    st.write("Choose a sector to explore:")

    sectors_list = list(sector_overview.keys())
    rows = [sectors_list[i:i+3] for i in range(0, len(sectors_list), 3)]

    for row in rows:
        cols = st.columns(3, gap="medium")
        for col, sector in zip(cols, row):
            with col:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)
                st.image(thumb_urls[sector], use_container_width=True)
                st.markdown(f"<h3 style='color:#064b86; margin-top:12px;'>{sector}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:14.5px; color:#444; text-align:justify;'>{sector_overview[sector]}</p>", unsafe_allow_html=True)
                
                tool_html = "".join([f"<span class='tool-btn'>{t}</span>" for t in sector_tools[sector]])
                st.markdown(f"<b>Tools & Tech:</b><br>{tool_html}", unsafe_allow_html=True)

                if st.button(f"Explore {sector}", key=f"btn_{sector}"):
                    st.session_state["sector"] = sector
                    st.experimental_rerun()
                st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# SECTOR PAGE
# -------------------------
else:
    sector_name = st.session_state["sector"]
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"## {sector_name} Navigation")
        if st.button("🏠 Home"):
            st.session_state["sector"] = None
            st.experimental_rerun()
        for s in sector_overview.keys():
            if s != sector_name:
                if st.button(f"➡ {s}"):
                    st.session_state["sector"] = s
                    st.experimental_rerun()

    st.header(f"{sector_name} – Projects / Use Cases")
    usecases = sectors[sector_name]

    for row_idx in range(0, len(usecases), 3):
        cols = st.columns(3, gap="medium")
        for col_idx, col in enumerate(cols):
            if row_idx + col_idx < len(usecases):
                uc = usecases[row_idx + col_idx]
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)
                st.image(thumb_urls[sector_name], use_container_width=True)
                st.markdown(f"<h4 style='color:#064b86; margin-top:8px;'>{uc['name']}</h4>", unsafe_allow_html=True)
                
                # Open button
                page_slug = uc['page'][:-3] if uc['page'].endswith(".py") else uc['page']
                deployed_url = f"https://analytics-avenue.streamlit.app/{page_slug}"
                st.markdown(f"""
                    <a href="{deployed_url}" target="_blank" style="text-decoration:none;">
                        <div style="background:#eef4ff; color:#064b86; padding:6px 12px; border-radius:6px; text-align:center; font-weight:600; margin-top:5px;">
                            Open
                        </div>
                    </a>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Back to Home"):
        st.session_state["sector"] = None
        st.experimental_rerun()
