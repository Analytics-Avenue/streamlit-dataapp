import streamlit as st
import os

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
<div style="display: flex; align-items: center; margin-bottom:20px;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# GLOBAL CSS
# -------------------------
st.markdown("""
<style>
.card-box {
    border: 1px solid #c9d7f0;
    border-radius: 14px;
    padding: 15px;
    background: #ffffff;
    transition: 0.25s ease-in-out;
    min-height: 5px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    margin-bottom: 25px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
}
.card-box:hover {
    transform: translateY(-6px);
    box-shadow: 0 6px 22px rgba(0,0,0,0.18);
    border-color: #7fa8ff;
    background: #f9fbff;
}
.card-box img {
    border-radius: 8px;
    outline: 1px solid #dce6ff;
    transition: 0.25s ease-in-out;
    max-height: 120px;
    object-fit: cover;
}
.card-box:hover img {
    outline-color: #7fa8ff;
    box-shadow: 0px 0px 10px rgba(130,160,255,0.5);
    transform: scale(1.02);
}
.tool-btn {
    background: #eef4ff;
    border-radius: 6px;
    padding: 5px 9px;
    font-size: 12px;
    border: 1px solid #c6d7ff;
    margin: 3px;
    display: inline-block;
    font-weight: 600;
    transition: 0.2s;
}
.tool-btn:hover {
    background: #d9e7ff;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Project Details (all projects fully filled)
# -------------------------
project_details = {

    # -------------------------
    "Marketing Analytics": {
        "Marketing Campaign Performance Analyzer": {
            "overview": "Track campaign reach, clicks, conversions, and ROI. Use dashboards to compare ad performance across channels and optimize spending decisions. Ideal for marketing managers and analysts seeking data-driven insights.",
            "tools": ["Python", "SQL", "Tableau", "Google Analytics 4", "A/B Testing"]
        },
        "Marketing Intelligence & Forecasting Lab": {
            "overview": "Forecast marketing outcomes using historical campaign data and predictive analytics. Identify high-potential segments and optimize resource allocation. Helps marketers plan more efficiently and increase ROI.",
            "tools": ["Python", "Power BI", "Scikit-Learn", "Pandas"]
        },
        "Click & Convertion Analytics": {
            "overview": "Analyze customer journeys from clicks to conversions. Identify drop-off points and optimize the conversion funnel. Supports data-driven UX and CRO strategies for e-commerce and SaaS platforms.",
            "tools": ["SQL", "Python", "Excel", "Google Analytics 4"]
        },
        "Marketing Performance Analysis": {
            "overview": "Evaluate marketing KPIs such as engagement, reach, and conversion rates. Use visualization and segmentation to understand performance trends over time. Enables smarter decision-making for campaign optimization.",
            "tools": ["Power BI", "Tableau", "Python", "Excel"]
        },
        "Content & SEO Performance Dashboard": {
            "overview": "Track website content performance, keyword rankings, and traffic sources. Identify top-performing pages and optimize SEO strategies to improve organic search visibility.",
            "tools": ["Google Analytics 4", "Python", "Excel", "Power BI"]
        },
        "Customer Retention & Churn Analysis": {
            "overview": "Measure customer loyalty, retention rates, and churn predictors. Implement proactive strategies to retain high-value customers and reduce attrition. Ideal for subscription-based businesses.",
            "tools": ["Python", "R", "SQL", "Machine Learning", "Tableau"]
        },
        "Customer Journey & Funnel Insights,": {
            "overview": "Map customer touchpoints across channels and analyze funnel conversion rates. Use insights to optimize engagement and marketing spend efficiently.",
            "tools": ["Python", "Power BI", "SQL", "Excel"]
        },
        "Google Ads Performance Analytics.": {
            "overview": "Monitor Google Ads campaigns in real time, identify top-performing keywords, and optimize bids. Improve campaign ROI with data-driven decisions.",
            "tools": ["Google Analytics 4", "Python", "Excel", "Power BI"]
        },
        "Email & WhatsApp Marketing Forecast Lab": {
            "overview": "Analyze engagement metrics for email and WhatsApp campaigns, forecast click-through and open rates, and improve messaging effectiveness. Supports marketing automation strategies.",
            "tools": ["Python", "SQL", "Excel", "Tableau"]
        },
    },

    # -------------------------
    "Real Estate Analytics": {
        "Real Estate Intelligence Suite": {
            "overview": "Gain a 360-degree view of properties, pricing trends, and neighborhood data. Analyze investment opportunities and predict market growth using comprehensive dashboards.",
            "tools": ["Python", "QGIS", "Tableau", "GeoPandas"]
        },
        "Real Estate Demand Forecasting System": {
            "overview": "Forecast property demand using historical transactions, local demographics, and market trends. Identify high-growth areas and optimize investment decisions.",
            "tools": ["Python", "SQL", "Power BI", "Time Series Models"]
        },
        "Price vs Property Features Analyzer": {
            "overview": "Analyze how property attributes affect pricing. Use regression and ML models to predict property values and support buyer/seller decision-making.",
            "tools": ["Python", "Excel", "Regression Models", "Scikit-Learn"]
        },
        "Agent & Market Insights Dashboard": {
            "overview": "Monitor agent performance, track property listings, and visualize market trends. Helps real estate firms optimize sales strategies and client targeting.",
            "tools": ["Power BI", "Python", "SQL", "Tableau"]
        },
        "Real Estate Investment Opportunity Analyzer": {
            "overview": "Evaluate potential real estate investments using predictive analytics and historical data. Prioritize properties with the highest ROI potential.",
            "tools": ["Python", "Excel", "Regression Models", "Power BI"]
        },
        "Tenant Risk & Market Trend Analyzer": {
            "overview": "Assess tenant credit risk and forecast rental trends. Helps landlords and property managers mitigate risks and maximize rental yield.",
            "tools": ["Python", "SQL", "Time Series Analysis", "Tableau"]
        },
        "Rental Yield & Investment Analyzer": {
            "overview": "Analyze rental yield and cash flow projections for properties. Support portfolio optimization and investment strategy decisions.",
            "tools": ["Python", "Excel", "Power BI", "GeoPandas"]
        },
        "Real Estate Buyer Sentiment Analyzer": {
            "overview": "Analyze buyer preferences, sentiment trends, and engagement across listings. Helps agents and developers tailor offerings to market demand.",
            "tools": ["Python", "R", "SQL", "Power BI"]
        },
        "Neighborhood Lifestyle & Risk Aware Analyzer": {
            "overview": "Evaluate neighborhood factors like crime rates, schools, and amenities. Supports risk-aware investment and residential planning decisions.",
            "tools": ["Python", "QGIS", "Excel", "Power BI"]
        },
        "Real Estate Intelligence — Hybrid Dashboard (Property + CRM)": {
            "overview": "Integrates property analytics with CRM data for a holistic view of clients and listings. Optimizes sales, marketing, and investment strategies.",
            "tools": ["Python", "SQL", "Tableau", "CRM Integration Tools"]
        },
    },

    # -------------------------
    "Health Care Analytics": {
        "Healthscope Insights": {
            "overview": "Monitor hospital KPIs, patient volumes, and operational efficiency. Identify bottlenecks and improve resource allocation for better healthcare delivery.",
            "tools": ["Python", "Power BI", "EMR/EHR Data"]
        },
        "Patient Visit Analytics & Hospital Performance": {
            "overview": "Analyze patient visit patterns, waiting times, and staff allocation. Supports data-driven operational improvements and workflow optimization.",
            "tools": ["R", "Python", "Time Series Forecasting"]
        },
        "PatientFlow Navigator": {
            "overview": "Visualize patient movement through departments to minimize delays and improve care quality. Optimize hospital processes using predictive models.",
            "tools": ["Python", "SQL", "Tableau", "NLP"]
        },
        "Ambulance Ops & Routing Lab": {
            "overview": "Optimize ambulance dispatch, routing, and response times using real-time data. Helps reduce delays in emergency services.",
            "tools": ["Python", "GIS Tools", "Power BI", "SQL"]
        },
        "Health Care Analytics1": {
            "overview": "Analyze clinical and operational data to improve hospital outcomes. Supports decision-making for management and quality assurance teams.",
            "tools": ["Python", "Excel", "Power BI", "Machine Learning"]
        },
        "Health Care Analytics2": {
            "overview": "Use predictive models to forecast patient influx, staffing needs, and resource allocation. Improves overall hospital efficiency.",
            "tools": ["Python", "R", "SQL", "Time Series Models"]
        },
    }

}

# -------------------------
# Thumbnail URLs
# -------------------------
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

# ============================================================
# HOME PAGE
# ============================================================
if st.session_state["sector"] is None:
    st.title("Data Analytics Solutions")
    st.write("Choose a sector to explore:")

    sector_list = list(project_details.keys())
    rows = [sector_list[i:i+3] for i in range(0, len(sector_list), 3)]

    for row in rows:
        cols = st.columns(3)
        for col, sector in zip(cols, row):
            with col:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)
                st.image(thumb_urls[sector], use_container_width=True)
                st.markdown(f"<h3 style='color:#064b86; margin-top:12px;'>{sector}</h3>", unsafe_allow_html=True)
                st.markdown("<b>Tools & Tech:</b><br>", unsafe_allow_html=True)
                tools = set()
                for proj in project_details[sector].values():
                    tools.update(proj["tools"])
                tool_html = "".join([f"<span class='tool-btn'>{t}</span>" for t in tools])
                st.markdown(tool_html, unsafe_allow_html=True)
                if st.button(f"Explore {sector}", key=f"btn_{sector}"):
                    st.session_state["sector"] = sector
                st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# SECTOR PAGE (Projects / Use Cases)
# ============================================================
else:
    sector_name = st.session_state["sector"]
    st.header(f"{sector_name} – Projects / Use Cases")

    usecases = project_details[sector_name]
    rows = [list(usecases.keys())[i:i+3] for i in range(0, len(usecases), 3)]

    for row in rows:
        cols = st.columns(3)
        for col, uc_name in zip(cols, row):
            uc = usecases[uc_name]
            with col:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)
                st.image(thumb_urls[sector_name], use_container_width=True)
                st.markdown(f"<h4 style='color:#064b86; margin-top:8px;'>{uc_name}</h4>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:14px; color:#444; text-align:justify;'>{uc['overview']}</p>", unsafe_allow_html=True)
                st.markdown("<b>Tools & Tech:</b><br>", unsafe_allow_html=True)
                tool_html = "".join([f"<span class='tool-btn'>{t}</span>" for t in uc["tools"]])
                st.markdown(tool_html, unsafe_allow_html=True)
                page_slug = uc_name.replace(" ", "_").lower()
                deployed_url = f"https://analytics-avenue.streamlit.app/{page_slug}"
                st.markdown(f"""
                    <a href="{deployed_url}" target="_blank" 
                       style="text-decoration:none;">
                       <div style="background:#eef4ff; color:#064b86; padding:6px 12px; border-radius:6px; text-align:center; font-weight:600; margin-top:5px;">
                           Open
                       </div>
                    </a>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Back to Home"):
        st.session_state["sector"] = None
        st.experimental_rerun()
