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
<div style="display:flex; align-items:center; margin-bottom:20px;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Global CSS
# -------------------------
st.markdown("""
<style>
.card-box {
    border:1px solid #c9d7f0;
    border-radius:14px;
    padding:15px;
    background:#fff;
    transition:0.25s ease-in-out;
    min-height:5px;
    box-shadow:0 2px 10px rgba(0,0,0,0.08);
    margin-bottom:25px;
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:flex-start;
}
.card-box:hover {
    transform:translateY(-6px);
    box-shadow:0 6px 22px rgba(0,0,0,0.18);
    border-color:#7fa8ff;
    background:#f9fbff;
}
.card-box img {
    border-radius:8px;
    outline:1px solid #dce6ff;
    transition:0.25s ease-in-out;
    max-height:120px;
    object-fit:cover;
}
.card-box:hover img {
    outline-color:#7fa8ff;
    box-shadow:0px 0px 10px rgba(130,160,255,0.5);
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
    "Marketing Analytics": """Analyze customer journeys, optimize ad spends, improve campaign ROAS, and track funnel drop-offs.
Use segmentation, forecasting, and attribution insights to supercharge marketing performance.
Gain a unified 360-degree view of customers across channels for smarter decisions.""",

    "Real Estate Analytics": """Understand locality demand, analyze pricing trends, compare property attributes, and forecast ROI.
Leverage geospatial intelligence and ML models to identify high-growth micro-markets.
Support investment decisions using rental yield and accurate price prediction analytics.""",

    "Health Care Analytics": """Improve patient flow, predict OPD/ER volumes, enhance doctor allocation, and reduce waiting times.
Use forecasting, classification, and EMR/EHR data to optimize hospital operations.
Boost care quality with real-time monitoring and clinical performance analytics.""",
 
    "Manufacturing Analytics": """Improve patient flow, predict OPD/ER volumes, enhance doctor allocation, and reduce waiting times.
Use forecasting, classification, and EMR/EHR data to optimize hospital operations.
Boost care quality with real-time monitoring and clinical performance analytics."""
}

sector_tools = {
    "Marketing Analytics": ["Python","SQL","Excel","Power BI","Tableau","Google Analytics 4","Pandas","NumPy","Scikit-Learn",
                            "A/B Testing","Attribution Models","Segmentation Models"],
    "Real Estate Analytics": ["Python","SQL","Excel","Power BI","Tableau","QGIS","GeoPandas","Google Maps API",
                              "Regression Models","Time Series","Clustering","Price Prediction","Rental Yield Models"],
    "Health Care Analytics": ["Python","R","SQL","Excel","Power BI","Tableau","EMR/EHR Data","Time Series Forecasting",
                              "Classification Models","NLP","Patient Flow Forecasting"],
    "Manufacturing Analytics": ["Python","R","SQL","Excel","Power BI","Tableau","EMR/EHR Data","Time Series Forecasting",
                              "Classification Models","NLP","Patient Flow Forecasting"]
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
        {"name":"HealthOps Dashboard","page":"healthcare_5.py"},
    ],
    "Manufacturing Analytics": [
        {"name":"Production Downtime & Predictive Maintenance","page":"manufacturing_1.py"},
        {"name":"Patient Visit Analytics & Hospital Performance","page":"manufacturing_2.py"},
        {"name":"PatientFlow Navigator","page":"manufacturing_3.py"},
    ]
}

thumb_urls = {
    "Marketing Analytics": "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/assets/marketing_thumb.jpeg",
    "Real Estate Analytics": "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/assets/real_estate_thumb.jpeg",
    "Health Care Analytics": "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/assets/healthcare_thumb.jpeg",
    "Manufacturing Analytics": "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/assets/manufacturing_thumb.jpeg",
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
        cols = st.columns(3)
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
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass

                st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# SECTOR PAGE
# -------------------------
else:
    sector_name = st.session_state["sector"]

    # Sidebar
    # -------------------------
    
    # -------------------------
    # Sidebar Navigation
    # -------------------------
    if "navigate_to" not in st.session_state:
        st.session_state["navigate_to"] = None
    
    with st.sidebar:
        st.markdown(f"## {sector_name} Navigation")
        
        if st.button("Home", key="sidebar_home"):
            st.session_state["navigate_to"] = "home"
        
        for s in sector_overview.keys():
            if s != sector_name:
                if st.button(f"➡ {s}", key=f"sidebar_{s}"):
                    st.session_state["navigate_to"] = s
    
    # -------------------------
    # Handle navigation safely
    # -------------------------
    if st.session_state.get("navigate_to") is not None:
        target = st.session_state["navigate_to"]
        st.session_state["navigate_to"] = None
        
        if target == "home":
            st.session_state["sector"] = None
        else:
            st.session_state["sector"] = target
    
        # Hide potential rerun errors
        try:
            st.experimental_rerun()
        except Exception:
            pass



 # -------------------


    st.header(f"{sector_name} – Projects / Use Cases")

    project_details = {
        "Marketing Analytics": [
            {"overview": "Analyze campaign effectiveness, optimize ad spends, track click-throughs, and forecast ROI using customer behavior insights.",
             "tools": ["Python","SQL","Pandas","NumPy","Power BI","Tableau","A/B Testing"]},
            {"overview": "Predict customer churn, improve retention strategies, and enhance loyalty programs through segmentation and predictive modeling.",
             "tools": ["Python","Scikit-Learn","Excel","HubSpot","CRM Data"]},
            {"overview": "Track funnel performance, conversion rates, and customer journeys to identify drop-offs and improve marketing ROI.",
             "tools": ["Python","Google Analytics 4","Tableau","Segmentation Models"]},
            {"overview": "Analyze content performance across channels and optimize SEO to boost engagement and organic traffic.",
             "tools": ["Excel","Python","Google Ads API","Meta Ads API","Power BI"]},
            {"overview": "Forecast campaign trends, test creatives using A/B experiments, and attribute conversions to proper channels.",
             "tools": ["Python","NumPy","Pandas","Attribution Models","Power BI"]},
            {"overview": "Measure customer retention, predict churn, and identify key drivers of customer loyalty using ML models.",
             "tools": ["Python","Scikit-Learn","Excel","Tableau","NLP"]},
            {"overview": "Analyze multi-channel customer journeys and funnel touchpoints to improve personalization and engagement.",
             "tools": ["Python","SQL","Power BI","Segmentation Models"]},
            {"overview": "Monitor Google Ads performance, optimize bidding strategies, and improve ad placements using predictive insights.",
             "tools": ["Python","Google Ads API","Power BI","Excel"]},
            {"overview": "Forecast email and WhatsApp campaign success, segment audiences, and improve targeted messaging efficiency.",
             "tools": ["Python","Excel","A/B Testing","Pandas","Tableau"]},
        ],
        "Real Estate Analytics": [
            {"overview": "Analyze property prices, identify investment hotspots, and forecast ROI using historical data and market trends.",
             "tools": ["Python","SQL","Power BI","Regression Models"]},
            {"overview": "Forecast real estate demand per location and segment, helping agents prioritize high-growth neighborhoods.",
             "tools": ["Python","Time Series","Tableau","GeoPandas"]},
            {"overview": "Compare property features and prices to identify undervalued properties and market opportunities.",
             "tools": ["Python","Excel","QGIS","Clustering"]},
            {"overview": "Provide agents with actionable insights on listings, buyer preferences, and market movements.",
             "tools": ["Power BI","Python","Google Maps API","Regression Models"]},
            {"overview": "Analyze investment potential for properties with rental yield and ROI prediction models.",
             "tools": ["Python","Rental Yield Models","Tableau","SQL"]},
            {"overview": "Assess tenant risk, forecast market trends, and provide data-driven insights for property management.",
             "tools": ["Python","SQL","Clustering","Excel"]},
            {"overview": "Identify high-yield rental properties and optimize investment decisions using predictive analytics.",
             "tools": ["Python","Time Series","Power BI","GeoPandas"]},
            {"overview": "Analyze buyer sentiment and preferences to guide property marketing and pricing strategies.",
             "tools": ["Python","NLP","Power BI","Excel"]},
            {"overview": "Understand neighborhood lifestyle trends, safety, and amenities for smarter property recommendations.",
             "tools": ["Python","GeoPandas","Tableau","Power BI"]},
            {"overview": "Combine CRM data and property metrics to deliver a hybrid intelligence dashboard for agents and investors.",
             "tools": ["Python","Power BI","SQL","Tableau"]},
        ],
        "Health Care Analytics": [
            {"overview": "Analyze patient visit data to optimize hospital operations and improve patient flow.",
             "tools": ["Python","SQL","Power BI","Time Series Forecasting"]},
            {"overview": "Predict patient volume, assess hospital performance metrics, and reduce wait times effectively.",
             "tools": ["R","Python","Tableau","Forecasting Models"]},
            {"overview": "Visualize patient journey, resource utilization, and treatment pathways to improve efficiency.",
             "tools": ["Python","Power BI","EMR/EHR Data","Excel"]},
            {"overview": "Optimize ambulance routes, reduce response times, and monitor emergency operations in real-time.",
             "tools": ["Python","QGIS","Time Series","Tableau"]},
            {"overview": "Track hospital KPIs, patient satisfaction, and operational efficiency using predictive models.",
             "tools": ["Python","R","Power BI","Classification Models"]},
            {"overview": "Forecast patient visits and resource requirements to improve staffing and care quality.",
             "tools": ["Python","SQL","NLP","Patient Flow Forecasting"]},
        ],
        "Manufacturing Analytics": [
        {"overview": "Analyze patient visit data to optimize hospital operations and improve patient flow.",
         "tools": ["Python","SQL","Power BI","Time Series Forecasting"]},
        {"overview": "Predict patient volume, assess hospital performance metrics, and reduce wait times effectively.",
         "tools": ["R","Python","Tableau","Forecasting Models"]},
        {"overview": "Visualize patient journey, resource utilization, and treatment pathways to improve efficiency.",
         "tools": ["Python","Power BI","EMR/EHR Data","Excel"]},
         ]
    }

    usecases = sectors[sector_name]
    details = project_details[sector_name]
    rows = [usecases[i:i+3] for i in range(0, len(usecases), 3)]

    for row_idx, row in enumerate(rows):
        cols = st.columns(3)
        for col_idx, (col, uc) in enumerate(zip(cols, row)):
            with col:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)
                st.image(thumb_urls[sector_name], use_container_width=True)
                st.markdown(f"<h4 style='color:#064b86; margin-top:8px;'>{uc['name']}</h4>", unsafe_allow_html=True)
                
                proj_overview = details[row_idx*3 + col_idx]['overview']
                st.markdown(f"<p style='font-size:14px; color:#444; text-align:justify;'>{proj_overview}</p>", unsafe_allow_html=True)
                
                proj_tools = details[row_idx*3 + col_idx]['tools']
                tool_html = "".join([f"<span class='tool-btn'>{t}</span>" for t in proj_tools])
                st.markdown(f"<b>Tools & Tech:</b><br>{tool_html}", unsafe_allow_html=True)

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

    if st.button("Back to Home", key="back_home_bottom"):
        st.session_state["sector"] = None
        st.experimental_rerun()
