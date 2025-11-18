import streamlit as st
from PIL import Image

st.set_page_config(layout="wide")

# Simple styling
st.markdown("""
    <style>
        .sector-box {
            border: 2px solid #dddddd;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            background-color: #f9f9f9;
        }

        .thumbnail {
            width: 100%;
            height: 180px;
            object-fit: cover;
            border-radius: 8px;
            margin-bottom: 15px;
        }

        .tool-card {
            border: 1px solid #cfcfcf;
            border-radius: 6px;
            padding: 10px;
            text-align: center;
            font-size: 15px;
            background: white;
        }

        .usecase-card {
            border: 1px solid #bbb;
            border-radius: 6px;
            padding: 15px;
            background: #fff;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# DATA MODEL
# ----------------------------------------------------------

sectors = [
    {
        "title": "Marketing Analytics",
        "thumbnail": "https://via.placeholder.com/600x300.png?text=Marketing",
        "overview": "Analytics solutions for campaign performance, forecasting, segmentation and digital optimization.",
        "tools": [
            "Campaign Analyzer", "Forecasting Lab", "SEO Dashboard", "Churn Predictor",
            "Lead Funnel Analyzer", "Customer Insight Tool", "Attribution Modeler"
        ],
        "use_cases": [
            "Campaign Spend Optimization",
            "Customer Segmentation Engine",
            "A/B Testing Intelligence",
            "Social Media Analytics"
        ]
    },
    {
        "title": "Retail Analytics",
        "thumbnail": "https://via.placeholder.com/600x300.png?text=Retail",
        "overview": "Offline retail intelligence including footfall analytics, consumer journey tracking and conversion improvement.",
        "tools": [
            "Footfall Tracker", "Heatmap Viewer", "Sales Predictor", "Inventory Forecaster",
            "POS Analyzer", "Store Ops Dashboard"
        ],
        "use_cases": [
            "Shelf Optimization",
            "Conversion Rate Benchmarking",
            "Queue Management Analytics"
        ]
    },
    {
        "title": "Healthcare Analytics",
        "thumbnail": "https://via.placeholder.com/600x300.png?text=Healthcare",
        "overview": "Patient analytics, hospital optimization, disease prediction models and operational insights.",
        "tools": [
            "Patient Flow Analyzer", "Diagnosis Predictor", "Bed Utilization Tracker",
            "Hospital Ops Dashboard", "Lab Report Analyzer"
        ],
        "use_cases": [
            "Disease Trend Prediction",
            "Patient Journey Mapping",
            "Doctor Efficiency Analysis",
            "Prescription Pattern Mining"
        ]
    }
]


# ----------------------------------------------------------
# LAYOUT
# ----------------------------------------------------------

st.title("Data Solutions Dashboard")

for sector in sectors:
    st.markdown(f"<div class='sector-box'>", unsafe_allow_html=True)

    # Thumbnail
    st.image(sector["thumbnail"], use_column_width=True)

    # Title
    st.subheader(sector["title"])

    # Overview
    st.write(sector["overview"])

    st.write("### Tools & Technologies")

    tools = sector["tools"]
    cols = st.columns(5)

    for i, tool in enumerate(tools):
        with cols[i % 5]:
            st.markdown(f"<div class='tool-card'>{tool}</div>", unsafe_allow_html=True)

    # Explore button
    st.button(f"Explore {sector['title']}", key=sector["title"])

    st.write("### Use Case Grid")

    uc_cols = st.columns(2)
    for i, uc in enumerate(sector["use_cases"]):
        with uc_cols[i % 2]:
            st.markdown(f"<div class='usecase-card'>{uc}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
