import streamlit as st

st.set_page_config(layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>

.card {
    border: 1.5px solid #d4d4d4;
    border-radius: 10px;
    padding: 15px;
    background: white;
    transition: all 0.3s ease;
    box-shadow: 0px 0px 4px rgba(0,0,0,0.05);
}

.card:hover {
    transform: scale(1.02);
    box-shadow: 0px 4px 18px rgba(0,0,0,0.15);
}

.thumbnail {
    width: 100%;
    height: 180px;
    object-fit: cover;
    border-radius: 8px;
}

.title {
    font-size: 20px;
    font-weight: 700;
    margin-top: 10px;
}

.overview {
    font-size: 13.5px;
    text-align: justify;
    margin-top: 5px;
}

.tool-tag {
    display: inline-block;
    background: #eef2ff;
    padding: 5px 10px;
    border-radius: 6px;
    margin: 4px 4px 0 0;
    font-size: 12px;
    color: #333;
    border: 1px solid #cdd3ff;
}

.explore-btn {
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------- SECTOR DATA ----------
sectors = [
    {
        "name": "Marketing Analytics",
        "thumbnail": "https://i.ibb.co/7VFr5vV/marketing.jpg",
        "overview": """Marketing analytics helps brands decode customer behavior, optimize campaigns, manage ad budgets smarter, 
        improve retention, and forecast demand. It blends data across channels to reveal why customers act the way they do 
        and how businesses can boost ROI with precision and speed.""",
        "tools": [
            "Google Analytics", "Power BI", "Tableau", "Python", "R", "SQL",
            "Google BigQuery", "Meta Ads Manager", "Google Ads", "Excel"
        ]
    },
    {
        "name": "Real Estate Analytics",
        "thumbnail": "https://i.ibb.co/4PfLsJz/realestate.jpg",
        "overview": """Real estate analytics drives smarter pricing, property valuation, rental forecasting, 
        buyer profiling, and investment decisions. It merges location intelligence with market trends to give builders, 
        investors, and brokers actionable insights for negotiation and strategic planning.""",
        "tools": [
            "GIS Tools", "Tableau", "Power BI", "Python", "SQL",
            "Excel", "ArcGIS", "QGIS", "GeoPandas", "Machine Learning Models"
        ]
    },
    {
        "name": "Health Care Analytics",
        "thumbnail": "https://i.ibb.co/jHrWcX8/healthcare.jpg",
        "overview": """Healthcare analytics strengthens patient outcomes, reduces operational costs, enhances doctor 
        efficiency, and supports early disease detection. It integrates EHR data, hospital workflows, and predictive 
        models to transform diagnosis accuracy and treatment planning.""",
        "tools": [
            "Power BI", "Python", "R", "SQL", "Tableau",
            "Healthcare Dashboards", "ML Models", "Excel", "FHIR APIs", "Medical Databases"
        ]
    }
]

# ---------- UI HEADER ----------
st.title("Data Analytics Solutions")
st.write("Choose a sector to explore:")

# ---------- LAYOUT ----------
cols = st.columns(3)

for idx, sector in enumerate(sectors):
    with cols[idx]:
        st.markdown(f"<div class='card'>", unsafe_allow_html=True)

        st.markdown(
            f"<img class='thumbnail' src='{sector['thumbnail']}'>",
            unsafe_allow_html=True
        )

        st.markdown(f"<div class='title'>{sector['name']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='overview'>{sector['overview']}</div>", unsafe_allow_html=True)

        # Tools in grid-like flow
        tool_html = "".join([f"<span class='tool-tag'>{tool}</span>" for tool in sector["tools"]])
        st.markdown(tool_html, unsafe_allow_html=True)

        st.button(f"Explore {sector['name']}", key=f"explore_{idx}")

        st.markdown("</div>", unsafe_allow_html=True)
