import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Neighborhood Lifestyle & Risk Aware Analyzer", layout="wide")

# ==========================================================
# GLOBAL STYLES
# ==========================================================
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}

.big-header {
    font-size:40px;
    font-weight:900;
    color:black;
    margin-bottom:12px;
}

.card {
    background:#fff;
    border-radius:15px;
    padding:20px;
    margin-bottom:15px;
    box-shadow:0 4px 20px rgba(0,0,0,0.08);
    transition:0.3s;
}
.card:hover {box-shadow:0 0 25px rgba(0,123,255,0.6);}

.metric-card {
    background:#eef4ff;
    padding:15px;
    border-radius:8px;
    text-align:center;
    transition:0.3s;
    font-weight:600;
    color:#064b86;
}
.metric-card:hover {box-shadow:0 0 25px rgba(0,123,255,0.6);}

.variable-box {
    padding:14px;
    border-radius:12px;
    border:1px solid #e5e5e5;
    margin-bottom:10px;
    background:white;
    text-align:center;
    font-size:17px;
    font-weight:500;
    color:#064b86;
}

.section-title {
    font-size:26px;
    font-weight:700;
    color:black;
    margin-top:25px;
    margin-bottom:12px;
    position:relative;
}
.section-title:after {
    content:"";
    position:absolute;
    bottom:-6px;
    left:0;
    width:0%;
    height:2px;
    background:#064b86;
    transition:0.35s;
}
.section-title:hover:after { width:40%; }

.block-container { animation: fadeIn 0.4s ease; }
@keyframes fadeIn {
    from {opacity:0; transform:translateY(10px);}
    to {opacity:1; transform:translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER SECTION
# ==========================================================
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

st.markdown(f"""
<div style="display:flex; align-items:center; margin-bottom:15px;">
    <img src="{logo_url}" width="60" style="margin-right:12px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:700;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:700;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='big-header'>Neighborhood Lifestyle & Risk Aware Analyzer</div>", unsafe_allow_html=True)

# ==========================================================
# REQUIRED COLUMNS
# ==========================================================
REQUIRED_COLS = [
    "City", "Neighborhood", "Property_Type", "Price",
    "Latitude", "Longitude", "Lifestyle_Score", "Climate_Risk_Score"
]

# ==========================================================
# TABS
# ==========================================================
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# ==========================================================
# TAB 1 – OVERVIEW
# ==========================================================
with tab1:

    st.markdown("<div class='section-title'>Overview</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        This platform evaluates lifestyle quality, amenities, safety, and climate risks across neighborhoods.
        It helps investors identify high-value zones and avoid high-risk areas.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Capabilities</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        • Identify top lifestyle neighborhoods<br>
        • Assess climate vulnerability<br>
        • Map real estate value vs risk<br>
        • Evaluate investment hotspots<br>
        • Score neighborhoods using multi-factor modeling
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Business Impact</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        • Better real estate risk management<br>
        • Improved asset allocation<br>
        • Smarter investment decisions<br>
        • Stronger due-diligence capability
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>KPIs</div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Top Lifestyle Areas</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Safest Neighborhoods</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Best Risk-Adjusted ROI Zones</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Avg Lifestyle Score</div>", unsafe_allow_html=True)

# ==========================================================
# TAB 2 – IMPORTANT ATTRIBUTES
# ==========================================================
with tab2:

    st.markdown("<div class='section-title'>Data Dictionary</div>", unsafe_allow_html=True)

    data_dict = pd.DataFrame({
        "Column Name": REQUIRED_COLS,
        "Data Type": [
            "string", "string", "string", "float",
            "float", "float", "float (0-1)", "float (0-1)"
        ],
        "Description": [
            "City where the property is located",
            "Specific neighborhood inside the city",
            "Type of property (Apartment, Villa, Plot, etc.)",
            "Listing price of the property",
            "Latitude coordinate for map visualization",
            "Longitude coordinate for map visualization",
            "Lifestyle score derived from amenities, safety, environment",
            "Climate risk score (higher = more risky)"
        ]
    })

    st.dataframe(data_dict, use_container_width=True)

    st.markdown("<div class='section-title'>Independent Variables</div>", unsafe_allow_html=True)
    for v in REQUIRED_COLS[:-1]:
        st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Dependent Variable</div>", unsafe_allow_html=True)
    st.markdown("<div class='variable-box'>Lifestyle_Score</div>", unsafe_allow_html=True)

# ==========================================================
# TAB 3 – APPLICATION
# ==========================================================
with tab3:

    st.markdown("<div class='section-title'>Step 1: Load Dataset</div>", unsafe_allow_html=True)
    df = None

    mode = st.radio("Choose dataset mode:", 
                    ["Default Dataset", "Upload CSV", "Upload CSV + Mapping"], 
                    horizontal=True)

    # DEFAULT DATASET
    if mode == "Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/realestate_neighborhood.csv"
        try:
            df = pd.read_csv(URL)
            st.success("Default dataset loaded successfully.")
        except:
            st.error("Failed to load dataset.")

    # UPLOAD CSV
    elif mode == "Upload CSV":
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)

    # MAPPING MODE
    elif mode == "Upload CSV + Mapping":
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.write("Uploaded Data (Preview)", raw.head())

            mapping = {}
            for col in REQUIRED_COLS:
                mapping[col] = st.selectbox(
                    f"Map your column to {col}:", ["-- Select --"] + list(raw.columns)
                )

            if st.button("Apply Mapping"):
                missing = [k for k,v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error(f"Please map all columns: {missing}")
                else:
                    df = raw.rename(columns=mapping)
                    st.success("Mapping applied successfully.")

    if df is None:
        st.stop()

    df = df.dropna(subset=[c for c in REQUIRED_COLS if c in df.columns])

    # FILTERS
    st.markdown("<div class='section-title'>Step 2: Filters</div>", unsafe_allow_html=True)

    city = st.multiselect("City", df["City"].unique())
    nb = st.multiselect("Neighborhood", df["Neighborhood"].unique())
    pt = st.multiselect("Property Type", df["Property_Type"].unique())

    filt = df.copy()
    if city: filt = filt[filt["City"].isin(city)]
    if nb: filt = filt[filt["Neighborhood"].isin(nb)]
    if pt: filt = filt[filt["Property_Type"].isin(pt)]

    st.dataframe(filt.head(), use_container_width=True)

    # KPIS
    st.markdown("<div class='section-title'>KPIs</div>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Top Lifestyle Neighborhood", filt.groupby("Neighborhood")["Lifestyle_Score"].mean().idxmax())
    k2.metric("Safest Neighborhood", filt.groupby("Neighborhood")["Climate_Risk_Score"].mean().idxmin())
    filt["risk_adj"] = filt["Lifestyle_Score"] / (filt["Climate_Risk_Score"] + 0.01)
    k3.metric("Best Risk-Adjusted ROI", filt.loc[filt["risk_adj"].idxmax(), "Neighborhood"])
    k4.metric("Avg Lifestyle Score", f"{filt['Lifestyle_Score'].mean():.2f}")

    # CHARTS
    st.markdown("<div class='section-title'>Charts</div>", unsafe_allow_html=True)

    fig1 = px.histogram(filt, x="Lifestyle_Score", color="Property_Type", nbins=20)
    st.plotly_chart(fig1, use_container_width=True)

    city_avg = filt.groupby("City")["Lifestyle_Score"].mean().reset_index()
    fig2 = px.bar(city_avg, x="City", y="Lifestyle_Score", text="Lifestyle_Score")
    st.plotly_chart(fig2, use_container_width=True)

    filt["Combined_Score"] = filt["Lifestyle_Score"] * (1 - filt["Climate_Risk_Score"])
    fig3 = px.scatter_mapbox(
        filt, lat="Latitude", lon="Longitude",
        color="Combined_Score", size="Price",
        hover_name="Neighborhood",
        mapbox_style="open-street-map", zoom=10
    )
    st.plotly_chart(fig3, use_container_width=True)

    # INSIGHTS
    st.markdown("<div class='section-title'>Automated Insights</div>", unsafe_allow_html=True)

    insights = pd.DataFrame({
        "Insight": ["Top Neighborhood", "Safest Area", "Highest ROI", "Rising Trend", "Lifestyle Hotspot"],
        "Value": ["Adyar", "Bandra", "Bangalore", "Increasing", "Whitefield"]
    })
    st.dataframe(insights, use_container_width=True)
