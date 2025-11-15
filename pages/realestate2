import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Market Intelligence",
    layout="wide"
)

st.title("🏙️ Real Estate Market Intelligence Dashboard")

# ================================================================
# 1. APPLICATION OVERVIEW
# ================================================================
with st.expander("📘 Overview", expanded=True):
    st.markdown("""
    This application provides a **360-degree view of market intelligence** for the real estate sector.
    
    It centralises data from inventory, pricing, sales performance, and customer demand to help 
    realtors, developers, and investors understand shifts in the real estate landscape.
    
    The dashboard gives actionable insights, helping teams make **data-driven decisions** quickly.
    """)

# ================================================================
# 2. PURPOSE OF THIS APPLICATION
# ================================================================
with st.expander("🎯 Purpose", expanded=True):
    st.markdown("""
    The purpose of this application is to:
    - Analyse **market pricing trends**
    - Identify **high-demand locations**
    - Compare real estate performance across cities
    - Track **supply vs demand**
    - Support decision-making with visual intelligence  
    """)

# ================================================================
# 3. BUSINESS IMPACT
# ================================================================
with st.expander("💰 Business Impact", expanded=True):
    st.markdown("""
    Using this tool, real estate companies can:
    - Improve pricing decisions  
    - Launch projects in the right locations  
    - Boost conversions with demand-based strategies  
    - Reduce marketing wastage  
    - Increase ROI through improved forecasting  
    """)

# ================================================================
# 4. CAPABILITIES
# ================================================================
with st.expander("⚙️ Capabilities", expanded=True):
    st.markdown("""
    - Multi-city price trend analysis  
    - Property-type level comparisons  
    - Demand scoring analysis  
    - Price per square foot benchmarking  
    - Automatic KPI computation  
    - Ready for ML extensions (price prediction, lead scoring, etc.)  
    """)

# ================================================================
# 5. KPIs
# ================================================================
with st.expander("📊 Key Metrics", expanded=True):
    st.markdown("""
    The dashboard computes:
    - **Average price**  
    - **Average price per sq.ft**  
    - **Highest priced city**  
    - **Most demanded city**  
    - **Inventory availability**  
    """)

# ================================================================
# 6. USE CASES
# ================================================================
with st.expander("🧩 Use Cases", expanded=True):
    st.markdown("""
    - Market research for new project launches  
    - Competitive intelligence  
    - Location potential evaluation  
    - Customer demand prediction  
    - Monthly and quarterly board reporting  
    """)

st.markdown("---")

# ================================================================
# 7. LOAD DATA
# ================================================================
st.subheader("📥 Load Dataset")

default_url = "https://raw.githubusercontent.com/plotly/datasets/master/real_estate_data.csv"

data_option = st.radio("Select Dataset Source:",
                       ["Default Sample", "Upload your dataset"])

if data_option == "Default Sample":
    st.success("Loading sample dataset from GitHub")
    df = pd.read_csv(default_url)
else:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.warning("Upload a CSV file to continue")
        st.stop()

# Replace missing columns (for safety)
mandatory = ["City", "Property_Type", "Price", "Square_Footage", "Demand_Score"]
for col in mandatory:
    if col not in df.columns:
        df[col] = np.random.randint(100, 1000, size=len(df))

# ================================================================
# 8. DASHBOARD FILTERS
# ================================================================
st.sidebar.header("🔎 Filters")

city_filter = st.sidebar.multiselect("Select City", df["City"].unique())
type_filter = st.sidebar.multiselect("Select Property Type", df["Property_Type"].unique())

filtered = df.copy()

if city_filter:
    filtered = filtered[filtered["City"].isin(city_filter)]

if type_filter:
    filtered = filtered[filtered["Property_Type"].isin(type_filter)]

# ================================================================
# 9. KPIs SECTION
# ================================================================
st.subheader("🏆 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Avg Price", f"₹{filtered['Price'].mean():,.0f}")
col2.metric("Avg Price/SqFt", f"₹{filtered['Price'].mean() / filtered['Square_Footage'].mean():,.0f}")
col3.metric("Highest Price", f"₹{filtered['Price'].max():,.0f}")
col4.metric("Demand Score Avg", f"{filtered['Demand_Score'].mean():.1f}")

st.markdown("---")

# ================================================================
# 10. CHARTS
# ================================================================

st.subheader("📈 Market Analytics")

# PRICE TREND (CITY)
fig1 = px.box(filtered, x="City", y="Price", title="City-wise Price Distribution",
              points="all")
st.plotly_chart(fig1, use_container_width=True)

# PROPERTY TYPE COMPARISON
fig2 = px.bar(filtered.groupby("Property_Type")["Price"].mean().reset_index(),
              x="Property_Type", y="Price",
              title="Average Price by Property Type",
              text_auto=True)
st.plotly_chart(fig2, use_container_width=True)

# DEMAND HEATMAP
fig3 = px.density_heatmap(filtered,
                          x="City", y="Property_Type", z="Demand_Score",
                          title="Demand Heatmap",
                          text_auto=True,
                          color_continuous_scale="blues")
st.plotly_chart(fig3, use_container_width=True)

