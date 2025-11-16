import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ---------------------------- Page Config ----------------------------
st.set_page_config(page_title="Rental Yield & Investment Analyzer", layout="wide")

# ---------------------------- CSS ----------------------------
hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
.big-header {
    font-size: 40px; font-weight: 900;
    background: linear-gradient(90deg,#FF6347,#FFD700);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.card {background:#fff;border-radius:15px;padding:20px;margin-bottom:15px;
box-shadow:0 4px 20px rgba(0,0,0,0.08);}
.metric-card {background:#eef4ff;padding:15px;border-radius:8px;text-align:center;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# ---------------------------- Header ----------------------------
st.markdown("<div class='big-header'>Rental Yield & Investment Analyzer</div>", unsafe_allow_html=True)

# ---------------------------- Required Columns ----------------------------
REQUIRED_COLS = [
    "City", "Property_Type", "Price", "Area_sqft",
    "Age_Years", "Conversion_Probability", "Latitude", "Longitude"
]

# ==========================================================
# TABS
# ==========================================================
tab1, tab2 = st.tabs(["Overview", "Application"])

# ==========================================================
# TAB 1 - OVERVIEW
# ==========================================================
with tab1:
    st.markdown("### Overview")
    st.markdown("""
    <div class='card'>
    Evaluate rental income and investment potential of properties. Calculate rental yield, city- and property-level segmentation, and highlight top-scoring investment properties.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Purpose")
    st.markdown("""
    <div class='card'>
    • Compare rental yields across properties and cities<br>
    • Identify top-performing property types<br>
    • Highlight high-potential investments<br>
    • Support data-driven investment decisions
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Capabilities")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='card'>
        <b>Technical</b><br>
        • Rental yield calculation<br>
        • Investment Score calculation<br>
        • Interactive charts & maps<br>
        • ML prediction for new properties
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='card'>
        <b>Business</b><br>
        • Maximize ROI<br>
        • Investment portfolio optimization<br>
        • Identify high-yield areas<br>
        • Transparent property comparison
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Top Rental Yield Property</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Highest Yield City</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Average Yield</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Top Property Type</div>", unsafe_allow_html=True)

# ==========================================================
# TAB 2 - APPLICATION
# ==========================================================
with tab2:
    st.markdown("### Step 1: Load Dataset")
    df = None

    mode = st.radio(
        "Select Option:",
        ["Default Dataset", "Upload CSV", "Upload CSV + Column Mapping"],
        horizontal=True
    )

    if mode == "Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        try:
            df = pd.read_csv(URL)
            st.success("Default dataset loaded successfully.")
        except Exception as e:
            st.error(f"Could not load dataset: {e}")

    elif mode == "Upload CSV":
        
    st.markdown("#### Download Sample CSV for Reference")
    URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"    
        try:
            # Load default dataset
            sample_df = pd.read_csv(URL).head(5)  # Take first 5 rows
            sample_csv = sample_df.to_csv(index=False)
            st.download_button("Download Sample CSV", sample_csv, "sample_dataset.csv", "text/csv")
        except Exception as e:
            st.info(f"Sample CSV unavailable: {e}")
    
        # Upload actual CSV
        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)
            
    elif mode == "Upload CSV + Column Mapping":
        file = st.file_uploader("Upload dataset", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.write("Uploaded Data", raw.head())
            st.markdown("### Map Required Columns Only")
            mapping = {}
            for col in REQUIRED_COLS:
                mapping[col] = st.selectbox(
                    f"Map your column to: {col}",
                    options=["-- Select --"] + list(raw.columns)
                )
            if st.button("Apply Mapping"):
                missing = [col for col, mapped in mapping.items() if mapped == "-- Select --"]
                if missing:
                    st.error(f"Please map all required columns: {missing}")
                else:
                    df = raw.rename(columns=mapping)
                    st.success("Mapping applied successfully.")
                    st.dataframe(df.head())

    if df is None:
        st.stop()

    if not all(col in df.columns for col in REQUIRED_COLS):
        st.error("Dataset missing required columns.")
        st.write("Required columns:", REQUIRED_COLS)
        st.stop()

    df = df.dropna()

    # ---------------- Filters ----------------
    st.markdown("### Step 2: Filters")
    f1, f2, f3 = st.columns(3)
    with f1:
        city = st.multiselect("City", df["City"].unique())
    with f2:
        ptype = st.multiselect("Property Type", df["Property_Type"].unique())
    with f3:
        age_filter = st.slider("Property Age (Years)", int(df["Age_Years"].min()), int(df["Age_Years"].max()), (int(df["Age_Years"].min()), int(df["Age_Years"].max())))

    filt = df.copy()
    if city:
        filt = filt[filt["City"].isin(city)]
    if ptype:
        filt = filt[filt["Property_Type"].isin(ptype)]
    filt = filt[(filt["Age_Years"] >= age_filter[0]) & (filt["Age_Years"] <= age_filter[1])]

    st.markdown("### Data Preview")
    st.dataframe(filt.head(), use_container_width=True)

    # ---------------- Calculate Rental Yield & Investment Score ----------------
    filt["Est_Rent"] = filt["Price"] * 0.005
    filt["Rental_Yield"] = (filt["Est_Rent"] * 12) / filt["Price"]
    filt["Investment_Score"] = filt["Rental_Yield"] * filt["Conversion_Probability"]

    # ---------------- KPIs ----------------
    st.markdown("### Key Metrics")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Top Rental Yield Property", filt.sort_values("Rental_Yield", ascending=False).iloc[0]["Property_Type"])
    k2.metric("Highest Yield City", filt.groupby("City")["Rental_Yield"].mean().sort_values(ascending=False).head(1).index[0])
    k3.metric("Median Rental Yield", f"{filt['Rental_Yield'].median():.2f}")
    k4.metric("Max Investment Score", f"{filt['Investment_Score'].max():.2f}")
    k5.metric("Average Price/SqFt", f"{(filt['Price']/filt['Area_sqft']).mean():.0f}")
    k6.metric("Top 3 Cities by Score", ", ".join(filt.groupby("City")["Investment_Score"].mean().sort_values(ascending=False).head(3).index.tolist()))

    # ---------------- Charts ----------------
    # Chart 1 - Rental Yield by Property Type
    st.markdown("### Rental Yield by Property Type")
    with st.expander("Purpose & Quick Tip"):
        st.markdown("""
        **Purpose:** Compare average rental yield across property types.<br>
        **Quick Tip:** Focus on property types with highest yield for investments.
        """, unsafe_allow_html=True)
    ptype_yield = filt.groupby("Property_Type")["Rental_Yield"].mean().reset_index()
    fig1 = px.bar(ptype_yield, x="Property_Type", y="Rental_Yield",
                  color="Rental_Yield", color_continuous_scale=px.colors.sequential.Plasma,
                  text="Rental_Yield")
    fig1.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2 - City-wise Rental Yield
    st.markdown("### City-wise Rental Yield")
    with st.expander("Purpose & Quick Tip"):
        st.markdown("""
        **Purpose:** Compare rental yields across cities.<br>
        **Quick Tip:** Prioritize investments in high-yield cities.
        """, unsafe_allow_html=True)
    city_yield = filt.groupby("City")["Rental_Yield"].mean().reset_index()
    fig2 = px.bar(city_yield, x="City", y="Rental_Yield", color="City",
                  text="Rental_Yield", color_discrete_sequence=px.colors.qualitative.Bold)
    fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3 - Investment Score Map
    st.markdown("### Top Investment Score Map")
    with st.expander("Purpose & Quick Tip"):
        st.markdown("""
        **Purpose:** Visualize properties with best combined yield & conversion probability.<br>
        **Quick Tip:** Target properties with highest Investment Score for best ROI.
        """, unsafe_allow_html=True)
    filt["Score_Normalized"] = (filt["Investment_Score"] - filt["Investment_Score"].min()) / (filt["Investment_Score"].max() - filt["Investment_Score"].min())
    fig3 = px.scatter_mapbox(filt, lat="Latitude", lon="Longitude",
                             size="Price", color="Score_Normalized",
                             hover_name="Property_Type",
                             hover_data=["City","Price","Rental_Yield","Conversion_Probability","Investment_Score"],
                             color_continuous_scale=px.colors.sequential.Viridis,
                             size_max=20, zoom=10)
    fig3.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig3, use_container_width=True)

    # ---------------- Top Investment Properties ----------------
    st.markdown("### Top Investment Properties by Score")
    top_inv = filt.sort_values("Investment_Score", ascending=False).head(10)
    st.dataframe(top_inv[["City","Property_Type","Price","Rental_Yield","Conversion_Probability","Investment_Score"]])

    csv = top_inv.to_csv(index=False)
    st.download_button("Download Top Investment Properties", csv, "top_investments.csv", "text/csv")
