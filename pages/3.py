import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

st.set_page_config(layout="wide")

hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# ----------------------------------------------------------
# REQUIRED COLUMNS ONLY FOR THIS APPLICATION
# ----------------------------------------------------------
REQUIRED_COLS = [
    "City",
    "Property_Type",
    "Area_sqft",
    "Price"
]

# ----------------------------------------------------------
# PAGE SETTINGS + CSS
# ----------------------------------------------------------
st.set_page_config(page_title="Real Estate Intelligence Suite", layout="wide")

st.markdown("""
<style>
.big-header {
    font-size: 40px; font-weight: 900;
    background: linear-gradient(90deg,#0A5EB0,#2E82FF);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.card {background:#fff;border-radius:15px;padding:20px;margin-bottom:15px;
box-shadow:0 4px 20px rgba(0,0,0,0.08);}
.metric-card {background:#eef4ff;padding:15px;border-radius:8px;text-align:center;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# MAIN HEADER
# ----------------------------------------------------------
st.markdown("<div class='big-header'>Real Estate Intelligence Suite</div>", unsafe_allow_html=True)

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
    This platform provides an enterprise-grade real estate intelligence framework covering valuation, 
    forecasting, performance analytics, and city-level dashboards.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Purpose")
    st.markdown("""
    <div class='card'>
    • Standardize property pricing<br>
    • Improve valuation accuracy<br>
    • Support investors, developers, agencies<br>
    • Fast decision-making with ML automation
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Capabilities")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='card'>
        <b>Technical</b><br>
        • Machine learning valuation<br>
        • Interactive dashboards<br>
        • Geo intelligence<br>
        • NLP Search<br>
        • Region-wise segmentation
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='card'>
        <b>Business</b><br>
        • Faster deal closures<br>
        • Transparent pricing<br>
        • Better negotiations<br>
        • Predictable demand mapping
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Model RMSE</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Avg Price Deviation%</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Lead Score Efficiency</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Market Alignment Score</div>", unsafe_allow_html=True)

# ==========================================================
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

    # --------------------------------------------------------------
    # 1. DEFAULT DATASET
    # --------------------------------------------------------------
    if mode == "Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        try:
            df = pd.read_csv(URL)
            st.success("Default dataset loaded successfully.")
            st.write(df.head())
        except Exception as e:
            st.error(f"Could not load dataset: {e}")

    # --------------------------------------------------------------
    # 2. UPLOAD CSV
    # --------------------------------------------------------------
    if mode == "Upload CSV":
        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)
            st.success("Dataset uploaded.")
            st.dataframe(df.head())

    # --------------------------------------------------------------
    # 3. UPLOAD + COLUMN MAPPING (ONLY REQUIRED COLS)
    # --------------------------------------------------------------
    if mode == "Upload CSV + Column Mapping":
        file = st.file_uploader("Upload dataset", type=["csv"])
        
        if file:
            raw = pd.read_csv(file)
            st.write("Uploaded Data", raw.head())

            st.markdown("### Map Required Columns Only")

            mapping = {}

            # Ask mapping ONLY for required cols
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

    # --------------------------------------------------------------
    # PROCEED ONLY IF DATA LOADED
    # --------------------------------------------------------------
    if df is None:
        st.stop()

    # Ensure required columns exist
    if not all(col in df.columns for col in REQUIRED_COLS):
        st.error("Dataset missing required columns.")
        st.write("Required columns:", REQUIRED_COLS)
        st.stop()

    df = df.dropna()

    # ==========================================================
    # FILTERS
    # ==========================================================
    st.markdown("### Step 2: Dashboard Filters")

    f1, f2 = st.columns(2)
    with f1:
        city = st.multiselect("City", df["City"].unique())
    with f2:
        ptype = st.multiselect("Property Type", df["Property_Type"].unique())

    filt = df.copy()
    if city:
        filt = filt[filt["City"].isin(city)]
    if ptype:
        filt = filt[filt["Property_Type"].isin(ptype)]

    st.markdown("### Data Preview")
    st.dataframe(filt.head(), use_container_width=True)

    # ==========================================================
    # KPIs
    # ==========================================================
    st.markdown("### Key Metrics")
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Properties", len(filt))
    k2.metric("Avg Price", f"₹ {filt['Price'].mean():,.0f}")
    k3.metric("Avg Area", f"{filt['Area_sqft'].mean():,.0f} sqft")

    # ==========================================================
    # CHARTS
    # ==========================================================
    st.markdown("### Price Distribution")
    fig = px.histogram(filt, x="Price")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### City-wise Average Price")
    city_avg = filt.groupby("City")["Price"].mean().reset_index()
    fig2 = px.bar(city_avg, x="City", y="Price", text="Price")
    fig2.update_traces(texttemplate="₹ %{text:,.0f}", textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    # ==========================================================
    # MACHINE LEARNING - PRICE PREDICTION
    # ==========================================================
    st.markdown("### Step 3: ML Price Prediction")

    X = df[["City", "Property_Type", "Area_sqft"]]
    y = df["Price"]

    transformer = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["City", "Property_Type"]),
        ("num", StandardScaler(), ["Area_sqft"])
    ])

    X_trans = transformer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.2)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    c1, c2, c3 = st.columns(3)
    with c1:
        p_city = st.selectbox("City", df["City"].unique())
    with c2:
        p_ptype = st.selectbox("Property Type", df["Property_Type"].unique())
    with c3:
        p_area = st.number_input("Area (sqft)", min_value=300, max_value=10000, value=1200)

    pred = transformer.transform(pd.DataFrame([[p_city, p_ptype, p_area]],
                     columns=["City","Property_Type","Area_sqft"]))

    price_pred = model.predict(pred)[0]

    st.metric("Estimated Price", f"₹ {price_pred:,.0f}")
