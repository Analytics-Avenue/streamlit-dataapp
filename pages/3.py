import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

st.set_page_config(page_title="Real Estate Price vs Property Features Analyzer", layout="wide")

# --------------------------
# Hide default sidebar
# --------------------------
hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# --------------------------
# Required columns for analysis
# --------------------------
REQUIRED_COLS = [
    "City",
    "Property_Type",
    "BHK",
    "Bathroom_Count",
    "Area_sqft",
    "Price",
    "Parking",
    "Age_Years"
]

# --------------------------
# CSS for headers & cards
# --------------------------
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

# --------------------------
# Main header
# --------------------------
st.markdown("<div class='big-header'>Real Estate Price vs Property Features Analyzer</div>", unsafe_allow_html=True)

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
    This analytics module explores real estate prices in relation to property features
    like bedrooms, bathrooms, area, age, and amenities. It supports valuation, investment,
    and market insights.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Purpose")
    st.markdown("""
    <div class='card'>
    • Analyze price trends across property features<br>
    • Compare city and property type valuations<br>
    • Predict estimated price using ML<br>
    • Support strategic investment decisions
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Capabilities")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='card'>
        <b>Technical</b><br>
        • Feature-wise price analysis<br>
        • Interactive dashboards<br>
        • ML-based price prediction<br>
        • Geo and city-level comparison
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='card'>
        <b>Business</b><br>
        • Improved negotiation insights<br>
        • Transparent pricing trends<br>
        • Better investment planning<br>
        • Fast feature-based comparison
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Avg Price</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Avg Price per SqFt</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Top Property Type</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Top City</div>", unsafe_allow_html=True)

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

    # ----------------------------------------------------------
    # Default dataset
    # ----------------------------------------------------------
    if mode == "Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        try:
            df = pd.read_csv(URL)
            st.success("Default dataset loaded successfully.")
            st.write(df.head())
        except Exception as e:
            st.error(f"Could not load dataset: {e}")

    # ----------------------------------------------------------
    # Upload CSV
    # ----------------------------------------------------------
    if mode == "Upload CSV":
        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)
            st.success("Dataset uploaded.")
            st.dataframe(df.head())

    # ----------------------------------------------------------
    # Upload + Column Mapping
    # ----------------------------------------------------------
    if mode == "Upload CSV + Column Mapping":
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

    # ----------------------------------------------------------
    if df is None:
        st.stop()

    # Ensure required columns exist
    if not all(col in df.columns for col in REQUIRED_COLS):
        st.error("Dataset missing required columns.")
        st.write("Required columns:", REQUIRED_COLS)
        st.stop()

    df = df.dropna()

    # ==========================================================
    # Step 2: Filters
    # ==========================================================
    st.markdown("### Step 2: Dashboard Filters")
    f1, f2, f3, f4 = st.columns(4)

    with f1:
        city = st.multiselect("City", df["City"].unique())
    with f2:
        ptype = st.multiselect("Property Type", df["Property_Type"].unique())
    with f3:
        bhk = st.multiselect("Bedrooms", sorted(df["BHK"].unique()))
    with f4:
        bath = st.multiselect("Bathrooms", sorted(df["Bathroom_Count"].unique()))

    filt = df.copy()
    if city:
        filt = filt[filt["City"].isin(city)]
    if ptype:
        filt = filt[filt["Property_Type"].isin(ptype)]
    if bhk:
        filt = filt[filt["BHK"].isin(bhk)]
    if bath:
        filt = filt[filt["Bathroom_Count"].isin(bath)]

    st.markdown("### Data Preview")
    st.dataframe(filt.head(), use_container_width=True)

    # ==========================================================
    # KPIs
    # ==========================================================
    st.markdown("### Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Average Price", f"₹ {filt['Price'].mean():,.0f}")
    k2.metric("Avg Price per SqFt", f"₹ {(filt['Price'] / filt['Area_sqft']).mean():,.0f}")
    k3.metric("Top Property Type", filt["Property_Type"].mode()[0] if not filt.empty else "NA")
    k4.metric("Top City", filt["City"].mode()[0] if not filt.empty else "NA")

    # ==========================================================
    # Charts
    # ==========================================================
    # Price vs Bedrooms
    st.markdown("### Price vs Bedrooms")
    grouped_bhk = filt.groupby("BHK")["Price"].mean().reset_index()
    fig = px.bar(grouped_bhk, x="BHK", y="Price", text="Price", color="BHK", color_discrete_sequence=px.colors.qualitative.Vivid)
    fig.update_traces(texttemplate="₹ %{text:,.0f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # Price vs Bathrooms
    st.markdown("### Price vs Bathrooms")
    grouped_bath = filt.groupby("Bathroom_Count")["Price"].mean().reset_index()
    fig2 = px.bar(grouped_bath, x="Bathroom_Count", y="Price", text="Price", color="Bathroom_Count", color_discrete_sequence=px.colors.qualitative.Bold)
    fig2.update_traces(texttemplate="₹ %{text:,.0f}", textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    # Price vs Area
    st.markdown("### Price vs Area")
    fig3 = px.scatter(filt, x="Area_sqft", y="Price", color="Property_Type", size="BHK",
                      color_discrete_sequence=px.colors.qualitative.Pastel, hover_data=["City"])
    st.plotly_chart(fig3, use_container_width=True)

    # ==========================================================
    # ML Price Prediction
    # ==========================================================
    # ==========================================================
    st.markdown("### Step 3: ML Price Prediction")
    
    # Features
    X = filt[["City", "Property_Type", "BHK", "Bathroom_Count", "Area_sqft"]]
    y = np.log1p(filt["Price"])  # log-transform to stabilize skew
    
    transformer = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["City", "Property_Type"]),
        ("num", "passthrough", ["BHK", "Bathroom_Count", "Area_sqft"])
    ])
    
    X_trans = transformer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    
    # Input from user
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        p_city = st.selectbox("City", filt["City"].unique())
    with c2:
        p_ptype = st.selectbox("Property Type", filt["Property_Type"].unique())
    with c3:
        p_bhk = st.selectbox("Bedrooms", sorted(filt["BHK"].unique()))
    with c4:
        p_bath = st.selectbox("Bathrooms", sorted(filt["Bathroom_Count"].unique()))
    
    p_area = st.number_input("Area (sqft)", min_value=300, max_value=10000, value=1200)
    
    pred_input = transformer.transform(pd.DataFrame([[p_city, p_ptype, p_bhk, p_bath, p_area]],
                         columns=["City","Property_Type","BHK","Bathroom_Count","Area_sqft"]))
    
    price_pred = np.expm1(model.predict(pred_input)[0])  # reverse log-transform
    st.metric("Estimated Price", f"₹ {price_pred:,.0f}")
    
    
    
    
