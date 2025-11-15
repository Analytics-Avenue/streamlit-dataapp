import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# PAGE SETUP
# -------------------------------------------------------------------
st.set_page_config(page_title="Real Estate Intelligence Suite - App 3", layout="wide")

hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# -------------------------------------------------------------------
# REQUIRED COLUMNS FOR APP 3
# -------------------------------------------------------------------
REQUIRED_COLS = [
    "City",
    "Property_Type",
    "BHK",
    "Bathroom_Count",
    "Parking",
    "Age_Years",
    "Price",
    "Area_sqft"
]

# -------------------------------------------------------------------
# GLOBAL STYLING
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# MAIN HEADER
# -------------------------------------------------------------------
st.markdown("<div class='big-header'>Real Estate Intelligence Suite - App 3</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------
tab1, tab2 = st.tabs(["Overview", "Application"])

# ==========================================================
# TAB 1 - OVERVIEW
# ==========================================================
with tab1:
    st.markdown("### Overview")
    st.markdown("""
    <div class='card'>
    This module analyzes how property features such as bedrooms, bathrooms, parking, 
    age and size drive price variations across cities and asset classes.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Purpose")
    st.markdown("""
    <div class='card'>
    • Benchmark property pricing based on features<br>
    • Identify price-value anomalies<br>
    • Enable transparent buyer-seller negotiations<br>
    • Support project positioning and segmentation
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Capabilities")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='card'>
        <b>Technical</b><br>
        • Price-feature analytics<br>
        • Matplotlib charts with insights<br>
        • Feature correlation patterns<br>
        • Outlier detection capabilities
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='card'>
        <b>Business</b><br>
        • Better unit pricing<br>
        • Accurate feature-based valuation<br>
        • Improved listing quality<br>
        • Faster decision-making
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Avg Price</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Price per Sqft</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Top Property Type</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>City with Max Listings</div>", unsafe_allow_html=True)

# ==========================================================
# TAB 2 - APPLICATION
# ==========================================================
with tab2:

    # --------------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------------
    st.markdown("### Step 1: Load Dataset")

    mode = st.radio(
        "Choose Option:",
        ["Default Dataset", "Upload CSV", "Upload CSV + Column Mapping"],
        horizontal=True
    )

    df = None

    # DEFAULT DATASET
    if mode == "Default Dataset":
        url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        try:
            df = pd.read_csv(url)
            st.success("Default dataset loaded.")
            st.dataframe(df.head())
        except:
            st.error("Could not load dataset")

    # DIRECT CSV UPLOAD
    if mode == "Upload CSV":
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)
            st.success("Dataset uploaded.")
            st.dataframe(df.head())

    # COLUMN MAPPING
    if mode == "Upload CSV + Column Mapping":
        file = st.file_uploader("Upload dataset", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.write("Uploaded Data", raw.head())

            st.markdown("### Map Required Columns")
            mapping = {}

            for col in REQUIRED_COLS:
                mapping[col] = st.selectbox(
                    f"Map your column to: {col}",
                    ["-- Select --"] + list(raw.columns)
                )

            if st.button("Apply Mapping"):
                if any(v == "-- Select --" for v in mapping.values()):
                    st.error("Map all required fields first.")
                else:
                    df = raw.rename(columns=mapping)
                    st.success("Mapping applied successfully.")
                    st.dataframe(df.head())

    if df is None:
        st.stop()

    # Validate required columns
    if not all(col in df.columns for col in REQUIRED_COLS):
        st.error("Dataset missing required columns.")
        st.write("Required:", REQUIRED_COLS)
        st.stop()

    df = df.dropna()

    # --------------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------------
    st.markdown("### Step 2: Dashboard Filters")

    f1, f2, f3 = st.columns(3)

    with f1:
        city = st.multiselect("City", df["City"].unique())
    with f2:
        ptype = st.multiselect("Property Type", df["Property_Type"].unique())
    with f3:
        bhk = st.multiselect("BHK", sorted(df["BHK"].unique()))

    f4, f5 = st.columns(2)
    with f4:
        bath = st.multiselect("Bathrooms", sorted(df["Bathroom_Count"].unique()))
    with f5:
        parking = st.multiselect("Parking", df["Parking"].unique())

    price_range = st.slider(
        "Price Range",
        int(df["Price"].min()),
        int(df["Price"].max()),
        (int(df["Price"].min()), int(df["Price"].max()))
    )

    filtered = df.copy()

    if city:
        filtered = filtered[filtered["City"].isin(city)]
    if ptype:
        filtered = filtered[filtered["Property_Type"].isin(ptype)]
    if bhk:
        filtered = filtered[filtered["BHK"].isin(bhk)]
    if bath:
        filtered = filtered[filtered["Bathroom_Count"].isin(bath)]
    if parking:
        filtered = filtered[filtered["Parking"].isin(parking)]

    filtered = filtered[(filtered["Price"] >= price_range[0]) & (filtered["Price"] <= price_range[1])]

    # --------------------------------------------------------------
    # KPIs
    # --------------------------------------------------------------
    st.markdown("### Key Metrics")
    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Avg Price", f"₹ {filtered.Price.mean():,.0f}")
    k2.metric("Avg Price per Sqft", f"{(filtered.Price / filtered.Area_sqft).mean():.0f}")
    k3.metric("Top Property Type", filtered.Property_Type.mode()[0] if not filtered.empty else "NA")
    k4.metric("City with Most Listings", filtered.City.mode()[0] if not filtered.empty else "NA")

    st.markdown("---")

    # --------------------------------------------------------------
    # CHART 1 – PRICE VS BHK
    # --------------------------------------------------------------
    st.markdown("### Price vs BHK")

    fig, ax = plt.subplots(figsize=(8,5))
    grouped = filtered.groupby("BHK")["Price"].mean()

    ax.bar(grouped.index.astype(str), grouped.values, color="skyblue", edgecolor="black")
    ax.set_xlabel("BHK", fontweight="bold")
    ax.set_ylabel("Avg Price", fontweight="bold")
    ax.set_title("Price vs BHK")

    for i, v in enumerate(grouped.values):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)

    st.pyplot(fig)

    st.markdown("**Purpose:** Understand how BHK count impacts valuation.")
    st.markdown("---")

    # --------------------------------------------------------------
    # CHART 2 – PRICE VS AGE
    # --------------------------------------------------------------
    st.markdown("### Price vs Property Age")

    fig2, ax2 = plt.subplots(figsize=(8,5))
    grouped2 = filtered.groupby("Age_Years")["Price"].mean()

    ax2.plot(grouped2.index, grouped2.values, marker="o", linewidth=2)
    ax2.set_xlabel("Age (Years)", fontweight="bold")
    ax2.set_ylabel("Avg Price", fontweight="bold")
    ax2.set_title("Price vs Age")

    for i, val in enumerate(grouped2.values):
        ax2.text(i, val, f"{int(val):,}", ha="center", va="bottom", fontsize=9)

    st.pyplot(fig2)
    st.markdown("**Purpose:** See depreciation or appreciation trend over age.")
    st.markdown("---")

    # --------------------------------------------------------------
    # CHART 3 – PRICE VS SQFT (SCATTER)
    # --------------------------------------------------------------
    st.markdown("### Price vs Area")

    fig3, ax3 = plt.subplots(figsize=(8,5))
    ax3.scatter(filtered["Area_sqft"], filtered["Price"], alpha=0.6)
    ax3.set_xlabel("Area (sqft)", fontweight="bold")
    ax3.set_ylabel("Price", fontweight="bold")
    ax3.set_title("Price vs Sqft")

    st.pyplot(fig3)
    st.markdown("---")

    # --------------------------------------------------------------
    # DATA TABLE + DOWNLOAD
    # --------------------------------------------------------------
    st.subheader("Filtered Dataset")
    st.dataframe(filtered, use_container_width=True)

    csv = filtered.to_csv(index=False)
    st.download_button("Download Filtered Data", csv, "App3_filtered_data.csv", "text/csv")
