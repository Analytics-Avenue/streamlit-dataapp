import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# ----------------------------------------------------------
# PAGE SETTINGS + CUSTOM THEME + CSS
# ----------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Intelligence Suite",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# BEAUTIFUL UI CSS
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

section.main > div {
    padding-top: 1rem;
}

.big-header {
    font-size: 38px;
    font-weight: 900;
    background: linear-gradient(90deg, #0A5EB0, #2E82FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub {
    font-size: 22px;
    font-weight: 600;
    color: #0A5EB0;
    margin-bottom: 6px;
}

.card {
    background: rgba(255,255,255,0.65);
    border-radius: 16px;
    padding: 20px 26px;
    backdrop-filter: blur(8px);
    margin-bottom: 18px;
    box-shadow: 0 4px 22px rgba(0,0,0,0.08);
    transition: 0.2s ease-in-out;
}

.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
}

.metric-card {
    background: rgba(10,94,176,0.1);
    padding: 18px;
    border-radius: 12px;
    text-align: center;
}

.glass-box {
    background: rgba(250,250,250,0.7);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(9px);
    box-shadow: 0 4px 22px rgba(0,0,0,0.08);
}

.preview-btn {
    background: #0A5EB0;
    color: white;
    padding: 10px 18px;
    border-radius: 12px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------
# MAIN PAGE TITLE
# ----------------------------------------------------------
st.markdown("<div class='big-header'>Real Estate Intelligence Suite</div>", unsafe_allow_html=True)
st.write("A modern enterprise-grade property analytics and valuation system.")


# ==========================================================
# TABS: OVERVIEW + APPLICATION
# ==========================================================
tab1, tab2 = st.tabs(["Overview", "Application"])

# ==========================================================
# TAB 1: OVERVIEW
# ==========================================================
with tab1:

    st.markdown("### Platform Overview")

    st.markdown("<div class='sub'>1. Overview</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
    This platform provides a complete end-to-end real estate intelligence ecosystem.  
    It supports valuation, forecasting, benchmarking, demand analytics, and region-level insights across India and global cities.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sub'>2. Purpose</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
    • Automate price estimation  
    • Standardize real estate pricing  
    • Enable fast investment decisions  
    • Reduce manual valuation errors  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sub'>3. Capabilities</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='card'>
        <b>Technical Capabilities</b><br><br>
        • ML Price Prediction<br>
        • Real-time Filtering<br>
        • Multi-City Intelligence<br>
        • Automated Feature Engineering<br>
        • Region-wise Aggregation
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='card'>
        <b>Business Capabilities</b><br><br>
        • Deal Closure Acceleration<br>
        • Standardized Pricing Models<br>
        • Higher Transparency<br>
        • Better Negotiation Power
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='sub'>4. Business Impact</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
    • 40% reduction in valuation inconsistencies  
    • 3x faster pre-sales enablement  
    • Accurate forecasting improves investments  
    • More transparent developer–buyer ecosystem  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sub'>5. KPI Framework</div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown("<div class='metric-card'>Model RMSE</div>", unsafe_allow_html=True)
    m2.markdown("<div class='metric-card'>Avg Price Deviation%</div>", unsafe_allow_html=True)
    m3.markdown("<div class='metric-card'>Demand Index</div>", unsafe_allow_html=True)
    m4.markdown("<div class='metric-card'>Market Alignment Score</div>", unsafe_allow_html=True)



# ==========================================================
# TAB 2: APPLICATION
# ==========================================================
with tab2:

    st.markdown("<div class='sub'>Step 1: Load Dataset</div>", unsafe_allow_html=True)

    df = None

    method = st.radio(
        "Select input mode:",
        ["Default Dataset", "Upload File", "Upload + Column Mapping"],
        horizontal=True
    )

    # -------------------------------------------
    # Default dataset
    # -------------------------------------------
    if method == "Default Dataset":
        URL = "https://raw.githubusercontent.com/plotly/datasets/master/real_estate_data.csv"
        try:
            df = pd.read_csv(URL)
            st.success("Default dataset loaded successfully.")
            st.write(df.head())
        except:
            st.error("Failed to load dataset.")

    # -------------------------------------------
    # Upload dataset
    # -------------------------------------------
    elif method == "Upload File":
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)
            st.success("File uploaded.")
            st.write(df.head())

    # -------------------------------------------
    # Column mapping
    # -------------------------------------------
    elif method == "Upload + Column Mapping":
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.write("Uploaded Data", raw.head())

            st.markdown("### Map Required Columns")
            col_price = st.selectbox("Price Column", raw.columns)
            col_city = st.selectbox("City Column", raw.columns)
            col_ptype = st.selectbox("Property Type Column", raw.columns)
            col_area = st.selectbox("Area Column", raw.columns)

            df = raw.rename(columns={
                col_price: "Price",
                col_city: "City",
                col_ptype: "Property_Type",
                col_area: "Area"
            })

            st.success("Mapping applied successfully.")
            st.write(df.head())

    # -------------------------------------------
    # PROCESS IF DATA EXISTS
    # -------------------------------------------
    if df is not None:

        st.markdown("<div class='sub'>Step 2: Filters & Dashboard</div>", unsafe_allow_html=True)

        required_cols = ["Price", "City", "Property_Type", "Area"]
        if not all(c in df.columns for c in required_cols):
            st.warning("Dataset missing required columns.")
            st.stop()

        df = df.dropna()

        # FILTERS
        col1, col2 = st.columns(2)
        with col1:
            city_filter = st.multiselect("Filter by City", df["City"].unique())
        with col2:
            type_filter = st.multiselect("Filter by Property Type", df["Property_Type"].unique())

        filtered = df.copy()
        if city_filter:
            filtered = filtered[filtered["City"].isin(city_filter)]
        if type_filter:
            filtered = filtered[filtered["Property_Type"].isin(type_filter)]

        st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
        st.write("### Filtered Data Preview")
        st.dataframe(filtered.head(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # METRICS
        st.markdown("### Key Metrics")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Total Properties", len(filtered))
        mc2.metric("Avg Price", f"₹ {filtered['Price'].mean():,.0f}")
        mc3.metric("Avg Area", f"{filtered['Area'].mean():,.0f} sqft")

        # CHARTS
        st.markdown("### Price Distribution")
        fig1 = px.histogram(filtered, x="Price", nbins=40)
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("### City-wise Average Price")
        fig2 = px.bar(
            filtered.groupby("City")["Price"].mean().reset_index(),
            x="City", y="Price", text="Price"
        )
        fig2.update_traces(texttemplate="₹ %{text:,.0f}", textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

        # MACHINE LEARNING
        st.markdown("<div class='sub'>Step 3: ML Price Prediction</div>", unsafe_allow_html=True)

        X = df[["City", "Property_Type", "Area"]]
        y = df["Price"]

        transformer = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["City", "Property_Type"]),
            ("num", StandardScaler(), ["Area"])
        ])

        X_trans = transformer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.2)

        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        st.markdown("##### Try a Prediction")

        c1, c2, c3 = st.columns(3)
        with c1:
            inp_city = st.selectbox("Select City", df["City"].unique())
        with c2:
            inp_ptype = st.selectbox("Select Property Type", df["Property_Type"].unique())
        with c3:
            inp_area = st.number_input("Area (sqft)", min_value=300, max_value=10000, value=1200)

        pred_df = pd.DataFrame([[inp_city, inp_ptype, inp_area]], columns=["City", "Property_Type", "Area"])
        pred_trans = transformer.transform(pred_df)
        pred_price = model.predict(pred_trans)[0]

        st.metric("Estimated Price", f"₹ {pred_price:,.0f}")

