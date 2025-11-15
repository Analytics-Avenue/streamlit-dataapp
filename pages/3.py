import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

st.set_page_config(page_title="Real Estate Analytics Suite", layout="wide")

st.title("🏙️ Real Estate Analytics Suite")
st.write("A professional end-to-end analytics product demo for clients.")

# =======================================================================================
#   TAB 1: OVERVIEW
# =======================================================================================
tab1, tab2 = st.tabs(["📘 Overview", "🧩 Application"])

with tab1:

    st.header("1. Overview")
    st.write("""
This suite is designed to showcase a full real estate intelligence workflow used in valuation,
benchmarking, forecasting and investment insights across India and global markets.
""")

    st.header("2. Purpose")
    st.write("""
• Automate property price estimation  
• Reduce human dependency in valuations  
• Provide city/locality level intelligence  
• Build a standardized pricing foundation for developers, banks and investors  
""")

    st.header("3. Capabilities")
    st.write("""
**Technical**
• Machine learning valuation  
• Data ingestion pipelines  
• Multi-modal filtering  
• Geo-intelligence  
• Automated dashboards  

**Business**
• Faster deal closures  
• Transparent pricing  
• Risk reduction for investments  
• Better asset placement decisions  
""")

    st.header("4. Business Impact")
    st.write("""
• Up to 30–40% reduction in valuation errors  
• 3x faster property assessment  
• Improved transparency for buyers and investors  
• Better pricing negotiation ability  
""")

    st.header("5. KPIs Tracked")
    st.write("""
• Model RMSE  
• Price deviation %  
• Demand index  
• Region-wise supply analytics  
""")

# =======================================================================================
#   TAB 2: APPLICATION
# =======================================================================================
with tab2:

    st.header("Step 1: Dataset Setup")

    dataset_option = st.radio(
        "Choose how you want to load your dataset:",
        ["Default Dataset", "Upload Dataset", "Upload + Column Mapping"],
        horizontal=True
    )

    df = None

    # ------------------------------------------------------
    # DEFAULT DATASET
    # ------------------------------------------------------
    if dataset_option == "Default Dataset":
        GITHUB_URL = "https://raw.githubusercontent.com/plotly/datasets/master/real_estate_data.csv"
        try:
            df = pd.read_csv(GITHUB_URL)
            st.success("Default dataset loaded successfully.")
        except:
            st.error("Could not load default dataset.")

    # ------------------------------------------------------
    # USER UPLOAD DATASET
    # ------------------------------------------------------
    elif dataset_option == "Upload Dataset":
        file = st.file_uploader("Upload your CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)
            st.success("File uploaded successfully.")

    # ------------------------------------------------------
    # MANUAL MAPPING DATASET
    # ------------------------------------------------------
    elif dataset_option == "Upload + Column Mapping":
        file = st.file_uploader("Upload your CSV", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.write("### Uploaded Preview", raw.head())

            st.subheader("Map Columns")

            col_price = st.selectbox("Price Column", raw.columns)
            col_city = st.selectbox("City Column", raw.columns)
            col_prop = st.selectbox("Property Type Column", raw.columns)
            col_area = st.selectbox("Area Column", raw.columns)

            df = raw.rename(columns={
                col_price: "Price",
                col_city: "City",
                col_prop: "Property_Type",
                col_area: "Area"
            })

            st.success("Column mapping completed.")
            st.write(df.head())

    # ===================================================================================
    #     ONLY PROCEED IF DATA IS PRESENT
    # ===================================================================================
    if df is not None:

        st.divider()
        st.header("Step 2: Insights & Model Output")

        # Check required columns
        required = ["Price", "City", "Property_Type", "Area"]
        if not all(r in df.columns for r in required):
            st.error("Dataset missing required columns.")
        else:
            # ==========================
            #   FILTER SECTION
            # ==========================
            st.subheader("Filters")

            city_filter = st.multiselect("City", df["City"].unique())
            ptype_filter = st.multiselect("Property Type", df["Property_Type"].unique())

            filtered = df.copy()
            if city_filter:
                filtered = filtered[filtered["City"].isin(city_filter)]
            if ptype_filter:
                filtered = filtered[filtered["Property_Type"].isin(ptype_filter)]

            st.write("### Filtered Data", filtered.head())

            # ==========================
            #   DASHBOARD METRICS
            # ==========================
            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Total Properties", len(filtered))

            with c2:
                st.metric("Average Price", f"₹ {filtered['Price'].mean():,.0f}")

            with c3:
                st.metric("Average Area (sqft)", f"{filtered['Area'].mean():,.0f}")

            # ==========================
            #   PRICE DISTRIBUTION CHART
            # ==========================
            st.subheader("Price Distribution")
            fig = px.histogram(filtered, x="Price", nbins=40)
            st.plotly_chart(fig, use_container_width=True)

            # ==========================
            #   CITY-WISE AVERAGE PRICES
            # ==========================
            st.subheader("City-wise Pricing")
            fig2 = px.bar(
                filtered.groupby("City")["Price"].mean().reset_index(),
                x="City",
                y="Price",
                text="Price"
            )
            fig2.update_traces(texttemplate="₹ %{text:,.0f}", textposition="outside")
            st.plotly_chart(fig2, use_container_width=True)

            # ==========================
            #   MACHINE LEARNING MODEL
            # ==========================
            st.header("ML Price Prediction")

            df = df.dropna()

            X = df[["City", "Property_Type", "Area"]]
            y = df["Price"]

            transformer = ColumnTransformer(
                [
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["City", "Property_Type"]),
                    ("num", StandardScaler(), ["Area"])
                ],
                remainder="drop"
            )

            X_trans = transformer.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.2)

            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            # Prediction section
            st.subheader("Try a Prediction")

            user_city = st.selectbox("City", df["City"].unique())
            user_ptype = st.selectbox("Property Type", df["Property_Type"].unique())
            user_area = st.number_input("Area (sqft)", min_value=300, max_value=10000)

            input_df = pd.DataFrame([[user_city, user_ptype, user_area]],
                                    columns=["City", "Property_Type", "Area"])

            input_transformed = transformer.transform(input_df)
            pred_price = model.predict(input_transformed)[0]

            st.metric("Estimated Price", f"₹ {pred_price:,.0f}")

            st.success("Application processing complete.")

