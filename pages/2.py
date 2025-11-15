import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from io import BytesIO

st.set_page_config(page_title="App 2: Real Estate Demand Forecasting", layout="wide")

# ---------------------------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------------------------
st.title("Real Estate Demand Forecasting System")
st.markdown("A professional analytics application for forecasting property demand using historic trends.")


# ---------------------------------------------------------------------
# LEFT SIDEBAR NAVIGATION (same layout as App 1)
# ---------------------------------------------------------------------
section = st.sidebar.radio(
    "Navigate",
    ["Overview", "Application"]
)


# ---------------------------------------------------------------------
# OVERVIEW PAGE
# ---------------------------------------------------------------------
if section == "Overview":
    st.header("Overview")

    with st.expander("1. Application Overview"):
        st.write("""
        This module predicts future real estate demand using sales, listings, and price trends.
        """)

    with st.expander("2. Purpose of This Application"):
        st.write("""
        Identifying demand trends across cities, property types, and price segments.
        """)

    with st.expander("3. Business Impact"):
        st.write("""
        - Improved inventory planning  
        - Better pricing strategy  
        - Higher lead-to-sale conversion  
        - Reduced holding cost  
        """)

    with st.expander("4. Capabilities"):
        st.write("""
        - Demand forecasting  
        - Dynamic filters  
        - Auto insights  
        - Trend pattern analysis  
        - Downloadable filtered data  
        """)

    st.header("Key KPIs")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Avg Monthly Sales", "134 units")
    col2.metric("Demand Growth Rate", "18%")
    col3.metric("Top Demanded Cities", "6")
    col4.metric("Price Sensitivity Index", "0.74")


# ---------------------------------------------------------------------
# APPLICATION PAGE
# ---------------------------------------------------------------------
if section == "Application":
    st.header("Application")

    # ------------------------------------------
    # DATASET SETUP
    # ------------------------------------------
    st.subheader("Dataset Setup")

    option = st.radio(
        "Select Dataset Input Method",
        ["Use Default Dataset", "Upload Your Own Dataset"]
    )

    if option == "Use Default Dataset":
        default_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        try:
            df = pd.read_csv(default_url)
            st.success("Default dataset loaded successfully")
        except:
            st.error("Unable to load dataset.")
            st.stop()

    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
        else:
            st.warning("Please upload a dataset to proceed.")
            st.stop()

    # ------------------------------------------
    # REQUIRED COLUMNS CHECK
    # ------------------------------------------
    required = ["City", "Date", "Property_Type", "Price"]
    if not all(x in df.columns for x in required):
        st.error(f"Dataset must include these columns: {required}")
        st.stop()

    df["Date"] = pd.to_datetime(df["Date"])

    # ------------------------------------------
    # FILTERS - same layout as App 1 (3-column row)
    # ------------------------------------------
    st.subheader("Filters")

    colA, colB, colC = st.columns(3)

    city_filter = colA.multiselect("City", df["City"].unique(), df["City"].unique())
    type_filter = colB.multiselect("Property Type", df["Property_Type"].unique(), df["Property_Type"].unique())
    date_filter = colC.date_input("Date Range")

    dff = df.copy()
    if city_filter:
        dff = dff[dff["City"].isin(city_filter)]
    if type_filter:
        dff = dff[dff["Property_Type"].isin(type_filter)]
    if len(date_filter) == 2:
        dff = dff[(dff["Date"] >= pd.to_datetime(date_filter[0])) &
                  (dff["Date"] <= pd.to_datetime(date_filter[1]))]

    st.write("Filtered Rows:", len(dff))

    # ------------------------------------------
    # DOWNLOAD FILTERED DATA
    # ------------------------------------------
    buffer = BytesIO()
    dff.to_csv(buffer, index=False)
    st.download_button("Download Filtered Data", buffer.getvalue(), "filtered_data.csv", "text/csv")

    # ------------------------------------------
    # CHART 1 — Monthly Demand Trend
    # ------------------------------------------
    st.subheader("Monthly Demand Trend")
    st.caption("Purpose: Shows month-over-month demand pattern.")
    st.caption("Quick Tip: Higher peaks indicate strong buyer sentiment.")

    dff["Month"] = dff["Date"].dt.to_period("M").astype(str)
    trend = dff.groupby("Month").size().reset_index(name="Demand")

    fig1 = px.line(
        trend,
        x="Month",
        y="Demand",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Dark24
    )

    fig1.update_layout(
        xaxis_title="Month",
        yaxis_title="Demand",
        xaxis=dict(title_font=dict(size=14, color="black")),
        yaxis=dict(title_font=dict(size=14, color="black")),
        font=dict(size=13),
    )

    fig1.update_traces(text=trend["Demand"], textposition="top center")

    st.plotly_chart(fig1, use_container_width=True)

    with st.expander("Insights"):
        if len(trend) > 1:
            growth = (trend["Demand"].iloc[-1] - trend["Demand"].iloc[0]) / trend["Demand"].iloc[0] * 100
            st.write(f"Overall demand trend changed by {growth:.2f}% during the period.")
        else:
            st.write("Not enough data for trend insight.")

    # ------------------------------------------
    # CHART 2 — Demand by Property Type
    # ------------------------------------------
    st.subheader("Demand by Property Type")
    type_demand = dff.groupby("Property_Type").size().reset_index(name="Count")

    fig2 = px.bar(
        type_demand,
        x="Property_Type",
        y="Count",
        color="Property_Type",
        color_discrete_sequence=px.colors.qualitative.Prism
    )

    fig2.update_layout(
        xaxis_title="Property Type",
        yaxis_title="Demand",
        xaxis=dict(title_font=dict(size=14, color="black")),
        yaxis=dict(title_font=dict(size=14, color="black")),
    )

    fig2.update_traces(text=type_demand["Count"], textposition="outside")

    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Insights"):
        top_type = type_demand.sort_values("Count", ascending=False).iloc[0]["Property_Type"]
        st.write(f"The most demanded property type is {top_type}.")

    # ------------------------------------------
    # FORECASTING
    # ------------------------------------------
    st.subheader("Demand Forecasting")

    if len(trend) >= 3:
        trend["Index"] = np.arange(len(trend))
        X = trend[["Index"]]
        y = trend["Demand"]

        model = LinearRegression()
        model.fit(X, y)

        future_idx = np.arange(len(trend), len(trend) + 6)
        prediction = model.predict(future_idx.reshape(-1, 1))

        forecast = pd.DataFrame({
            "Month": [f"Future {i+1}" for i in range(6)],
            "Forecasted_Demand": prediction
        })

        fig3 = px.line(
            forecast,
            x="Month",
            y="Forecasted_Demand",
            markers=True,
            color_discrete_sequence=px.colors.qualitative.Bold
        )

        fig3.update_traces(
            text=np.round(prediction, 1),
            textposition="top center"
        )

        st.plotly_chart(fig3, use_container_width=True)

        with st.expander("Insights"):
            st.write("This projected trend helps anticipate demand for the next 6 months.")

    else:
        st.warning("Not enough historical data to forecast.")


