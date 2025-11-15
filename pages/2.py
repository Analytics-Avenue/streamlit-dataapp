import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from io import BytesIO

st.set_page_config(page_title="App 2: Real Estate Demand Forecasting", layout="wide")


# ---------------------------------------------------------------------
# Page Title
# ---------------------------------------------------------------------
st.title("Real Estate Demand Forecasting System")
st.markdown("A professional analytics application for forecasting property demand using historic trends.")


# ---------------------------------------------------------------------
# SIDEBAR NAVIGATION
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
        This analytics module predicts future real estate demand using historical sales,
        listings, and price movement. The tool helps businesses understand market cycles,
        anticipate customer needs, and strategically manage inventory and pricing.
        """)

    with st.expander("2. Purpose of This Application"):
        st.write("""
        The purpose is to identify demand trends across cities, property types, and price segments.
        The forecasting engine helps sales, marketing, and strategy teams take proactive decisions.
        """)

    with st.expander("3. Business Impact"):
        st.write("""
        - Better inventory planning  
        - Improved price strategy  
        - Higher lead-to-sale conversion  
        - Reduced holding cost  
        - Increased market responsiveness  
        """)

    with st.expander("4. Our Capabilities"):
        st.write("""
        - Forecasting using regression time-series  
        - Dynamic city and property filters  
        - Auto-generated insights  
        - Trend detection on demand cycles  
        - Export-ready filtered datasets  
        """)

    st.header("Key Business KPIs")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Average Monthly Sales", "134 units")
    col2.metric("Demand Growth Rate", "18 percent")
    col3.metric("High-demand Cities", "6")
    col4.metric("Avg Price Sensitivity", "0.74")


# ---------------------------------------------------------------------
# APPLICATION PAGE
# ---------------------------------------------------------------------
if section == "Application":
    st.header("Application")

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
            st.error("Unable to load dataset from GitHub URL.")
            st.stop()

    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
        else:
            st.warning("Upload a CSV file to continue.")
            st.stop()


    # -----------------------------------------------------------------
    # REQUIRED COLUMNS CHECK
    # -----------------------------------------------------------------
    required_cols = ["City", "Date", "Property_Type", "Price"]

    if not all([col in df.columns for col in required_cols]):
        st.error(f"Dataset must contain columns: {required_cols}")
        st.stop()

    df["Date"] = pd.to_datetime(df["Date"])

    # -----------------------------------------------------------------
    # FILTER SECTION
    # -----------------------------------------------------------------
    st.subheader("Filters")

    colA, colB, colC = st.columns(3)

    city_filter = colA.multiselect("City", df["City"].unique(), df["City"].unique())
    ptype_filter = colB.multiselect("Property Type", df["Property_Type"].unique(), df["Property_Type"].unique())
    date_filter = colC.date_input("Date Filter", [])

    dff = df.copy()

    if city_filter:
        dff = dff[dff["City"].isin(city_filter)]
    if ptype_filter:
        dff = dff[dff["Property_Type"].isin(ptype_filter)]
    if date_filter:
        if len(date_filter) == 2:
            dff = dff[(dff["Date"] >= pd.to_datetime(date_filter[0])) & (dff["Date"] <= pd.to_datetime(date_filter[1]))]

    st.write("Filtered Rows:", len(dff))

    # -----------------------------------------------------------------
    # DOWNLOAD FILTERED DATA
    # -----------------------------------------------------------------
    def download_button(df):
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        st.download_button("Download Filtered Data", buffer.getvalue(), "filtered_data.csv",
                           "text/csv")

    download_button(dff)


    # -----------------------------------------------------------------
    # CHART 1: Monthly Demand Trend
    # -----------------------------------------------------------------
    st.subheader("Monthly Demand Trend")

    st.caption("Purpose: Shows how property demand changes month by month.")
    st.caption("Quick Tip: A rising curve signals an upcoming seller's market.")

    dff["Month"] = dff["Date"].dt.to_period("M").astype(str)
    trend = dff.groupby("Month").size().reset_index(name="Demand")

    fig1 = px.line(
        trend,
        x="Month",
        y="Demand",
        markers=True,
        title="Monthly Property Demand Trend",
        color_discrete_sequence=px.colors.qualitative.Dark24
    )

    fig1.update_layout(
        xaxis_title="Month",
        yaxis_title="Demand",
        xaxis=dict(title_font=dict(size=14, color="black")),
        yaxis=dict(title_font=dict(size=14, color="black")),
        font=dict(size=13)
    )

    fig1.update_traces(
        text=trend["Demand"],
        textposition="top center"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Insights
    with st.expander("Insights"):
        if len(trend) > 1:
            growth = round(((trend["Demand"].iloc[-1] - trend["Demand"].iloc[0]) / trend["Demand"].iloc[0]) * 100, 2)
            st.write(f"Demand changed by {growth} percent from the first to last month.")
        else:
            st.write("Not enough data to generate insights.")


    # -----------------------------------------------------------------
    # CHART 2: Demand by Property Type
    # -----------------------------------------------------------------
    st.subheader("Demand by Property Type")

    type_demand = dff.groupby("Property_Type").size().reset_index(name="Count")

    fig2 = px.bar(
        type_demand,
        x="Property_Type",
        y="Count",
        color="Property_Type",
        title="Property Type Demand",
        color_discrete_sequence=px.colors.qualitative.Prism
    )

    fig2.update_layout(
        xaxis_title="Property Type",
        yaxis_title="Demand",
        xaxis=dict(title_font=dict(size=14, color="black")),
        yaxis=dict(title_font=dict(size=14, color="black")),
    )

    fig2.update_traces(
        text=type_demand["Count"],
        textposition="outside"
    )

    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Insights"):
        top_type = type_demand.sort_values("Count", ascending=False).iloc[0]["Property_Type"]
        st.write(f"Most demanded property type is {top_type}.")



    # -----------------------------------------------------------------
    # FORECASTING MODEL
    # -----------------------------------------------------------------
    st.subheader("Demand Forecasting")

    if len(trend) < 3:
        st.warning("Not enough historical points for forecasting.")
        st.stop()

    trend["Index"] = np.arange(len(trend))

    X = trend[["Index"]]
    y = trend["Demand"]

    model = LinearRegression()
    model.fit(X, y)

    future_indexes = np.arange(len(trend), len(trend) + 6)
    forecast_values = model.predict(future_indexes.reshape(-1, 1))

    forecast_df = pd.DataFrame({
        "Month": [f"Future {i+1}" for i in range(6)],
        "Forecasted_Demand": forecast_values
    })

    fig3 = px.line(
        forecast_df,
        x="Month",
        y="Forecasted_Demand",
        markers=True,
        title="6-Month Demand Forecast",
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    fig3.update_traces(
        text=np.round(forecast_values, 1),
        textposition="top center"
    )

    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Insights"):
        st.write("The forecasted demand values show expected market trajectory for 6 upcoming months.")
