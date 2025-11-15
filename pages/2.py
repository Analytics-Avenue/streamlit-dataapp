import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from io import BytesIO

# ----------------------------------------------------------
# PAGE LAYOUT + HIDE SIDEBAR
# ----------------------------------------------------------
st.set_page_config(page_title="Demand Forecasting Suite", layout="wide")

hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# ----------------------------------------------------------
# CSS FOR CARDS + HEADERS (Same as App1)
# ----------------------------------------------------------
st.markdown("""
<style>
.big-header {
    font-size: 40px; font-weight: 900;
    background: linear-gradient(90deg,#0A5EB0,#2E82FF);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.card {
    background:#fff;
    border-radius:15px;
    padding:20px;
    margin-bottom:15px;
    box-shadow:0 4px 20px rgba(0,0,0,0.08);
}
.metric-card {
    background:#eef4ff;
    padding:15px;
    border-radius:8px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# MAIN TITLE
# ----------------------------------------------------------
st.markdown("<div class='big-header'>Real Estate Demand Forecasting System</div>", unsafe_allow_html=True)

# ==========================================================
# TABS (Same as App1)
# ==========================================================
tab1, tab2 = st.tabs(["Overview", "Application"])

# ==========================================================
# OVERVIEW TAB
# ==========================================================
with tab1:
    st.markdown("### Overview")
    st.markdown("""
    <div class='card'>
    This platform forecasts future real estate demand using historical price and listing signals.
    It equips business teams with insights for planning, pricing and inventory decisions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Purpose")
    st.markdown("""
    <div class='card'>
    • Identify demand cycles<br>
    • Detect property trends<br>
    • Forecast supply needs<br>
    • Support sales and strategy decisions
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Capabilities")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='card'>
        <b>Technical</b><br>
        • Regression forecasting<br>
        • Interactive filtering<br>
        • Multi-property analytics<br>
        • Automated insights
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='card'>
        <b>Business</b><br>
        • Predict market cycles<br>
        • Optimize inventory<br>
        • Guide pricing<br>
        • Improve conversions
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Avg Monthly Sales</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Demand Growth Rate</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>High Demand Cities</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Price Sensitivity</div>", unsafe_allow_html=True)


# ==========================================================
# APPLICATION TAB
# ==========================================================
with tab2:

    st.markdown("### Step 1: Load Dataset")
    df = None

    mode = st.radio(
        "Select Option:",
        ["Default Dataset", "Upload CSV"],
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
            st.dataframe(df.head(), use_container_width=True)
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
            st.dataframe(df.head(), use_container_width=True)

    if df is None:
        st.stop()

    # Required columns
    required = ["City", "Date", "Property_Type", "Price"]
    if not all(col in df.columns for col in required):
        st.error(f"Dataset must contain required columns: {required}")
        st.stop()

    df["Date"] = pd.to_datetime(df["Date"])

    # ==========================================================
    # FILTERS
    # ==========================================================
    st.markdown("### Step 2: Dashboard Filters")
    f1, f2, f3 = st.columns(3)

    city = f1.multiselect("City", df["City"].unique(), df["City"].unique())
    ptype = f2.multiselect("Property Type", df["Property_Type"].unique(), df["Property_Type"].unique())
    dates = f3.date_input("Select Date Range", [])

    dff = df.copy()
    if city:
        dff = dff[dff["City"].isin(city)]
    if ptype:
        dff = dff[dff["Property_Type"].isin(ptype)]
    if len(dates) == 2:
        dff = dff[(dff["Date"] >= pd.to_datetime(dates[0])) &
                  (dff["Date"] <= pd.to_datetime(dates[1]))]

    st.markdown("### Data Preview")
    st.dataframe(dff.head(), use_container_width=True)

    # ----------------------------------------------------------
    # DOWNLOAD FILTERED DATA
    # ----------------------------------------------------------
    buffer = BytesIO()
    dff.to_csv(buffer, index=False)
    st.download_button(
        "Download Filtered Data",
        data=buffer.getvalue(),
        file_name="filtered_data.csv",
        mime="text/csv"
    )

    # ==========================================================
    # CHART 1: Monthly Demand Trend
    # ==========================================================
    st.markdown("### Monthly Demand Trend")

    dff["Month"] = dff["Date"].dt.to_period("M").astype(str)
    trend = dff.groupby("Month").size().reset_index(name="Demand")

    fig1 = px.line(
        trend,
        x="Month",
        y="Demand",
        markers=True,
        text="Demand",
        color_discrete_sequence=px.colors.qualitative.Dark24
    )

    # Bold black axes
    fig1.update_layout(
        xaxis=dict(showline=True, linewidth=2, linecolor="black"),
        yaxis=dict(showline=True, linewidth=2, linecolor="black"),
        xaxis_title="<b>Month</b>",
        yaxis_title="<b>Demand</b>"
    )

    fig1.update_traces(textposition="top center")

    st.plotly_chart(fig1, use_container_width=True)

    with st.expander("Insights"):
        if len(trend) > 1:
            growth = ((trend["Demand"].iloc[-1] - trend["Demand"].iloc[0]) /
                      trend["Demand"].iloc[0]) * 100
            st.write(f"Demand changed by {growth:.2f} percent overall.")
        else:
            st.write("Not enough data for insights.")

    # ==========================================================
    # CHART 2: Demand by Property Type
    # ==========================================================
    st.markdown("### Demand by Property Type")

    tdf = dff.groupby("Property_Type").size().reset_index(name="Count")

    fig2 = px.bar(
        tdf,
        x="Property_Type",
        y="Count",
        text="Count",
        color="Property_Type",
        color_discrete_sequence=px.colors.qualitative.Prism
    )

    fig2.update_layout(
        xaxis=dict(showline=True, linewidth=2, linecolor="black"),
        yaxis=dict(showline=True, linewidth=2, linecolor="black"),
        xaxis_title="<b>Property Type</b>",
        yaxis_title="<b>Demand</b>"
    )

    fig2.update_traces(textposition="outside")

    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Insights"):
        top_type = tdf.sort_values("Count", ascending=False).iloc[0]["Property_Type"]
        st.write(f"The most demanded property type is {top_type}.")

    # ==========================================================
    # FORECASTING
    # ==========================================================
    st.markdown("### 6-Month Demand Forecast")

    if len(trend) >= 3:
        trend["Index"] = np.arange(len(trend))
        model = LinearRegression().fit(trend[["Index"]], trend["Demand"])

        future_idx = np.arange(len(trend), len(trend) + 6)
        forecast_vals = model.predict(future_idx.reshape(-1, 1))

        fdf = pd.DataFrame({
            "Month": [f"Future {i+1}" for i in range(6)],
            "Forecasted": forecast_vals
        })

        fig3 = px.line(
            fdf,
            x="Month",
            y="Forecasted",
            markers=True,
            text=np.round(forecast_vals, 1),
            color_discrete_sequence=px.colors.qualitative.Bold
        )

        fig3.update_layout(
            xaxis=dict(showline=True, linewidth=2, linecolor="black"),
            yaxis=dict(showline=True, linewidth=2, linecolor="black"),
            xaxis_title="<b>Month</b>",
            yaxis_title="<b>Forecasted Demand</b>"
        )

        fig3.update_traces(textposition="top center")

        st.plotly_chart(fig3, use_container_width=True)

        with st.expander("Insights"):
            st.write("The forecast reveals the expected short-term demand trend for the next 6 months.")
    else:
        st.warning("Not enough monthly data available for forecasting.")
