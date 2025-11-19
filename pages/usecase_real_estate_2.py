import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from io import BytesIO

# ----------------------------------------------------------
# PAGE LAYOUT
# ----------------------------------------------------------
st.set_page_config(page_title="Demand Forecasting Suite", layout="wide")

hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# -------------------------
# Company Logo + Name
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display: flex; align-items: center;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0; padding:0;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0; padding:0;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# CSS
# ----------------------------------------------------------
st.markdown("""
<style>
.big-header {font-size:40px; font-weight:900; color:black;}
.card, .metric-card, .hover-card {background:#fff; border-radius:15px; padding:20px; margin-bottom:15px; box-shadow:0 4px 20px rgba(0,0,0,0.08); transition: all 0.25s ease; text-align:left;}
.card:hover, .metric-card:hover, .hover-card:hover {box-shadow:0 0 18px rgba(0,0,0,0.4); transform:scale(1.03);}
.metric-card {font-weight:600;}
.grid-container {display:flex; gap:20px; flex-wrap:wrap;}
.grid-item {flex:1;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# TITLE
# ----------------------------------------------------------
st.markdown("<div class='big-header'>Real Estate Demand Forecasting System</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# TABS
# ----------------------------------------------------------
tab1, tab2 = st.tabs(["Overview", "Application"])

# ----------------------------------------------------------
# OVERVIEW
# ----------------------------------------------------------
with tab1:
    st.markdown("### Overview")
    st.markdown("""
    <div class='hover-card'>
    This platform forecasts real estate demand using market signals like price, location, and listing trends.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Purpose")
    st.markdown("""
    <div class='hover-card'>
    • Identify demand cycles<br>
    • Detect property trends<br>
    • Forecast future inventory needs<br>
    • Support pricing and sales strategy
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Capabilities & Business Impact")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='hover-card'>
        <b>Capabilities</b><br>
        • Regression forecasting<br>
        • Dynamic filters<br>
        • Time-based analytics<br>
        • Automated insights engine
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='hover-card'>
        <b>Business Impact</b><br>
        • Predict market cycles<br>
        • Optimize inventory<br>
        • Guide pricing<br>
        • Improve conversion outcomes
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Avg Monthly Sales</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Demand Growth Rate</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>High Demand Cities</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Price Sensitivity</div>", unsafe_allow_html=True)

    st.markdown("### Use of App")
    st.markdown("""
    <div class='hover-card'>
    • Monitor real estate demand trends<br>
    • Forecast upcoming market activity<br>
    • Support strategic pricing and inventory decisions
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Who Should Use")
    st.markdown("""
    <div class='hover-card'>
    • Real estate developers<br>
    • Property investors<br>
    • Market analysts<br>
    • Agencies handling property transactions
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------
# APPLICATION
# ----------------------------------------------------------
with tab2:

    st.markdown("### Step 1: Load Dataset")

    df = None
    REQUIRED = {
        "City": "City",
        "Listing_Date": "Date",
        "Property_Type": "Property Type",
        "Price": "Price",
    }

    mode = st.radio(
        "Select Input Method",
        ["Default Dataset", "Upload CSV (Auto Detect)", "Upload CSV (Manual Mapping)"],
        horizontal=True
    )

    if mode == "Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        try:
            df = pd.read_csv(URL)
            st.success("Default dataset loaded.")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Could not load dataset: {e}")
            st.stop()

    if mode == "Upload CSV (Auto Detect)":
        st.markdown("#### Download Sample CSV for Reference")
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        sample_df = pd.read_csv(URL).head(5)
        sample_csv = sample_df.to_csv(index=False)
        st.download_button("Download Sample CSV", sample_csv, "sample_dataset.csv", "text/csv")
        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)
            missing = [c for c in REQUIRED if c not in df.columns]
            if missing:
                st.error(f"Dataset missing required columns: {missing}")
                st.stop()
            st.success("Dataset uploaded.")
            st.dataframe(df.head(), use_container_width=True)
        else:
            st.stop()

    if mode == "Upload CSV (Manual Mapping)":
        file = st.file_uploader("Upload dataset to map", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.success("Dataset uploaded. Map required columns below.")
            col1, col2, col3, col4 = st.columns(4)
            city_col = col1.selectbox("City Column", raw.columns)
            date_col = col2.selectbox("Date Column", raw.columns)
            type_col = col3.selectbox("Property Type Column", raw.columns)
            price_col = col4.selectbox("Price Column", raw.columns)
            df = raw.rename(columns={
                city_col: "City",
                date_col: "Listing_Date",
                type_col: "Property_Type",
                price_col: "Price"
            })
            st.info("Mapped Columns → City, Date, Property_Type, Price")
            st.dataframe(df.head(), use_container_width=True)
        else:
            st.stop()

    if not all(col in df.columns for col in REQUIRED):
        missing = [c for c in REQUIRED if c not in df.columns]
        st.error(f"Dataset must contain required columns: {missing}")
        st.stop()

    df["Date"] = pd.to_datetime(df["Listing_Date"], errors="coerce")
    df = df.dropna(subset=["Listing_Date"])

    # ----------------------------------------------------------
    # FILTERS
    # ----------------------------------------------------------
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
        dff = dff[(dff["Date"] >= pd.to_datetime(dates[0])) & (dff["Date"] <= pd.to_datetime(dates[1]))]

    st.markdown("### Data Preview")
    st.dataframe(dff.head(), use_container_width=True)

    # ----------------------------------------------------------
    # DOWNLOAD FILTERED DATA
    # ----------------------------------------------------------
    buf = BytesIO()
    dff.to_csv(buf, index=False)
    st.download_button("Download Filtered Data", data=buf.getvalue(), file_name="filtered_data.csv", mime="text/csv")

    # ----------------------------------------------------------
    # CHARTS
    # ----------------------------------------------------------
    st.markdown("### Monthly Demand Trend")
    dff["Month"] = dff["Date"].dt.to_period("M").astype(str)
    trend = dff.groupby("Month").size().reset_index(name="Demand")
    fig1 = px.line(trend, x="Month", y="Demand", markers=True, text="Demand", color_discrete_sequence=px.colors.qualitative.Dark24)
    fig1.update_traces(textposition="top center", textfont=dict(size=12))
    fig1.update_layout(xaxis=dict(showline=True, linewidth=2, linecolor="black"), yaxis=dict(showline=True, linewidth=2, linecolor="black"), xaxis_title="<b>Month</b>", yaxis_title="<b>Demand</b>")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### Demand by Property Type")
    tdf = dff.groupby("Property_Type").size().reset_index(name="Count")
    fig2 = px.bar(tdf, x="Property_Type", y="Count", text="Count", color="Property_Type", color_discrete_sequence=px.colors.qualitative.Prism)
    fig2.update_traces(textposition="outside", textfont=dict(size=12))
    fig2.update_layout(xaxis=dict(showline=True, linewidth=2, linecolor="black"), yaxis=dict(showline=True, linewidth=2, linecolor="black"), xaxis_title="<b>Property Type</b>", yaxis_title="<b>Demand</b>")
    st.plotly_chart(fig2, use_container_width=True)

    # ----------------------------------------------------------
    # FORECASTING / ML PREDICTIONS TABLE + DOWNLOAD
    # ----------------------------------------------------------
    st.markdown("### 6-Month Demand Forecast & ML Table")

    if len(trend) >= 3:
        trend["Index"] = np.arange(len(trend))
        model = LinearRegression().fit(trend[["Index"]], trend["Demand"])
        future_idx = np.arange(len(trend), len(trend)+6)
        forecast_vals = model.predict(future_idx.reshape(-1,1))

        fdf = pd.DataFrame({"Month":[f"Future {i+1}" for i in range(6)], "Forecasted":forecast_vals})
        st.dataframe(fdf, use_container_width=True)

        # Download ML Predictions CSV
        buf_ml = BytesIO()
        fdf.to_csv(buf_ml, index=False)
        st.download_button("Download Forecast CSV", buf_ml.getvalue(), file_name="ml_predictions.csv", mime="text/csv")

        # Automated Insights Table
        insights = dff.groupby(["City","Property_Type"]).agg({"Price":["mean","max","min"]}).reset_index()
        insights.columns = ["City","Property_Type","Avg_Price","Max_Price","Min_Price"]
        st.markdown("### Automated Insights Table")
        st.dataframe(insights, use_container_width=True)

        buf_ai = BytesIO()
        insights.to_csv(buf_ai, index=False)
        st.download_button("Download Automated Insights CSV", buf_ai.getvalue(), file_name="automated_insights.csv", mime="text/csv")
