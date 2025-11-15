import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from io import BytesIO

# ---------------------------------------------
# PAGE CONFIG
# ---------------------------------------------
st.set_page_config(page_title="Real Estate Demand Forecasting", layout="wide")

hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display:none;}
section[data-testid="stSidebar"] {display:none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# ---------------------------------------------
# GLOBAL CSS
# ---------------------------------------------
st.markdown("""
<style>
.big-header {
    font-size: 40px; 
    font-weight: 900;
    background: linear-gradient(90deg,#0A5EB0,#2E82FF);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.card {
    background:#fff;
    border-radius:15px;
    padding:20px;
    margin-bottom:15px;
    box-shadow:0 4px 18px rgba(0,0,0,0.08);
}
.metric-card {
    background:#eef4ff;
    padding:15px;
    border-radius:8px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# TITLE
# ---------------------------------------------
st.markdown("<div class='big-header'>Real Estate Demand Forecasting System</div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Overview", "Application"])


# ================================================================
# OVERVIEW PAGE
# ================================================================
with tab1:

    st.markdown("### Overview")
    st.markdown("""
    <div class='card'>
    This analytical engine predicts real estate demand using historical pricing,
    listing activity and behavioural trends across different cities and property categories.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Purpose")
    st.markdown("""
    <div class='card'>
    • Identify demand patterns<br>
    • Predict future market momentum<br>
    • Support pricing and inventory planning<br>
    • Guide sales strategy for high-impact areas
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Capabilities")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class='card'>
        <b>Technical</b><br>
        • Linear regression forecasting<br>
        • Dynamic filtering<br>
        • Interactive visual analytics<br>
        • Automatic insights generation
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='card'>
        <b>Business</b><br>
        • Demand cycle prediction<br>
        • Price-sensitivity tracking<br>
        • Region-level growth insights<br>
        • Better sales conversions
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Avg Monthly Demand</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Demand Growth Rate</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Top Cities</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Price Elasticity</div>", unsafe_allow_html=True)



# ================================================================
# APPLICATION PAGE
# ================================================================
with tab2:

    st.markdown("### Step 1: Load Dataset")

    df = None

    mode = st.radio("Select Option:", ["Default Dataset", "Upload CSV"], horizontal=True)

    if mode == "Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        try:
            df = pd.read_csv(URL)
            st.success("Default dataset loaded.")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Could not load dataset: {e}")

    if mode == "Upload CSV":
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up:
            df = pd.read_csv(up)
            st.success("Dataset uploaded.")
            st.dataframe(df.head(), use_container_width=True)

    if df is None:
        st.stop()

    # -------------------------------------------------------------
    # USER COLUMN MAPPING (only required fields)
    # -------------------------------------------------------------
    st.markdown("### Step 2: Map Required Columns")

    col_map = {}

    required_cols = {
        "City": "City",
        "Date": "Listing_Date",
        "Property Type": "Property_Type",
        "Price": "Price"
    }

    for label, default in required_cols.items():
        col_map[label] = st.selectbox(
            f"Select column for {label}",
            options=df.columns,
            index=list(df.columns).index(default) if default in df.columns else 0
        )

    df = df.rename(columns={
        col_map["City"]: "City",
        col_map["Date"]: "Date",
        col_map["Property Type"]: "Property_Type",
        col_map["Price"]: "Price"
    })

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])


    # -------------------------------------------------------------
    # FILTERS
    # -------------------------------------------------------------
    st.markdown("### Step 3: Filters")

    f1, f2, f3 = st.columns(3)

    city = f1.multiselect("City", df["City"].unique(), df["City"].unique())
    ptype = f2.multiselect("Property Type", df["Property_Type"].unique(), df["Property_Type"].unique())
    dates = f3.date_input("Date Range")

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

    # Download filtered data
    buff = BytesIO()
    dff.to_csv(buff, index=False)
    st.download_button("Download Filtered Data", buff.getvalue(), "filtered_data.csv", "text/csv")


    # ================================================================
    # CHART 1: Monthly Demand Trend
    # ================================================================
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
            st.write(f"Demand changed by {growth:.2f} percent.")
        else:
            st.write("Not enough data for insights.")



    # ================================================================
    # CHART 2: Demand by Property Type
    # ================================================================
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


    # ================================================================
    # FORECASTING
    # ================================================================
    st.markdown("### 6-Month Demand Forecast")

    if len(trend) >= 3:

        trend["Index"] = np.arange(len(trend))
        model = LinearRegression().fit(trend[["Index"]], trend["Demand"])

        future_idx = np.arange(len(trend), len(trend) + 6)
        preds = model.predict(future_idx.reshape(-1, 1))

        fdf = pd.DataFrame({
            "Month": [f"Future {i+1}" for i in range(6)],
            "Forecasted": preds
        })

        fig3 = px.line(
            fdf,
            x="Month",
            y="Forecasted",
            markers=True,
            text=np.round(preds, 2),
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
            st.write("Short-term demand trend projected using regression.")
    else:
        st.warning("Not enough monthly data to forecast.")
