import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from io import BytesIO

st.set_page_config(layout="wide")

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
    color:black;
}
.card {background:#fff;border-radius:15px;padding:20px;margin-bottom:15px;
box-shadow:0 4px 20px rgba(0,0,0,0.08);}
.metric-card {background:#eef4ff;padding:15px;border-radius:8px;text-align:left;
transition: all 0.25s ease; font-weight:600;}
.metric-card:hover {box-shadow:0 0 18px rgba(0,0,0,0.4); transform:scale(1.03);}
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
    st.markdown("""
    <div class='card'>
    <b>Technical</b><br>
    • Machine learning valuation<br>
    • Interactive dashboards<br>
    • Geo intelligence<br>
    • NLP Search<br>
    • Region-wise segmentation<br><br>
    <b>Business</b><br>
    • Faster deal closures<br>
    • Transparent pricing<br>
    • Better negotiations<br>
    • Predictable demand mapping
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Key KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Model RMSE</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Avg Price Deviation%</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Lead Score Efficiency</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Market Alignment Score</div>", unsafe_allow_html=True)

    st.markdown("### Use of App")
    st.markdown("""
    <div class='card'>
    • Monitor property pricing trends<br>
    • Forecast property valuations<br>
    • Compare city-wise and property-type trends
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Who Should Use")
    st.markdown("""
    <div class='card'>
    • Real estate investors<br>
    • Property developers<br>
    • Market analysts<br>
    • Agencies handling property transactions
    </div>
    """, unsafe_allow_html=True)

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
        st.markdown("#### Download Sample CSV for Reference")
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        try:
            sample_df = pd.read_csv(URL).head(5)
            sample_csv = sample_df.to_csv(index=False)
            st.download_button("Download Sample CSV", sample_csv, "sample_dataset.csv", "text/csv")
        except Exception as e:
            st.info(f"Sample CSV unavailable: {e}")
        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)

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

    if df is None:
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
    counts, bins = np.histogram(filt["Price"], bins=40)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    hist_df = pd.DataFrame({"bin_center": bin_centers, "count": counts})
    fig = px.bar(hist_df, x="bin_center", y="count", text="count", color_discrete_sequence=px.colors.qualitative.Vivid)
    fig.update_traces(textposition="outside", marker=dict(line=dict(width=1, color="black")), opacity=0.9)
    fig.update_layout(xaxis_title="<b>Price</b>", yaxis_title="<b>Count</b>", xaxis=dict(showline=True, linewidth=2, linecolor="black"), yaxis=dict(showline=True, linewidth=2, linecolor="black"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### City-wise Average Price")
    city_avg = filt.groupby("City")["Price"].mean().reset_index()
    fig2 = px.bar(city_avg, x="City", y="Price", text="Price", color="City", color_discrete_sequence=px.colors.qualitative.Bold)
    fig2.update_traces(texttemplate="₹ %{text:,.0f}", textposition="outside", marker=dict(line=dict(width=1, color="black")), opacity=0.9)
    fig2.update_layout(xaxis_title="<b>City</b>", yaxis_title="<b>Average Price</b>", xaxis=dict(showline=True, linewidth=2, linecolor="black"), yaxis=dict(showline=True, linewidth=2, linecolor="black"))
    st.plotly_chart(fig2, use_container_width=True)

    # ==========================================================
    # MACHINE LEARNING - PRICE PREDICTION
    # ==========================================================
    st.markdown("### Step 3: ML Price Prediction (Actual vs Predicted Table & Download)")

    X = df[["City", "Property_Type", "Area_sqft"]]
    y = df["Price"]

    transformer = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["City", "Property_Type"]),
        ("num", StandardScaler(), ["Area_sqft"])
    ])
    X_trans = transformer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Create table with features, actual, predicted
    test_df = pd.DataFrame(X_test, columns=transformer.get_feature_names_out())
    test_df["Price_Actual"] = y_test.values
    test_df["Price_Predicted"] = y_pred
    st.dataframe(test_df.head(), use_container_width=True)

    # Download button
    b = BytesIO()
    b.write(test_df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button("Download ML Predictions CSV", b, file_name="ml_predictions.csv", mime="text/csv")

    # ==========================================================
    # AUTOMATED INSIGHTS
    # ==========================================================
    st.markdown("### Automated Insights Table & Download")
    insights = filt.groupby(["City", "Property_Type"]).agg({"Price":["mean","max","min"]}).reset_index()
    insights.columns = ["City","Property_Type","Avg_Price","Max_Price","Min_Price"]
    st.dataframe(insights, use_container_width=True)
    b2 = BytesIO()
    b2.write(insights.to_csv(index=False).encode("utf-8"))
    b2.seek(0)
    st.download_button("Download Automated Insights CSV", b2, file_name="automated_insights.csv", mime="text/csv")
