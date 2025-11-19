import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from io import BytesIO

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

# --------------------------
# Required columns
# --------------------------
REQUIRED_COLS = ["City","Property_Type","BHK","Bathroom_Count","Area_sqft","Price","Parking","Age_Years"]

# --------------------------
# CSS for headers & hover cards
# --------------------------
st.markdown("""
<style>
.big-header {font-size:40px; font-weight:900; color:black;}
.card, .metric-card, .hover-card {background:#fff; border-radius:15px; padding:20px; margin-bottom:15px; box-shadow:0 4px 20px rgba(0,0,0,0.08); transition: all 0.25s ease; text-align:left;}
.card:hover, .metric-card:hover, .hover-card:hover {box-shadow:0 0 18px rgba(0,0,0,0.4); transform:scale(1.03);}
.metric-card {font-weight:600;}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Main header
# --------------------------
st.markdown("<div class='big-header'>Real Estate Price vs Property Features Analyzer</div>", unsafe_allow_html=True)

# ==========================================================
# Tabs
# ==========================================================
tab1, tab2 = st.tabs(["Overview", "Application"])

# ==========================================================
# Overview tab
# ==========================================================
with tab1:
    st.markdown("### Overview")
    st.markdown("<div class='hover-card'>This analytics module explores real estate prices in relation to property features like bedrooms, bathrooms, area, age, and amenities. It supports valuation, investment, and market insights.</div>", unsafe_allow_html=True)

    st.markdown("### Purpose")
    st.markdown("<div class='hover-card'>• Analyze price trends across property features<br>• Compare city and property type valuations<br>• Predict estimated price using ML<br>• Support strategic investment decisions</div>", unsafe_allow_html=True)

    st.markdown("### Capabilities & Business Impact")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='hover-card'><b>Technical</b><br>• Feature-wise price analysis<br>• Interactive dashboards<br>• ML-based price prediction<br>• Geo and city-level comparison</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='hover-card'><b>Business</b><br>• Improved negotiation insights<br>• Transparent pricing trends<br>• Better investment planning<br>• Fast feature-based comparison</div>", unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Avg Price</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Avg Price per SqFt</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Top Property Type</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Top City</div>", unsafe_allow_html=True)

    st.markdown("### Use of App")
    st.markdown("<div class='hover-card'>• Monitor property pricing trends<br>• Estimate property value<br>• Support strategic investment decisions<br>• Compare market prices across cities</div>", unsafe_allow_html=True)

    st.markdown("### Who Should Use")
    st.markdown("<div class='hover-card'>• Real estate developers<br>• Investors<br>• Market analysts<br>• Agencies handling property transactions</div>", unsafe_allow_html=True)

# ==========================================================
# Application tab
# ==========================================================
with tab2:
    st.markdown("### Step 1: Load Dataset")
    df = None

    mode = st.radio("Select Option:", ["Default Dataset", "Upload CSV", "Upload CSV + Column Mapping"], horizontal=True)

    if mode=="Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        try:
            df = pd.read_csv(URL)
            st.success("Default dataset loaded.")
            st.dataframe(df.head(), use_container_width=True)
        except:
            st.error("Could not load default dataset.")

    if mode=="Upload CSV":
        st.markdown("#### Download Sample CSV for Reference")
        sample_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        sample_df = pd.read_csv(sample_url).head(5)
        st.download_button("Download Sample CSV", sample_df.to_csv(index=False), "sample.csv", "text/csv")
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file: df = pd.read_csv(file)

    if mode=="Upload CSV + Column Mapping":
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.write(raw.head())
            mapping = {}
            for col in REQUIRED_COLS:
                mapping[col] = st.selectbox(f"Map your column to {col}", options=["-- Select --"]+list(raw.columns))
            if st.button("Apply Mapping"):
                missing = [c for c,v in mapping.items() if v=="-- Select --"]
                if missing: st.error(f"Map all columns: {missing}")
                else:
                    df = raw.rename(columns=mapping)
                    st.success("Mapping applied")
                    st.dataframe(df.head())

    if df is None:
        st.stop()
    if not all(col in df.columns for col in REQUIRED_COLS):
        st.error("Dataset missing required columns."); st.stop()

    df = df.dropna()

    # Step 2: Filters
    st.markdown("### Step 2: Filters")
    f1, f2, f3, f4 = st.columns(4)
    with f1: city = st.multiselect("City", df["City"].unique())
    with f2: ptype = st.multiselect("Property Type", df["Property_Type"].unique())
    with f3: bhk = st.multiselect("Bedrooms", sorted(df["BHK"].unique()))
    with f4: bath = st.multiselect("Bathrooms", sorted(df["Bathroom_Count"].unique()))
    filt = df.copy()
    if city: filt=filt[filt["City"].isin(city)]
    if ptype: filt=filt[filt["Property_Type"].isin(ptype)]
    if bhk: filt=filt[filt["BHK"].isin(bhk)]
    if bath: filt=filt[filt["Bathroom_Count"].isin(bath)]
    st.dataframe(filt.head(), use_container_width=True)

    # KPIs
    st.markdown("### Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Average Price", f"₹ {filt['Price'].mean():,.0f}")
    k2.metric("Avg Price per SqFt", f"₹ {(filt['Price']/filt['Area_sqft']).mean():,.0f}")
    k3.metric("Top Property Type", filt["Property_Type"].mode()[0] if not filt.empty else "NA")
    k4.metric("Top City", filt["City"].mode()[0] if not filt.empty else "NA")

    # Charts
    st.markdown("### Price vs Bedrooms")
    grp = filt.groupby("BHK")["Price"].mean().reset_index()
    fig = px.bar(grp, x="BHK", y="Price", text="Price", color="BHK", color_discrete_sequence=px.colors.qualitative.Vivid)
    fig.update_traces(texttemplate="₹ %{text:,.0f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Price vs Bathrooms")
    grp2 = filt.groupby("Bathroom_Count")["Price"].mean().reset_index()
    fig2 = px.bar(grp2, x="Bathroom_Count", y="Price", text="Price", color="Bathroom_Count", color_discrete_sequence=px.colors.qualitative.Bold)
    fig2.update_traces(texttemplate="₹ %{text:,.0f}", textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Price vs Area")
    fig3 = px.scatter(filt, x="Area_sqft", y="Price", color="Property_Type", size="BHK",
                      color_discrete_sequence=px.colors.qualitative.Pastel, hover_data=["City"])
    st.plotly_chart(fig3, use_container_width=True)

    # ML Prediction
    st.markdown("### Step 3: ML Price Prediction")
    X = filt[["City","Property_Type","BHK","Bathroom_Count","Area_sqft"]]
    y = np.log1p(filt["Price"])
    transformer = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["City","Property_Type"]),
        ("num","passthrough", ["BHK","Bathroom_Count","Area_sqft"])
    ])
    X_trans = transformer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_trans,y,test_size=0.2,random_state=42)
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train,y_train)

    # Predict for full dataset
    y_pred = np.expm1(model.predict(X_trans))
    results = X.copy()
    results["Actual_Price"] = np.expm1(y)
    results["Predicted_Price"] = y_pred
    st.markdown("### Predicted vs Actual Prices")
    st.dataframe(results, use_container_width=True)

    # Download predictions
    buf_pred = BytesIO()
    results.to_csv(buf_pred,index=False)
    st.download_button("Download Predictions CSV", buf_pred.getvalue(), "ml_predictions.csv","text/csv")

    # Automated Insights
    insights = filt.groupby(["City","Property_Type"]).agg({"Price":["mean","max","min"]}).reset_index()
    insights.columns = ["City","Property_Type","Avg_Price","Max_Price","Min_Price"]
    st.markdown("### Automated Insights")
    st.dataframe(insights, use_container_width=True)

    buf_ai = BytesIO()
    insights.to_csv(buf_ai,index=False)
    st.download_button("Download Insights CSV", buf_ai.getvalue(), "automated_insights.csv","text/csv")
