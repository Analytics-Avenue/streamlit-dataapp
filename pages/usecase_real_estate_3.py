import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from io import BytesIO

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Real Estate Price vs Property Features Analyzer", layout="wide")

# Hide sidebar
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display:none;}
section[data-testid="stSidebar"] {display:none;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# LOGO + HEADER
# ------------------------------------------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

st.markdown(f"""
<div style="display:flex; align-items:center; margin-bottom:16px;">
    <img src="{logo_url}" width="60" style="margin-right:12px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:700;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:700;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# GLOBAL CSS — Exact Marketing Lab UI
# ------------------------------------------------------------
st.markdown("""
<style>
* { font-family: 'Inter', sans-serif; }
body, [class*="css"] { color:#000 !important; font-size:17px; }

/* Fade animation */
.block-container { animation: fadeIn 0.5s ease; }
@keyframes fadeIn { from {opacity:0; transform:translateY(10px);} to {opacity:1;} }

/* Section title */
.section-title {
    font-size:24px !important;
    font-weight:600 !important;
    margin-top:30px;
    margin-bottom:12px;
    color:#000;
    position:relative;
}
.section-title:after {
    content:"";
    position:absolute;
    bottom:-5px;
    left:0;
    height:2px;
    width:0%;
    background:#064b86;
    transition:0.35s ease;
}
.section-title:hover:after { width:40%; }

/* Card */
.card {
    background:#fff;
    padding:22px;
    border-radius:14px;
    border:1px solid #e6e6e6;
    box-shadow:0 3px 14px rgba(0,0,0,0.08);
    transition:0.25s ease;
    color:#000;
}
.card:hover {
    transform:translateY(-4px);
    border-color:#064b86;
    box-shadow:0 12px 25px rgba(6,75,134,0.18);
}

/* KPI - Blue */
.kpi {
    background:#fff;
    padding:22px;
    border-radius:14px;
    border:1px solid #e2e2e2;
    text-align:center;
    color:#064b86 !important;
    font-size:20px !important;
    font-weight:600 !important;
    box-shadow:0 3px 14px rgba(0,0,0,0.07);
    transition:0.25s ease;
}
.kpi:hover {
    transform:translateY(-4px);
    border-color:#064b86;
    box-shadow:0 12px 26px rgba(6,75,134,0.18);
}

/* Variable-box - Blue */
.variable-box {
    background:white;
    padding:16px;
    border-radius:14px;
    color:#064b86 !important;
    border:1px solid #e5e5e5;
    font-size:17px !important;
    font-weight:500 !important;
    box-shadow:0 2px 10px rgba(0,0,0,0.10);
    transition:0.25s ease;
    margin-bottom:10px;
}
.variable-box:hover {
    transform:translateY(-5px);
    border-color:#064b86;
    box-shadow:0 12px 22px rgba(6,75,134,0.18);
}

/* Table */
.required-table th {
    background:#ffffff !important;
    color:#000 !important;
    font-size:21px !important;
    border-bottom:2px solid #000 !important;
}
.required-table td {
    background:#ffffff !important;
    color:#000 !important;
    font-size:19px !important;
    padding:10px !important;
    border-bottom:1px solid #dcdcdc !important;
}
.required-table tr:hover td {
    background:#f8f8f8 !important;
}

/* Buttons */
.stButton>button, .stDownloadButton>button {
    background:#064b86 !important;
    color:white !important;
    border:none;
    padding:10px 22px;
    border-radius:8px !important;
    font-size:15.5px !important;
    font-weight:600;
    transition:0.25s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    transform:translateY(-3px);
    background:#0a6eb3 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-header' style='font-size:36px;font-weight:800;'>Real Estate Price vs Property Features Analyzer</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# Required Column Dictionary
# ------------------------------------------------------------
REQUIRED_COLS = ["City","Property_Type","BHK","Bathroom_Count","Area_sqft","Price","Parking","Age_Years"]

REQUIRED_DICT = {
    "City": "City where the property is located.",
    "Property_Type": "Category such as Apartment, Villa, Plot, etc.",
    "BHK": "Number of bedrooms.",
    "Bathroom_Count": "Number of bathrooms.",
    "Area_sqft": "Built-up or saleable area.",
    "Price": "Listing or sale price.",
    "Parking": "Parking availability or number of slots.",
    "Age_Years": "Age of the property in years."
}

# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# ------------------------------------------------------------
# OVERVIEW TAB
# ------------------------------------------------------------
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        Explore real estate valuation dynamics across features like area, BHK count, bathrooms, amenities, age, and city-level factors.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        • Feature-level pricing analysis<br>
        • City and property type comparisons<br>
        • ML-driven price estimation<br>
        • Interactive charts and insights
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        • Stronger negotiation insights<br>
        • Better pricing transparency<br>
        • Improved investment analysis<br>
        • Faster, data-backed decisions
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Avg Price</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Price per SqFt</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Top Property Type</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Top City</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# IMPORTANT ATTRIBUTES TAB
# ------------------------------------------------------------
with tab2:

    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    dict_df = pd.DataFrame(
        [{"Attribute": k, "Description": v} for k,v in REQUIRED_DICT.items()]
    )

    st.dataframe(dict_df.style.set_table_attributes('class="required-table"'),
                 use_container_width=True)

    st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
    indep = ["City","Property_Type","BHK","Bathroom_Count","Area_sqft","Parking","Age_Years"]
    for i in indep:
        st.markdown(f"<div class='variable-box'>{i}</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Dependent Variable</div>', unsafe_allow_html=True)
    st.markdown("<div class='variable-box'>Price</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# APPLICATION TAB
# ------------------------------------------------------------
with tab3:

    st.markdown('<div class="section-title">Step 1: Load Dataset</div>', unsafe_allow_html=True)

    df = None
    mode = st.radio("Select Option:", ["Default Dataset", "Upload CSV", "Upload CSV + Mapping"], horizontal=True)

    # Default
    if mode == "Default Dataset":
        try:
            URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
            df = pd.read_csv(URL)
            st.success("Dataset loaded.")
            st.dataframe(df.head(), use_container_width=True)
        except:
            st.error("Failed to load default dataset.")
            st.stop()

    # Upload CSV
    if mode == "Upload CSV":
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)
            st.dataframe(df.head(), use_container_width=True)

    # Mapping
    if mode == "Upload CSV + Mapping":
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.write(raw.head())
            mapping = {}
            for col in REQUIRED_COLS:
                mapping[col] = st.selectbox(f"Map → {col}", ["-- Select --"]+list(raw.columns))
            if st.button("Apply Mapping"):
                missing = [c for c,v in mapping.items() if v=="-- Select --"]
                if missing:
                    st.error(f"Map all: {missing}")
                else:
                    df = raw.rename(columns=mapping)
                    st.dataframe(df.head())

    if df is None:
        st.stop()

    if not all(col in df.columns for col in REQUIRED_COLS):
        st.error("Dataset missing required columns.")
        st.stop()

    df = df.dropna()

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------
    st.markdown('<div class="section-title">Step 2: Filters</div>', unsafe_allow_html=True)

    f1,f2,f3,f4 = st.columns(4)
    with f1: c1 = st.multiselect("City", df["City"].unique())
    with f2: c2 = st.multiselect("Property Type", df["Property_Type"].unique())
    with f3: c3 = st.multiselect("BHK", sorted(df["BHK"].unique()))
    with f4: c4 = st.multiselect("Bathrooms", sorted(df["Bathroom_Count"].unique()))

    filtered = df.copy()
    if c1: filtered = filtered[filtered["City"].isin(c1)]
    if c2: filtered = filtered[filtered["Property_Type"].isin(c2)]
    if c3: filtered = filtered[filtered["BHK"].isin(c3)]
    if c4: filtered = filtered[filtered["Bathroom_Count"].isin(c4)]

    st.dataframe(filtered.head(), use_container_width=True)

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------
    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)

    k1.metric("Avg Price", f"₹ {filtered['Price'].mean():,.0f}")
    k2.metric("Avg Price/SqFt", f"₹ {(filtered['Price']/filtered['Area_sqft']).mean():,.0f}")
    k3.metric("Top Property Type", filtered["Property_Type"].mode()[0] if not filtered.empty else "NA")
    k4.metric("Top City", filtered["City"].mode()[0] if not filtered.empty else "NA")

    # --------------------------------------------------------
    # Charts
    # --------------------------------------------------------
    st.markdown('<div class="section-title">Price vs Bedrooms</div>', unsafe_allow_html=True)
    grp1 = filtered.groupby("BHK")["Price"].mean().reset_index()
    st.plotly_chart(px.bar(grp1, x="BHK", y="Price", text="Price", color="BHK"), use_container_width=True)

    st.markdown('<div class="section-title">Price vs Bathrooms</div>', unsafe_allow_html=True)
    grp2 = filtered.groupby("Bathroom_Count")["Price"].mean().reset_index()
    st.plotly_chart(px.bar(grp2, x="Bathroom_Count", y="Price", text="Price", color="Bathroom_Count"), use_container_width=True)

    st.markdown('<div class="section-title">Price vs Area</div>', unsafe_allow_html=True)
    fig3 = px.scatter(filtered, x="Area_sqft", y="Price", color="Property_Type", size="BHK")
    st.plotly_chart(fig3, use_container_width=True)

    # --------------------------------------------------------
    # ML MODEL
    # --------------------------------------------------------
    st.markdown('<div class="section-title">ML — Price Prediction</div>', unsafe_allow_html=True)

    X = filtered[["City","Property_Type","BHK","Bathroom_Count","Area_sqft"]]
    y = np.log1p(filtered["Price"])

    transformer = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["City","Property_Type"]),
        ("num", "passthrough", ["BHK","Bathroom_Count","Area_sqft"])
    ])

    X_t = transformer.fit_transform(X)
    X_train,X_test,y_train,y_test = train_test_split(X_t,y,test_size=0.2,random_state=42)

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train,y_train)

    y_pred = np.expm1(model.predict(X_t))

    results = X.copy()
    results["Actual_Price"] = filtered["Price"].values
    results["Predicted_Price"] = y_pred

    st.dataframe(results.head(), use_container_width=True)

    buf = BytesIO()
    results.to_csv(buf,index=False)
    st.download_button("Download ML Predictions CSV", buf.getvalue(), "ml_predictions.csv", "text/csv")

    # --------------------------------------------------------
    # Automated Insights
    # --------------------------------------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)

    insights = filtered.groupby(["City","Property_Type"]).agg({"Price":["mean","max","min"]}).reset_index()
    insights.columns = ["City","Property_Type","Avg_Price","Max_Price","Min_Price"]
    st.dataframe(insights, use_container_width=True)

    buf2 = BytesIO()
    insights.to_csv(buf2,index=False)
    st.download_button("Download Insights CSV", buf2.getvalue(), "automated_insights.csv","text/csv")
