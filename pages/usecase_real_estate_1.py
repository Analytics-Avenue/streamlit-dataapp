import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Real Estate Intelligence Suite", layout="wide")
LOGO = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

# -------------------------
# Marketing Lab CSS
# -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
body, [class*="css"] { color:#000 !important; font-size:17px; }

.big-header {
    font-size:36px !important;
    font-weight:700 !important;
    margin-bottom:12px;
}

/* Section Title */
.section-title {
    font-size:24px !important;
    font-weight:600 !important;
    margin-top:30px;
    margin-bottom:12px;
    position:relative;
}
.section-title:after {
    content:"";
    position:absolute;
    bottom:-5px; left:0;
    height:2px; width:0%;
    background:#064b86;
    transition: width 0.35s ease;
}
.section-title:hover:after { width:40%; }

/* Card */
.card {
    background:#fff; padding:22px; border-radius:14px;
    border:1px solid #e6e6e6; font-weight:500;
    box-shadow:0 3px 14px rgba(0,0,0,0.08);
    transition:0.25s ease;
}
.card:hover {
    transform:translateY(-4px);
    box-shadow:0 12px 24px rgba(6,75,134,0.18);
    border-color:#064b86;
}

/* KPI Boxes */
.kpi {
    background:#fff; padding:20px; border-radius:14px;
    border:1px solid #e2e2e2;
    color:#064b86 !important; font-weight:600;
    font-size:20px !important; text-align:center;
    box-shadow:0 3px 14px rgba(0,0,0,0.07);
}
.kpi:hover { transform:translateY(-3px); }

/* Variable Box */
.variable-box {
    padding:16px; border-radius:14px;
    border:1px solid #e5e5e5; background:white;
    color:#064b86; font-weight:500;
    box-shadow:0 2px 10px rgba(0,0,0,0.10);
    transition:0.25s ease; margin-bottom:10px;
}
.variable-box:hover { transform:translateY(-5px); }

/* Table */
.required-table th {
    background:#fff !important; color:#000 !important;
    border-bottom:2px solid #000 !important;
    padding:10px !important;
}
.required-table td {
    border-bottom:1px solid #e6e6e6 !important;
    padding:10px !important;
}

/* Buttons */
.stButton>button, .stDownloadButton>button {
    background:#064b86 !important; color:white !important;
    padding:9px 20px; border-radius:8px;
    font-weight:600; border:none;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background:#0a6eb3 !important;
}

/* Fade-in */
.block-container { animation: fadeIn 0.5s ease; }
@keyframes fadeIn { from {opacity:0;} to {opacity:1;} }
</style>
""", unsafe_allow_html=True)




# Header & Logo
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display: flex; align-items: center; margin-bottom:16px;">
    <img src="{logo_url}" width="60" style="margin-right:12px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:700;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:700;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='big-header'>Real Estate Intelligence Suite</div>", unsafe_allow_html=True)






# -------------------------
# Duplicate column fix
# -------------------------
def dedupe_columns(cols):
    seen = {}
    new_cols = []
    for col in cols:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
    return new_cols




# -------------------------
# Required Columns
# -------------------------
REQUIRED_COLS = ["City", "Property_Type", "Area_sqft", "Price"]

AUTO_MAP = {
    "City": ["city","location","town"],
    "Property_Type": ["property_type","ptype","type"],
    "Area_sqft": ["area","sqft","area_sqft","size"],
    "Price": ["price","amount","listing_price"]
}




# -------------------------
# Column Auto-map
# -------------------------
def auto_map_columns(df):
    rename = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for req, alts in AUTO_MAP.items():
        for alt in alts:
            if alt.lower() in lower_cols:
                rename[lower_cols[alt.lower()]] = req
    df = df.rename(columns=rename)
    return df




# -------------------------
# Utility
# -------------------------
def ensure_numeric(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(np.nan)
    return df

def download_df(df, filename):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button("Download CSV", b, file_name=filename, mime="text/csv")

def to_currency(x):
    return f"₹ {x:,.0f}"




# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])





# -------------------------
# TAB 1 — OVERVIEW
# -------------------------
with tab1:

    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    A complete real estate valuation, ML prediction and market insights app built in the Marketing Lab UI standard.
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)
    left.markdown("""
    <div class="section-title">Capabilities</div>
    <div class="card">
    • ML-based price prediction<br>
    • City-level dashboards<br>
    • Automated insights<br>
    • Full filtering system<br>
    • Data export utilities
    </div>
    """, unsafe_allow_html=True)

    right.markdown("""
    <div class="section-title">Business Impact</div>
    <div class="card">
    • Faster decision-making<br>
    • Accurate valuations<br>
    • Support for investors & developers<br>
    • Market benchmarking
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown("<div class='kpi'>Model RMSE</div>", unsafe_allow_html=True)
    c2.markdown("<div class='kpi'>Avg Price Deviation%</div>", unsafe_allow_html=True)
    c3.markdown("<div class='kpi'>Median Price</div>", unsafe_allow_html=True)
    c4.markdown("<div class='kpi'>Total Properties</div>", unsafe_allow_html=True)




# -------------------------
# TAB 2 — IMPORTANT ATTRIBUTES
# -------------------------
with tab2:

    st.markdown('<div class="section-title">Required Column Dictionary</div>', unsafe_allow_html=True)

    dict_df = pd.DataFrame([
        ["City", "City where the property is located"],
        ["Property_Type", "Type of property (Apartment, Villa, Plot, etc.)"],
        ["Area_sqft", "Area in square feet"],
        ["Price", "Listing or sale price (INR)"],
    ], columns=["Attribute","Description"])

    st.markdown(dict_df.to_html(index=False, classes="required-table"), unsafe_allow_html=True)

    l, r = st.columns(2)

    with l:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        for v in ["City","Property_Type","Area_sqft"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with r:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        for v in ["Price","Price_per_sqft"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)




# -------------------------
# TAB 3 — APPLICATION
# -------------------------
with tab3:

    st.markdown('<div class="section-title">Step 1 — Load Dataset</div>', unsafe_allow_html=True)
    mode = st.radio("Dataset option:", ["Default dataset","Upload CSV","Upload CSV + Column mapping"], horizontal=True)

    df = None
    raw = None
    DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"

    # DEFAULT
    if mode == "Default dataset":
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = dedupe_columns(df.columns)
            df = auto_map_columns(df)
            st.success("Dataset loaded")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Could not load default dataset: {e}")
            st.stop()

    # Upload
    elif mode == "Upload CSV":
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up:
            df = pd.read_csv(up)
            df.columns = dedupe_columns(df.columns)
            df = auto_map_columns(df)
            st.success("File loaded")
            st.dataframe(df.head(), use_container_width=True)

    # Mapping
    else:
        raw_up = st.file_uploader("Upload CSV for mapping", type=["csv"])
        if raw_up:
            raw = pd.read_csv(raw_up)
            raw.columns = dedupe_columns(raw.columns)
            st.dataframe(raw.head(), use_container_width=True)

            mapping = {}
            for req in REQUIRED_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", ["-- Select --"] + list(raw.columns))

            if st.button("Apply mapping"):
                missing = [k for k, v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Map all required fields")
                else:
                    df = raw.rename(columns={v:k for k,v in mapping.items()})
                    st.success("Mapping applied")
                    st.dataframe(df.head(), use_container_width=True)

    if df is None:
        st.stop()




    # -------------------------
    # Clean core fields
    # -------------------------
    df = df[[c for c in REQUIRED_COLS if c in df.columns]].copy()
    df["City"] = df["City"].astype(str)
    df["Property_Type"] = df["Property_Type"].astype(str)

    df = ensure_numeric(df, "Area_sqft")
    df = ensure_numeric(df, "Price")

    df = df.dropna(subset=["Area_sqft","Price"])
    df["Price_per_sqft"] = df["Price"] / df["Area_sqft"]




    # -------------------------
    # FILTERS
    # -------------------------
    st.markdown('<div class="section-title">Step 2 — Filters</div>', unsafe_allow_html=True)

    f1,f2,f3 = st.columns(3)

    cities = sorted(df["City"].unique())
    ptypes = sorted(df["Property_Type"].unique())

    with f1:
        sel_city = st.multiselect("City", cities, default=cities[:5])

    with f2:
        sel_ptype = st.multiselect("Property Type", ptypes, default=ptypes[:5])

    with f3:
        min_area, max_area = float(df["Area_sqft"].min()), float(df["Area_sqft"].max())
        area_range = st.slider("Area range (sqft)", min_area, max_area, (min_area, max_area))

    filt = df.copy()

    if sel_city:
        filt = filt[filt["City"].isin(sel_city)]

    if sel_ptype:
        filt = filt[filt["Property_Type"].isin(sel_ptype)]

    filt = filt[(filt["Area_sqft"]>=area_range[0]) & (filt["Area_sqft"]<=area_range[1])]




    # -------------------------
    # Preview
    # -------------------------
    st.markdown('<div class="section-title">Filtered Preview</div>', unsafe_allow_html=True)
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt, "filtered_real_estate.csv")




    # -------------------------
    # KPIs
    # -------------------------
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)

    total = len(filt)
    median_price = filt["Price"].median() if total>0 else 0
    avg_psf = filt["Price_per_sqft"].mean() if total>0 else 0
    avg_area = filt["Area_sqft"].mean() if total>0 else 0

    c1.metric("Total Properties", total)
    c2.metric("Median Price", to_currency(median_price))
    c3.metric("Avg Price / sqft", f"₹ {avg_psf:,.0f}")
    c4.metric("Avg Area (sqft)", f"{avg_area:,.0f}")




    # -------------------------
    # Charts
    # -------------------------
    st.markdown('<div class="section-title">Charts</div>', unsafe_allow_html=True)

    if not filt.empty:

        st.markdown("#### Price Distribution")
        hist_fig = px.histogram(filt, x="Price", nbins=30)
        st.plotly_chart(hist_fig, use_container_width=True)

        st.markdown("#### City-wise Avg Price")
        city_avg = filt.groupby("City")["Price"].mean().reset_index()
        fig2 = px.bar(city_avg, x="City", y="Price", text="Price")
        fig2.update_traces(texttemplate="₹ %{text:,.0f}")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Price vs Area")
        fig3 = px.scatter(filt, x="Area_sqft", y="Price", color="Property_Type")
        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.info("No data for charts.")




    # -------------------------
    # ML — Price Prediction
    # -------------------------
    st.markdown('<div class="section-title">ML — Price Prediction</div>', unsafe_allow_html=True)

    if len(df) >= 30:

        X = df[["City","Property_Type","Area_sqft"]]
        y = df["Price"]

        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["City","Property_Type"]),
            ("num", StandardScaler(), ["Area_sqft"])
        ])

        X_t = pre.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        rmse = mean_squared_error(y_test, preds)**0.5
        r2 = r2_score(y_test, preds)

        st.write(f"RMSE: {rmse:.2f} | R²: {r2:.3f}")

        feat_names = pre.get_feature_names_out()
        ml_df = pd.DataFrame(X_test, columns=feat_names)
        ml_df["Actual_Price"] = y_test.values
        ml_df["Predicted_Price"] = preds

        st.dataframe(ml_df.head(), use_container_width=True)
        download_df(ml_df, "realestate_ml_predictions.csv")
    else:
        st.info("Need at least 30 rows for ML model.")




    # -------------------------
    # Automated Insights
    # -------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)

    if not filt.empty:
        ins = filt.groupby(["City","Property_Type"]).agg(
            Avg_Price=("Price","mean"),
            Max_Price=("Price","max"),
            Min_Price=("Price","min"),
            Count=("Price","count")
        ).reset_index()

        ins["Avg_Price"] = ins["Avg_Price"].round(0)
        st.dataframe(ins, use_container_width=True)
        download_df(ins, "realestate_insights.csv")
    else:
        st.info("No insights available.")


