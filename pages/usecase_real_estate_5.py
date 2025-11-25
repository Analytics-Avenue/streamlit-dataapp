# marketing_lab_real_estate.py
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
import math
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Real Estate Investment Opportunity Analyzer", layout="wide")
st.markdown("""<style>[data-testid="stSidebarNav"]{display:none;}</style>""", unsafe_allow_html=True)

# -------------------------
# Header & Branding
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display: flex; align-items: center; margin-bottom:12px;">
    <img src="{logo_url}" width="60" style="margin-right:12px;">
    <div style="line-height:1;">
        <div style="font-size:36px; font-weight:700; color:#000;">Analytics Avenue &</div>
        <div style="font-size:36px; font-weight:700; color:#000;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)



# -------------------------
# CSS (Marketing Lab rules: Inter font, pure-black text, blue KPI & variable boxes, fade-in)
# -------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap" rel="stylesheet">
<style>
* { font-family: 'Inter', sans-serif; }

/* GLOBAL TEXT - PURE BLACK */
body, [class*="css"] { color:#000 !important; font-size:16.5px; }

/* MAIN HEADER (keeps spacing consistent) */
.big-header { font-size: 36px !important; font-weight:700 !important; color:#000 !important; margin-bottom:8px; }

/* SECTION TITLE */
.section-title {
    font-size: 22px !important;
    font-weight: 600 !important;
    margin-top:18px;
    margin-bottom:10px;
    color:#000 !important;
    position:relative;
}
.section-title:after {
    content:"";
    position:absolute;
    bottom:-6px;
    left:0;
    height:2px;
    width:0%;
    background:#064b86;
    transition: width 0.35s ease;
}
.section-title:hover:after { width:36%; }

/* CARD LAYOUT */
.card {
    background:#ffffff;
    padding:20px;
    border-radius:12px;
    border:1px solid #e8e8e8;
    font-size:15.8px;
    color:#000 !important;
    font-weight:500;
    box-shadow:0 3px 16px rgba(0,0,0,0.06);
    transition: all 0.20s ease;
}
.card:hover { transform: translateY(-4px); box-shadow:0 10px 28px rgba(6,75,134,0.10); border-color:#064b86; }

/* KPI CARDS - BLUE TEXT */
.kpi {
    background:#ffffff;
    padding:18px;
    border-radius:12px;
    border:1px solid #e6eef6;
    font-size:18px !important;
    font-weight:700 !important;
    text-align:center;
    color:#064b86 !important;
    box-shadow:0 3px 12px rgba(0,0,0,0.05);
}
.kpi:hover { transform: translateY(-4px); box-shadow:0 14px 28px rgba(6,75,134,0.12); }

/* VARIABLE BOXES - BLUE TEXT */
.variable-box {
    padding:14px;
    border-radius:12px;
    background:#ffffff;
    border:1px solid #e6eef6;
    box-shadow:0 2px 10px rgba(0,0,0,0.05);
    text-align:center;
    font-size:16px !important;
    font-weight:600 !important;
    color:#064b86 !important;
    margin-bottom:10px;
}

/* TABLE (index-safe) */
.stDataFrame>div>div>div>table { border-collapse:collapse; }
.dataframe th {
    background:#ffffff !important;
    color:#000 !important;
    padding:10px !important;
    font-size:15px !important;
}
.dataframe td {
    font-size:14.8px !important;
    color:#000 !important;
    padding:8px !important;
    border-bottom:1px solid #efefef !important;
}

/* MAIN HEADER */
.big-header {
    font-size:36px !important;
    font-weight:700 !important;
    color:#000 !important;
    margin-bottom:12px;
}

/* Buttons */
.stButton>button, .stDownloadButton>button {
    background:#064b86 !important;
    color:white !important;
    border:none;
    padding:8px 18px;
    border-radius:8px !important;
    font-size:15px !important;
    font-weight:600 !important;
}
.stButton>button:hover, .stDownloadButton>button:hover { background:#0a6eb3 !important; transform:translateY(-2px); }

/* Fade-in */
.block-container { animation: fadeIn 0.45s ease; }
@keyframes fadeIn { from {opacity:0; transform:translateY(8px);} to {opacity:1; transform:translateY(0);} }

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-header'>Marketing Intelligence & Forecasting Lab</div>", unsafe_allow_html=True)

# -------------------------
# Required columns + autos
# -------------------------
REQUIRED_COLS = [
    "City", "Property_Type", "Price", "Area_sqft", "Agent_Name",
    "Conversion_Probability", "Latitude", "Longitude"
]

AUTO_MAPS = {
    "City": ["city"],
    "Property_Type": ["property_type","type","property type"],
    "Price": ["price","amount","listing_price"],
    "Area_sqft": ["area","area_sqft","sqft","area_sqft"],
    "Agent_Name": ["agent","agent_name","broker"],
    "Conversion_Probability": ["conversion","conversion_probability","probability"],
    "Latitude": ["lat","latitude"],
    "Longitude": ["lon","lng","long","longitude"]
}

def auto_map_columns(df):
    rename = {}
    cols = [c.strip() for c in df.columns]
    for req, candidates in AUTO_MAPS.items():
        for c in cols:
            low = c.lower().strip()
            for cand in candidates:
                cand_low = cand.lower().strip()
                if cand_low == low or cand_low in low or low in cand_low:
                    rename[c] = req
                    break
            if c in rename:
                break
    if rename:
        df = df.rename(columns=rename)
    return df

def ensure_datetime_safe(df, col):
    try:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    except:
        pass
    return df

def download_df(df, filename):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button("Download CSV", b, file_name=filename, mime="text/csv")

# -------------------------
# Tabs (exact 3)
# -------------------------
tab_overview, tab_attributes, tab_application = st.tabs(["Overview", "Important Attributes", "Application"])

# -------------------------
# Overview Tab (detailed)
# -------------------------
with tab_overview:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    The Real Estate Investment Opportunity Analyzer combines property listing attributes, agent performance and lead conversion signals
    to produce a single, comparable Investment Score for each property. The Investment Score synthesizes rental yield, conversion likelihood
    and location signals so investors can prioritize deals with high expected risk-adjusted return.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Purpose</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    • Standardize evaluation across cities and property types.<br>
    • Surface high-ROI listings by combining price, conversion probability and yield proxies.<br>
    • Provide interactive maps and charts for quick decision-making.<br>
    • Supply downloadable datasets and ML-driven predictions to operational teams.
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Technical Capabilities</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Investment Score = Rental Yield × Conversion Probability (configurable).<br>
        • Map visualizations with normalized conversion heat and score mapping.<br>
        • ML model to predict Expected ROI for new listings.<br>
        • Cluster & segment analysis for city-level patterns.
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Prioritise high-performing properties to increase portfolio returns.<br>
        • Reduce time-to-decision with standardized ranking and exportable playbooks.<br>
        • Improve agent incentives by identifying high-conversion performers.<br>
        • Make acquisition & marketing spend decisions backed by measurable ROI signals.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Key KPIs</div>', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Top Rental Yield Property</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Highest Yield City</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Median Rental Yield</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Top Property Type</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Ideal Users</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    Real estate investors, portfolio managers, property analysts, developers and agency sales teams who need a repeatable, transparent
    method to evaluate and compare investment opportunities across regions and asset types.
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# Important Attributes Tab
# -------------------------
with tab_attributes:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    dict_rows = [
        ("City", "City where the property is listed — used for geo-aggregation, city-level KPIs and mapping."),
        ("Property_Type", "Category of property (Apartment, Villa, Studio, Commercial etc.) used for type-level benchmarking."),
        ("Price", "Listing or ask price for the property (INR). Used as the primary denominator for yield & ROI calculations."),
        ("Area_sqft", "Built area or carpet area in sqft. Used to calculate price per sqft and derived yield metrics."),
        ("Agent_Name", "Name of the listing agent or broker. Used to measure agent-level conversion performance."),
        ("Conversion_Probability", "Estimated probability that the lead will convert for this listing (0-1). Used to weight expected ROI."),
        ("Latitude", "Latitude coordinate for map visualizations and geospatial clustering."),
        ("Longitude", "Longitude coordinate for map visualizations and geospatial clustering.")
    ]
    req_df = pd.DataFrame([{"Attribute":r[0],"Description":r[1]} for r in dict_rows])
    st.dataframe(req_df.style.set_table_attributes('class="dataframe"'), use_container_width=True)

    st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
    # Left column: independent vars (blue boxes)
    indep = ["City","Property_Type","Area_sqft","Agent_Name","Latitude","Longitude"]
    c1,c2 = st.columns(2)
    with c1:
        for v in indep[:3]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)
    with c2:
        for v in indep[3:]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
    dep = ["Price","Conversion_Probability","Expected_ROI","Rental_Yield"]
    d1,d2 = st.columns(2)
    with d1:
        st.markdown(f"<div class='variable-box'>{dep[0]}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='variable-box'>{dep[1]}</div>", unsafe_allow_html=True)
    with d2:
        st.markdown(f"<div class='variable-box'>{dep[2]}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='variable-box'>{dep[3]}</div>", unsafe_allow_html=True)

# -------------------------
# Application Tab
# -------------------------
with tab_application:
    st.markdown('<div class="section-title">Application</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">Load data, map columns if needed, filter, view KPIs, charts, ML predictions and download insights.</div>', unsafe_allow_html=True)

    # -------------------------
    # Data load modes
    # -------------------------
    st.markdown('<div class="section-title">Step 1 — Load dataset</div>', unsafe_allow_html=True)
    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)
    df = None
    raw = None

    if mode == "Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("Default dataset loaded")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    elif mode == "Upload CSV":
        st.markdown("#### Download Sample CSV for Reference")
        SAMPLE_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        try:
            sample_df = pd.read_csv(SAMPLE_URL).head(5)
            st.download_button("Download Sample CSV", sample_df.to_csv(index=False), "sample_dataset.csv", "text/csv")
        except Exception:
            st.info("Sample CSV unavailable")
        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("Uploaded dataset loaded (auto-mapping attempted).")

    else:
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head(), use_container_width=True)
            st.markdown("Map your columns to required fields.")
            mapping = {}
            for req in REQUIRED_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + list(raw.columns))
            if st.button("Apply mapping"):
                missing = [k for k,v in mapping.items() if v=="-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v:k for k,v in mapping.items()})
                    st.success("Mapping applied.")
                    st.dataframe(df.head(), use_container_width=True)

    if df is None:
        st.stop()

    # -------------------------
    # Basic cleaning & conversions
    # -------------------------
    # Keep only required columns present
    df = df[[c for c in REQUIRED_COLS if c in df.columns]].copy()
    # numeric conversions
    for col in ["Price","Area_sqft","Conversion_Probability","Latitude","Longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Price","Area_sqft"])  # ensure fundamentals exist

    # derive basic metrics
    # rental yield proxy (assume monthly rent = 0.5% of price by default, configurable later)
    st.markdown('<div class="section-title">Step 2 — Filters & Preview</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns([2,2,1])
    cities = sorted(df["City"].dropna().unique()) if "City" in df.columns else []
    ptypes = sorted(df["Property_Type"].dropna().unique()) if "Property_Type" in df.columns else []
    agents = sorted(df["Agent_Name"].dropna().unique()) if "Agent_Name" in df.columns else []

    with c1:
        sel_cities = st.multiselect("City", options=cities, default=cities[:6])
    with c2:
        sel_ptypes = st.multiselect("Property Type", options=ptypes, default=ptypes[:4])
    with c3:
        sel_agents = st.multiselect("Agent", options=agents, default=agents[:4])

    filt = df.copy()
    if sel_cities: filt = filt[filt["City"].isin(sel_cities)]
    if sel_ptypes: filt = filt[filt["Property_Type"].isin(sel_ptypes)]
    if sel_agents: filt = filt[filt["Agent_Name"].isin(sel_agents)]

    st.markdown("#### Data Preview (first 10 rows)")
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(500), "filtered_preview.csv")

    # -------------------------
    # Key Metrics (blue KPI cards)
    # -------------------------
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)

    # compute rental yield proxy and expected ROI
    rent_pct = st.number_input("Assumed monthly rent % of price (0.1% - 2.0%)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    filt["Est_Rent"] = filt["Price"] * (rent_pct/100)
    filt["Rental_Yield"] = np.where(filt["Price"]>0, (filt["Est_Rent"]*12) / filt["Price"], 0)
    filt["Expected_ROI"] = filt["Price"] * np.where(filt.get("Conversion_Probability", pd.Series(0)) > 0,
                                                   filt["Conversion_Probability"], 0)

    total_listings = len(filt)
    top_rental = filt.sort_values("Rental_Yield", ascending=False).iloc[0] if total_listings>0 else None
    highest_yield_city = filt.groupby("City")["Rental_Yield"].mean().sort_values(ascending=False).head(1).index[0] if total_listings>0 else "N/A"

    k1.markdown(f"<div class='kpi'>Top Rental Yield Property<br><b>{top_rental['Property_Type'] if top_rental is not None else 'N/A'}</b></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'>Highest Yield City<br><b>{highest_yield_city}</b></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'>Median Rental Yield<br><b>{filt['Rental_Yield'].median():.2f}</b></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'>Top Property Type<br><b>{filt.groupby('Property_Type')['Expected_ROI'].mean().sort_values(ascending=False).head(1).index[0] if total_listings>0 else 'N/A'}</b></div>", unsafe_allow_html=True)

    # -------------------------
    # Charts
    # -------------------------
    st.markdown('<div class="section-title">Charts & Visuals</div>', unsafe_allow_html=True)

    # 1) Rental Yield by Property Type
    st.markdown("##### Rental Yield by Property Type")
    if not filt.empty:
        ptype_yield = filt.groupby("Property_Type")["Rental_Yield"].mean().reset_index().sort_values("Rental_Yield", ascending=False)
        fig1 = px.bar(ptype_yield, x="Property_Type", y="Rental_Yield", text="Rental_Yield", template="plotly_white")
        fig1.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig1.update_layout(xaxis_title="Property Type", yaxis_title="Rental Yield")
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("No data to plot Rental Yield by Property Type.")

    # 2) Expected ROI by City
    st.markdown("##### Expected ROI by City")
    if not filt.empty:
        city_roi = filt.groupby("City")["Expected_ROI"].mean().reset_index().sort_values("Expected_ROI", ascending=False)
        fig2 = px.bar(city_roi, x="City", y="Expected_ROI", text="Expected_ROI", template="plotly_white")
        fig2.update_traces(texttemplate="₹ %{text:,.0f}", textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No data to plot Expected ROI by City.")

    # 3) Map: Investment hotspots
    st.markdown("##### Investment Hotspot Map")
    map_df = filt.dropna(subset=["Latitude","Longitude","Expected_ROI"])
    if map_df.empty:
        st.info("No geolocation data available for map.")
    else:
        # small jitter
        map_df = map_df.copy()
        map_df["Latitude"] += np.random.uniform(-0.0003,0.0003,size=len(map_df))
        map_df["Longitude"] += np.random.uniform(-0.0003,0.0003,size=len(map_df))
        center_lat = map_df["Latitude"].mean()
        center_lon = map_df["Longitude"].mean()
        map_fig = px.scatter_mapbox(map_df, lat="Latitude", lon="Longitude",
                                    size="Expected_ROI", color="Rental_Yield",
                                    hover_name="Property_Type",
                                    hover_data=["City","Price","Rental_Yield","Conversion_Probability","Expected_ROI"],
                                    color_continuous_scale=px.colors.sequential.Viridis,
                                    size_max=18, zoom=11, center={"lat":center_lat,"lon":center_lon})
        map_fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(map_fig, use_container_width=True)

    # -------------------------
    # ML: Predict Expected_ROI (RandomForest)
    # -------------------------
    st.markdown('<div class="section-title">ML: Predict Expected ROI</div>', unsafe_allow_html=True)
    st.markdown("<div class='card'>Trains a RandomForest to predict Expected_ROI using selected features (requires >= 40 rows).</div>", unsafe_allow_html=True)

    ml_df = filt.copy().dropna(subset=["Expected_ROI"])
    feat_cols = ["City","Property_Type","Area_sqft","Agent_Name"]
    feat_cols = [c for c in feat_cols if c in ml_df.columns]

    if len(ml_df) < 40 or len(feat_cols) < 2:
        st.info("Not enough data to train ML model (>=40 rows and >=2 features needed).")
    else:
        X = ml_df[feat_cols].copy()
        y = ml_df["Expected_ROI"].astype(float)
        cat_cols = [c for c in X.columns if X[c].dtype == "object"]
        num_cols = [c for c in X.columns if c not in cat_cols]

        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols) if cat_cols else ("noop","passthrough",[]),
            ("num", StandardScaler(), num_cols) if num_cols else ("noop2","passthrough",[])
        ], remainder="drop")
        try:
            X_t = preprocessor.fit_transform(X)
            X_train,X_test,y_train,y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)
            rf = RandomForestRegressor(n_estimators=200, random_state=42)
            with st.spinner("Training RandomForest..."):
                rf.fit(X_train, y_train)
            preds_test = rf.predict(X_test)
            rmse = math.sqrt(((y_test - preds_test)**2).mean())
            r2 = 1 - (((y_test - preds_test)**2).sum() / ((y_test - y_test.mean())**2).sum()) if len(y_test)>1 else 0.0
            st.write(f"ML results — RMSE: {rmse:.2f}, R² (approx): {r2:.3f}")
            # combine predictions with sample features for download
            X_test_df = pd.DataFrame(X_test, columns=[f"F_{i}" for i in range(X_test.shape[1])])
            X_test_df["Actual_Expected_ROI"] = y_test.reset_index(drop=True)
            X_test_df["Predicted_Expected_ROI"] = preds_test
            st.dataframe(X_test_df.head(), use_container_width=True)
            download_df(X_test_df, "ml_expected_roi_predictions.csv")
        except Exception as e:
            st.error("ML training failed: " + str(e))

    # -------------------------
    # Automated Insights
    # -------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)
    insights = []
    if not filt.empty:
        # top city
        city_top = filt.groupby("City")["Expected_ROI"].mean().reset_index().sort_values("Expected_ROI", ascending=False)
        if not city_top.empty:
            insights.append({"Insight":"Top City by Expected ROI","Value":city_top.iloc[0]["City"], "Metric": city_top.iloc[0]["Expected_ROI"]})
        # top property type
        ptype_top = filt.groupby("Property_Type")["Expected_ROI"].mean().reset_index().sort_values("Expected_ROI", ascending=False)
        if not ptype_top.empty:
            insights.append({"Insight":"Top Property Type by ROI","Value":ptype_top.iloc[0]["Property_Type"], "Metric": ptype_top.iloc[0]["Expected_ROI"]})
        # agent top
        agent_top = filt.groupby("Agent_Name")["Conversion_Probability"].mean().reset_index().sort_values("Conversion_Probability", ascending=False)
        if not agent_top.empty:
            insights.append({"Insight":"Top Agent by Conversion","Value":agent_top.iloc[0]["Agent_Name"], "Metric": agent_top.iloc[0]["Conversion_Probability"]})
        # high ROI listing
        high_prop = filt.sort_values("Expected_ROI", ascending=False).head(1)
        if not high_prop.empty:
            insights.append({"Insight":"Highest Expected ROI Listing","Value":high_prop.iloc[0]["Property_Type"], "Metric": high_prop.iloc[0]["Expected_ROI"]})

    insights_df = pd.DataFrame(insights)
    if insights_df.empty:
        st.info("No insights generated for the selected filters.")
    else:
        st.dataframe(insights_df, use_container_width=True)
        download_df(insights_df, "automated_insights.csv")

    st.markdown("### Done — download filtered dataset or insights as needed", unsafe_allow_html=True)
    st.download_button("Download filtered dataset (full)", filt.to_csv(index=False), "filtered_dataset_full.csv", "text/csv")

# End of file
