# real_estate_marketing_lab.py
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
st.set_page_config(page_title="Real Estate Intelligence Suite — Marketing Lab", layout="wide")
LOGO = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

# -------------------------
# Marketing Lab CSS (Inter font, pure black text, blue KPIs etc)
# -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
body, [class*="css"] { color:#000 !important; font-size:17px; }

/* MAIN HEADER */
.big-header { font-size:36px !important; font-weight:700 !important; color:#000 !important; margin-bottom:12px; }

/* SECTION TITLE */
.section-title {
    font-size:24px !important;
    font-weight:600 !important;
    margin-top:30px;
    margin-bottom:12px;
    color:#000 !important;
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
    transition: width 0.35s ease;
}
.section-title:hover:after { width:40%; }

/* CARD */
.card {
    background:#ffffff;
    padding:22px;
    border-radius:14px;
    border:1px solid #e6e6e6;
    font-size:16.5px;
    color:#000 !important;
    font-weight:500;
    box-shadow:0 3px 14px rgba(0,0,0,0.08);
    transition: all 0.25s ease;
}
.card:hover { transform:translateY(-4px); box-shadow:0 12px 25px rgba(6,75,134,0.18); border-color:#064b86; }

/* KPI CARDS - blue text */
.kpi {
    background:#ffffff;
    padding:22px;
    border-radius:14px;
    border:1px solid #e2e2e2;
    font-size:20px !important;
    font-weight:600 !important;
    text-align:center;
    color:#064b86 !important;
    box-shadow:0 3px 14px rgba(0,0,0,0.07);
    transition:0.25s ease;
}
.kpi:hover { transform:translateY(-4px); box-shadow:0 13px 26px rgba(6,75,134,0.20); border-color:#064b86; }

/* VARIABLE BOXES - blue text */
.variable-box {
    padding:18px;
    border-radius:14px;
    background:white;
    border:1px solid #e5e5e5;
    box-shadow:0 2px 10px rgba(0,0,0,0.10);
    transition:0.25s ease;
    text-align:center;
    font-size:17.5px !important;
    font-weight:500 !important;
    color:#064b86 !important;
    margin-bottom:12px;
}
.variable-box:hover { transform:translateY(-5px); box-shadow:0 12px 22px rgba(6,75,134,0.18); border-color:#064b86; }

/* index-safe HTML table */
.required-table thead th { background:#ffffff !important; color:#000 !important; font-size:18px !important; border-bottom:2px solid #000 !important; padding:10px !important; text-align:left; }
.required-table tbody td { color:#000 !important; font-size:15.5px !important; padding:10px !important; border-bottom:1px solid #efefef !important; }
.required-table tbody tr:hover td { background:#f8f8f8 !important; }

/* Buttons */
.stButton>button, .stDownloadButton>button {
    background:#064b86 !important;
    color:white !important;
    border:none;
    padding:10px 22px;
    border-radius:8px !important;
    font-size:15.5px !important;
    font-weight:600 !important;
    transition:0.25s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover { transform:translateY(-3px); background:#0a6eb3 !important; }

/* Page fade */
.block-container { animation: fadeIn 0.5s ease; }
@keyframes fadeIn { from {opacity:0; transform:translateY(10px);} to {opacity:1; transform:translateY(0);} }

.small-muted { color:#666; font-size:13px !important; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header / Branding (Marketing Lab header)
# -------------------------
st.markdown(f"""
<div style="display:flex; align-items:center; margin-bottom:8px;">
    <img src="{LOGO}" width="56" style="border-radius:6px; margin-right:12px;">
    <div>
        <div style="color:#064b86; font-size:36px; font-weight:700;">Analytics Avenue & Advanced Analytics</div>
        <div style="color:#666; font-size:13px; margin-top:2px;">Real Estate Intelligence Suite — Marketing Lab UI</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='big-header'>Real Estate Demand Forecasting Lab</div>", unsafe_allow_html=True)

# -------------------------
# Required columns & automap helpers
# -------------------------
REQUIRED_COLS = ["City","Property_Type","Area_sqft","Price"]

AUTO_MAP = {
    "City": ["city","town","location","area"],
    "Property_Type": ["property_type","type","propertytype","property"],
    "Area_sqft": ["area_sqft","area","sqft","size"],
    "Price": ["price","amount","listing_price","value"]
}

def auto_map_columns(df):
    rename = {}
    cols = [c.strip() for c in df.columns]
    for req, candidates in AUTO_MAP.items():
        for c in cols:
            low = c.lower().strip()
            for cand in candidates:
                if cand == low or cand in low or low in cand:
                    rename[c] = req
                    break
            if c in rename:
                break
    if rename:
        df = df.rename(columns=rename)
    return df

def ensure_numeric(df, col, fill=0):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill)
    return df

def to_currency(x):
    try:
        return "₹ " + f"{float(x):,.0f}"
    except:
        return str(x)

def render_index_safe_table(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("No data to display.")
        return
    html = df.to_html(index=False, classes="required-table")
    st.markdown(html, unsafe_allow_html=True)

def download_df(df, filename, label="Download CSV"):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")

# -------------------------
# Tabs (exact 3-tab structure)
# -------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# -------------------------
# TAB 1 — OVERVIEW
# -------------------------
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <b>Purpose:</b> Provide a centralized real estate pricing, valuation and forecasting lab with ML-backed predictions, market dashboards and exportable insights.
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • ML property valuation and prediction.<br>
        • City-level and property-type dashboards.<br>
        • Exportable automated insights and ML predictions.<br>
        • Filter-driven exploration (city, property type, area).
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Faster, more accurate valuations.<br>
        • Transparent pricing for negotiations.<br>
        • Data-driven city expansion and investment planning.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Model RMSE</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Avg Price Deviation%</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Median Price</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Total Properties</div>", unsafe_allow_html=True)

# -------------------------
# TAB 2 — IMPORTANT ATTRIBUTES
# -------------------------
with tab2:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)
    required_dict = {
        "City": "City or town name where the property is located.",
        "Property_Type": "Type of property (Apartment, Villa, Plot, Office, etc.).",
        "Area_sqft": "Area in square feet.",
        "Price": "Listing or transacted price (INR)."
    }
    dict_df = pd.DataFrame([{"Attribute": k, "Description": v} for k, v in required_dict.items()])
    render_index_safe_table(dict_df)

    st.markdown('<div class="section-title">Attributes Overview</div>', unsafe_allow_html=True)
    lcol, rcol = st.columns(2)
    with lcol:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        indep = ["City","Property_Type","Area_sqft"]
        for v in indep:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)
    with rcol:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        dep = ["Price","Price_per_sqft","Price_Deviation"]
        for v in dep:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# -------------------------
# TAB 3 — APPLICATION
# -------------------------
with tab3:
    st.markdown('<div class="section-title">Step 1 — Load dataset</div>', unsafe_allow_html=True)

    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)
    df = None
    raw = None
    DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"

    if mode == "Default dataset":
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("Default dataset loaded")
            st.dataframe(df.head(5), use_container_width=True)
        except Exception as e:
            st.error("Could not load default dataset: " + str(e))
            st.stop()

    elif mode == "Upload CSV":
        st.markdown("#### Download sample CSV for reference")
        try:
            sample_df = pd.read_csv(DEFAULT_URL).head(5)
            st.download_button("Download Sample CSV", sample_df.to_csv(index=False), "sample_real_estate.csv", "text/csv")
        except Exception:
            pass
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("File uploaded")
            st.dataframe(df.head(5), use_container_width=True)

    else:  # mapping mode
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"], key="map_upload")
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.markdown("<div class='small-muted'>Preview (first 5 rows)</div>", unsafe_allow_html=True)
            st.dataframe(raw.head(5), use_container_width=True)
            st.markdown('<div class="section-title">Map your columns to required fields</div>', unsafe_allow_html=True)
            mapping = {}
            cols_list = list(raw.columns)
            for req in REQUIRED_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + cols_list, key=f"map_{req}")
            if st.button("Apply mapping"):
                missing = [k for k,v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    inv = {v:k for k,v in mapping.items()}
                    df = raw.rename(columns=inv)
                    st.success("Mapping applied")
                    st.dataframe(df.head(5), use_container_width=True)

    if df is None:
        st.stop()

    # Keep only required columns present
    df = df[[c for c in REQUIRED_COLS if c in df.columns]].copy()

    # Coerce types & basic cleaning
    df["City"] = df["City"].astype(str)
    df["Property_Type"] = df["Property_Type"].astype(str)
    df = ensure_numeric(df, "Area_sqft", fill=np.nan)
    df = ensure_numeric(df, "Price", fill=np.nan)

    # Derive price per sqft where possible
    if "Area_sqft" in df.columns and "Price" in df.columns:
        df["Price_per_sqft"] = np.where(df["Area_sqft"]>0, df["Price"]/df["Area_sqft"], np.nan)
    else:
        df["Price_per_sqft"] = np.nan

    # Remove rows missing core numeric fields
    df = df.dropna(subset=["Price","Area_sqft"], how="any").reset_index(drop=True)

    # -------------------------
    # Step 2 — Filters & Preview
    # -------------------------
    st.markdown('<div class="section-title">Step 2 — Filters & preview</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2,2,1])
    cities = sorted(df["City"].unique())
    ptypes = sorted(df["Property_Type"].unique())
    with c1:
        sel_city = st.multiselect("City", options=cities, default=cities[:5] if cities else [])
    with c2:
        sel_ptype = st.multiselect("Property Type", options=ptypes, default=ptypes[:5] if ptypes else [])
    with c3:
        try:
            min_area = float(df["Area_sqft"].min())
            max_area = float(df["Area_sqft"].max())
            area_range = st.slider("Area (sqft)", min_value=min_area, max_value=max_area, value=(min_area, max_area))
        except Exception:
            area_range = None

    filt = df.copy()
    if sel_city:
        filt = filt[filt["City"].isin(sel_city)]
    if sel_ptype:
        filt = filt[filt["Property_Type"].isin(sel_ptype)]
    if area_range:
        filt = filt[(filt["Area_sqft"] >= area_range[0]) & (filt["Area_sqft"] <= area_range[1])]

    st.markdown('<div class="section-title">Filtered preview (first 10 rows)</div>', unsafe_allow_html=True)
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(500), "realestate_filtered_preview.csv", label="Download filtered preview (up to 500 rows)")

    # -------------------------
    # KPIs
    # -------------------------
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    kcol1, kcol2, kcol3, kcol4 = st.columns(4)
    total_props = len(filt)
    median_price = filt["Price"].median() if total_props>0 else 0
    avg_price = filt["Price"].mean() if total_props>0 else 0
    avg_area = filt["Area_sqft"].mean() if total_props>0 else 0

    kcol1.markdown("<div class='kpi'>Total Properties</div>", unsafe_allow_html=True)
    kcol2.markdown("<div class='kpi'>Median Price</div>", unsafe_allow_html=True)
    kcol3.markdown("<div class='kpi'>Avg Price / sqft</div>", unsafe_allow_html=True)
    kcol4.markdown("<div class='kpi'>Avg Area (sqft)</div>", unsafe_allow_html=True)

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Total Properties", f"{total_props:,}")
    n2.metric("Median Price", to_currency(median_price))
    n3.metric("Avg Price / sqft", f"₹ {filt['Price_per_sqft'].mean():.0f}" if "Price_per_sqft" in filt.columns else "N/A")
    n4.metric("Avg Area (sqft)", f"{avg_area:,.0f}")

    # -------------------------
    # Charts
    # -------------------------
    st.markdown('<div class="section-title">Charts</div>', unsafe_allow_html=True)

    # Price distribution histogram
    st.markdown("#### Price distribution")
    if not filt.empty:
        try:
            counts, bins = np.histogram(filt["Price"].dropna(), bins=30)
            bin_centers = 0.5*(bins[:-1] + bins[1:])
            hist_df = pd.DataFrame({"bin_center": bin_centers, "count": counts})
            fig = px.bar(hist_df, x="bin_center", y="count", labels={"bin_center":"Price","count":"Count"})
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Unable to plot price distribution for selected filters.")
    else:
        st.info("No data for charts.")

    # City-level average price
    st.markdown("#### City-wise average price (top 15)")
    city_avg = filt.groupby("City").agg(Avg_Price=("Price","mean"), Count=("Price","count")).reset_index().sort_values("Count", ascending=False).head(15)
    if not city_avg.empty:
        fig2 = px.bar(city_avg, x="City", y="Avg_Price", text="Avg_Price")
        fig2.update_traces(texttemplate="₹ %{text:,.0f}", textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No city-level data available.")

    # Price vs area scatter
    st.markdown("#### Price vs Area (scatter)")
    if not filt.empty:
        fig3 = px.scatter(filt, x="Area_sqft", y="Price", color="Property_Type", hover_data=["City"], trendline="ols")
        st.plotly_chart(fig3, use_container_width=True)

    # -------------------------
    # ML: Price Prediction
    # -------------------------
    st.markdown('<div class="section-title">ML — Price prediction (RandomForest)</div>', unsafe_allow_html=True)
    if len(df) < 30:
        st.info("Not enough rows to build a robust ML model (need 30+).")
    else:
        # Features and target
        features = ["City","Property_Type","Area_sqft"]
        X = df[features].copy()
        y = df["Price"].copy()

        cat_cols = ["City","Property_Type"]
        num_cols = ["Area_sqft"]

        transformer = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols)
        ])
        X_t = transformer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        with st.spinner("Training RandomForest regressor..."):
            model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = (mean_squared_error(y_test, preds)) ** 0.5
        r2 = r2_score(y_test, preds)
        st.write(f"Model performance — RMSE: {rmse:.2f} | R²: {r2:.3f}")

        # Attach feature names if available
        try:
            feature_names = transformer.get_feature_names_out()
        except Exception:
            feature_names = [f"f_{i}" for i in range(X_test.shape[1])]

        test_df = pd.DataFrame(X_test, columns=feature_names)
        test_df["Price_Actual"] = y_test.values
        test_df["Price_Predicted"] = preds
        st.markdown("Sample predictions (actual vs predicted)")
        st.dataframe(test_df.head(10), use_container_width=True)
        download_df(test_df, "realestate_ml_predictions.csv", label="Download ML predictions (CSV)")

    # -------------------------
    # Automated Insights
    # -------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)
    if not filt.empty:
        insights = filt.groupby(["City","Property_Type"]).agg(
            Avg_Price=("Price","mean"),
            Max_Price=("Price","max"),
            Min_Price=("Price","min"),
            Count=("Price","count")
        ).reset_index().sort_values("Avg_Price", ascending=False)
        insights["Avg_Price"] = insights["Avg_Price"].round(0)
        st.dataframe(insights, use_container_width=True)
        download_df(insights, "realestate_automated_insights.csv", label="Download automated insights (CSV)")
    else:
        st.info("No insights for selected filters.")

    # -------------------------
    # Export filtered dataset
    # -------------------------
    st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
    download_df(filt, "realestate_filtered_dataset.csv", label="Download filtered dataset (all rows)")

    st.markdown("### Done — export what you need", unsafe_allow_html=True)

# End of file
