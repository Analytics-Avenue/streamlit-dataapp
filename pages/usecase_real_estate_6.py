# tenant_risk_marketing_lab.py
# Tenant Risk & Market Trend Analyzer — Marketing Lab UI (full app)
# Drop this into a file and run with: streamlit run tenant_risk_marketing_lab.py

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

hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# -------------------------
# Page config + Branding
# -------------------------
st.set_page_config(page_title="Tenant Risk & Market Trend Analyzer — Marketing Lab", layout="wide")
LOGO = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

# -------------------------
# Marketing Lab CSS (Inter font, pure-black text, blue KPIs/variable boxes, fade-in, table)
# -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }

/* GLOBAL TEXT */
body, [class*="css"] { color:#000 !important; font-size:16.5px; }

/* Header */
.header-row { display:flex; align-items:center; gap:12px; margin-bottom:8px; }
.header-title { color:#064b86; font-size:34px; font-weight:700; margin:0; }
.header-sub { color:#666; font-size:13px; margin-top:2px; }

/* Section Title */
.section-title {
    font-size:22px !important;
    font-weight:600 !important;
    margin-top:20px;
    margin-bottom:10px;
    position:relative;
}
/* MAIN HEADER */
.big-header {
    font-size:36px !important;
    font-weight:700 !important;
    color:#000 !important;
    margin-bottom:12px;
}

.section-title:after {
    content:"";
    position:absolute; bottom:-6px; left:0;
    height:2px; width:0%;
    background:#064b86; transition: width 0.35s ease;
}
.section-title:hover:after { width:36%; }

/* Card */
.card {
    background:#fff; padding:20px; border-radius:14px;
    border:1px solid #e6e6e6; box-shadow:0 3px 14px rgba(0,0,0,0.06);
}
.card:hover { transform:translateY(-3px); box-shadow:0 12px 24px rgba(6,75,134,0.12); }

/* Blue KPI */
.kpi {
    background:#fff; padding:16px; border-radius:12px;
    border:1px solid #e2e2e2; color:#064b86 !important;
    font-weight:700; text-align:center; box-shadow:0 3px 10px rgba(0,0,0,0.04);
}

/* Variable boxes (blue text) */
.variable-box {
    padding:14px; border-radius:12px; border:1px solid #e5e5e5;
    background:#fff; color:#064b86 !important; font-weight:600;
    box-shadow:0 2px 10px rgba(0,0,0,0.04); margin-bottom:10px;
}

/* Index-safe HTML table */
.required-table thead th { background:#fff !important; color:#000 !important; padding:10px !important; border-bottom:2px solid #000 !important; text-align:left; }
.required-table td { padding:10px !important; border-bottom:1px solid #efefef !important; color:#000 !important; }

/* Buttons */
.stButton>button, .stDownloadButton>button { background:#064b86 !important; color:white !important; padding:9px 18px; border-radius:8px; font-weight:600; }

/* Fade in */
.block-container { animation: fadeIn 0.45s ease; }
@keyframes fadeIn { from { opacity:0; transform:translateY(6px);} to { opacity:1; transform:none; } }
</style>
""", unsafe_allow_html=True)

# -------------------------
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

st.markdown("<div class='big-header'>Tenant Risk & Market Trend Analyzer</div>", unsafe_allow_html=True)

# -------------------------
# Required columns & auto-map dictionary
# -------------------------
REQUIRED_COLS = [
    "City", "Property_Type", "Price", "Area_sqft",
    "Age_Years", "Conversion_Probability", "Latitude", "Longitude"
]

AUTO_MAP = {
    "City": ["city", "location", "town", "municipality"],
    "Property_Type": ["property_type", "ptype", "type", "listing_type"],
    "Price": ["price", "listing_price", "amount", "value"],
    "Area_sqft": ["area_sqft", "area", "size_sqft", "sqft"],
    "Age_Years": ["age_years", "age", "building_age"],
    "Conversion_Probability": ["conversion_probability", "occupancy_prob", "conversion_prob", "occupancy_probability"],
    "Latitude": ["latitude", "lat"],
    "Longitude": ["longitude", "lon", "lng"]
}

# -------------------------
# Helpers
# -------------------------
def dedupe_columns(cols):
    """Make duplicate column names unique by appending suffixes."""
    seen = {}
    out = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
    return out

def auto_map_columns(df):
    """Try to map common alternate column names to required names."""
    rename = {}
    lower_to_col = {c.lower(): c for c in df.columns}
    for req, variants in AUTO_MAP.items():
        for v in variants:
            if v.lower() in lower_to_col:
                rename[lower_to_col[v.lower()]] = req
                break
    if rename:
        df = df.rename(columns=rename)
    return df

def ensure_numeric(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def download_df(df, filename, label="Download CSV"):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")

def render_index_safe_table(df: pd.DataFrame, max_rows=500):
    if df is None or df.empty:
        st.info("No data to show")
        return
    html = df.head(max_rows).to_html(index=False, classes="required-table")
    st.markdown(html, unsafe_allow_html=True)

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# -------------------------
# Tab 1 — Overview
# -------------------------
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">This app identifies tenant/occupancy risk and surfaces emerging market trends for real-estate portfolios. Built to Marketing Lab UI standards.</div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">• Occupancy risk scoring by listing<br>• Market and city-level occupancy trends<br>• Map visualization of high-risk listings<br>• ML prediction for new property risk</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">• Proactive portfolio risk mitigation<br>• Target investments into high-growth pockets<br>• Data-driven property acquisition & pricing</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    K1, K2, K3, K4 = st.columns(4)
    K1.markdown("<div class='kpi'>High Risk Properties</div>", unsafe_allow_html=True)
    K2.markdown("<div class='kpi'>Low Risk Properties</div>", unsafe_allow_html=True)
    K3.markdown("<div class='kpi'>Top Growth City</div>", unsafe_allow_html=True)
    K4.markdown("<div class='kpi'>Avg Occupancy Prob.</div>", unsafe_allow_html=True)

# -------------------------
# Tab 2 — Important Attributes
# -------------------------
with tab2:
    st.markdown('<div class="section-title">Required Column Dictionary</div>', unsafe_allow_html=True)
    dict_df = pd.DataFrame([
        ["City", "City or town of the property"],
        ["Property_Type", "Type of property (Apartment, Villa, Office, Plot, etc.)"],
        ["Price", "Listing or sale price (INR)"],
        ["Area_sqft", "Area in square feet"],
        ["Age_Years", "Age of property in years"],
        ["Conversion_Probability", "Occupancy / conversion probability (0-1)"],
        ["Latitude", "Latitude coordinate for mapping"],
        ["Longitude", "Longitude coordinate for mapping"]
    ], columns=["Attribute", "Description"])
    render_index_safe_table(dict_df)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        for v in ["City", "Property_Type", "Area_sqft", "Age_Years"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        for v in ["Conversion_Probability", "Price", "Price_per_sqft"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# -------------------------
# Tab 3 — Application
# -------------------------
with tab3:
    st.markdown('<div class="section-title">Step 1 — Load dataset</div>', unsafe_allow_html=True)

    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)
    df = None
    raw = None
    DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"

    # Default dataset
    if mode == "Default dataset":
        try:
            df = pd.read_csv(DEFAULT_URL)
            # dedupe duplicates and auto-map
            df.columns = dedupe_columns(list(df.columns))
            df = auto_map_columns(df)
            st.success("Default dataset loaded")
            st.dataframe(df.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Could not load default dataset: {e}")
            st.stop()

    # Upload CSV
    elif mode == "Upload CSV":
        st.markdown("#### Download Sample CSV for Reference")
        try:
            sample_df = pd.read_csv(DEFAULT_URL).head(5)
            st.download_button("Download Sample CSV", sample_df.to_csv(index=False), "sample_realestate.csv", "text/csv")
        except Exception:
            pass
        uploaded = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            df.columns = dedupe_columns(list(df.columns))
            df = auto_map_columns(df)
            st.success("File uploaded")
            st.dataframe(df.head(5), use_container_width=True)

    # Upload + mapping
    else:
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"], key="mapping_upload")
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = dedupe_columns(list(raw.columns))
            st.markdown("<div class='section-title'>Preview (first 5 rows)</div>", unsafe_allow_html=True)
            st.dataframe(raw.head(5), use_container_width=True)
            st.markdown('<div class="section-title">Map your columns to required fields</div>', unsafe_allow_html=True)
            mapping = {}
            cols_list = list(raw.columns)
            for req in REQUIRED_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + cols_list, key=f"map_{req}")
            if st.button("Apply mapping"):
                missing = [k for k, v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    inv = {v: k for k, v in mapping.items()}
                    df = raw.rename(columns=inv)
                    st.success("Mapping applied")
                    st.dataframe(df.head(5), use_container_width=True)

    if df is None:
        st.stop()

    # Keep only relevant columns if present, and coerce types
    present = [c for c in REQUIRED_COLS if c in df.columns]
    if len(present) < 5:
        st.warning(f"Found {len(present)}/{len(REQUIRED_COLS)} required columns. Present: {present}. Some sections may be disabled.")

    # Ensure numeric columns
    for col in ["Price", "Area_sqft", "Age_Years", "Conversion_Probability", "Latitude", "Longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute price_per_sqft when possible
    if "Price" in df.columns and "Area_sqft" in df.columns:
        df["Price_per_sqft"] = df["Price"] / df["Area_sqft"]
    else:
        df["Price_per_sqft"] = np.nan

    # Drop rows without any essential numeric data to avoid complete failure
    df = df.dropna(subset=[c for c in ["Price", "Conversion_Probability"] if c in df.columns], how="all").reset_index(drop=True)
    if df.empty:
        st.error("No usable rows after cleaning.")
        st.stop()

    # -------------------------
    # Step 2 — Filters & Preview
    # -------------------------
    st.markdown('<div class="section-title">Step 2 — Filters & preview</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2,2,1])

    cities = sorted(df["City"].dropna().unique()) if "City" in df.columns else []
    ptypes = sorted(df["Property_Type"].dropna().unique()) if "Property_Type" in df.columns else []

    with c1:
        sel_city = st.multiselect("City", options=cities, default=cities[:5] if len(cities) else [])
    with c2:
        sel_ptype = st.multiselect("Property Type", options=ptypes, default=ptypes[:5] if len(ptypes) else [])
    with c3:
        if "Age_Years" in df.columns:
            min_age = int(np.nanmin(df["Age_Years"].fillna(0)))
            max_age = int(np.nanmax(df["Age_Years"].fillna(0)))
            age_range = st.slider("Property Age (years)", min_age, max_age, (min_age, max_age))
        else:
            age_range = None

    filt = df.copy()
    if sel_city:
        filt = filt[filt["City"].isin(sel_city)]
    if sel_ptype:
        filt = filt[filt["Property_Type"].isin(sel_ptype)]
    if age_range and "Age_Years" in filt.columns:
        filt = filt[(filt["Age_Years"] >= age_range[0]) & (filt["Age_Years"] <= age_range[1])]

    st.markdown('<div class="section-title">Filtered preview</div>', unsafe_allow_html=True)
    render_index_safe_table(filt.head(10))

    download_df(filt.head(500), "tenant_risk_filtered_preview.csv", label="Download filtered preview (CSV)")

    # -------------------------
    # KPIs (blue KPI cards)
    # -------------------------
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    def safe_count(cond_series):
        try:
            return int(cond_series.sum())
        except Exception:
            return 0

    high_risk_count = safe_count((filt["Conversion_Probability"] < 0.4) if "Conversion_Probability" in filt.columns else pd.Series(dtype=float))
    low_risk_count = safe_count((filt["Conversion_Probability"] > 0.7) if "Conversion_Probability" in filt.columns else pd.Series(dtype=float))
    avg_occ = float(np.nanmean(filt["Conversion_Probability"])) if "Conversion_Probability" in filt.columns else float("nan")
    top_city = None
    if "City" in filt.columns and "Conversion_Probability" in filt.columns:
        try:
            top_city = filt.groupby("City")["Conversion_Probability"].mean().idxmax()
        except Exception:
            top_city = None

    k1.markdown(f"<div class='kpi'>High Risk Properties<br><b>{high_risk_count}</b></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'>Low Risk Properties<br><b>{low_risk_count}</b></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'>Top Growth City<br><b>{top_city if top_city else 'N/A'}</b></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'>Avg Occupancy Prob.<br><b>{avg_occ:.2f}</b></div>", unsafe_allow_html=True)

    # -------------------------
    # Charts (recreated in Marketing-Lab style)
    # -------------------------
    st.markdown('<div class="section-title">Charts</div>', unsafe_allow_html=True)

    if not filt.empty:
        # Occupancy by property type
        if "Property_Type" in filt.columns and "Conversion_Probability" in filt.columns:
            st.markdown("#### Occupancy Probability by Property Type")
            ptype_risk = filt.groupby("Property_Type")["Conversion_Probability"].mean().reset_index().sort_values("Conversion_Probability", ascending=False)
            fig1 = px.bar(ptype_risk, x="Property_Type", y="Conversion_Probability", text="Conversion_Probability")
            fig1.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            st.plotly_chart(fig1, use_container_width=True)

        # Age vs Price_per_sqft
        if "Age_Years" in filt.columns and "Price_per_sqft" in filt.columns:
            st.markdown("#### Age vs Price per sqft")
            fig2 = px.scatter(filt, x="Age_Years", y="Price_per_sqft", color="Conversion_Probability" if "Conversion_Probability" in filt.columns else None,
                              hover_data=["City", "Property_Type", "Price"])
            st.plotly_chart(fig2, use_container_width=True)

        # City-wise occupancy trend
        if "City" in filt.columns and "Conversion_Probability" in filt.columns:
            st.markdown("#### City-wise Occupancy Trend")
            city_trend = filt.groupby("City")["Conversion_Probability"].mean().reset_index().sort_values("Conversion_Probability", ascending=False)
            fig3 = px.bar(city_trend, x="City", y="Conversion_Probability", text="Conversion_Probability")
            fig3.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            st.plotly_chart(fig3, use_container_width=True)

        # Map of high-risk properties (if lat/lon exist)
        if "Latitude" in filt.columns and "Longitude" in filt.columns:
            st.markdown("#### High Risk Properties Map")
            # create a normalized risk score for color scaling
            if "Conversion_Probability" in filt.columns:
                filt["Risk_Score"] = 1 - (filt["Conversion_Probability"] - filt["Conversion_Probability"].min()) / (filt["Conversion_Probability"].max() - filt["Conversion_Probability"].min() + 1e-9)
            else:
                filt["Risk_Score"] = 0
            # ensure price_per_sqft exists
            if "Price_per_sqft" not in filt.columns:
                filt["Price_per_sqft"] = np.nan
            fig4 = px.scatter_mapbox(
                filt,
                lat="Latitude",
                lon="Longitude",
                size="Price_per_sqft" if "Price_per_sqft" in filt.columns else None,
                color="Risk_Score",
                hover_name="Property_Type" if "Property_Type" in filt.columns else None,
                hover_data=["City","Price","Age_Years","Conversion_Probability"] if all(c in filt.columns for c in ["Price","Age_Years","Conversion_Probability"]) else None,
                color_continuous_scale=px.colors.sequential.OrRd,
                zoom=8,
                size_max=18,
                height=500
            )
            fig4.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No data to chart.")

    # -------------------------
    # ML: Predict occupancy risk for a new property
    # -------------------------
    st.markdown('<div class="section-title">Step 3 — Predict occupancy probability for a new property</div>', unsafe_allow_html=True)
    st.markdown("<small>Note: Model trains only if dataset has at least 30 rows and required predictors.</small>", unsafe_allow_html=True)

    # Quick inputs for prediction
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            inp_city = st.selectbox("City", options=sorted(df["City"].dropna().unique()), index=0)
        with col2:
            inp_ptype = st.selectbox("Property Type", options=sorted(df["Property_Type"].dropna().unique()), index=0)
        with col3:
            inp_age = st.number_input("Age (years)", min_value=0, max_value=100, value=5)
        inp_area = st.number_input("Area (sqft)", min_value=100, max_value=20000, value=1000)
        submitted = st.form_submit_button("Predict Occupancy Probability")

    # Train model if possible
    model_ready = all(c in df.columns for c in ["City", "Property_Type", "Age_Years", "Area_sqft", "Conversion_Probability"]) and len(df) >= 30
    if model_ready:
        features = ["City", "Property_Type", "Age_Years", "Area_sqft"]
        X = df[features].copy()
        y = df["Conversion_Probability"].copy()

        # Column transformer
        cat_cols = ["City", "Property_Type"]
        num_cols = ["Age_Years", "Area_sqft"]
        transformer = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols)
        ], remainder="drop")

        X_t = transformer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)

        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        with st.spinner("Training occupancy prediction model..."):
            rf.fit(X_train, y_train)

        # Show model performance
        preds_test = rf.predict(X_test)
       
        mse = mean_squared_error(y_test, preds_test)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, preds_test)
        st.markdown(f"**Model performance:** RMSE: {rmse:.3f} | R²: {r2:.3f}")

        if submitted:
            new_row = pd.DataFrame([[inp_city, inp_ptype, inp_age, inp_area]], columns=features)
            new_X = transformer.transform(new_row)
            pred_occ = rf.predict(new_X)[0]
            st.metric("Predicted Occupancy Probability", f"{pred_occ:.3f}")
            # allow download of prediction
            out_df = pd.DataFrame([{
                "City": inp_city,
                "Property_Type": inp_ptype,
                "Age_Years": inp_age,
                "Area_sqft": inp_area,
                "Predicted_Conversion_Probability": pred_occ
            }])
            download_df(out_df, "predicted_occupancy.csv", label="Download prediction (CSV)")
    else:
        st.info("ML model not available. Need at least 30 rows and columns: City, Property_Type, Age_Years, Area_sqft, Conversion_Probability.")

    # -------------------------
    # Automated Insights & Export
    # -------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)
    insight_ready = all(c in filt.columns for c in ["City", "Property_Type", "Price"])
    if insight_ready:
        ins = filt.groupby(["City", "Property_Type"]).agg(
            Avg_Price=("Price", "mean"),
            Avg_Occupancy=("Conversion_Probability", "mean") if "Conversion_Probability" in filt.columns else ("Conversion_Probability", lambda x: np.nan),
            Count=("Price", "count")
        ).reset_index()
        ins["Avg_Price"] = ins["Avg_Price"].round(0)
        render_index_safe_table(ins)
        download_df(ins, "tenant_risk_automated_insights.csv", label="Download automated insights (CSV)")
    else:
        st.info("Not enough fields for automated insights (need City, Property_Type, Price).")

    st.markdown('<div class="section-title">Export filtered dataset</div>', unsafe_allow_html=True)
    if not filt.empty:
        download_df(filt, "tenant_risk_filtered_dataset.csv", label="Download filtered dataset (CSV)")
    else:
        st.info("No rows to export.")
