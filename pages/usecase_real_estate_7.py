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
# Page config + Branding
# -------------------------
st.set_page_config(page_title="Rental Yield & Investment Analyzer — Marketing Lab", layout="wide")
LOGO = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

# -------------------------
# Marketing Lab CSS
# (Inter font, pure-black text, blue KPI & variable boxes, fade-in, index-safe table)
# -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }

/* GLOBAL TEXT */
body, [class*="css"] { color:#000 !important; font-size:16.5px; }

/* Header area */
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
st.markdown("<div class='big-header'>Rental Yield & Investment Analyzer</div>", unsafe_allow_html=True)


# -------------------------
# Required columns & mapping hints
# -------------------------
REQUIRED_COLS = [
    "City", "Property_Type", "Price", "Area_sqft",
    "Age_Years", "Conversion_Probability", "Latitude", "Longitude"
]

AUTO_MAP = {
    "City": ["city", "location", "town"],
    "Property_Type": ["property_type", "ptype", "type"],
    "Price": ["price", "listing_price", "amount"],
    "Area_sqft": ["area_sqft", "area", "size_sqft", "sqft"],
    "Age_Years": ["age_years", "age"],
    "Conversion_Probability": ["conversion_probability", "occupancy_prob", "conversion_prob"],
    "Latitude": ["latitude", "lat"],
    "Longitude": ["longitude", "lon", "lng"]
}

def dedupe_columns(cols):
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
    rename = {}
    lower_map = {c.lower(): c for c in df.columns}
    for req, variants in AUTO_MAP.items():
        for v in variants:
            if v.lower() in lower_map:
                rename[lower_map[v.lower()]] = req
                break
    if rename:
        df = df.rename(columns=rename)
    return df

def render_index_safe_table(df: pd.DataFrame, max_rows=500):
    if df is None or df.empty:
        st.info("No data to show")
        return
    html = df.head(max_rows).to_html(index=False, classes="required-table")
    st.markdown(html, unsafe_allow_html=True)

def download_df(df, filename, label="Download CSV"):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# -------------------------
# Overview tab
# -------------------------
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">Estimate rental yield and investment potential across cities and property types. Designed with the Marketing Lab UI system.</div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">• Rental yield calculation<br>• Investment score combining yield & conversion probability<br>• City & property-type segmentation<br>• Map visualization and exportable top-investments</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">• Prioritise high-ROI investments<br>• Reduce time to identify buy-to-let candidates<br>• Data-driven portfolio optimization</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    K1, K2, K3, K4 = st.columns(4)
    K1.markdown("<div class='kpi'>Top Rental Yield Property</div>", unsafe_allow_html=True)
    K2.markdown("<div class='kpi'>Highest Yield City</div>", unsafe_allow_html=True)
    K3.markdown("<div class='kpi'>Median Rental Yield</div>", unsafe_allow_html=True)
    K4.markdown("<div class='kpi'>Top Property Type</div>", unsafe_allow_html=True)

# -------------------------
# Important Attributes tab
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
        ["Latitude", "Latitude coordinate (for maps)"],
        ["Longitude", "Longitude coordinate (for maps)"]
    ], columns=["Attribute", "Description"])
    render_index_safe_table(dict_df)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        for v in ["City", "Property_Type", "Area_sqft", "Age_Years"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        for v in ["Rental_Yield", "Investment_Score", "Conversion_Probability"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# -------------------------
# Application tab
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
            df.columns = dedupe_columns(list(df.columns))
            df = auto_map_columns(df)
            st.success("Default dataset loaded")
            st.dataframe(df.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Could not load default dataset: {e}")
            st.stop()

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

    else:  # mapping mode
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

    # Ensure numeric columns
    for col in ["Price", "Area_sqft", "Age_Years", "Conversion_Probability", "Latitude", "Longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute fields
    if "Price" in df.columns and "Area_sqft" in df.columns:
        df["Price_per_sqft"] = df["Price"] / df["Area_sqft"]
    else:
        df["Price_per_sqft"] = np.nan

    # Est Rent assumption, Rental_Yield, Investment Score
    # Est_Rent: conservative assumption - 0.5% monthly of price (0.005)
    df["Est_Rent"] = np.where(df["Price"].notna(), df["Price"] * 0.005, np.nan)
    df["Rental_Yield"] = np.where(df["Price"].notna(), (df["Est_Rent"] * 12) / df["Price"], np.nan)
    # Investment_Score = normalized Rental_Yield * Conversion_Probability (0-1)
    df["Conversion_Probability"] = df["Conversion_Probability"].fillna(0)
    df["Investment_Score"] = df["Rental_Yield"].fillna(0) * df["Conversion_Probability"]

    # Basic cleaning drop rows missing critical values
    df = df.dropna(subset=[c for c in ["Price", "Area_sqft"] if c in df.columns], how="any").reset_index(drop=True)
    if df.empty:
        st.error("No usable rows after cleaning.")
        st.stop()

    # -------------------------
    # Step 2 — Filters & preview
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
    download_df(filt.head(500), "rental_yield_filtered_preview.csv", label="Download filtered preview (CSV)")

    # -------------------------
    # KPIs
    # -------------------------
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    def safe_agg(series, func, default="N/A"):
        try:
            val = func(series.dropna())
            if pd.isna(val):
                return default
            return val
        except Exception:
            return default

    top_prop = safe_agg(filt.sort_values("Rental_Yield", ascending=False)["Property_Type"].head(1), lambda s: s.iloc[0], "N/A")
    top_city = safe_agg(filt.groupby("City")["Rental_Yield"].mean().sort_values(ascending=False).index, lambda s: s[0], "N/A")
    median_yield = safe_agg(filt["Rental_Yield"], np.nanmedian, "N/A")
    top_type = safe_agg(filt.groupby("Property_Type")["Investment_Score"].mean().sort_values(ascending=False).index, lambda s: s[0], "N/A")

    k1.markdown(f"<div class='kpi'>Top Rental Yield Property<br><b>{top_prop}</b></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'>Highest Yield City<br><b>{top_city}</b></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'>Median Rental Yield<br><b>{median_yield:.4f}</b></div>" if isinstance(median_yield, (int,float)) else f"<div class='kpi'>Median Rental Yield<br><b>{median_yield}</b></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'>Top Property Type (by score)<br><b>{top_type}</b></div>", unsafe_allow_html=True)

    # -------------------------
    # Charts (Marketing Lab style)
    # -------------------------
    st.markdown('<div class="section-title">Charts</div>', unsafe_allow_html=True)

    if not filt.empty:
        # Rental Yield by Property Type
        if "Property_Type" in filt.columns:
            st.markdown("#### Rental Yield by Property Type")
            ptype_yield = filt.groupby("Property_Type")["Rental_Yield"].mean().reset_index().sort_values("Rental_Yield", ascending=False)
            fig1 = px.bar(ptype_yield, x="Property_Type", y="Rental_Yield", text="Rental_Yield")
            fig1.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            st.plotly_chart(fig1, use_container_width=True)

        # City-wise Rental Yield
        if "City" in filt.columns:
            st.markdown("#### City-wise Rental Yield")
            city_yield = filt.groupby("City")["Rental_Yield"].mean().reset_index().sort_values("Rental_Yield", ascending=False)
            fig2 = px.bar(city_yield, x="City", y="Rental_Yield", text="Rental_Yield")
            fig2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            st.plotly_chart(fig2, use_container_width=True)

        # Investment Score map (if coords present)
        if "Latitude" in filt.columns and "Longitude" in filt.columns:
            st.markdown("#### Investment Score Map")
            if filt["Investment_Score"].max() - filt["Investment_Score"].min() > 0:
                filt["Score_Normalized"] = (filt["Investment_Score"] - filt["Investment_Score"].min()) / (filt["Investment_Score"].max() - filt["Investment_Score"].min())
            else:
                filt["Score_Normalized"] = 0.0
            fig3 = px.scatter_mapbox(filt, lat="Latitude", lon="Longitude",
                                     size="Investment_Score", color="Score_Normalized",
                                     hover_name="Property_Type",
                                     hover_data=["City","Price","Rental_Yield","Conversion_Probability","Investment_Score"],
                                     color_continuous_scale=px.colors.sequential.Viridis,
                                     size_max=18, zoom=8, height=500)
            fig3.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No data to chart.")

    # -------------------------
    # Top Investments table & export
    # -------------------------
    st.markdown('<div class="section-title">Top Investment Properties</div>', unsafe_allow_html=True)
    top_inv = filt.sort_values("Investment_Score", ascending=False).head(20)
    if not top_inv.empty:
        render_index_safe_table(top_inv[["City","Property_Type","Price","Rental_Yield","Conversion_Probability","Investment_Score"]])
        download_df(top_inv, "top_investments.csv", label="Download Top Investment Properties (CSV)")
    else:
        st.info("No top investments to display.")

    # -------------------------
    # Optional ML: Predict Investment_Score for new property
    # -------------------------
    st.markdown('<div class="section-title">Step 3 — Predict Investment Score (optional)</div>', unsafe_allow_html=True)
    st.markdown("<small>Model trains when dataset has >= 40 rows and required features.</small>", unsafe_allow_html=True)

    can_train = all(c in df.columns for c in ["City", "Property_Type", "Age_Years", "Area_sqft", "Investment_Score"]) and len(df) >= 40
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            inp_city = st.selectbox("City", options=sorted(df["City"].dropna().unique()), index=0)
        with c2:
            inp_ptype = st.selectbox("Property Type", options=sorted(df["Property_Type"].dropna().unique()), index=0)
        with c3:
            inp_age = st.number_input("Age (years)", min_value=0, max_value=100, value=5)
        inp_area = st.number_input("Area (sqft)", min_value=100, max_value=20000, value=1000)
        submit = st.form_submit_button("Train & Predict")

    if can_train:
        features = ["City", "Property_Type", "Age_Years", "Area_sqft"]
        X = df[features].copy()
        y = df["Investment_Score"].copy()

        cat_cols = ["City", "Property_Type"]
        num_cols = ["Age_Years", "Area_sqft"]
        transformer = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols)
        ], remainder="drop")

        X_t = transformer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)

        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        with st.spinner("Training model for Investment_Score..."):
            rf.fit(X_train, y_train)

        preds_test = rf.predict(X_test)
        # compatibility: compute RMSE manually
        mse = mean_squared_error(y_test, preds_test)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, preds_test)
        st.markdown(f"**Model performance:** RMSE: {rmse:.4f} | R²: {r2:.3f}")

        if submit:
            new_row = pd.DataFrame([[inp_city, inp_ptype, inp_age, inp_area]], columns=features)
            new_X = transformer.transform(new_row)
            pred_score = rf.predict(new_X)[0]
            st.metric("Predicted Investment Score", f"{pred_score:.4f}")
            out_df = pd.DataFrame([{
                "City": inp_city,
                "Property_Type": inp_ptype,
                "Age_Years": inp_age,
                "Area_sqft": inp_area,
                "Predicted_Investment_Score": pred_score
            }])
            download_df(out_df, "predicted_investment_score.csv", label="Download prediction (CSV)")
    else:
        st.info("ML unavailable: need >=40 rows and fields City, Property_Type, Age_Years, Area_sqft, Investment_Score.")

    # -------------------------
    # Automated insights & export of filtered dataset
    # -------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)
    if all(c in filt.columns for c in ["City","Property_Type","Investment_Score"]):
        ins = filt.groupby(["City","Property_Type"]).agg(
            Avg_Price=("Price","mean"),
            Avg_Rental_Yield=("Rental_Yield","mean"),
            Avg_Investment_Score=("Investment_Score","mean"),
            Count=("Price","count")
        ).reset_index()
        ins["Avg_Price"] = ins["Avg_Price"].round(0)
        render_index_safe_table(ins)
        download_df(ins, "rental_yield_automated_insights.csv", label="Download automated insights (CSV)")
    else:
        st.info("Not enough fields for automated insights (need City, Property_Type, Investment_Score).")

    st.markdown('<div class="section-title">Export filtered dataset</div>', unsafe_allow_html=True)
    if not filt.empty:
        download_df(filt, "rental_yield_filtered_dataset.csv", label="Download filtered dataset (CSV)")
    else:
        st.info("No rows to export.")
