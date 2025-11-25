# agent_market_insights_marketing_lab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Page config + branding
# -------------------------
st.set_page_config(page_title="Agent & Market Insights — Marketing Lab", layout="wide")
LOGO = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

# -------------------------
# Marketing Lab CSS (Inter, pure black text, blue KPIs, variable boxes, fade-in)
# -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
body, [class*="css"] { color:#000 !important; font-size:17px; }

/* Header */
.big-header { font-size:36px !important; font-weight:700 !important; color:#000 !important; margin-bottom:12px; }

/* SECTION TITLE */
.section-title {
    font-size:24px !important;
    font-weight:600 !important;
    margin-top:24px;
    margin-bottom:12px;
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
    position:absolute; bottom:-5px; left:0; height:2px; width:0%;
    background:#064b86; transition: width 0.35s ease;
}
.section-title:hover:after { width:40%; }

/* CARD */
.card { background:#fff; padding:20px; border-radius:12px; border:1px solid #e6e6e6; box-shadow:0 3px 14px rgba(0,0,0,0.06); }
.card:hover { transform:translateY(-3px); box-shadow:0 12px 24px rgba(6,75,134,0.12); }

/* KPI */
.kpi { background:#fff; padding:18px; border-radius:12px; border:1px solid #e2e2e2; color:#064b86 !important; font-weight:700; text-align:center; box-shadow:0 3px 10px rgba(0,0,0,0.05); }

/* Variable box */
.variable-box { padding:14px; border-radius:12px; border:1px solid #e5e5e5; color:#064b86 !important; font-weight:600; background:#fff; box-shadow:0 2px 8px rgba(0,0,0,0.04); margin-bottom:10px; }

/* index-safe table */
.required-table thead th { background:#fff !important; color:#000 !important; padding:10px !important; border-bottom:2px solid #000 !important; text-align:left; }
.required-table td { padding:10px !important; border-bottom:1px solid #efefef !important; color:#000 !important; }

/* Buttons */
.stButton>button, .stDownloadButton>button { background:#064b86 !important; color:white !important; padding:9px 18px; border-radius:8px; font-weight:600; }

/* fade in */
.block-container { animation: fadeIn 0.45s ease; }
@keyframes fadeIn { from { opacity:0; transform:translateY(8px);} to {opacity:1; transform:none;} }
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
st.markdown("<div class='big-header'>Real Estate Agent & Market Insights</div>", unsafe_allow_html=True)


# -------------------------
# Required columns and automap dict
# -------------------------
REQUIRED_COLS = ["City","Property_Type","Agent_Name","Price","Lead_Score","Conversion_Probability","Days_On_Market"]

AUTO_MAP = {
    "City": ["city","location","town"],
    "Property_Type": ["property_type","ptype","type"],
    "Agent_Name": ["agent_name","agent","broker"],
    "Price": ["price","listing_price","amount","value"],
    "Lead_Score": ["lead_score","leadscore","lead_score_val"],
    "Conversion_Probability": ["conversion_probability","conv_prob","conversion_prob"],
    "Days_On_Market": ["days_on_market","dom","days_on_market_val"]
}

# -------------------------
# Helpers: dedupe columns, auto-map, downloads, small renderers
# -------------------------
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
    lower_to_col = {c.lower(): c for c in df.columns}
    for req, variants in AUTO_MAP.items():
        for v in variants:
            if v.lower() in lower_to_col:
                rename[lower_to_col[v.lower()]] = req
                break
    return df.rename(columns=rename)

def download_df(df, filename, label="Download CSV"):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")

def render_index_safe_table(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("No data to show")
        return
    st.markdown(df.to_html(index=False, classes="required-table"), unsafe_allow_html=True)

# -------------------------
# Tabs (exact 3-tab structure requested earlier; here we follow Marketing Lab: Overview, Important Attributes, Application)
# -------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# -------------------------
# TAB 1 — Overview
# -------------------------
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">Agent & market analytics platform for monitoring agent performance, segmentation, and revenue prediction. Built to Marketing Lab UI standards.</div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">• Agent performance dashboards<br>• Lead conversion analytics<br>• Market segmentation & clustering<br>• ML revenue/price prediction<br>• Automated insights & CSV export</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">• Better incentive allocation<br>• Focus on high-conversion agents<br>• Identify promising markets & listings<br>• Data-driven marketing spend</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Avg Lead Score</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Avg Conversion %</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Avg Days on Market</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Top Agent (by conv %)</div>", unsafe_allow_html=True)

# -------------------------
# TAB 2 — Important Attributes
# -------------------------
with tab2:
    st.markdown('<div class="section-title">Required Column Dictionary</div>', unsafe_allow_html=True)
    dict_df = pd.DataFrame([
        ["City","City or town of the property"],
        ["Property_Type","Type of property (Apartment, Villa, Office, Plot)"],
        ["Agent_Name","Name of listing agent or broker"],
        ["Price","Listing or sale price (INR)"],
        ["Lead_Score","Lead quality score (0-100)"],
        ["Conversion_Probability","Probability of conversion (0-1)"],
        ["Days_On_Market","Days property has been listed"]
    ], columns=["Attribute","Description"])
    render_index_safe_table(dict_df)

    st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        for v in ["City","Property_Type","Agent_Name","Lead_Score","Days_On_Market"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        for v in ["Price","Conversion_Probability"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# -------------------------
# TAB 3 — Application
# -------------------------
with tab3:
    st.markdown('<div class="section-title">Step 1 — Load dataset</div>', unsafe_allow_html=True)

    mode = st.radio("Dataset option:", ["Default dataset","Upload CSV","Upload CSV + Column mapping"], horizontal=True)
    df = None
    raw = None
    DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"

    # Load default
    if mode == "Default dataset":
        try:
            df = pd.read_csv(DEFAULT_URL)
            # dedupe duplicate columns if any
            df.columns = dedupe_columns(list(df.columns))
            df = auto_map_columns(df)
            st.success("Default dataset loaded")
            st.dataframe(df.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Could not load default dataset: {e}")
            st.stop()

    # Upload raw
    elif mode == "Upload CSV":
        st.markdown("#### Download sample CSV")
        try:
            sample_df = pd.read_csv(DEFAULT_URL).head(5)
            st.download_button("Download Sample CSV", sample_df.to_csv(index=False), "sample_real_estate.csv", "text/csv")
        except Exception:
            pass
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            df.columns = dedupe_columns(list(df.columns))
            df = auto_map_columns(df)
            st.success("File uploaded")
            st.dataframe(df.head(5), use_container_width=True)

    # Upload + mapping
    else:
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"], key="map_upload")
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

    # Keep only required columns if present
    present_required = [c for c in REQUIRED_COLS if c in df.columns]
    if len(present_required) < 4:
        st.warning(f"Only found {len(present_required)} of required columns. Found: {present_required}. Some features/sections may be disabled.")
    df = df.copy()

    # Coerce numeric columns and basic cleaning
    for col in ["Price","Lead_Score","Conversion_Probability","Days_On_Market"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing core numerical data to avoid crashes
    core_numeric = [c for c in ["Price","Conversion_Probability","Lead_Score","Days_On_Market"] if c in df.columns]
    if core_numeric:
        df = df.dropna(subset=core_numeric, how="all").reset_index(drop=True)

    # -------------------------
    # Step 2 — Filters & preview
    # -------------------------
    st.markdown('<div class="section-title">Step 2 — Filters & preview</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns([2,2,1])

    cities = sorted(df["City"].dropna().unique()) if "City" in df.columns else []
    ptypes = sorted(df["Property_Type"].dropna().unique()) if "Property_Type" in df.columns else []
    agents = sorted(df["Agent_Name"].dropna().unique()) if "Agent_Name" in df.columns else []

    with c1:
        sel_city = st.multiselect("City", options=cities, default=cities[:5] if cities else [])
    with c2:
        sel_ptype = st.multiselect("Property Type", options=ptypes, default=ptypes[:5] if ptypes else [])
    with c3:
        sel_agent = st.multiselect("Agent Name", options=agents, default=agents[:5] if agents else [])

    filt = df.copy()
    if sel_city:
        filt = filt[filt["City"].isin(sel_city)]
    if sel_ptype:
        filt = filt[filt["Property_Type"].isin(sel_ptype)]
    if sel_agent:
        filt = filt[filt["Agent_Name"].isin(sel_agent)]

    st.markdown('<div class="section-title">Filtered preview (first 10 rows)</div>', unsafe_allow_html=True)
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(500), "agent_market_filtered_preview.csv")

    # -------------------------
    # KPIs (blue KPI cards)
    # -------------------------
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)

    def safe_mean(col):
        try:
            return np.nanmean(filt[col])
        except Exception:
            return np.nan

    avg_lead_score = safe_mean("Lead_Score") if "Lead_Score" in filt.columns else np.nan
    avg_conv_pct = safe_mean("Conversion_Probability") if "Conversion_Probability" in filt.columns else np.nan
    avg_dom = safe_mean("Days_On_Market") if "Days_On_Market" in filt.columns else np.nan
    top_agent = None
    if "Agent_Name" in filt.columns and "Conversion_Probability" in filt.columns:
        try:
            top_agent = filt.groupby("Agent_Name")["Conversion_Probability"].mean().idxmax()
        except Exception:
            top_agent = None

    k1.markdown(f"<div class='kpi'>Avg Lead Score<br><b>{avg_lead_score:.2f}</b></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'>Avg Conversion %<br><b>{(avg_conv_pct*100 if not np.isnan(avg_conv_pct) else np.nan):.2f}%</b></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'>Avg Days on Market<br><b>{avg_dom:.1f}</b></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'>Top Performing Agent<br><b>{top_agent if top_agent else 'N/A'}</b></div>", unsafe_allow_html=True)

    # -------------------------
    # Charts
    # -------------------------
    st.markdown('<div class="section-title">Charts</div>', unsafe_allow_html=True)

    if not filt.empty:
        if "Agent_Name" in filt.columns and "Conversion_Probability" in filt.columns:
            st.markdown("#### Agent-wise Conversion Probability")
            agent_conv = filt.groupby("Agent_Name")["Conversion_Probability"].mean().reset_index().sort_values("Conversion_Probability", ascending=False)
            fig1 = px.bar(agent_conv, x="Agent_Name", y="Conversion_Probability", text="Conversion_Probability", title="Agent conversion probability")
            fig1.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            st.plotly_chart(fig1, use_container_width=True)

        if "City" in filt.columns and "Price" in filt.columns:
            st.markdown("#### City-wise Average Price")
            city_price = filt.groupby("City")["Price"].mean().reset_index().sort_values("Price", ascending=False)
            fig2 = px.bar(city_price.head(20), x="City", y="Price", text="Price", title="City average price (top 20)")
            fig2.update_traces(texttemplate="₹ %{text:,.0f}", textposition="outside")
            st.plotly_chart(fig2, use_container_width=True)

        # Market segmentation (KMeans) if features exist
        if all(c in filt.columns for c in ["Price","Days_On_Market","Conversion_Probability"]):
            st.markdown("#### Market Segmentation (KMeans)")
            seg_features = filt[["Price","Days_On_Market","Conversion_Probability"]].fillna(0)
            scaler = StandardScaler()
            seg_scaled = scaler.fit_transform(seg_features)
            n_clusters = st.slider("Number of segments (K)", 2, 8, 4)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            filt["Segment"] = kmeans.fit_predict(seg_scaled)
            fig3 = px.scatter(filt, x="Price", y="Conversion_Probability", color="Segment", hover_data=["Agent_Name","City","Property_Type"], title="Segments: Price vs Conversion")
            st.plotly_chart(fig3, use_container_width=True)

    else:
        st.info("No data available for charts")

    # -------------------------
    # ML: Revenue/Price Prediction
    # -------------------------
    st.markdown('<div class="section-title">ML — Price / Revenue Prediction</div>', unsafe_allow_html=True)

    ml_possible = all(col in filt.columns for col in ["Price"]) and any(col in filt.columns for col in ["Lead_Score","Days_On_Market","Conversion_Probability"])
    if ml_possible and len(filt) >= 30:
        # choose features available
        feat_cols = []
        cat_cols = []
        num_cols = []
        for c in ["City","Property_Type","Lead_Score","Days_On_Market","Conversion_Probability"]:
            if c in filt.columns:
                feat_cols.append(c)
                if filt[c].dtype == "object":
                    cat_cols.append(c)
                else:
                    num_cols.append(c)

        X = filt[feat_cols].copy()
        y = filt["Price"].copy()

        # ColumnTransformer
        transformers = []
        if cat_cols:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
        if num_cols:
            transformers.append(("num", "passthrough", num_cols))
        if not transformers:
            st.info("No usable features for ML after checking types.")
        else:
            col_transformer = ColumnTransformer(transformers=transformers, remainder="drop")
            X_t = col_transformer.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            with st.spinner("Training RandomForest..."):
                model.fit(X_train, y_train)
            preds = model.predict(X_test)
            try:
                from sklearn.metrics import mean_squared_error, r2_score
                rmse = mean_squared_error(y_test, preds, squared=False)
                r2 = r2_score(y_test, preds)
            except Exception:
                rmse, r2 = None, None
            st.write(f"Model performance — RMSE: {rmse:.2f} | R²: {r2:.3f}" if rmse is not None else "Model trained")

            # create downloadable predictions
            # map X_test back to feature names if possible
            try:
                feat_names = col_transformer.get_feature_names_out()
            except Exception:
                feat_names = [f"f_{i}" for i in range(X_test.shape[1])]
            ml_df = pd.DataFrame(X_test, columns=feat_names)
            ml_df["Actual_Price"] = y_test.values
            ml_df["Predicted_Price"] = preds
            st.markdown("Sample predictions")
            st.dataframe(ml_df.head(10), use_container_width=True)
            download_df(ml_df, "agent_market_ml_predictions.csv", label="Download ML predictions (CSV)")
    else:
        st.info("ML disabled — need Price plus at least one numeric predictor (Lead_Score/Days_On_Market/Conversion_Probability) and >=30 rows.")

    # -------------------------
    # Automated Insights & Exports
    # -------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)

    insights_ready = ("City" in filt.columns and "Property_Type" in filt.columns and "Price" in filt.columns)
    if insights_ready:
        ins = filt.groupby(["City","Property_Type"]).agg(Avg_Price=("Price","mean"), Max_Price=("Price","max"), Min_Price=("Price","min"), Count=("Price","count")).reset_index()
        ins["Avg_Price"] = ins["Avg_Price"].round(0)
        st.dataframe(ins, use_container_width=True)
        download_df(ins, "agent_market_automated_insights.csv", label="Download automated insights")
    else:
        st.info("Not enough columns for automated insights (City, Property_Type, Price required).")

    st.markdown('<div class="section-title">Export filtered dataset</div>', unsafe_allow_html=True)
    if not filt.empty:
        download_df(filt, "agent_market_filtered_dataset.csv", label="Download filtered dataset (CSV)")
    else:
        st.info("No rows to export.")

# End of file
