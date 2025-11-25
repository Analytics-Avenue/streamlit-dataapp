import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import math
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# App config + logo
# -------------------------
st.set_page_config(page_title="Customer Journey & Funnel Analytics — Lab", layout="wide", initial_sidebar_state="collapsed")

logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

# -------------------------
# UI constants
# -------------------------
BLUE = "#064b86"
BLACK = "#000000"
BASE_FONT_PX = 17

# -------------------------
# Helpers (must be defined early)
# -------------------------
def to_currency(x):
    try:
        return "₹ " + f"{float(x):,.2f}"
    except Exception:
        return "₹ 0.00"

def download_df_bytes(df: pd.DataFrame, filename: str, label: str = "Download CSV"):
    """Generic download button for a DataFrame (no index)."""
    if df is None or df.empty:
        st.info("No data to download.")
        return
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")

def ensure_datetime(df: pd.DataFrame, col: str = "Date"):
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            pass
    return df

def safe_numeric(df: pd.DataFrame, cols: list):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def render_required_table(df: pd.DataFrame):
    """
    Index-safe table renderer (no .hide_index() used because some Streamlit builds break).
    """
    styled = (
        df.style
        .set_table_attributes('class="required-table"')
        .set_properties(**{
            "color": "#000000",
            "font-size": "17px",
        })
        .set_table_styles([
            {"selector": "th", "props": [("color", "#000000"), ("font-size", "18px"), ("font-weight", "600")]}
        ])
        .format_index(lambda x: "", axis=0)  # <— SAFEST cross-version alternative
    )

    html = styled.to_html()
    html = html.replace("<th></th>", "").replace("<td></td>", "")  # clean empty index cells
    st.write(html, unsafe_allow_html=True)



def auto_map_columns(df: pd.DataFrame, mapping_dict: dict):
    """
    Light auto-mapper. mapping_dict: key -> list of candidate substrings.
    Returns renamed DataFrame (does not force presence of all required fields).
    """
    rename = {}
    cols = [c.strip() for c in df.columns]
    for req, candidates in mapping_dict.items():
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

# -------------------------
# Required column spec (source: your Customer Journey app)
# -------------------------
REQUIRED_COLS = [
    "Date","Campaign","Channel","Stage","Conversion_Flag","Revenue",
    "Impressions","Clicks","Leads","CTR","CPC","CPA",
    "Video_50%","Video_75%","Video_100%","ThruPlay_Rate",
    "Country","Device","AgeGroup","Gender"
]

AUTO_MAPS = {
    "Date": ["date", "day", "event date"],
    "Campaign": ["campaign", "campaign_name", "campaign name"],
    "Channel": ["channel", "source", "platform"],
    "Stage": ["stage", "funnel_stage"],
    "Conversion_Flag": ["conversion_flag", "is_converted", "converted"],
    "Revenue": ["revenue", "amount"],
    "Impressions": ["impressions", "impr"],
    "Clicks": ["clicks", "link clicks"],
    "Leads": ["leads", "results"],
    "CTR": ["ctr", "click through rate"],
    "CPC": ["cpc", "cost per click"],
    "CPA": ["cpa", "cost per action", "cost per acquisition"],
    "Video_50%": ["video_50", "video_50%","video50"],
    "Video_75%": ["video_75", "video_75%","video75"],
    "Video_100%": ["video_100", "video_100%","video100","thruplay"],
    "ThruPlay_Rate": ["thruplay_rate", "thruplay"],
    "Country": ["country", "region"],
    "Device": ["device", "platform"],
    "AgeGroup": ["agegroup", "age group", "age"],
    "Gender": ["gender", "sex"]
}

# -------------------------
# Style / CSS — Marketing Lab look (pure-black text, blue KPIs/vars)
# -------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {{
  background: #ffffff;
  color: {BLACK};
  font-family: 'Inter', sans-serif;
  font-size: {BASE_FONT_PX}px;
}}

/* header */
.header-row {{
    display:flex;
    align-items:center;
    gap:12px;
    margin-bottom:8px;
}}
.header-title {{
    color: {BLUE};
    font-size:28px;
    font-weight:700;
    margin:0;
}}
.header-sub {{ color: {BLACK}; font-size:13px; margin:0; }}



/* section title */
.section-title {{
    color: {BLACK};
    font-size:22px;
    font-weight:600;
    margin: 14px 0 10px 0;
    position:relative;
    display:inline-block;
}}

/* MAIN HEADER */
.big-header {{
    font-size:36px !important;
    font-weight:700 !important;
    color:#000 !important;
    margin-bottom:12px;
}}
.section-title:hover::after {{
    content: "";
    position:absolute;
    left:0;
    bottom:-6px;
    width:36%;
    height:3px;
    background: {BLUE};
    border-radius:2px;
}}

/* card (pure black content) */
.card {{
    background:#ffffff;
    color: {BLACK};
    border-radius:12px;
    padding:18px;
    border:1px solid #e8e8e8;
    box-shadow:0 6px 20px rgba(0,0,0,0.04);
    margin-bottom:12px;
}}

/* KPI card (blue) */
.kpi {{
    background:#ffffff;
    color:{BLUE};
    border-radius:12px;
    padding:16px;
    border:1px solid #e8e8e8;
    text-align:center;
    font-weight:700;
    box-shadow:0 4px 12px rgba(0,0,0,0.05);
}}
.kpi .kpi-value {{ font-size:20px; margin-top:6px; color:{BLUE}; }}

/* variable box (blue) */
.variable-box {{
    background:#ffffff;
    color:{BLUE};
    border-radius:12px;
    padding:12px;
    border:1px solid #e8e8e8;
    text-align:center;
    font-weight:600;
    box-shadow:0 4px 10px rgba(0,0,0,0.04);
}}

/* required-table (index-safe renderer) - PURE BLACK text */
.required-table {{
    width:100%;
    border-collapse:collapse;
    font-size:17px;
    color:{BLACK};
    background:#fff;
}}
.required-table thead th {{
    border-bottom:2px solid #000;
    padding:10px;
    text-align:left;
    font-weight:700;
}}
.required-table tbody td {{
    padding:10px;
    border-bottom:1px solid #efefef;
}}
.required-table tbody tr:hover {{ background:#fafafa; }}

/* standard dataframe fallback */
.dataframe th {{
    background:{BLUE} !important;
    color:#fff !important;
    padding:10px !important;
    font-size:15.5px !important;
}}
.dataframe td {{
    color:{BLACK} !important;
    font-size:15.5px !important;
    padding:8px !important;
}}

/* Buttons */
.stButton>button, .stDownloadButton>button {{
    background:{BLUE} !important;
    color:white !important;
    padding:10px 18px;
    border-radius:8px !important;
    font-weight:600 !important;
}}
.stButton>button:hover, .stDownloadButton>button:hover {{ background:#0a6eb3 !important; }}

/* page fade */
.block-container {{ animation: fadeIn 0.45s ease; }}
@keyframes fadeIn {{ from {{ opacity:0; transform:translateY(8px); }} to {{ opacity:1; transform:translateY(0); }} }}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
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

st.markdown("<div class='big-header'>Customer Journey & Funnel Analytics</div>", unsafe_allow_html=True)

# -------------------------
# Tabs (Overview, Important Attributes, Application)
# -------------------------
tab_overview, tab_attrs, tab_app = st.tabs(["Overview", "Important Attributes", "Application"])

# -------------------------
# Overview Tab
# -------------------------
with tab_overview:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        Centralized dashboard for tracking customer journey, funnel stages, video engagement, and revenue-driven predictions.
        Designed for performance teams and growth analysts to make fast, evidence-based decisions.
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            • Funnel visualization & stage-level conversion analysis<br>
            • Video engagement metrics (50/75/100% & ThruPlay)<br>
            • Predictive ML for revenue and conversion signals<br>
            • Exportable prediction and insights tables for reporting
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            • Identify leaky funnel stages and prioritize fixes<br>
            • Improve creative budget allocation based on engagement<br>
            • Reduce CPA by scaling high-performing segments<br>
            • Create executive-ready downloads for decision meetings
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    kk1, kk2, kk3, kk4 = st.columns(4)
    kk1.markdown('<div class="kpi">Total Revenue<div class="kpi-value">—</div></div>', unsafe_allow_html=True)
    kk2.markdown('<div class="kpi">ROAS<div class="kpi-value">—</div></div>', unsafe_allow_html=True)
    kk3.markdown('<div class="kpi">Total Leads<div class="kpi-value">—</div></div>', unsafe_allow_html=True)
    kk4.markdown('<div class="kpi">Conversion Rate<div class="kpi-value">—</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Forecasting & ML</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        Revenue & conversion forecasting with RandomForest and linear fallback for short-term planning.
        Model metrics (RMSE, R²) are shown and predictions available as CSV exports.
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# Important Attributes Tab
# -------------------------
with tab_attrs:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    # Data dictionary derived from the user's required cols — descriptions tailored to the customer journey dataset
    required_dict = {
        "Date": "Event date for the record. Must be parseable to a datetime.",
        "Campaign": "Marketing campaign name or initiative identifier.",
        "Channel": "Acquisition channel or traffic source (e.g., Facebook, Google, Email).",
        "Stage": "Funnel stage label (Awareness, Consideration, Purchase, etc.).",
        "Conversion_Flag": "Binary or boolean flag indicating conversion (1/0 or True/False).",
        "Revenue": "Revenue attributable to the row (currency units).",
        "Impressions": "Number of times the creative/ad was shown.",
        "Clicks": "Number of clicks recorded.",
        "Leads": "Number of leads captured (form fills, signups).",
        "CTR": "Click-through rate = Clicks / Impressions.",
        "CPC": "Cost per click (if spend data is available).",
        "CPA": "Cost per acquisition / action (if spend and conversion available).",
        "Video_50%": "Count or % of viewers who watched 50% of the video.",
        "Video_75%": "Count or % of viewers who watched 75% of the video.",
        "Video_100%": "Count or % of viewers who watched the full video (100%).",
        "ThruPlay_Rate": "Rate of ThruPlays (views that count as completed views).",
        "Country": "Country or geography for the record.",
        "Device": "Device type (Mobile, Desktop, Tablet, etc.).",
        "AgeGroup": "Age bucket for audience segmentation.",
        "Gender": "Gender of audience segment (Male/Female/Other)."
    }

    dict_df = pd.DataFrame([{"Column": c, "Description": required_dict.get(c, "")} for c in REQUIRED_COLS])
    render_required_table(dict_df)

    st.markdown('<div class="section-title">Independent / Dependent Variables</div>', unsafe_allow_html=True)
    L, R = st.columns(2)
    with L:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        indep = ["Campaign","Channel","Date","Impressions","Clicks","Video_50%","Video_75%","Video_100%","Device","AgeGroup","Gender"]
        for v in indep:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)
    with R:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        dep = ["Leads","Conversion_Flag","Revenue","CTR","CPC","CPA","ThruPlay_Rate"]
        for v in dep:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# -------------------------
# Application Tab
# -------------------------
with tab_app:
    st.markdown('<div class="section-title">Step 1 — Load dataset</div>', unsafe_allow_html=True)

    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Mapping"], horizontal=True)
    df = None

    if mode == "Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/customer_journey_funnel_video.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df, AUTO_MAPS)
            st.success("Default dataset loaded")
            # Preview small
            st.dataframe(df.head(5), use_container_width=True)
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    elif mode == "Upload CSV":
        st.markdown("#### Download sample CSV (format reference)")
        SAMPLE_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/customer_journey_funnel_video.csv"
        try:
            sample_df = pd.read_csv(SAMPLE_URL).head(5)
            st.download_button("Download Sample CSV", sample_df.to_csv(index=False), "sample_customer_journey.csv", "text/csv")
        except Exception:
            pass

        uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df, AUTO_MAPS)
            st.success("File uploaded and columns auto-mapped where possible.")
            st.dataframe(df.head(5), use_container_width=True)

    else:
        uploaded = st.file_uploader("Upload CSV for manual mapping", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.markdown("Preview (first 5 rows)")
            st.dataframe(raw.head(5), use_container_width=True)

            st.markdown("Map your columns to required fields")
            mapping = {}
            cols = list(raw.columns)
            options = ["-- Select --"] + cols
            for req in REQUIRED_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=options, key=f"map_{req}")

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

    # -------------------------
    # Basic cleaning & derived metrics
    # -------------------------
    df = ensure_datetime(df, "Date")
    df = safe_numeric(df, ["Impressions","Clicks","Leads","Revenue","Video_50%","Video_75%","Video_100%","ThruPlay_Rate"])

    # derived metrics where possible
    if "Clicks" in df.columns and "Impressions" in df.columns:
        df["CTR"] = np.where(df["Impressions"] > 0, df["Clicks"] / df["Impressions"], 0)
    if "Revenue" in df.columns and "Leads" in df.columns:
        df["Revenue_per_Lead"] = np.where(df["Leads"]>0, df["Revenue"] / df["Leads"], np.nan)
    if "Clicks" in df.columns and "Leads" in df.columns:
        df["Conversion_Rate"] = np.where(df["Clicks"] > 0, df["Leads"] / df["Clicks"], 0)

    st.markdown('<div class="section-title">Step 2 — Filters</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2,2,1])
    campaigns = sorted(df["Campaign"].dropna().unique().tolist()) if "Campaign" in df.columns else []
    channels = sorted(df["Channel"].dropna().unique().tolist()) if "Channel" in df.columns else []

    with c1:
        sel_campaigns = st.multiselect("Campaign", options=campaigns, default=campaigns[:5])
    with c2:
        sel_channels = st.multiselect("Channel", options=channels, default=channels[:3])
    with c3:
        try:
            min_d = df["Date"].min().date()
            max_d = df["Date"].max().date()
            date_range = st.date_input("Date range", value=(min_d, max_d))
        except Exception:
            date_range = st.date_input("Date range")

    filt = df.copy()
    if sel_campaigns:
        filt = filt[filt["Campaign"].isin(sel_campaigns)]
    if sel_channels:
        filt = filt[filt["Channel"].isin(sel_channels)]
    try:
        if date_range and len(date_range) == 2:
            start_dt = pd.to_datetime(date_range[0])
            end_dt = pd.to_datetime(date_range[1])
            filt = filt[(filt["Date"] >= start_dt) & (filt["Date"] <= end_dt)]
    except Exception:
        pass

    st.markdown('<div class="section-title">Filtered preview</div>', unsafe_allow_html=True)
    st.dataframe(filt.head(5), use_container_width=True)
    download_df_bytes(filt.head(500), "filtered_customer_journey_preview.csv", label="Download filtered preview (up to 500 rows)")

    # -------------------------
    # KPIs (blue KPI cards)
    # -------------------------
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    kk1, kk2, kk3, kk4, kk5, kk6 = st.columns(6)
    total_impr = int(filt["Impressions"].sum()) if "Impressions" in filt.columns else 0
    total_clicks = int(filt["Clicks"].sum()) if "Clicks" in filt.columns else 0
    total_leads = int(filt["Leads"].sum()) if "Leads" in filt.columns else 0
    avg_conv_rate = filt["Conversion_Rate"].mean() if "Conversion_Rate" in filt.columns else 0.0
    total_revenue = float(filt["Revenue"].sum()) if "Revenue" in filt.columns else 0.0
    avg_thru = filt["ThruPlay_Rate"].mean() if "ThruPlay_Rate" in filt.columns else 0.0

    kk1.markdown(f"<div class='kpi'>Impressions<div class='kpi-value'>{total_impr:,}</div></div>", unsafe_allow_html=True)
    kk2.markdown(f"<div class='kpi'>Clicks<div class='kpi-value'>{total_clicks:,}</div></div>", unsafe_allow_html=True)
    kk3.markdown(f"<div class='kpi'>Leads<div class='kpi-value'>{total_leads:,}</div></div>", unsafe_allow_html=True)
    kk4.markdown(f"<div class='kpi'>Conversion Rate<div class='kpi-value'>{avg_conv_rate:.2%}</div></div>", unsafe_allow_html=True)
    kk5.markdown(f"<div class='kpi'>Revenue<div class='kpi-value'>{to_currency(total_revenue)}</div></div>", unsafe_allow_html=True)
    kk6.markdown(f"<div class='kpi'>ThruPlay Rate<div class='kpi-value'>{avg_thru:.2%}</div></div>", unsafe_allow_html=True)

    # -------------------------
    # Funnel visualization
    # -------------------------
    st.markdown('<div class="section-title">Funnel Stage Distribution</div>', unsafe_allow_html=True)
    if "Stage" in filt.columns and "Leads" in filt.columns:
        funnel = filt.groupby("Stage", as_index=False).agg({"Leads": "sum"}).sort_values("Leads", ascending=False)
        if not funnel.empty:
            fig_funnel = px.funnel(funnel, x="Leads", y="Stage", text="Leads")
            st.plotly_chart(fig_funnel, use_container_width=True)
        else:
            st.info("No funnel data to plot for the selected filters.")
    else:
        st.info("Stage or Leads column missing; cannot render funnel.")

    # -------------------------
    # Video engagement chart
    # -------------------------
    st.markdown('<div class="section-title">Video Engagement Metrics</div>', unsafe_allow_html=True)
    video_metrics = [c for c in ["Video_50%","Video_75%","Video_100%"] if c in filt.columns]
    if video_metrics:
        video_df = filt[video_metrics].mean().reset_index()
        video_df.columns = ["Metric", "Average"]
        fig_video = px.bar(video_df, x="Metric", y="Average", text=video_df["Average"].round(2))
        st.plotly_chart(fig_video, use_container_width=True)
    else:
        st.info("Video engagement columns missing.")

    # -------------------------
    # ML — Predict Revenue (preview + full download)
    # -------------------------
    st.markdown('<div class="section-title">ML — Predict Revenue (RandomForest)</div>', unsafe_allow_html=True)
    ml_df = filt.copy()
    if "Revenue" not in ml_df.columns or ml_df["Revenue"].dropna().shape[0] < 30:
        st.info("Not enough Revenue data to train ML model (need >=30 rows).")
    else:
        feat_cols = [c for c in ["Channel","Campaign","Impressions","Clicks","Leads","Video_50%","Video_75%","Video_100%","Device","AgeGroup","Gender"] if c in ml_df.columns]
        if len(feat_cols) < 2:
            st.info("Not enough features available to train model.")
        else:
            X = ml_df[feat_cols].copy()
            y = ml_df["Revenue"].copy()

            cat_cols = [c for c in X.columns if X[c].dtype == "object"]
            num_cols = [c for c in X.columns if c not in cat_cols]

            transformers = []
            if cat_cols:
                transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
            if num_cols:
                transformers.append(("num", StandardScaler(), num_cols))

            preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

            try:
                X_t = preprocessor.fit_transform(X)
            except Exception as e:
                st.error("Preprocessing failed: " + str(e))
                X_t = None

            if X_t is not None:
                X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)
                rf = RandomForestRegressor(n_estimators=220, random_state=42)
                with st.spinner("Training RandomForest..."):
                    rf.fit(X_train, y_train)

                preds = rf.predict(X_test)
                rmse = math.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)

                st.markdown(f"""
                    <div class="card">
                        <b>Model performance</b><br>
                        RMSE: {rmse:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; R²: {r2:.3f}
                    </div>
                """, unsafe_allow_html=True)

                out_df = pd.DataFrame({
                    "Actual_Revenue": y_test.reset_index(drop=True),
                    "Predicted_Revenue": preds
                })

                # preview only (first 10 rows) rendered with required-table
                st.markdown('<div class="section-title">ML predictions preview (first 10 rows)</div>', unsafe_allow_html=True)
                render_required_table(out_df.head(10))

                # full predictions download (features not included to keep payload small here)
                download_df_bytes(out_df.reset_index(drop=True), "ml_revenue_predictions.csv", label="Download full ML predictions (Actual vs Predicted)")

    # -------------------------
    # Forecasting (linear fallback only)
    # -------------------------
    st.markdown('<div class="section-title">Forecasting (linear fallback)</div>', unsafe_allow_html=True)
    if "Date" in filt.columns and "Revenue" in filt.columns:
        daily = filt.groupby(pd.Grouper(key="Date", freq="D"))["Revenue"].sum().reset_index().dropna()
        if daily.shape[0] >= 10:
            daily = daily.reset_index(drop=True)
            daily["t"] = np.arange(len(daily))
            lr = LinearRegression()
            lr.fit(daily[["t"]], daily["Revenue"])
            future_idx = np.arange(len(daily), len(daily)+30).reshape(-1,1)
            preds_future = lr.predict(future_idx)
            future_dates = pd.date_range(daily["Date"].max() + pd.Timedelta(days=1), periods=30)

            df_forecast = pd.DataFrame({"Date": future_dates, "Forecast_Revenue": preds_future})
            figf = go.Figure()
            figf.add_trace(go.Scatter(x=daily["Date"], y=daily["Revenue"], name="Actual"))
            figf.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["Forecast_Revenue"], name="Forecast"))
            figf.update_layout(template="plotly_white")
            st.plotly_chart(figf, use_container_width=True)
            download_df_bytes(df_forecast, "revenue_30day_forecast.csv", label="Download 30-day revenue forecast")
        else:
            st.info("Not enough daily data to produce a forecast (need >=10 days).")
    else:
        st.info("Date or Revenue column missing for forecasting.")

    # -------------------------
    # Automated Insights (table + download)
    # -------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)
    insights_list = []
    if "Channel" in filt.columns and "Revenue" in filt.columns and "Impressions" in filt.columns:
        ch_perf = filt.groupby("Channel").agg({"Revenue":"sum","Impressions":"sum","Leads":"sum"}).reset_index()
        if not ch_perf.empty:
            ch_perf["Rev_per_Impr"] = np.where(ch_perf["Impressions"]>0, ch_perf["Revenue"] / ch_perf["Impressions"], 0)
            best = ch_perf.sort_values("Rev_per_Impr", ascending=False).head(1)
            worst = ch_perf.sort_values("Rev_per_Impr", ascending=True).head(1)
            if not best.empty:
                insights_list.append({"Insight":"Best Channel Rev/Impr", "Channel": best.iloc[0]["Channel"], "Value": float(best.iloc[0]["Rev_per_Impr"])})
            if not worst.empty:
                insights_list.append({"Insight":"Lowest Channel Rev/Impr", "Channel": worst.iloc[0]["Channel"], "Value": float(worst.iloc[0]["Rev_per_Impr"])})

    if "Campaign" in filt.columns and "Leads" in filt.columns:
        camp = filt.groupby("Campaign").agg({"Leads":"sum","Revenue":"sum"}).reset_index()
        if not camp.empty:
            top = camp.sort_values("Revenue", ascending=False).head(3)
            for _, row in top.iterrows():
                insights_list.append({"Insight":"Top campaign by revenue", "Campaign": row["Campaign"], "Revenue": float(row["Revenue"]), "Leads": int(row["Leads"])})

    if insights_list:
        ins_df = pd.DataFrame(insights_list)
        render_required_table(ins_df)
        download_df_bytes(ins_df, "automated_insights.csv", label="Download automated insights")
    else:
        st.markdown('<div class="card">No automated insights available for the selected filters.</div>', unsafe_allow_html=True)
