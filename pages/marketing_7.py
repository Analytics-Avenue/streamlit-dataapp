# marketing_performance_customer_journey_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import math
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Helpers - MUST be defined before use
# -------------------------
def download_df(df: pd.DataFrame, filename: str, label: str = "Download CSV"):
    """
    Safe download: write df as UTF-8 CSV without index and expose a Streamlit download button.
    """
    if df is None or df.empty:
        st.info("No data to download.")
        return
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    b = BytesIO()
    b.write(csv_bytes)
    b.seek(0)
    st.download_button(label=label, data=b, file_name=filename, mime="text/csv")

def render_required_table(df: pd.DataFrame):
    """
    Render an index-safe table with the 'required-table' class.
    Uses pandas Styler -> HTML, then strips accidental index placeholders.
    """
    styled = df.style.set_table_attributes('class="required-table"')
    html = styled.to_html()
    html = html.replace("<th></th>", "").replace("<td></td>", "")
    st.write(html, unsafe_allow_html=True)

# Auto-column mapping dictionary (used by mapping flows)
AUTO_MAPS = {
    "Campaign": ["campaign", "campaign_name", "campaign name"],
    "Channel": ["channel", "platform", "source", "page", "page name"],
    "Date": ["date", "day", "start date", "end date"],
    "Impressions": ["impressions", "impression"],
    "Clicks": ["clicks", "link clicks", "all clicks", "total clicks"],
    "Leads": ["leads", "results", "result"],
    "Conversions": ["conversions", "purchase", "add to cart", "complete registration"],
    "Spend": ["spend", "budget", "cost", "amount spent", "amount spent (inr)"],
    "Revenue": ["revenue", "amount"],
    "ROAS": ["roas"],
    "Device": ["device", "platform"],
    "AgeGroup": ["agegroup", "age group", "age"],
    "Gender": ["gender", "sex"],
    "AdSet": ["adset", "ad set"],
    "Creative": ["creative", "ad creative"],
    "Stage": ["stage", "funnel_stage"],
    "Conversion_Flag": ["conversion_flag", "is_converted", "converted"]
}

REQUIRED_COLS = [
    "Date","Campaign","Channel","Stage","Conversion_Flag","Revenue",
    "Impressions","Clicks","Leads","CTR","CPC","CPA",
    "Video_50%","Video_75%","Video_100%","ThruPlay_Rate",
    "Country","Device","AgeGroup","Gender"
]

def auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to automatically rename columns in df according to AUTO_MAPS.
    Conservative mapping: matches if candidate or candidate substring appears in column lower-case.
    """
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

def ensure_datetime(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            pass
    return df

def safe_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def to_currency(x):
    try:
        return "₹ " + f"{float(x):,.2f}"
    except:
        return "₹ 0.00"

# -------------------------
# Page config & Header
# -------------------------
st.set_page_config(page_title="Customer Journey & Funnel Analytics", layout="wide", initial_sidebar_state="collapsed")

logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

# CSS: exact marketing lab rules: Inter font, pure black body text, blue KPIs/variable boxes,
# required-table class, fade-in, hover underline etc.
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {{
    background: #ffffff;
    color: #000000;
    font-family: 'Inter', sans-serif;
    font-size: 17px;
}}

/* Fade-in for container */
.block-container {{ animation: fadeIn 0.45s ease-in-out; }}
@keyframes fadeIn {{ from {{ opacity:0; transform: translateY(6px); }} to {{ opacity:1; transform: translateY(0); }} }}

/* Header row */
.logo-row {{ display:flex; align-items:center; gap:12px; margin-bottom:8px; }}
.app-title {{ color:#064b86; font-size:28px; font-weight:700; margin:0; }}
.app-sub {{ color:#000; margin:0; font-size:14px; }}

/* Section title */
.section-title {{
    font-size:22px;
    font-weight:500;
    color:#000;
    margin-top:18px;
    margin-bottom:10px;
    position:relative;
    display:inline-block;
}}
.section-title:hover::after {{
    content: "";
    position: absolute;
    bottom: -6px;
    left: 0;
    height: 3px;
    width: 40%;
    background: #064b86;
    border-radius: 2px;
}}

/* Cards (white, blue content text) */
.card {{
    background: #ffffff;
    color: #064b86;
    border: 1px solid #e6e6e6;
    border-radius: 13px;
    padding: 18px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.06);
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
    margin-bottom: 16px;
}}
.card:hover {{
    transform: translateY(-6px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.10);
    border-color: #064b86;
}}
.card h4, .card p, .card li {{ color: #064b86; margin:0; }}

/* KPI card */
.kpi-card {{
    background: #ffffff;
    color: #064b86;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    border: 1px solid #e6e6e6;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}}
.kpi-card:hover {{
    transform: translateY(-6px);
    box-shadow: 0 10px 28px rgba(0,0,0,0.10);
    filter: drop-shadow(0 0 8px rgba(6,75,134,0.12));
}}

/* Variable boxes */
.variable-box {{
    background: #ffffff;
    color: #064b86;
    border-radius: 12px;
    padding: 12px;
    border: 1px solid #e6e6e6;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    text-align:center;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    font-weight:600;
    margin-bottom:12px;
}}
.variable-box:hover {{
    transform: translateY(-6px);
    box-shadow: 0 18px 40px rgba(6,75,134,0.12);
    border-color: #064b86;
}}

/* Required-table: index-safe renderer target */
.required-table {{
    width:100%;
    border-collapse:collapse;
    font-size:17px;
    color:#000;
    background:#fff;
}}
.required-table thead th {{
    border-bottom:2px solid #000;
    padding:10px;
    font-weight:600;
    text-align:left;
}}
.required-table tbody td {{
    padding:10px;
    border-bottom:1px solid #efefef;
}}
.required-table tbody tr:hover {{ background:#fafafa; }}

/* Buttons */
.stButton>button, .stDownloadButton>button {{
    background:#064b86 !important;
    color:white !important;
    border-radius:8px !important;
    padding:8px 18px !important;
    font-weight:600 !important;
    border:none !important;
}}

/* Small muted helper text */
.small-muted {{ color:#666666; font-size:13px; }}

/* Keep container spacing consistent with master app */
.block-container > .main {{
    padding-left: 24px;
    padding-right: 24px;
    max-width: 1400px;
    margin-left:auto;
    margin-right:auto;
}}

</style>
""", unsafe_allow_html=True)

# Header area
st.markdown(f"""
<div class="fade-in logo-row">
    <img src="{logo_url}" width="52" style="border-radius:8px;"/>
    <div>
        <div class="app-title">Analytics Avenue & Advanced Analytics</div>
        <div class="app-sub small-muted">Customer Journey & Funnel Analytics — standardized UI system</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Tabs: Overview / Important Attributes / Application
# -------------------------
tabs = st.tabs(["Overview", "Important Attributes", "Application"])

# =======================================================================================
# TAB 1 — OVERVIEW
# =======================================================================================
with tabs[0]:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4>Purpose</h4>
        <p>
            End-to-end customer journey and funnel analytics focused on video engagement, funnel stages,
            conversion tracking and revenue predictions. Built to give performance teams actionable intelligence.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        • Funnel stage breakdowns and drop-off diagnostics<br>
        • Video engagement analysis (50% / 75% / 100% / ThruPlay)<br>
        • Predictive revenue modelling using RandomForest (with linear fallback for forecasting)<br>
        • Automated insights and export-ready CSVs for exec reports
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Impact</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        • Reduce funnel leaks by identifying weak stages and creatives<br>
        • Improve video-to-conversion flow by optimizing high-impact moments<br>
        • Increase ROAS by prioritizing channels and campaigns with positive revenue-per-spend
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown('<div class="kpi-card">Total Revenue</div>', unsafe_allow_html=True)
    k2.markdown('<div class="kpi-card">ROAS</div>', unsafe_allow_html=True)
    k3.markdown('<div class="kpi-card">Total Leads</div>', unsafe_allow_html=True)
    k4.markdown('<div class="kpi-card">Conversion Rate</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Forecasting & ML Capabilities</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        • Revenue and conversion predictions using RandomForest.<br>
        • Time-series forecasting via linear fallback (Prophet optional if installed).<br>
        • ML performance metrics (RMSE, R²) and downloadable prediction tables.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        • Channel ROI comparisons, top-performing creatives, and stage drop-off alerts.<br>
        • Exportable insights for leadership reporting.
    </div>
    """, unsafe_allow_html=True)

# =======================================================================================
# TAB 2 — IMPORTANT ATTRIBUTES
# =======================================================================================
with tabs[1]:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    # Use same content you had — content is same as your marketing_performance app
    required_dict = {
        "Date": "Date of the event/row (parseable to datetime).",
        "Campaign": "Marketing campaign identifier or name.",
        "Channel": "Traffic/acquisition channel (Facebook, Google, Email, etc.).",
        "Stage": "Funnel stage label (Awareness, Consideration, Purchase, etc.).",
        "Conversion_Flag": "Binary flag (1/0) indicating whether the row represents a conversion.",
        "Revenue": "Revenue attributable to the row/campaign.",
        "Impressions": "Number of ad impressions shown.",
        "Clicks": "Number of clicks recorded.",
        "Leads": "Leads generated / form submissions.",
        "CTR": "Click-through rate = Clicks / Impressions.",
        "CPC": "Cost per click.",
        "CPA": "Cost per acquisition (spend / conversions).",
        "Video_50%": "Count or percent who watched 50% of the video.",
        "Video_75%": "Count or percent who watched 75% of the video.",
        "Video_100%": "Count or percent who watched the full video.",
        "ThruPlay_Rate": "Rate of thruplays (plays lasting >= a threshold).",
        "Country": "Country of the user/impression.",
        "Device": "Device category (Mobile, Desktop, Tablet).",
        "AgeGroup": "Age bucket for audience segmentation.",
        "Gender": "Gender of the audience (if available)."
    }

    dict_df = pd.DataFrame([{"Column": k, "Description": v} for k, v in required_dict.items()])

    # show the required dict table with index-safe renderer
    render_required_table(dict_df)

    # Independent / Dependent variables boxes (exact two-column design)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        for v in ["Campaign", "Channel", "Date", "Impressions", "Clicks", "Video_50%", "Video_75%", "Video_100%"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        for v in ["Leads", "Conversions", "Revenue", "ThruPlay_Rate", "CTR", "CPA"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# =======================================================================================
# TAB 3 — APPLICATION
# =======================================================================================
with tabs[2]:
    st.markdown('<div class="section-title">Application</div>', unsafe_allow_html=True)

    # Step 1 – Load dataset (three modes)
    st.markdown('<div class="section-title">Step 1 — Load dataset</div>', unsafe_allow_html=True)
    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Mapping"], horizontal=True)
    df = None

    if mode == "Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/customer_journey_funnel_video.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            # attempt auto mapping (non-destructive)
            df = auto_map_columns(df)
            st.success("Default dataset loaded")
            render_required_table(df.head(5))
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    elif mode == "Upload CSV":
        st.markdown("#### Download Sample CSV for Reference")
        SAMPLE_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/customer_journey_funnel_video.csv"
        try:
            sample_df = pd.read_csv(SAMPLE_URL).head(5)
            st.download_button("Download Sample CSV", sample_df.to_csv(index=False), "sample_customer_journey.csv", "text/csv")
        except Exception:
            pass

        uploaded = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("Uploaded dataset loaded")
            render_required_table(df.head(5))

    else:
        uploaded = st.file_uploader("Upload CSV for manual mapping", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.markdown("### Preview (first 5 rows)")
            render_required_table(raw.head(5))

            st.markdown("### Map your columns to required fields")
            mapping = {}
            cols = list(raw.columns)
            options = ["-- Select --"] + cols
            for req in REQUIRED_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=options, key=f"map_{req}")

            if st.button("Apply mapping"):
                missing = [k for k,v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    # invert mapping: user picks raw-col as value for each required key
                    inv = {v:k for k,v in mapping.items()}
                    df = raw.rename(columns=inv)
                    st.success("Mapping applied successfully.")
                    render_required_table(df.head(5))

    # stop if no df
    if df is None:
        st.stop()

    # keep only columns that exist (don't crash if optional columns missing)
    # ensure datatypes and derived metrics
    df = ensure_datetime(df, "Date")
    df = safe_numeric(df, ["Impressions","Clicks","Leads","Revenue","Video_50%","Video_75%","Video_100%","ThruPlay_Rate","CTR","CPC","CPA"])
    # Derived metrics: Conversion_Rate if possible
    if "Clicks" in df.columns and "Leads" in df.columns:
        df["Conversion_Rate"] = np.where(df["Clicks"]>0, df["Leads"]/df["Clicks"], 0)
    else:
        df["Conversion_Rate"] = 0

    st.markdown('<div class="section-title">Step 2 — Filters & Preview</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2,2,1])
    campaigns = sorted(df["Campaign"].dropna().unique().tolist()) if "Campaign" in df.columns else []
    channels = sorted(df["Channel"].dropna().unique().tolist()) if "Channel" in df.columns else []

    with c1:
        sel_campaigns = st.multiselect("Campaign", options=campaigns, default=campaigns[:5])
    with c2:
        sel_channels = st.multiselect("Channel", options=channels, default=channels[:3])
    with c3:
        try:
            min_date = df["Date"].min().date()
            max_date = df["Date"].max().date()
            date_range = st.date_input("Date range", value=(min_date, max_date))
        except Exception:
            date_range = st.date_input("Date range")

    filt = df.copy()
    if sel_campaigns:
        filt = filt[filt["Campaign"].isin(sel_campaigns)]
    if sel_channels:
        filt = filt[filt["Channel"].isin(sel_channels)]
    try:
        if date_range and len(date_range) == 2:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            filt = filt[(filt["Date"] >= start) & (filt["Date"] <= end)]
    except Exception:
        pass

    st.markdown("Filtered preview (first 10 rows)")
    render_required_table(filt.head(10))
    download_df(filt.head(500), "filtered_customer_journey_preview.csv", label="Download filtered preview (up to 500 rows)")

    # KPIs (blue KPI cards)
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    kcols = st.columns(6)
    try:
        total_impr = int(filt["Impressions"].sum()) if "Impressions" in filt.columns else 0
    except:
        total_impr = 0
    try:
        total_clicks = int(filt["Clicks"].sum()) if "Clicks" in filt.columns else 0
    except:
        total_clicks = 0
    try:
        total_leads = int(filt["Leads"].sum()) if "Leads" in filt.columns else 0
    except:
        total_leads = 0
    try:
        conv_rate_mean = filt["Conversion_Rate"].mean() if "Conversion_Rate" in filt.columns else 0
    except:
        conv_rate_mean = 0
    try:
        revenue_sum = float(filt["Revenue"].sum()) if "Revenue" in filt.columns else 0.0
    except:
        revenue_sum = 0.0
    try:
        thruplay_rate = filt["ThruPlay_Rate"].mean() if "ThruPlay_Rate" in filt.columns else 0
    except:
        thruplay_rate = 0

    kcols[0].markdown(f"<div class='kpi-card'>Total Impressions<br><span style='font-size:18px'>{total_impr:,}</span></div>", unsafe_allow_html=True)
    kcols[1].markdown(f"<div class='kpi-card'>Total Clicks<br><span style='font-size:18px'>{total_clicks:,}</span></div>", unsafe_allow_html=True)
    kcols[2].markdown(f"<div class='kpi-card'>Total Leads<br><span style='font-size:18px'>{total_leads:,}</span></div>", unsafe_allow_html=True)
    kcols[3].markdown(f"<div class='kpi-card'>Conversion Rate<br><span style='font-size:18px'>{conv_rate_mean:.2%}</span></div>", unsafe_allow_html=True)
    kcols[4].markdown(f"<div class='kpi-card'>Revenue<br><span style='font-size:18px'>{to_currency(revenue_sum)}</span></div>", unsafe_allow_html=True)
    kcols[5].markdown(f"<div class='kpi-card'>ThruPlay Rate<br><span style='font-size:18px'>{thruplay_rate:.2%}</span></div>", unsafe_allow_html=True)

    # Charts & EDA
    st.markdown('<div class="section-title">Charts & EDA</div>', unsafe_allow_html=True)

    # Funnel Stage Distribution
    if "Stage" in filt.columns and "Leads" in filt.columns:
        funnel = filt.groupby("Stage").agg({"Leads":"sum"}).reset_index()
        if not funnel.empty:
            fig_funnel = px.funnel(funnel, x="Leads", y="Stage", text="Leads")
            st.plotly_chart(fig_funnel, use_container_width=True)
        else:
            st.info("No funnel data to plot.")
    else:
        st.info("Stage or Leads column missing for funnel visualization.")

    # Video engagement
    if all(c in filt.columns for c in ["Video_50%","Video_75%","Video_100%"]):
        video_df = filt[["Video_50%","Video_75%","Video_100%"]].mean().reset_index()
        video_df.columns = ["Metric","Average"]
        fig_video = px.bar(video_df, x="Metric", y="Average", text=video_df["Average"].round(2))
        st.plotly_chart(fig_video, use_container_width=True)
    else:
        st.info("Video engagement columns missing (Video_50%, Video_75%, Video_100%).")

    # ML — Predict Revenue (RandomForest)
    st.markdown('<div class="section-title">ML — Predict Revenue (RandomForest)</div>', unsafe_allow_html=True)

    ml_df = filt.copy()
    if "Revenue" not in ml_df.columns or ml_df["Revenue"].dropna().shape[0] < 30:
        st.info("Not enough data to train ML model (need >=30 rows with Revenue).")
    else:
        # features
        feat_cols = ["Channel","Campaign","Impressions","Clicks","Leads","Video_50%","Video_75%","Video_100%"]
        feat_cols = [c for c in feat_cols if c in ml_df.columns]
        if len(feat_cols) < 2:
            st.info("Not enough feature columns available for training.")
        else:
            X = ml_df[feat_cols].copy()
            y = ml_df["Revenue"].copy()

            cat_cols = [c for c in X.columns if X[c].dtype == "object"]
            num_cols = [c for c in X.columns if c not in cat_cols]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols) if cat_cols else ("noop_cat","passthrough", []),
                    ("num", StandardScaler(), num_cols) if num_cols else ("noop_num","passthrough", [])
                ],
                remainder="drop"
            )

            try:
                X_t = preprocessor.fit_transform(X)
            except Exception as e:
                st.error("Preprocessing failed: " + str(e))
                X_t = None

            if X_t is not None:
                X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)
                model = RandomForestRegressor(n_estimators=200, random_state=42)
                with st.spinner("Training RandomForest..."):
                    model.fit(X_train, y_train)

                preds = model.predict(X_test)
                rmse = math.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)

                st.markdown(f"""
                <div class="card">
                    <h4>Model Performance</h4>
                    RMSE: {rmse:.2f}<br>
                    R² Score: {r2:.3f}
                </div>
                """, unsafe_allow_html=True)

                # Build out_df for preview + full download
                out_df = pd.DataFrame({
                    "Actual_Revenue": y_test.reset_index(drop=True),
                    "Predicted_Revenue": preds
                })

                # preview only (first 10 rows) using required renderer
                st.markdown('<div class="section-title">ML Predictions Preview</div>', unsafe_allow_html=True)
                render_required_table(out_df.head(10))

                # provide download for full predictions
                download_df(out_df.reset_index(drop=True), "ml_revenue_predictions_full.csv", label="Download full ML predictions")

    # Forecasting (linear fallback)
    st.markdown('<div class="section-title">Forecasting (30-day linear fallback)</div>', unsafe_allow_html=True)
    if "Date" in filt.columns and "Revenue" in filt.columns:
        daily = filt.groupby(pd.Grouper(key="Date", freq="D"))["Revenue"].sum().reset_index().dropna()
        if daily.shape[0] >= 10:
            daily = daily.reset_index(drop=True)
            daily["t"] = np.arange(len(daily))
            lr = LinearRegression()
            lr.fit(daily[["t"]], daily["Revenue"])
            future_idx = np.arange(len(daily), len(daily)+30).reshape(-1,1)
            preds_future = lr.predict(future_idx)
            future_dates = pd.date_range(daily["Date"].max()+pd.Timedelta(days=1), periods=30)
            df_forecast = pd.DataFrame({"Date": future_dates, "Forecast_Revenue": preds_future})
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily["Date"], y=daily["Revenue"], name="Actual"))
            fig.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["Forecast_Revenue"], name="Forecast"))
            fig.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
            download_df(df_forecast, "revenue_30day_forecast.csv", label="Download 30-day forecast")
        else:
            st.info("Not enough daily data (>=10 days required) to produce forecast.")
    else:
        st.info("Date or Revenue column missing for forecasting.")

    # Automated Insights
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)
    insights = []

    if "Channel" in filt.columns and "Revenue" in filt.columns and "Leads" in filt.columns:
        ch_perf = filt.groupby("Channel")[["Revenue","Leads"]].sum().reset_index()
        if not ch_perf.empty:
            ch_perf["Revenue_per_Lead"] = np.where(ch_perf["Leads"]>0, ch_perf["Revenue"]/ch_perf["Leads"], 0)
            best = ch_perf.sort_values("Revenue_per_Lead", ascending=False).iloc[0]
            worst = ch_perf.sort_values("Revenue_per_Lead", ascending=True).iloc[0]
            insights.append({"Insight":"Best Channel (rev/lead)","Channel":best["Channel"],"Value":float(best["Revenue_per_Lead"])})
            insights.append({"Insight":"Weak Channel (rev/lead)","Channel":worst["Channel"],"Value":float(worst["Revenue_per_Lead"])})

    if "Creative" in filt.columns and "Revenue" in filt.columns:
        cr_perf = filt.groupby("Creative")[["Revenue"]].sum().reset_index()
        if not cr_perf.empty:
            top_cre = cr_perf.sort_values("Revenue", ascending=False).iloc[0]
            insights.append({"Insight":"Top Creative by Revenue","Creative":top_cre["Creative"], "Revenue": float(top_cre["Revenue"])})

    if insights:
        ins_df = pd.DataFrame(insights)
        render_required_table(ins_df)
        download_df(ins_df, "automated_insights.csv", label="Download automated insights")
    else:
        st.markdown('<div class="card"><div class="small-muted">No automated insights available for selected filters.</div></div>', unsafe_allow_html=True)

    # Export final filtered dataset
    st.markdown('<div class="section-title">Exports</div>', unsafe_allow_html=True)
    download_df(filt.reset_index(drop=True), "customer_journey_filtered_full.csv", label="Download full filtered dataset (CSV)")

# End of file
