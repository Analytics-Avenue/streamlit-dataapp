# marketing_customer_journey_app.py
# Customer Journey & Funnel Analytics
# Rebuilt to follow the exact Marketing Intelligence & Forecasting Lab UI system:
# - Inter font, pure black body text (#000)
# - Blue (#064b86) for KPI cards and variable boxes
# - White background everywhere
# - Cards, KPI cards, variable boxes, section titles, fade-in animation
# - Index-safe required-table renderer for important tables
#
# Content kept exactly as your provided overview/application content.
# Upload + Mapping, Filters, KPIs, Charts, ML preview + full-download implemented.
# Preview-only ML table shown on UI; full ML predictions available for download.

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
# Config & tokens
# -------------------------
st.set_page_config(page_title="Customer Journey & Funnel Analytics", layout="wide", initial_sidebar_state="collapsed")

BLUE = "#064b86"
BLACK = "#000000"
BASE_FONT_SIZE = 17  # between 16-17px requested; using 17

REQUIRED_COLS = [
    "Date","Campaign","Channel","Stage","Conversion_Flag","Revenue",
    "Impressions","Clicks","Leads","CTR","CPC","CPA",
    "Video_50%","Video_75%","Video_100%","ThruPlay_Rate",
    "Country","Device","AgeGroup","Gender"
]

# -------------------------
# Helpers (must be above usage)
# -------------------------
def to_currency(x):
    try:
        return "₹ " + f"{float(x):,.2f}"
    except:
        return "₹ 0.00"

def download_df(df: pd.DataFrame, filename: str, label: str = "Download CSV"):
    if df is None or df.empty:
        st.info("No data to download.")
        return
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")

def ensure_datetime(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            pass
    return df

def safe_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def render_required_table(df: pd.DataFrame):
    """
    Index-safe HTML renderer for required tables.
    Uses class 'required-table' for CSS styling defined below.
    """
    styled = df.style.set_table_attributes('class="required-table"')
    html = styled.to_html()
    html = html.replace("<th></th>", "").replace("<td></td>", "")
    st.write(html, unsafe_allow_html=True)

# -------------------------
# UI CSS: Follow master spec strictly
# -------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* {{ font-family: 'Inter', sans-serif; }}
html, body, [data-testid="stAppViewContainer"] {{ background:#ffffff; color:{BLACK}; font-size:{BASE_FONT_SIZE}px; }}
.block-container {{ animation: fadeIn 0.45s ease; }}

/* Fade-in */
@keyframes fadeIn {{ from {{opacity:0; transform:translateY(8px);}} to {{opacity:1; transform:translateY(0);}} }}

/* Header */
.logo-row {{ display:flex; align-items:center; gap:12px; margin-bottom:8px; }}
.app-title {{ color:{BLUE}; font-size:28px; font-weight:700; margin:0; }}
.app-subtitle {{ color:{BLACK}; margin:0; font-size:14px; }}

/* Section title */
.section-title {{
    font-size:22px;
    font-weight:500;
    color:{BLACK};
    margin-top:18px;
    margin-bottom:10px;
    position:relative;
    display:inline-block;
}}
.section-title::after {{
    content: "";
    position: absolute;
    bottom: -6px;
    left: 0;
    height: 3px;
    width: 0%;
    background: {BLUE};
    transition: width 0.35s ease;
}}
.section-title:hover::after {{ width:40%; }}

/* Card - glass style white card (card content blue) */
.card {{
    background: #ffffff;
    color: {BLUE};
    border: 1px solid #e6e6e6;
    border-radius: 13px;
    padding: 18px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.06);
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
    margin-bottom: 14px;
}}
.card:hover {{
    transform: translateY(-6px);
    box-shadow: 0 14px 36px rgba(0,0,0,0.10);
    border-color: {BLUE};
    cursor: pointer;
}}
.card h4, .card p, .card li {{ color:{BLUE}; margin:0; }}

/* KPI card */
.kpi-card {{
    background: #ffffff;
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    border: 1px solid #e6e6e6;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    font-weight:600;
    color: {BLUE};
}}
.kpi-card:hover {{
    transform: translateY(-6px);
    box-shadow: 0 16px 36px rgba(6,75,134,0.14);
    cursor: pointer;
}}
.kpi-card .val {{ display:block; margin-top:6px; font-size:18px; color:{BLUE}; }}

/* Variable box (two-column) */
.variable-box {{
    background:#ffffff;
    color:{BLUE};
    border-radius:12px;
    padding:12px;
    border:1px solid #e6e6e6;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    transition: transform 0.18s ease;
    text-align:center;
    font-weight:600;
}}
.variable-box:hover {{
    transform: translateY(-6px);
    box-shadow: 0 18px 36px rgba(6,75,134,0.12);
    border-color:{BLUE};
}}

/* Required table - index safe */
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
    font-weight:600;
}}
.required-table tbody td {{
    padding:10px;
    border-bottom:1px solid #f2f2f2;
}}
.required-table tbody tr:hover {{ background:#fafafa; }}

/* Buttons */
.stButton>button, .stDownloadButton>button {{
    background:{BLUE} !important;
    color:#fff !important;
    border-radius:8px !important;
    padding:8px 18px !important;
    font-weight:600 !important;
    border:none !important;
}}

/* Small muted text */
.small-muted {{ color:#666666; font-size:13px; }}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div class="logo-row">
    <img src="{logo_url}" width="52" style="border-radius:8px;"/>
    <div>
        <div class="app-title">Analytics Avenue & Advanced Analytics</div>
        <div class="app-subtitle small-muted">Customer Journey & Funnel Analytics — standardized UI</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Tabs: Overview / Important Attributes / Application
# -------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# -------------------------
# TAB 1: Overview (content same as provided)
# -------------------------
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        This app delivers <b>end-to-end marketing performance tracking</b>, across campaigns, channels, creatives, and audience segments. 
        It aggregates campaign data, measures effectiveness, predicts revenue and conversions using <b>machine learning</b>, 
        and provides <b>forecasting</b> for short- and medium-term decision-making. 
        Built for <b>data-driven marketing teams</b>, the app gives actionable insights at a glance.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        • Multi-channel campaign tracking with breakdowns by channel, device, audience segment<br>
        • Audience analysis by Age, Gender, Device, and other demographic segments<br>
        • Creative performance insights: AdSet & Creative level ROI<br>
        • Predictive analytics: Revenue & Conversion forecasting using <b>RandomForest</b> & <b>Linear Regression</b><br>
        • Campaign optimization suggestions & ROI comparisons<br>
        • Automated insights highlighting best and worst-performing segments
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Impact</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        • Make <b>data-driven marketing decisions</b> faster<br>
        • Identify high-ROI campaigns & avoid wasted spend<br>
        • Prioritize channels, creatives, and audience segments based on predicted performance<br>
        • Improve conversion efficiency and revenue per spend unit<br>
        • Align marketing strategy with real-time insights and predictive trends
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown('<div class="kpi-card">Total Revenue<span class="val">—</span></div>', unsafe_allow_html=True)
    k2.markdown('<div class="kpi-card">ROAS<span class="val">—</span></div>', unsafe_allow_html=True)
    k3.markdown('<div class="kpi-card">Total Leads<span class="val">—</span></div>', unsafe_allow_html=True)
    k4.markdown('<div class="kpi-card">Conversion Rate<span class="val">—</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Forecasting & ML Capabilities</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        • Revenue & Conversion predictions using <b>RandomForest Regression</b><br>
        • Trend forecasting for next 30 days with <b>linear regression fallback</b><br>
        • Automatic identification of top-performing campaigns, channels, and audience segments<br>
        • Model performance metrics (R², RMSE) displayed for transparency and trust<br>
        • Downloadable ML predictions (Actual vs Predicted + features) for further analysis
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        • Channel-level ROI comparisons<br>
        • Identification of best and worst performing channels, creatives, and segments<br>
        • Downloadable insights tables for executive reporting<br>
        • Supports multi-dimensional filtering for campaigns, channels, device types, age-groups, and gender
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Who Should Use This App?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        • <b>Marketing Analysts</b> who want predictive insights and campaign breakdowns<br>
        • <b>CMOs / Marketing Heads</b> needing executive-ready dashboards<br>
        • <b>Digital Marketing Teams</b> optimizing ad spend across channels<br>
        • <b>Growth Teams</b> tracking conversion efficiency and revenue trends
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# TAB 2: Important Attributes (data dictionary, variables)
# -------------------------
with tab2:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    # Data dictionary descriptions (aligned with prior Marketing app wording)
    required_dict = {
        "Date": "Date of the campaign record. Must be parseable as datetime.",
        "Campaign": "Name of the marketing campaign or initiative.",
        "Channel": "Acquisition platform (Facebook, Google, Email, Organic, etc.).",
        "Stage": "Funnel stage label (e.g., Awareness, Consideration, Conversion).",
        "Conversion_Flag": "Binary or flag indicating whether the row represents a conversion event.",
        "Revenue": "Revenue attributed to the campaign/activity for the row.",
        "Impressions": "Total number of ad impressions.",
        "Clicks": "Number of user clicks on ads.",
        "Leads": "Number of users who expressed interest (form submits, signups).",
        "CTR": "Click-through rate = Clicks / Impressions.",
        "CPC": "Cost per click = Spend / Clicks (if Spend column available).",
        "CPA": "Cost per acquisition = Spend / Conversions or Leads.",
        "Video_50%": "Percent of viewers who watched 50% of the video (or count normalized).",
        "Video_75%": "Percent of viewers who watched 75% of the video (or count normalized).",
        "Video_100%": "Percent of viewers who watched entire video (or count normalized).",
        "ThruPlay_Rate": "Rate of completing video views (ThruPlays / Impressions or similar).",
        "Country": "Country of the user/session.",
        "Device": "Device category (Desktop, Mobile, Tablet, etc.).",
        "AgeGroup": "Age bucket of the audience (e.g., 18-24, 25-34).",
        "Gender": "Gender of the audience segment where available."
    }

    dict_df = pd.DataFrame([{"Column": k, "Description": v} for k, v in required_dict.items()])

    # Render the data dictionary using the required-table renderer
    render_required_table(dict_df)

    # Independent / Dependent variable boxes
    st.markdown('<div class="section-title">Independent / Dependent Variables</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="variable-box"><b>Independent Variables</b><br>Campaign, Channel, Device, AgeGroup, Gender, Impressions, Clicks, Video metrics</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="variable-box"><b>Dependent Variables</b><br>Leads, Conversions, Revenue, ROAS, CTR, CPA</div>', unsafe_allow_html=True)

# -------------------------
# TAB 3: Application (load, mapping, filters, KPIs, charts, ML preview + download, insights, export)
# -------------------------
with tab3:
    st.markdown('<div class="section-title">Application</div>', unsafe_allow_html=True)

    # Step 1 — Load dataset (3 modes)
    st.markdown('<div class="section-title">Step 1 — Load dataset</div>', unsafe_allow_html=True)
    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Mapping"], horizontal=True)
    df = None

    if mode == "Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/customer_journey_funnel_video.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            st.success("Default dataset loaded")
            # show small preview
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
            st.success("Uploaded dataset loaded")
            render_required_table(df.head(5))

    else:  # Upload + Mapping
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
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
                missing = [k for k, v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    # invert mapping to rename raw->required name
                    inv = {v: k for k, v in mapping.items()}
                    df = raw.rename(columns=inv)
                    st.success("Mapping applied.")
                    render_required_table(df.head(5))

    # If no dataframe loaded, stop
    if df is None:
        st.stop()

    # Data cleaning & derived metrics (keeps original columns)
    df = ensure_datetime(df, "Date")
    df = safe_numeric(df, ["Impressions","Clicks","Leads","Revenue","Video_50%","Video_75%","Video_100%","ThruPlay_Rate","CTR","CPC","CPA"])
    # Derived
    if "Clicks" in df.columns and "Leads" in df.columns:
        df["Conversion_Rate"] = np.where(df["Clicks"]>0, df["Leads"]/df["Clicks"], 0)
    else:
        df["Conversion_Rate"] = 0

    # Step 2 — Filters & preview
    st.markdown('<div class="section-title">Step 2 — Filters & preview</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2,2,1])
    campaigns = sorted(df["Campaign"].dropna().unique()) if "Campaign" in df.columns else []
    channels = sorted(df["Channel"].dropna().unique()) if "Channel" in df.columns else []
    devices = sorted(df["Device"].dropna().unique()) if "Device" in df.columns else []

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

    # apply date filter safely
    try:
        if date_range and len(date_range) == 2:
            start = pd.to_datetime(date_range[0])
            end = pd.to_datetime(date_range[1])
            if "Date" in filt.columns:
                filt = filt[(filt["Date"] >= start) & (filt["Date"] <= end)]
    except Exception:
        pass

    st.markdown("Preview (first 5 rows)")
    render_required_table(filt.head(5))
    download_df(filt.head(500), "filtered_customer_journey_preview.csv", label="Download filtered preview (up to 500 rows)")

    # Key Metrics (KPI cards)
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    kk1, kk2, kk3, kk4, kk5, kk6 = st.columns(6)
    total_impr = int(filt["Impressions"].sum()) if "Impressions" in filt.columns else 0
    total_clicks = int(filt["Clicks"].sum()) if "Clicks" in filt.columns else 0
    total_leads = int(filt["Leads"].sum()) if "Leads" in filt.columns else 0
    avg_conv = filt["Conversion_Rate"].mean() if "Conversion_Rate" in filt.columns else 0
    total_rev = float(filt["Revenue"].sum()) if "Revenue" in filt.columns else 0.0
    thruplay = filt["ThruPlay_Rate"].mean() if "ThruPlay_Rate" in filt.columns else 0

    kk1.markdown(f'<div class="kpi-card">Total Impressions<span class="val">{total_impr:,}</span></div>', unsafe_allow_html=True)
    kk2.markdown(f'<div class="kpi-card">Total Clicks<span class="val">{total_clicks:,}</span></div>', unsafe_allow_html=True)
    kk3.markdown(f'<div class="kpi-card">Total Leads<span class="val">{total_leads:,}</span></div>', unsafe_allow_html=True)
    kk4.markdown(f'<div class="kpi-card">Conversion Rate<span class="val">{avg_conv:.2%}</span></div>', unsafe_allow_html=True)
    kk5.markdown(f'<div class="kpi-card">Revenue<span class="val">{to_currency(total_rev)}</span></div>', unsafe_allow_html=True)
    kk6.markdown(f'<div class="kpi-card">ThruPlay Rate<span class="val">{thruplay:.2%}</span></div>', unsafe_allow_html=True)

    # Charts & EDA
    st.markdown('<div class="section-title">Funnel Stage Distribution</div>', unsafe_allow_html=True)
    if "Stage" in filt.columns and "Leads" in filt.columns:
        funnel = filt.groupby("Stage").agg({"Leads":"sum"}).reset_index().sort_values("Leads", ascending=False)
        if not funnel.empty:
            fig_funnel = px.funnel(funnel, x="Leads", y="Stage", text="Leads")
            st.plotly_chart(fig_funnel, use_container_width=True)
        else:
            st.info("No funnel stage data available for selected filters.")
    else:
        st.info("Stage or Leads column missing for funnel chart.")

    st.markdown('<div class="section-title">Video Engagement Metrics</div>', unsafe_allow_html=True)
    if set(["Video_50%","Video_75%","Video_100%"]).issubset(set(filt.columns)):
        video_df = filt[["Video_50%","Video_75%","Video_100%"]].mean().reset_index()
        video_df.columns = ["Metric","Average"]
        fig_video = px.bar(video_df, x="Metric", y="Average", text=video_df["Average"].round(2))
        st.plotly_chart(fig_video, use_container_width=True)
    else:
        st.info("Video engagement columns missing for video metrics chart.")

    # ML — Predict Revenue (RandomForest)
    st.markdown('<div class="section-title">ML — Predict Revenue (RandomForest)</div>', unsafe_allow_html=True)
    ml_df = filt.copy()
    if "Revenue" not in ml_df.columns:
        st.info("Revenue column missing — cannot train ML model.")
    else:
        ml_df = ml_df.dropna(subset=["Revenue"])
        # minimal feature set (only if present)
        feat_cols = [c for c in ["Channel","Campaign","Impressions","Clicks","Leads","Video_50%","Video_75%","Video_100%"] if c in ml_df.columns]
        if ml_df.shape[0] < 30 or len(feat_cols) < 2:
            st.info("Not enough data to train ML model (>=30 rows and >1 feature needed).")
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
                rf = RandomForestRegressor(n_estimators=200, random_state=42)
                with st.spinner("Training RandomForest..."):
                    rf.fit(X_train, y_train)

                preds = rf.predict(X_test)
                rmse = math.sqrt(((y_test - preds) ** 2).mean())
                r2 = rf.score(X_test, y_test)

                # Performance card
                st.markdown(f"""
                <div class="card">
                    <h4>Model Performance</h4>
                    RMSE: {rmse:.2f}<br>
                    R² Score: {r2:.3f}
                </div>
                """, unsafe_allow_html=True)

                # Build out_df with actual & predicted
                out_df = pd.DataFrame({
                    "Actual_Revenue": y_test.reset_index(drop=True),
                    "Predicted_Revenue": preds
                })

                # preview only (first 10 rows)
                st.markdown('<div class="section-title">ML Predictions Preview (first 10 rows)</div>', unsafe_allow_html=True)
                render_required_table(out_df.head(10))

                # prepare full downloadable predictions: include feature columns where possible
                # attempt to reconstruct feature names for X_test
                try:
                    # If OneHotEncoder used, get feature names (best-effort)
                    feature_names = []
                    if cat_cols:
                        ohe = preprocessor.named_transformers_.get("cat")
                        if hasattr(ohe, "get_feature_names_out"):
                            cat_names = list(ohe.get_feature_names_out(cat_cols))
                        else:
                            cat_names = [f"cat_{i}" for i in range(len(cat_cols))]
                        feature_names.extend(cat_names)
                    feature_names.extend(num_cols)
                    X_test_df = pd.DataFrame(X_test, columns=[f"feat_{i}" for i in range(X_test.shape[1])])
                except Exception:
                    X_test_df = pd.DataFrame(X_test, columns=[f"feat_{i}" for i in range(X_test.shape[1])])

                full_ml_df = pd.concat([out_df.reset_index(drop=True), X_test_df.reset_index(drop=True)], axis=1)
                download_df(full_ml_df, "ml_revenue_predictions.csv", label="Download full ML predictions (CSV)")

    # Forecasting (linear fallback only)
    st.markdown('<div class="section-title">Forecasting (30-day, linear fallback)</div>', unsafe_allow_html=True)
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
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily["Date"], y=daily["Revenue"], name="Actual"))
            fig.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["Forecast_Revenue"], name="Forecast"))
            fig.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
            download_df(df_forecast, "revenue_30day_forecast.csv", label="Download 30-day forecast")
        else:
            st.info("Not enough daily data (>=10) to produce forecast.")
    else:
        st.info("Date or Revenue column missing for forecasting.")

    # Automated Insights table and download
    st.markdown('<div class="section-title">Automated Insights Table</div>', unsafe_allow_html=True)
    if set(["Channel","Campaign","Leads","Revenue"]).issubset(set(filt.columns)):
        insights_df = filt.groupby(["Channel", "Campaign"]).agg({
            "Leads": "sum",
            "Revenue": "sum",
            "Conversion_Rate": "mean"
        }).reset_index().sort_values("Revenue", ascending=False)
        render_required_table(insights_df.head(200))
        download_df(insights_df, "automated_insights.csv", label="Download automated insights")
    else:
        st.info("Required columns (Channel, Campaign, Leads, Revenue) missing for automated insights.")

    # Exports
    st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
    download_df(filt, "marketing_customer_journey_filtered.csv", label="Download full filtered dataset (CSV)")

# End of file
