import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import math
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Helpers - MUST BE DEFINED BEFORE ANY USAGE
# -------------------------
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
    "Creative": ["creative", "ad creative"]
}

REQUIRED_MARKETING_COLS = [
    "Campaign", "Channel", "Date", "Impressions", "Clicks", "Leads",
    "Conversions", "Spend", "Revenue", "ROAS", "Device", "AgeGroup",
    "Gender", "AdSet", "Creative"
]

def auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to auto-rename columns based on AUTO_MAPS.
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

def download_df(df: pd.DataFrame, filename: str, label: str = "Download CSV"):
    """
    Download button for DataFrame. Removes index and writes UTF-8 CSV to buffer.
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
    Render index-safe table using unified font color + size,
    fully overriding Streamlit table styles.
    """
    if df is None:
        st.info("No table to render.")
        return
    styled = (
        df.style
        .set_table_attributes('class="required-table"')
        .set_properties(**{
            "color": "#000000",
            "font-size": "17px",
            "text-align": "left"
        })
        .set_table_styles([
            {"selector": "th", "props": [("color", "#000000"), ("font-size", "18px"), ("font-weight", "600"), ("text-align", "left")]},
            {"selector": "td", "props": [("color", "#000000"), ("font-size", "17px"), ("text-align", "left")]},
        ])
    )
    html = styled.to_html()
    html = html.replace("<th></th>", "").replace("<td></td>", "")
    st.write(html, unsafe_allow_html=True)

def to_currency(x):
    try:
        return "₹ " + f"{float(x):,.2f}"
    except:
        return "₹ 0.00"

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Marketing Performance Analysis", layout="wide")

# -------------------------
# Header & Logo (sample layout)
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

# -------------------------
# Global CSS — EXACT from your sample
# -------------------------
st.markdown("""
<style>
* { font-family: 'Inter', sans-serif; }

/* PURE BLACK global text */
body, [class*="css"] { color:#000 !important; font-size:17px; }

/* Section Title */
.section-title {
    font-size: 24px !important;
    font-weight: 600 !important;
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

/* Cards (pure black text) */
.card {
    background:#ffffff;
    padding:22px;
    border-radius:14px;
    border:1px solid #e6e6e6;
    font-size:16.5px;
    color:#000 !important;
    font-weight:500;
    box-shadow:0 3px 14px rgba(0,0,0,0.08);
    transition:0.25s ease;
}
.card:hover {
    transform:translateY(-4px);
    box-shadow:0 12px 25px rgba(6,75,134,0.18);
    border-color:#064b86;
}

/* KPI CARDS (blue text) */
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
.kpi:hover {
    transform:translateY(-4px);
    box-shadow:0 13px 26px rgba(6,75,134,0.20);
    border-color:#064b86;
}

/* Variable boxes (blue text) */
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
.variable-box:hover {
    transform:translateY(-5px);
    box-shadow:0 12px 22px rgba(6,75,134,0.18);
    border-color:#064b86;
}

/* Required-table CSS (index-safe renderer uses this class) */
.required-table {
    width:100%;
    border-collapse:collapse;
    font-size:17px;
    color:#000;
    background:#fff;
}
.required-table thead th {
    border-bottom:2px solid #000;
    padding:10px;
    font-weight:600;
}
.required-table tbody td {
    padding:10px;
    border-bottom:1px solid #efefef;
}
.required-table tbody tr:hover { background:#fafafa; }

/* Table styling fallback */
.dataframe th {
    background:#064b86 !important;
    color:#fff !important;
    padding:11px !important;
    font-size:15.5px !important;
}
.dataframe td {
    font-size:15.5px !important;
    color:#000 !important;
    padding:9px !important;
    border-bottom:1px solid #efefef !important;
}
.dataframe tbody tr:hover { background:#f4f9ff !important; }

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
.stButton>button:hover, .stDownloadButton>button:hover {
    transform:translateY(-3px);
    background:#0a6eb3 !important;
}

/* Fade-in animation */
.block-container { animation: fadeIn 0.5s ease; }
@keyframes fadeIn { 
    from {opacity:0; transform:translateY(10px);} 
    to {opacity:1; transform:translateY(0);} 
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# UI small helpers
# -------------------------
def to_money(v):
    try:
        return "₹ " + format(float(v), ",.2f")
    except:
        return "₹ 0.00"

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# =======================================================================================
# TAB 1 — OVERVIEW
# =======================================================================================
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    A complete marketing performance intelligence system that unifies channel data, 
    produces insights, forecasts revenue, and provides ML-based predictions.
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Multi-channel analytics<br>
        • KPI-driven performance benchmarking<br>
        • Machine Learning revenue prediction<br>
        • Spend optimization recommendations<br>
        • Automated insights engine<br>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Improve ROAS<br>
        • Reduce wasted spend<br>
        • Identify top converting audiences<br>
        • Strengthen budgeting & forecasting<br>
        • Enhance cross-channel visibility<br>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">High-Level KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Total Revenue</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>ROAS</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Leads</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Conversions</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Who Should Use This</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    Performance marketers, growth leads, CMOs, marketing analysts, 
    and teams needing unified revenue-driven marketing intelligence.
    </div>
    """, unsafe_allow_html=True)

# =======================================================================================
# TAB 2 — IMPORTANT ATTRIBUTES
# =======================================================================================
with tab2:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    required_dict = {
        "Campaign": "Name of the marketing campaign or initiative.",
        "Channel": "Acquisition platform (Facebook, Google, Meta Ads, Email, etc.).",
        "Date": "Date of the campaign record. Must be parseable as datetime.",
        "Impressions": "Total number of times the ad was shown to users.",
        "Clicks": "Total number of user clicks recorded for the ad.",
        "Leads": "Number of users who submitted a form / expressed intent.",
        "Conversions": "Number of users who completed the final business goal (purchase / signup / booking).",
        "Spend": "Total marketing cost for that row/period.",
        "Revenue": "Revenue generated from the campaign activity.",
        "ROAS": "Return on Ad Spend = Revenue / Spend."
    }

    dict_df = pd.DataFrame([{"Column": k, "Description": v} for k, v in required_dict.items()])
    render_required_table(dict_df)

    L, R = st.columns(2)

    with L:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        for v in ["Campaign","Channel","Date","Impressions","Clicks","Spend"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with R:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        for v in ["Leads","Conversions","Revenue","ROAS"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# =======================================================================================
# TAB 3 — APPLICATION
# =======================================================================================
with tab3:
    st.markdown('<div class="section-title">Step 1 — Load dataset</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Dataset option:",
        ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"],
        horizontal=True
    )

    df = None

    # DEFAULT DATASET
    if mode == "Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/marketing.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("Default dataset loaded successfully.")
            render_required_table(df.head(5))
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    # UPLOAD SIMPLE CSV (AUTO MAP)
    elif mode == "Upload CSV":
        st.markdown("#### Download sample CSV (optional)")
        SAMPLE_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/marketing.csv"
        try:
            smp = pd.read_csv(SAMPLE_URL).head(5)
            st.download_button(
                "Download Sample CSV",
                smp.to_csv(index=False),
                "sample_marketing.csv",
                "text/csv"
            )
        except:
            pass

        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("File uploaded.")
            render_required_table(df.head(5))

    # UPLOAD + MAPPING
    else:
        uploaded = st.file_uploader("Upload CSV for manual mapping", type=["csv"])
        if uploaded:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()

            st.markdown("### Preview (first 5 rows)")
            render_required_table(raw.head(5))

            st.markdown("### Map your columns to required fields")
            mapping = {}
            options = ["-- Select --"] + list(raw.columns)

            for req_col in REQUIRED_MARKETING_COLS:
                mapping[req_col] = st.selectbox(
                    f"Map → {req_col}",
                    options=options,
                    key=f"map_{req_col}"
                )

            if st.button("Apply Mapping"):
                missing = [k for k,v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v: k for k, v in mapping.items()})
                    st.success("Mapping applied successfully.")
                    render_required_table(df.head(5))

    # STOP IF NO DATAFRAME
    if df is None:
        st.stop()

    # Basic cleaning & derived metrics
    df = ensure_datetime(df, "Date")
    df = safe_numeric(df, ["Impressions","Clicks","Leads","Conversions","Spend","Revenue","ROAS"])


    # Step 2 — Filters
    st.markdown('<div class="section-title">Step 2 — Filters</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2,2,1])

    campaigns = sorted(df["Campaign"].dropna().unique()) if "Campaign" in df.columns else []
    channels  = sorted(df["Channel"].dropna().unique()) if "Channel" in df.columns else []

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

    st.markdown("Filtered preview:")
    render_required_table(filt.head(4))
    download_df(filt.head(500), "filtered_preview.csv", label="Download filtered preview (up to 500 rows)")

    # KPIs
    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    kk1, kk2, kk3, kk4 = st.columns(4)
    total_rev = float(filt["Revenue"].sum()) if "Revenue" in filt.columns else 0.0
    avg_roas = float(filt["ROAS"].mean()) if "ROAS" in filt.columns else 0.0
    total_leads = int(filt["Leads"].sum()) if "Leads" in filt.columns else 0
    conv_rate = (filt["Conversions"].sum() / max(filt["Clicks"].sum(), 1)) if ("Conversions" in filt.columns and "Clicks" in filt.columns) else 0.0

    kk1.markdown(f"<div class='kpi'>Total Revenue<br>{to_money(total_rev)}</div>", unsafe_allow_html=True)
    kk2.markdown(f"<div class='kpi'>ROAS<br>{avg_roas:.2f}</div>", unsafe_allow_html=True)
    kk3.markdown(f"<div class='kpi'>Total Leads<br>{total_leads:,}</div>", unsafe_allow_html=True)
    kk4.markdown(f"<div class='kpi'>Conversion Rate<br>{conv_rate:.2%}</div>", unsafe_allow_html=True)

    # Charts & EDA
    st.markdown('<div class="section-title">Charts & EDA</div>', unsafe_allow_html=True)
    if "Campaign" in filt.columns and (("Revenue" in filt.columns) or ("Conversions" in filt.columns)):
        agg = filt.groupby("Campaign")[["Revenue","Conversions"]].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Revenue"], name="Revenue"))
        fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Conversions"], name="Conversions"))
        fig.update_layout(barmode='group', template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Campaign-level revenue/conversion data missing for charting.")

    # ML — Predict Revenue (show preview only + download)
    st.markdown('<div class="section-title">ML — Predict Revenue (RandomForest)</div>', unsafe_allow_html=True)
    ml_df = filt.copy().dropna(subset=["Revenue"]) if "Revenue" in filt.columns else pd.DataFrame()
    feat_cols = ["Channel","Campaign","Device","AgeGroup","Gender","AdSet","Impressions","Clicks","Spend"]
    feat_cols = [c for c in feat_cols if c in ml_df.columns]

    if ml_df.shape[0] < 30 or len(feat_cols) < 2:
        st.info("Not enough data to train ML model (>=30 rows and >1 feature required).")
    else:
        X = ml_df[feat_cols]
        y = ml_df["Revenue"]

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
            X_train, X_test, y_train, y_test = train_test_split(
                X_t, y, test_size=0.2, random_state=42
            )

            rf = RandomForestRegressor(n_estimators=220, random_state=42)
            with st.spinner("Training RandomForest…"):
                rf.fit(X_train, y_train)

            preds = rf.predict(X_test)
            rmse = math.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)

            st.markdown(f"""
                <div class="card">
                <h4>Model Performance</h4>
                RMSE: {rmse:.2f}<br>
                R² Score: {r2:.3f}
                </div>
                """, unsafe_allow_html=True)

            out_df = pd.DataFrame({
                "Actual_Revenue": y_test.reset_index(drop=True),
                "Predicted_Revenue": preds
            })

            # preview only
            preview_df = out_df.head(10)
            st.markdown("<div class='section-title'>ML Predictions Preview</div>", unsafe_allow_html=True)
            render_required_table(preview_df)

            # download full predictions
            download_df(out_df, "ml_revenue_predictions.csv", label="Download full ML predictions")

    # Forecasting (linear fallback)
    st.markdown('<div class="section-title">Forecasting</div>', unsafe_allow_html=True)
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
            st.info("Not enough daily data (>=10) to produce forecast.")
    else:
        st.info("Date or Revenue column missing for forecasting.")

    # Automated Insights
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)
    insights = []
    if "Channel" in filt.columns and "Revenue" in filt.columns and "Spend" in filt.columns:
        ch_perf = filt.groupby("Channel")[["Revenue","Spend"]].sum().reset_index()
        ch_perf["Revenue_per_Rs"] = np.where(ch_perf["Spend"]>0, ch_perf["Revenue"]/ch_perf["Spend"], 0)
        if not ch_perf.empty:
            best = ch_perf.sort_values("Revenue_per_Rs", ascending=False).iloc[0]
            worst = ch_perf.sort_values("Revenue_per_Rs", ascending=True).iloc[0]
            insights.append({"Insight":"Best Channel ROI","Channel":best['Channel'],"Revenue_per_Rs":float(best['Revenue_per_Rs'])})
            insights.append({"Insight":"Lowest Channel ROI","Channel":worst['Channel'],"Revenue_per_Rs":float(worst['Revenue_per_Rs'])})
    if "Creative" in filt.columns and "Revenue" in filt.columns:
        cr_perf = filt.groupby("Creative")[["Revenue"]].sum().reset_index()
        if not cr_perf.empty:
            top_creative = cr_perf.sort_values("Revenue", ascending=False).iloc[0]
            insights.append({"Insight":"Top Creative by Revenue","Creative": top_creative['Creative'], "Revenue": float(top_creative['Revenue'])})

    if insights:
        ins_df = pd.DataFrame(insights)
        render_required_table(ins_df)
        download_df(ins_df, "automated_insights.csv", label="Download automated insights")
    else:
        st.markdown('<div class="card"><div class="small-muted">No automated insights available for selected filters.</div></div>', unsafe_allow_html=True)

    # Exports
    st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
    download_df(filt, "marketing_filtered_full.csv", label="Download full filtered dataset (CSV)")

# END OF FILE
