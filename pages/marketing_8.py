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
# App config
# -------------------------
st.set_page_config(page_title="Google Ads & SEO Performance Lab", layout="wide", initial_sidebar_state="collapsed")

# -------------------------
# Constants / UI tokens
# -------------------------
BLUE = "#064b86"
BLACK = "#000000"
BASE_FONT_SIZE_PX = 17

REQUIRED_COLS = [
    "Date","Campaign","AdGroup","Keyword","Device","Country",
    "Impressions","Clicks","CTR","CPC","Cost","Conversions","ConversionRate","Revenue","ROAS",
    "Organic_Impressions","Organic_Clicks","Organic_CTR","Bounce_Rate","Avg_Time_on_Page_sec",
    "Page_Position","Backlinks"
]

AUTO_MAPS = {
    "Date": ["date","day","report_date","reporting date"],
    "Campaign": ["campaign","campaign name","campaign_name"],
    "AdGroup": ["adgroup","ad group","ad_group"],
    "Keyword": ["keyword","query","search term"],
    "Device": ["device","platform"],
    "Country": ["country","region"],
    "Impressions": ["impressions","impr"],
    "Clicks": ["clicks","click"],
    "CTR": ["ctr"],
    "CPC": ["cpc","cost per click"],
    "Cost": ["cost","spend","amount"],
    "Conversions": ["conversions","orders","purchases"],
    "ConversionRate": ["conversion rate","conversion_rate","conv_rate"],
    "Revenue": ["revenue","amount","value"],
    "ROAS": ["roas"],
    "Organic_Impressions": ["organic impressions","organic_impressions"],
    "Organic_Clicks": ["organic clicks","organic_clicks"],
    "Organic_CTR": ["organic ctr","organic_ctr"],
    "Bounce_Rate": ["bounce","bounce_rate"],
    "Avg_Time_on_Page_sec": ["avg time","avg_time","time on page","avg_time_on_page_sec"],
    "Page_Position": ["position","page_position"],
    "Backlinks": ["backlinks","inbound links"]
}

# -------------------------
# Helpers - must be defined before usage
# -------------------------
def auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    except Exception:
        return "₹ 0.00"

def download_df(df: pd.DataFrame, filename: str, label: str = "Download CSV"):
    if df is None or df.empty:
        st.info("No data to download.")
        return
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label=label, data=b, file_name=filename, mime="text/csv")

def render_required_table(df: pd.DataFrame):
    """
    Index-free HTML table renderer. Resets/drops index to avoid invisible index columns.
    Uses class 'required-table' so CSS can style it.
    """
    df2 = df.reset_index(drop=True).copy()
    styled = df2.style.set_table_attributes('class="required-table"')
    html = styled.to_html()
    # remove extra empty th/td produced by pandas styler in some versions
    html = html.replace("<th></th>", "").replace("<td></td>", "")
    st.write(html, unsafe_allow_html=True)

# -------------------------
# Global CSS (Inter font, pure black text, blue KPIs/var boxes)
# -------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {{
    background: #ffffff;
    color: {BLACK};
    font-family: 'Inter', sans-serif;
    font-size: {BASE_FONT_SIZE_PX}px;
}}

/* Fade-in */
.fade-in {{
  animation: fadeIn 0.45s ease-in-out;
}}
@keyframes fadeIn {{
  from {{ opacity: 0; transform: translateY(6px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}

/* MAIN HEADER */
.big-header {{
    font-size:36px !important;
    font-weight:700 !important;
    color:#000 !important;
    margin-bottom:12px;
}}

/* Section titles */
.section-title {{
    color: {BLACK};
    font-size: 22px;
    font-weight: 600;
    text-align: left;
    margin: 12px 0 12px 0;
    position: relative;
    display: inline-block;
}}
.section-title:hover::after {{
    content: "";
    position: absolute;
    left: 0;
    bottom: -6px;
    width: 40%;
    height: 3px;
    background: {BLUE};
    border-radius: 2px;
}}

/* Cards (pure black body text) */
.card {{
    background: #ffffff;
    color: {BLACK};
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
    border-color: {BLUE};
    cursor: pointer;
}}
.card h4, .card p, .card li {{
    color: {BLACK};
    margin: 0;
}}

/* KPI cards (blue text) */
.kpi-row {{ display:flex; gap:16px; margin-bottom:16px; }}
.kpi-card {{
    flex:1;
    background:#ffffff;
    color:{BLUE};
    border-radius:12px;
    padding:16px;
    text-align:center;
    border:1px solid #e6e6e6;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    font-weight:600;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}}
.kpi-card:hover {{
    transform: translateY(-6px);
    box-shadow: 0 10px 28px rgba(0,0,0,0.10);
    filter: drop-shadow(0 0 8px rgba(6,75,134,0.12));
    cursor:pointer;
}}
.kpi-card .kpi-value {{ font-size:20px; color:{BLUE}; margin-top:6px; display:block; }}

/* Variable boxes (blue text) */
.variable-box {{
    background: #ffffff;
    color: {BLUE};
    border-radius: 12px;
    padding: 12px;
    border: 1px solid #e6e6e6;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    text-align:center;
}}
.variable-title {{ font-weight:600; color:{BLUE}; margin-bottom:6px; }}

/* Required table (pure black) */
.required-table {{
    border-collapse: collapse;
    width:100%;
    font-size:14px;
    color: {BLACK};
    background:#fff;
}}
.required-table thead th {{
    border-bottom: 2px solid #000;
    padding: 10px;
    text-align:left;
    font-weight:600;
}}
.required-table tbody td {{
    padding:10px;
    border-bottom:1px solid #f2f2f2;
}}
.required-table tbody tr:hover {{ background:#f7f7f7; }}

/* Buttons */
.stButton>button, .stDownloadButton>button {{
    background:{BLUE} !important;
    color:white !important;
    border:none;
    padding:10px 18px;
    border-radius:8px !important;
    font-size:15px !important;
    font-weight:600 !important;
}}
.stButton>button:hover, .stDownloadButton>button:hover {{
    transform: translateY(-3px);
    background:#0a6eb3 !important;
}}

/* small-muted utility */
.small-muted {{ color:#666666; font-size:13px; }}
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

st.markdown("<div class='big-header'>Google Ads & SEO Performance Lab — unified paid + organic insights</div>", unsafe_allow_html=True)


# -------------------------
# Tabs (Overview / Important Attributes / Application)
# -------------------------
tab_overview, tab_attributes, tab_app = st.tabs(["Overview", "Important Attributes", "Application"])

# -------------------------
# Overview tab
# -------------------------
with tab_overview:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <h4>Purpose</h4>
        <p>Unify paid search (Google Ads) performance with SEO signals to measure combined impact on clicks, conversions and revenue.</p>
        <ul style="margin:6px 0 0 18px;">
            <li>Keyword-level & landing page performance comparison (paid vs organic)</li>
            <li>Revenue & ROAS attribution, ML-driven revenue predictions, and short-term forecasts</li>
            <li>Exportable predictions and automated insight tables for leadership</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        • Multi-channel paid + organic performance analytics<br>
        • Keyword- and page-level revenue/ROAS slicing<br>
        • SERP position & backlink signal integration<br>
        • Automated insights and model explainability for trust
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">High-level KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi-card'>Paid Clicks</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi-card'>Organic Clicks</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi-card'>Revenue</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi-card'>ROAS</div>", unsafe_allow_html=True)

# -------------------------
# Important Attributes tab
# -------------------------
with tab_attributes:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    required_dict = {
        "Date": "Date of the record (parseable).",
        "Campaign": "Paid campaign name (Google Ads).",
        "AdGroup": "Paid ad group name.",
        "Keyword": "Keyword or query text.",
        "Device": "Device type (Desktop / Mobile / Tablet).",
        "Impressions": "Number of times ad/page was displayed.",
        "Clicks": "Number of clicks.",
        "CTR": "Click-through rate.",
        "CPC": "Cost per click.",
        "Cost": "Paid cost/spend.",
        "Conversions": "Number of conversions/actions.",
        "Revenue": "Revenue attributed to the row.",
        "ROAS": "Revenue / Cost.",
        "Organic_Impressions": "Search console impressions (organic).",
        "Organic_Clicks": "Search console clicks (organic).",
        "Bounce_Rate": "Bounce rate for landing pages.",
        "Avg_Time_on_Page_sec": "Average time on page in seconds."
    }

    dict_df = pd.DataFrame([{"Column": k, "Description": v} for k, v in required_dict.items()])
    st.markdown('<div class="card">', unsafe_allow_html=True)
    render_required_table(dict_df)
    st.markdown('</div>', unsafe_allow_html=True)

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        indep = ["Campaign","AdGroup","Keyword","Device","Impressions","Clicks","Cost","Organic_Impressions","Organic_Clicks"]
        for v in indep:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)
    with right_col:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        dep = ["Conversions","Revenue","ConversionRate","ROAS","Bounce_Rate","Avg_Time_on_Page_sec"]
        for v in dep:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# -------------------------
# Application tab
# -------------------------
with tab_app:
    st.markdown('<div class="section-title">Application</div>', unsafe_allow_html=True)

    # Step 1 — Load dataset
    st.markdown('<div class="section-title">Step 1 — Load dataset</div>', unsafe_allow_html=True)
    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)
    df = None

    if mode == "Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/google_ads_seo_performance.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("Default dataset loaded")
            # small preview via index-free renderer
            render_required_table(df.head(5))
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    elif mode == "Upload CSV":
        st.markdown("#### Download Sample CSV for Reference", unsafe_allow_html=True)
        SAMPLE_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/google_ads_seo_performance.csv"
        try:
            sample_df = pd.read_csv(SAMPLE_URL).head(5)
            sample_csv = sample_df.to_csv(index=False)
            st.download_button("Download Sample CSV", sample_csv, "sample_google_ads_seo.csv", "text/csv")
        except Exception:
            pass

        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("File uploaded")
            render_required_table(df.head(5))

    else:
        uploaded = st.file_uploader("Upload CSV for manual mapping", type=["csv"])
        if uploaded:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.markdown("Preview (first 5 rows)", unsafe_allow_html=True)
            render_required_table(raw.head(5))

            st.markdown("Map your columns to required fields", unsafe_allow_html=True)
            mapping = {}
            options = ["-- Select --"] + list(raw.columns)
            for req in REQUIRED_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=options, key=f"map_{req}")

            if st.button("Apply mapping"):
                missing = [k for k, v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v: k for k, v in mapping.items()})
                    st.success("Mapping applied successfully.")
                    render_required_table(df.head(5))

    if df is None:
        st.stop()

    # Basic cleaning
    df = ensure_datetime(df, "Date")
    numeric_cols = [
        "Impressions","Clicks","CTR","CPC","Cost","Conversions","ConversionRate","Revenue","ROAS",
        "Organic_Impressions","Organic_Clicks","Organic_CTR","Bounce_Rate","Avg_Time_on_Page_sec",
        "Page_Position","Backlinks"
    ]
    df = safe_numeric(df, numeric_cols)

    # Step 2 — Filters & Preview
    st.markdown('<div class="section-title">Step 2 — Filters & Preview</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2,2,1])
    campaigns = sorted(df["Campaign"].dropna().unique()) if "Campaign" in df.columns else []
    adgroups = sorted(df["AdGroup"].dropna().unique()) if "AdGroup" in df.columns else []
    keywords = sorted(df["Keyword"].dropna().unique()) if "Keyword" in df.columns else []

    with c1:
        sel_campaigns = st.multiselect("Campaign", options=campaigns, default=campaigns[:3])
    with c2:
        sel_adgroups = st.multiselect("AdGroup", options=adgroups, default=adgroups[:3])
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
    if sel_adgroups:
        filt = filt[filt["AdGroup"].isin(sel_adgroups)]
    try:
        if date_range and len(date_range) == 2:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            filt = filt[(filt["Date"] >= start) & (filt["Date"] <= end)]
    except Exception:
        pass

    st.markdown("Filtered preview (first 10 rows):", unsafe_allow_html=True)
    render_required_table(filt.head(10))
    download_df(filt.head(500), "filtered_preview.csv", label="Download filtered preview (up to 500 rows)")

    # KPIs
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    paid_clicks = int(filt["Clicks"].sum()) if "Clicks" in filt.columns else 0
    organic_clicks = int(filt["Organic_Clicks"].sum()) if "Organic_Clicks" in filt.columns else 0
    revenue_val = float(filt["Revenue"].sum()) if "Revenue" in filt.columns else 0.0
    roas_avg = float(filt["ROAS"].mean()) if "ROAS" in filt.columns else 0.0
    bounce_avg = float(filt["Bounce_Rate"].mean()) if "Bounce_Rate" in filt.columns else 0.0
    avg_time = float(filt["Avg_Time_on_Page_sec"].mean()) if "Avg_Time_on_Page_sec" in filt.columns else 0.0

    k1.markdown(f"<div class='kpi-card'>Paid Clicks<div class='kpi-value'>{paid_clicks:,}</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi-card'>Organic Clicks<div class='kpi-value'>{organic_clicks:,}</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi-card'>Revenue<div class='kpi-value'>{to_currency(revenue_val)}</div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi-card'>ROAS<div class='kpi-value'>{roas_avg:.2f}</div></div>", unsafe_allow_html=True)
    k5.markdown(f"<div class='kpi-card'>Bounce Rate<div class='kpi-value'>{bounce_avg:.2%}</div></div>", unsafe_allow_html=True)
    k6.markdown(f"<div class='kpi-card'>Avg Time / Page<div class='kpi-value'>{avg_time:.1f} s</div></div>", unsafe_allow_html=True)

    # Charts & EDA
    st.markdown('<div class="section-title">Charts & EDA</div>', unsafe_allow_html=True)
    if "Campaign" in filt.columns:
        agg = filt.groupby("Campaign").agg({"Clicks":"sum","Organic_Clicks":"sum"}).reset_index()
        if not agg.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Clicks"], name="Paid Clicks"))
            fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Organic_Clicks"], name="Organic Clicks"))
            fig.update_layout(barmode='group', template="plotly_white", xaxis_title="Campaign", yaxis_title="Clicks")
            st.plotly_chart(fig, use_container_width=True)

    # CTR trend
    if "Date" in filt.columns:
        ts = filt.groupby("Date").agg({"CTR":"mean","Organic_CTR":"mean"}).reset_index()
        if not ts.empty:
            fig2 = px.line(ts, x="Date", y=["CTR","Organic_CTR"], labels={"value":"CTR","variable":"Series"})
            st.plotly_chart(fig2, use_container_width=True)

    # Revenue by Keyword (top N)
    if "Keyword" in filt.columns and "Revenue" in filt.columns:
        agg_kw = filt.groupby("Keyword").agg({"Revenue":"sum","ROAS":"mean"}).reset_index().sort_values("Revenue", ascending=False).head(30)
        if not agg_kw.empty:
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=agg_kw["Keyword"], y=agg_kw["Revenue"], name="Revenue"))
            fig3.update_layout(xaxis_title="Keyword", yaxis_title="Revenue", template="plotly_white")
            st.plotly_chart(fig3, use_container_width=True)

    # ML — Predict Revenue (preview + downloadable)
    st.markdown('<div class="section-title">ML — Predict Revenue (RandomForest) — Preview + Download</div>', unsafe_allow_html=True)
    ml_df = filt.copy()
    if "Revenue" not in ml_df.columns or ml_df["Revenue"].dropna().shape[0] < 30:
        st.info("Not enough Revenue history (>=30 rows) to train a model.")
    else:
        feat_cols = ["Clicks","Impressions","Organic_Clicks","Conversions","CPC","CTR"]
        feat_cols = [c for c in feat_cols if c in ml_df.columns]
        if len(feat_cols) < 2:
            st.info("Insufficient feature columns for ML. Need at least 2 numeric features.")
        else:
            X = ml_df[feat_cols].fillna(0)
            y = ml_df["Revenue"].fillna(0)
            # simple preprocessing: scale numeric features
            scaler = StandardScaler()
            try:
                X_t = scaler.fit_transform(X)
            except Exception as e:
                st.error("Feature scaling failed: " + str(e))
                X_t = None

            if X_t is not None and len(X_t) >= 30:
                X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)
                rf = RandomForestRegressor(n_estimators=200, random_state=42)
                with st.spinner("Training RandomForest..."):
                    rf.fit(X_train, y_train)
                preds = rf.predict(X_test)
                rmse = math.sqrt(np.mean((y_test - preds) ** 2))
                r2 = rf.score(X_test, y_test)
                st.markdown(f"<div class='card'><b>Model performance</b><br>RMSE: {rmse:.2f} &nbsp;&nbsp; R²: {r2:.3f}</div>", unsafe_allow_html=True)

                out_df = pd.DataFrame({
                    "Actual_Revenue": y_test.reset_index(drop=True),
                    "Predicted_Revenue": preds
                })
                # preview only
                render_required_table(out_df.head(8))
                # full downloadable
                download_df(out_df.reset_index(drop=True), "ml_revenue_predictions.csv", label="Download full ML predictions")

    # Automated insights (table + download)
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)
    insights = []
    if "Campaign" in filt.columns and "Revenue" in filt.columns and "Cost" in filt.columns:
        ch = filt.groupby("Campaign")[["Revenue","Cost"]].sum().reset_index()
        ch["ROI"] = np.where(ch["Cost"]>0, ch["Revenue"]/ch["Cost"], 0)
        if not ch.empty:
            best = ch.loc[ch["ROI"].idxmax()]
            worst = ch.loc[ch["ROI"].idxmin()]
            insights.append({"Insight":"Best ROI Campaign","Campaign":best["Campaign"], "ROI":float(best["ROI"])})
            insights.append({"Insight":"Lowest ROI Campaign","Campaign":worst["Campaign"], "ROI":float(worst["ROI"])})

    if "Keyword" in filt.columns and "Revenue" in filt.columns:
        kw = filt.groupby("Keyword")[["Revenue","Clicks"]].sum().reset_index().sort_values("Revenue", ascending=False)
        if not kw.empty:
            topkw = kw.iloc[0]
            insights.append({"Insight":"Top Keyword by Revenue","Keyword":topkw["Keyword"], "Revenue":float(topkw["Revenue"])})

    if insights:
        ins_df = pd.DataFrame(insights)
        render_required_table(ins_df)
        download_df(ins_df, "automated_insights.csv", label="Download automated insights")
    else:
        st.markdown('<div class="card"><div class="small-muted">No automated insights available for selected filters.</div></div>', unsafe_allow_html=True)
