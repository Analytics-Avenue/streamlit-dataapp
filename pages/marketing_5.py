# content_seo_dashboard_marketing_lab.py
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
from sklearn.metrics import mean_squared_error, r2_score
import math
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Page / App Config
# -------------------------
st.set_page_config(page_title="Content & SEO Dashboard — Marketing Lab Style", layout="wide")
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

# -------------------------
# Global CSS (Inter, pure black text, KPI/variable-blue, cards, table, fade-in)
# -------------------------
st.markdown(f"""
<style>
/* Load Inter via Google Fonts (best-effort; offline fallback will use sans-serif) */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* {{ font-family: 'Inter', sans-serif; }}

:root {{
    --brand-blue: #064b86;
    --pure-black: #000000;
    --card-bg: #ffffff;
    --muted-border: #e6e6e6;
}}

/* Ensure most text is pure black */
[data-testid="stAppViewContainer"] * {{
    color: var(--pure-black) !important;
}}

/* Page container fade-in */
.block-container {{ animation: fadeIn 0.45s ease; }}
@keyframes fadeIn {{ from {{opacity:0; transform:translateY(8px)}} to {{opacity:1; transform:translateY(0)}} }}

/* Header */
.header-wrap {{ display:flex; align-items:center; gap:12px; margin-bottom:8px; }}
.header-title-main {{ font-size:34px !important; font-weight:700 !important; color:var(--pure-black) !important; margin:0; }}
.header-sub {{ font-size:16px !important; color:var(--pure-black) !important; margin:0; }}

/* Card */
.card {{
    background: var(--card-bg);
    padding: 18px;
    border-radius: 12px;
    border: 1px solid var(--muted-border);
    box-shadow: 0 4px 18px rgba(0,0,0,0.06);
    transition: all 0.20s ease;
    color: var(--pure-black) !important;
}}
.card:hover {{
    transform: translateY(-6px);
    box-shadow: 0 14px 30px rgba(6,75,134,0.08);
    border-color: var(--brand-blue);
}}

/* Section title */
.section-title {{
    font-size:20px !important;
    font-weight:600 !important;
    margin-top:16px;
    margin-bottom:10px;
    color:var(--pure-black) !important;
}}

/* KPI (blue text) */
.kpi {{
    background: var(--card-bg);
    padding:16px;
    border-radius:12px;
    border:1px solid #eaeaea;
    text-align:center;
    font-size:18px;
    font-weight:700;
    color: var(--brand-blue) !important;
    box-shadow: 0 3px 14px rgba(0,0,0,0.05);
}}
.kpi:hover {{ transform: translateY(-4px); box-shadow: 0 12px 22px rgba(6,75,134,0.08); }}

/* Variable boxes (blue text) */
.variable-box {{
    padding:12px 14px;
    border-radius:12px;
    background: var(--card-bg);
    border:1px solid #ececec;
    color: var(--brand-blue) !important;
    font-weight:600;
    text-align:center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-bottom:10px;
}}

/* Tables - pure black text, index-safe */
.required-table thead th {{
    background: #ffffff !important;
    color: var(--pure-black) !important;
    font-size:18px !important;
    border-bottom:2px solid var(--pure-black) !important;
    padding:10px !important;
    text-align:left;
}}
.required-table tbody td {{
    color: var(--pure-black) !important;
    font-size:15.5px !important;
    padding:10px !important;
    border-bottom:1px solid #efefef !important;
}}
.required-table tbody tr:hover td {{ background:#f8f8f8 !important; }}

/* Dataframe general styling (fallback) */
.css-ffhzg2 {{ color: var(--pure-black) !important; }}

/* Buttons styling */
.stButton>button, .stDownloadButton>button {{
    background: var(--brand-blue) !important;
    color: white !important;
    border: none;
    padding: 8px 18px;
    border-radius:8px;
    font-weight:700;
}}
.stButton>button:hover, .stDownloadButton>button:hover {{
    background:#0a6eb3 !important;
    transform: translateY(-2px);
}}

/* Small helpers */
.small-muted {{ color: #6b6b6b !important; font-size:13px !important; }}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Helpers & Constants
# -------------------------
REQUIRED_CONTENT_COLS = [
    "Date","Page","Content_Type","Keyword","Device","Country",
    "Impressions","Clicks","CTR","Bounce_Rate","Time_on_Page_sec",
    "Backlinks","Conversions","Revenue"
]

AUTO_MAPS = {
    "Date": ["date"],
    "Page": ["page","url","landing page","page url"],
    "Content_Type": ["content_type","type","format"],
    "Keyword": ["keyword","search term","query"],
    "Device": ["device","platform"],
    "Country": ["country","region"],
    "Impressions": ["impressions","impression","impr"],
    "Clicks": ["clicks","click"],
    "CTR": ["ctr","click through rate"],
    "Bounce_Rate": ["bounce","bouncerate","bounce_rate"],
    "Time_on_Page_sec": ["time_on_page","time_sec","duration","time on page"],
    "Backlinks": ["backlinks","links"],
    "Conversions": ["conversions","leads","goals"],
    "Revenue": ["revenue","earnings","rev"]
}

DEFAULT_CONTENT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/content_seo_dataset.csv"

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

def ensure_datetime(df, col="Date"):
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        except:
            pass
    return df

def to_currency(x):
    try:
        return "₹ " + f"{float(x):,.2f}"
    except:
        return "₹ 0.00"

def download_df(df, filename, label="Download CSV"):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")

def render_index_safe_table(df: pd.DataFrame, classes="required-table"):
    """
    Renders an HTML table from a dataframe with index omitted (index-safe).
    Uses classes so CSS from above applies.
    """
    if df is None or df.empty:
        st.info("No table data to display.")
        return
    html = df.to_html(index=False, classes=classes, escape=False)
    st.markdown(html, unsafe_allow_html=True)

# -------------------------
# Header / Title
# -------------------------
st.markdown(
    f"""
    <div class="header-wrap">
        <img src="{logo_url}" width="56" style="border-radius:6px;">
        <div>
            <div class="header-title-main">Analytics Avenue &amp; Advanced Analytics</div>
            <div class="header-sub small-muted">Content & SEO Performance Dashboard — marketing lab UI</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------
# 3 Tabs: Overview | Important Attributes | Application
# -------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# -------------------------
# TAB 1 — Overview
# -------------------------
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <b>Purpose:</b> Centralize Content & SEO page-level metrics, provide engagement and conversion insights, and enable simple ML-driven revenue predictions and quick recommendations.
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([2,1])
    with left:
        st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Page & keyword performance breakdown across devices and countries.<br>
        • Engagement metrics: bounce rate, time on page, CTR.<br>
        • Revenue & conversion attribution and prediction.<br>
        • Automated ROI rules and suggested actions for low-performing pages.<br>
        • Exportable filtered datasets and ML predictions.
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Improve content prioritization by ROI per page.<br>
        • Identify pages needing UX or content refresh due to high bounce or low time-on-page.<br>
        • Allocate SEO efforts to high-conversion keyword groups.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Total Revenue</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Total Conversions</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Average CTR</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Average Bounce Rate</div>", unsafe_allow_html=True)

# -------------------------
# TAB 2 — Important Attributes
# -------------------------
with tab2:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    required_dict = {
        "Date": "Date of the record (yyyy-mm-dd).",
        "Page": "Landing page URL or page ID.",
        "Content_Type": "Article, blog, product, category, landing, etc.",
        "Keyword": "Primary search keyword or query.",
        "Device": "Desktop, Mobile, Tablet, etc.",
        "Country": "Country or region of traffic.",
        "Impressions": "Search impressions or pageview proxy.",
        "Clicks": "Clicks from search or internal CTA clicks.",
        "CTR": "Click-through rate (Clicks / Impressions).",
        "Bounce_Rate": "Percentage of single-page sessions.",
        "Time_on_Page_sec": "Average time on page in seconds.",
        "Backlinks": "Number of referring backlinks (if available).",
        "Conversions": "Completed conversions attributed to page.",
        "Revenue": "Revenue attributed to conversions on this page."
    }

    dict_df = pd.DataFrame([{"Attribute": k, "Description": v} for k,v in required_dict.items()])

    # Render index-safe HTML table with required styling
    render_index_safe_table(dict_df)

    st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
    indep_col1, indep_col2 = st.columns(2)
    indep_list = [
        "Page", "Content_Type", "Keyword", "Device", "Country",
        "Impressions", "Clicks", "Backlinks", "Time_on_Page_sec"
    ]
    with indep_col1:
        for v in indep_list[:len(indep_list)//2]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)
    with indep_col2:
        for v in indep_list[len(indep_list)//2:]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
    dep_vars = ["Conversions", "Revenue", "CTR", "Bounce_Rate"]
    c1,c2 = st.columns(2)
    with c1:
        for v in dep_vars[:2]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)
    with c2:
        for v in dep_vars[2:]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# -------------------------
# TAB 3 — Application
# -------------------------
with tab3:
    st.markdown('<div class="section-title">Step 1 — Load dataset</div>', unsafe_allow_html=True)
    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)
    df = None
    raw = None

    if mode == "Default dataset":
        try:
            df = pd.read_csv(DEFAULT_CONTENT_URL)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("Default dataset loaded.")
            st.markdown('<div class="small-muted">Preview (first 5 rows)</div>', unsafe_allow_html=True)
            st.dataframe(df.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load default dataset: {e}")
            st.stop()

    elif mode == "Upload CSV":
        st.markdown("#### Download sample CSV for reference (first 5 rows)")
        try:
            sample_df = pd.read_csv(DEFAULT_CONTENT_URL).head(5)
            st.download_button("Download Sample CSV", sample_df.to_csv(index=False), "sample_content_seo.csv", "text/csv")
        except Exception:
            st.info("Sample unavailable.")

        uploaded = st.file_uploader("Upload your dataset (CSV)", type=["csv"], accept_multiple_files=False)
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                df.columns = df.columns.str.strip()
                df = auto_map_columns(df)
                st.success("File uploaded and columns auto-mapped where possible.")
                st.dataframe(df.head(5), use_container_width=True)
            except Exception as e:
                st.error("Failed to read uploaded CSV: " + str(e))

    else:  # Upload + mapping
        uploaded = st.file_uploader("Upload CSV to map (CSV)", type=["csv"], key="mapping_upload")
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.markdown("<div class='small-muted'>Preview (first 5 rows)</div>", unsafe_allow_html=True)
            st.dataframe(raw.head(5), use_container_width=True)

            st.markdown('<div class="section-title">Map your columns to required fields</div>', unsafe_allow_html=True)
            mapping = {}
            cols_list = list(raw.columns)
            for req in REQUIRED_CONTENT_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + cols_list, key=f"map_{req}")
            if st.button("Apply mapping"):
                missing = [k for k,v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    inv = {v:k for k,v in mapping.items()}
                    df = raw.rename(columns=inv)
                    st.success("Mapping applied.")
                    st.dataframe(df.head(5), use_container_width=True)

    if df is None:
        st.stop()

    # Keep only required columns that exist
    df = df[[c for c in REQUIRED_CONTENT_COLS if c in df.columns]].copy()
    df = ensure_datetime(df, "Date")
    # Numeric coercion
    for col in ["Impressions","Clicks","CTR","Bounce_Rate","Time_on_Page_sec","Backlinks","Conversions","Revenue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # -------------------------
    # Step 2 — Filters & Preview
    # -------------------------
    st.markdown('<div class="section-title">Step 2 — Filters & preview</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns([2,2,1])
    pages = sorted(df["Page"].dropna().unique()) if "Page" in df.columns else []
    devices = sorted(df["Device"].dropna().unique()) if "Device" in df.columns else []
    countries = sorted(df["Country"].dropna().unique()) if "Country" in df.columns else []
    content_types = sorted(df["Content_Type"].dropna().unique()) if "Content_Type" in df.columns else []

    with c1:
        sel_pages = st.multiselect("Page", options=pages, default=pages[:5] if pages else [])
    with c2:
        sel_devices = st.multiselect("Device", options=devices, default=devices[:3] if devices else [])
    with c3:
        # date bounds
        try:
            min_d = df["Date"].min().date()
            max_d = df["Date"].max().date()
            date_range = st.date_input("Date range", value=(min_d, max_d))
        except Exception:
            date_range = st.date_input("Date range")

    filt = df.copy()
    if sel_pages:
        filt = filt[filt["Page"].isin(sel_pages)]
    if sel_devices:
        filt = filt[filt["Device"].isin(sel_devices)]
    try:
        if date_range and len(date_range) == 2:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            filt = filt[(filt["Date"] >= start) & (filt["Date"] <= end)]
    except Exception:
        pass

    st.markdown('<div class="section-title">Filtered preview (first 10 rows)</div>', unsafe_allow_html=True)
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(500), "filtered_preview_content.csv", label="Download filtered preview (first 500 rows)")

    # -------------------------
    # Key Metrics display (as numbers + KPI cards)
    # -------------------------
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)

    def safe_sum(s):
        try:
            return pd.to_numeric(s, errors="coerce").sum()
        except:
            return 0

    k1.metric("Total Revenue", to_currency(filt["Revenue"].sum() if "Revenue" in filt.columns else 0))
    k2.metric("Total Conversions", int(safe_sum(filt["Conversions"]) if "Conversions" in filt.columns else 0))
    # Avoid divide by zero; CTR stored as fractional or percentage? user earlier treated as fraction -> format
    avg_ctr = (filt["CTR"].mean() if "CTR" in filt.columns else 0)
    try:
        # if CTR appears >1 assume it's percentage and convert
        if avg_ctr > 1:
            avg_ctr_display = f"{avg_ctr:.2f}%"
        else:
            avg_ctr_display = f"{avg_ctr:.2%}"
    except:
        avg_ctr_display = "0.00%"
    k3.metric("Average CTR", avg_ctr_display)
    avg_bounce = (filt["Bounce_Rate"].mean() if "Bounce_Rate" in filt.columns else 0)
    try:
        if avg_bounce > 1:
            b_display = f"{avg_bounce:.2f}%"
        else:
            b_display = f"{avg_bounce:.2%}"
    except:
        b_display = "0.00%"
    k4.metric("Avg Bounce Rate", b_display)

    # -------------------------
    # Charts
    # -------------------------
    st.markdown('<div class="section-title">Revenue & Conversions per Page</div>', unsafe_allow_html=True)
    if "Page" in filt.columns and not filt.empty:
        page_agg = filt.groupby("Page")[["Revenue","Conversions"]].sum().reset_index().sort_values("Revenue", ascending=False)
        if not page_agg.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=page_agg["Page"], y=page_agg["Revenue"], name="Revenue", text=page_agg["Revenue"], textposition="outside"))
            fig.add_trace(go.Bar(x=page_agg["Page"], y=page_agg["Conversions"], name="Conversions", text=page_agg["Conversions"], textposition="outside"))
            fig.update_layout(barmode="group", xaxis_title="Page", yaxis_title="Value", uniformtext_minsize=8, uniformtext_mode='hide')
            fig.update_xaxes(tickangle=-45, showgrid=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No page-level aggregation available.")
    else:
        st.info("No 'Page' column available for page-level charts.")

    st.markdown('<div class="section-title">Device / Country / Content_Type Performance</div>', unsafe_allow_html=True)
    group_cols = ["Device","Country","Content_Type"]
    for g in group_cols:
        if g in filt.columns:
            grp = filt.groupby(g)[["Revenue","Conversions"]].sum().reset_index().sort_values("Revenue", ascending=False)
            if not grp.empty:
                fig = px.bar(grp, x=g, y="Revenue", text="Revenue", title=f"{g} Revenue")
                fig.update_traces(textposition="outside")
                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # ML: Predict Revenue (RandomForest)
    # -------------------------
    st.markdown('<div class="section-title">ML — Predict Revenue (RandomForest)</div>', unsafe_allow_html=True)
    ml_df = filt.copy().dropna(subset=["Revenue"]) if "Revenue" in filt.columns else pd.DataFrame()
    feat_cols = ["Page","Content_Type","Device","Country","Impressions","Clicks","Time_on_Page_sec","Backlinks"]
    feat_cols = [c for c in feat_cols if c in ml_df.columns]

    if ml_df.empty or len(ml_df) < 30 or len(feat_cols) < 2:
        st.info("Not enough data to train ML model (need >=30 rows and >=2 features).")
    else:
        X = ml_df[feat_cols]
        y = ml_df["Revenue"]
        cat_cols = [c for c in X.columns if X[c].dtype == "object"]
        num_cols = [c for c in X.columns if c not in cat_cols]
        # ColumnTransformer: OneHotEncoder (dense output) for cats, StandardScaler for nums
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
            rf = RandomForestRegressor(n_estimators=200, random_state=42)
            with st.spinner("Training RandomForest..."):
                rf.fit(X_train, y_train)
            preds = rf.predict(X_test)
            rmse = math.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            st.write(f"Revenue prediction — RMSE: {rmse:.2f}, R²: {r2:.3f}")

            # Combine predictions with features for preview and download
            X_test_df = pd.DataFrame(X_test, columns=[f"Feature_{i}" for i in range(X_test.shape[1])])
            X_test_df["Actual_Revenue"] = y_test.reset_index(drop=True)
            X_test_df["Predicted_Revenue"] = preds
            st.markdown('<div class="small-muted">ML test predictions (sample)</div>', unsafe_allow_html=True)
            st.dataframe(X_test_df.head(10), use_container_width=True)
            download_df(X_test_df.head(500), "ml_revenue_predictions.csv", label="Download ML predictions (sample)")

    # -------------------------
    # Automated Insights
    # -------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)
    insights = []
    if "Country" in filt.columns and ("Revenue" in filt.columns and "Conversions" in filt.columns):
        country_perf = filt.groupby("Country")[["Revenue","Conversions"]].sum().reset_index()
        country_perf["Revenue_per_Conversion"] = np.where(country_perf["Conversions"]>0,
                                                         country_perf["Revenue"]/country_perf["Conversions"], 0)
        if not country_perf.empty:
            best = country_perf.sort_values("Revenue_per_Conversion", ascending=False).iloc[0]
            worst = country_perf.sort_values("Revenue_per_Conversion").iloc[0]
            insights.append({"Insight":"Best Country ROI", "Country":best['Country'], "Revenue_per_Conversion":best['Revenue_per_Conversion']})
            insights.append({"Insight":"Lowest Country ROI", "Country":worst['Country'], "Revenue_per_Conversion":worst['Revenue_per_Conversion']})

    # Page-level engagement flags
    if "Page" in filt.columns and "Bounce_Rate" in filt.columns and "Time_on_Page_sec" in filt.columns:
        page_eng = filt.groupby("Page")[["Bounce_Rate","Time_on_Page_sec","Conversions"]].mean().reset_index()
        high_bounce = page_eng[page_eng["Bounce_Rate"] > page_eng["Bounce_Rate"].median()].sort_values("Bounce_Rate", ascending=False).head(3)
        low_time = page_eng[page_eng["Time_on_Page_sec"] < page_eng["Time_on_Page_sec"].median()].sort_values("Time_on_Page_sec").head(3)
        if not high_bounce.empty:
            insights.append({"Insight":"High Bounce Pages (top 3)", "Pages": ", ".join(high_bounce["Page"].astype(str).tolist())})
        if not low_time.empty:
            insights.append({"Insight":"Low Time-On-Page (top 3)", "Pages": ", ".join(low_time["Page"].astype(str).tolist())})

    # Keyword-level quick ROI (if Keyword present)
    if "Keyword" in filt.columns and "Conversions" in filt.columns and "Revenue" in filt.columns:
        kw = filt.groupby("Keyword")[["Clicks","Conversions","Revenue"]].sum().reset_index()
        kw["Rev_per_Click"] = np.where(kw["Clicks"]>0, kw["Revenue"]/kw["Clicks"], 0)
        top_kw = kw.sort_values("Rev_per_Click", ascending=False).head(3)
        if not top_kw.empty:
            insights.append({"Insight":"Top Keywords by Rev/Click", "Keywords": ", ".join(top_kw["Keyword"].astype(str).tolist())})

    if not insights:
        st.info("No automated insights available for the selected filters.")
    else:
        # Present insights as a styled dataframe & provide download
        ins_df = pd.DataFrame(insights)
        # Clean small dicts for display
        st.markdown('<div class="small-muted">Insights</div>', unsafe_allow_html=True)
        render_index_safe_table(ins_df)
        download_df(ins_df, "automated_insights.csv", label="Download insights (CSV)")

    # -------------------------
    # Export full filtered dataset
    # -------------------------
    st.markdown('<div class="section-title">Export & Predict</div>', unsafe_allow_html=True)
    st.download_button("Download filtered dataset (all rows)", filt.to_csv(index=False), "content_seo_filtered.csv", "text/csv")

    # Naive conversion predictor (median conversion-rate)
    if "Clicks" in filt.columns and "Conversions" in filt.columns:
        if st.button("Predict conversions (naive median conversion rate)"):
            try:
                filt = filt.copy()
                filt["Conversion_Rate"] = np.where(filt["Clicks"]>0, filt["Conversions"]/filt["Clicks"], 0)
                median_conv = filt["Conversion_Rate"].median()
                preds = (filt["Clicks"] * median_conv).round().astype(int)
                out = filt.copy()
                out["Predicted_Conversions"] = preds
                st.markdown('<div class="small-muted">Predicted conversions (naive)</div>', unsafe_allow_html=True)
                st.dataframe(out.head(10), use_container_width=True)
                download_df(out[["Page","Date","Clicks","Predicted_Conversions"]].head(500), "predicted_conversions.csv", label="Download predicted conversions (sample)")
            except Exception as e:
                st.error("Prediction failed: " + str(e))
    else:
        st.markdown('<div class="small-muted">Clicks & Conversions columns required for naive prediction.</div>', unsafe_allow_html=True)

    st.markdown("### Done — export what you need")

# End of file
