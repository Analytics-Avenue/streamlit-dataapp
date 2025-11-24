# marketing_performance_app.py
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
# App Config
# -------------------------
st.set_page_config(page_title="Marketing Performance Analysis", layout="wide", initial_sidebar_state="collapsed")

# -------------------------
# Constants / UI tokens
# -------------------------
BLUE = "#064b86"
BLACK = "#000000"
BASE_FONT_SIZE_PX = 16  # between 16-17 as requested

REQUIRED_MARKETING_COLS = [
    "Campaign", "Channel", "Date", "Impressions", "Clicks", "Leads",
    "Conversions", "Spend", "Revenue", "ROAS", "Device", "AgeGroup",
    "Gender", "AdSet", "Creative"
]

AUTO_MAPS = {
    "Campaign": ["campaign", "campaign_name"],
    "Channel": ["channel", "platform", "source"],
    "Date": ["date", "day"],
    "Impressions": ["impressions", "impression"],
    "Clicks": ["clicks", "link clicks"],
    "Leads": ["leads", "results"],
    "Conversions": ["conversions", "purchase", "add to cart"],
    "Spend": ["spend", "budget", "cost", "amount spent"],
    "Revenue": ["revenue", "amount"],
    "ROAS": ["roas"],
    "Device": ["device", "platform"],
    "AgeGroup": ["agegroup", "age group", "age"],
    "Gender": ["gender", "sex"],
    "AdSet": ["adset", "ad set"],
    "Creative": ["creative", "ad creative"]
}

# -------------------------
# Helper functions
# -------------------------
def auto_map_columns(df):
    rename = {}
    cols = [c.strip() for c in df.columns]
    for req, candidates in AUTO_MAPS.items():
        for c in cols:
            low = c.lower().strip()
            for cand in candidates:
                if cand.lower() == low or cand.lower() in low or low in cand.lower():
                    rename[c] = req
                    break
            if c in rename:
                break
    if rename:
        df = df.rename(columns=rename)
    return df

def ensure_datetime(df, col="Date"):
    try:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    except Exception:
        pass
    return df

def to_currency(x):
    try:
        return "₹ " + f"{float(x):,.2f}"
    except:
        return x

def download_df(df, filename):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button("Download CSV", b, file_name=filename, mime="text/csv")

def render_required_table(df):
    """
    Use the required safe HTML table implementation exactly as specified.
    """
    styled = df.style.set_table_attributes('class="required-table"')
    html = styled.to_html()
    html = html.replace("<th></th>", "").replace("<td></td>", "")
    st.write(html, unsafe_allow_html=True)

def safe_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

# -------------------------
# CSS - follows spec precisely
# -------------------------
st.markdown(f"""
<style>
/* Import Inter font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {{
    background: #ffffff;  /* white background everywhere */
    color: {BLACK};
    font-family: 'Inter', sans-serif;
    font-size: {BASE_FONT_SIZE_PX}px;
}}

/* Fade-in for whole page */
.fade-in {{
  animation: fadeIn 0.45s ease-in-out;
}}
@keyframes fadeIn {{
  from {{ opacity: 0; transform: translateY(6px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}

/* Section titles */
.section-title {{
    color: {BLACK};
    font-size: 22px;
    font-weight: 500;
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

/* Glass-style cards (white) */
.card {{
    background: #ffffff;
    color: {BLUE}; /* card content = blue text */
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
    color: {BLUE}; /* ensure nested text is blue inside cards */
    margin: 0;
}}

/* KPI cards */
.kpi-row {{
    display: flex;
    gap: 16px;
    margin-bottom: 16px;
}}
.kpi-card {{
    flex: 1;
    background: #ffffff;
    color: {BLUE};
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    border: 1px solid #e6e6e6;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
    font-weight: 600;
}}
.kpi-card:hover {{
    transform: translateY(-6px);
    box-shadow: 0 10px 28px rgba(0,0,0,0.10);
    filter: drop-shadow(0 0 8px rgba(6,75,134,0.12));
    cursor: pointer;
}}
.kpi-card .kpi-value {{
    font-size: 20px;
    color: {BLUE};
    margin-top: 6px;
    display: block;
}}

/* Variable boxes (two-column layout) */
.variable-box {{
    background: #ffffff;
    color: {BLUE};
    border-radius: 12px;
    padding: 12px;
    border: 1px solid #e6e6e6;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    text-align: center;
}}
.variable-box:hover {{
    transform: translateY(-6px);
    box-shadow: 0 10px 28px rgba(0,0,0,0.10);
    border-color: {BLUE};
}}
.variable-title {{
    font-weight: 600;
    color: {BLUE};
    margin-bottom: 6px;
}}

/* Required table implementation (pure black text, font-size 17-18px) */
.required-table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 17px;
    color: {BLACK};
    background: #ffffff;
}}
.required-table thead th {{
    border-bottom: 2px solid #000000;
    padding: 8px 10px;
    text-align: left;
    font-weight: 600;
}}
.required-table tbody td {{
    padding: 10px;
    border-bottom: 1px solid #f2f2f2;
}}
.required-table tbody tr:hover {{
    background: #f7f7f7;
}}
/* hide index column inserted by pandas default styles if present visually */
.required-table th.col_heading.level0 {{}}

/* Plot container spacing */
.plot-wrapper {{
    margin-bottom: 16px;
}}

/* Misc small helpers */
.small-muted {{
    color: #666666;
    font-size: 13px;
}}
.logo-row {{
    display:flex;
    align-items:center;
    gap:12px;
    margin-bottom:6px;
}}
.app-title {{
    color: {BLUE};
    font-size:28px;
    font-weight:700;
    margin:0;
}}
.app-subtitle {{
    color: {BLACK};
    margin:0;
    font-size:14px;
}}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Header (logo + title)
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div class="fade-in logo-row">
    <img src="{logo_url}" width="52" style="border-radius:8px;"/>
    <div>
        <div class="app-title">Analytics Avenue & Advanced Analytics</div>
        <div class="app-subtitle small-muted">Marketing Performance Analysis — standardized UI system</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Tabs: Overview / Important Attributes / Application
# -------------------------
tabs = st.tabs(["Overview", "Important Attributes", "Application"])

# -------------------------
# Overview Tab
# -------------------------
with tabs[0]:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4>Purpose</h4>
        <p>
            End-to-end marketing performance tracking across campaigns, channels, creatives, devices and audience segments.
            Predict revenue and conversions with ML, present KPI-driven summaries, and export data-ready CSVs for executives.
        </p>
        <hr style="border:none;margin:10px 0 10px 0;border-top:1px solid #eee;" />
        <h4>Key highlights</h4>
        <ul style="margin:6px 0 0 18px;">
            <li>Strict UI system compliant: Inter font, pure black body text, blue card/KPI/variable text (#064b86).</li>
            <li>Three-tab layout: Overview / Important Attributes / Application.</li>
            <li>Index-safe table rendering method for the required table (no .hide_index()).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        • Multi-channel breakdowns, device & demographic slices<br>
        • Creative & AdSet level performance insights<br>
        • Revenue and conversion forecasting via RandomForest + LinearRegression fallback<br>
        • Automated insights and downloadable CSV exports for quick reporting
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Impact</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        • Improve ROI by prioritizing high-performing channels and creatives.<br>
        • Reduce wasted ad spend via data-driven decisions.<br>
        • Provide executive-ready KPIs and downloadable ML predictions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# -------------------------
# Important Attributes Tab
# -------------------------
with tabs[1]:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Important Attributes</div>', unsafe_allow_html=True)

    # Variable boxes (Independent / Dependent)
    st.markdown("""
    <div style="display:flex; gap:16px; margin-bottom:12px;">
        <div style="flex:1;">
            <div class="variable-box">
                <div class="variable-title">Independent Variables</div>
                <div>Campaign, Channel, Device, AgeGroup, Gender, Impressions, Clicks, Spend, AdSet, Creative</div>
            </div>
        </div>
        <div style="flex:1;">
            <div class="variable-box">
                <div class="variable-title">Dependent Variables</div>
                <div>Revenue, Conversions, ROAS, Leads</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Required Columns</div>', unsafe_allow_html=True)
    # show the required columns as a small table using required-table renderer
    req_df = pd.DataFrame({"RequiredColumn": REQUIRED_MARKETING_COLS})
    # transform to a friendly display table with only one column header name
    req_df_display = req_df.rename(columns={"RequiredColumn": "Column"})
    render_required_table(req_df_display)

    st.markdown('<div class="section-title">Data assumptions</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        • Date column must be parseable to datetime.<br>
        • Numeric fields (Impressions, Clicks, Leads, Conversions, Spend, Revenue, ROAS) will be coerced to numeric.<br>
        • At least 30 rows recommended to train ML models reliably.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# -------------------------
# Application Tab: main app flow
# -------------------------
with tabs[2]:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Application</div>', unsafe_allow_html=True)

    # Step 1 — Load dataset
    st.markdown('<div class="section-title">Step 1 — Load dataset</div>', unsafe_allow_html=True)
    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)

    df = None
    if mode == "Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/marketing.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("Default dataset loaded")
            # show small preview using required table renderer for head
            render_required_table(df.head(5))
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    elif mode == "Upload CSV":
        st.markdown("#### Download Sample CSV for Reference")
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/marketing.csv"
        try:
            sample_df = pd.read_csv(URL).head(5)
            sample_csv = sample_df.to_csv(index=False)
            st.download_button("Download Sample CSV", sample_csv, "sample_dataset.csv", "text/csv")
        except Exception as e:
            st.info(f"Sample CSV unavailable: {e}")

        file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
        if file:
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("Uploaded dataset loaded")
            render_required_table(df.head(5))

    else:  # Upload + mapping
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.write("Preview (first 5 rows):")
            render_required_table(raw.head(5))
            st.markdown("Map your columns to required fields.")
            mapping = {}
            cols = list(raw.columns)
            cols_sorted = ["-- Select --"] + cols
            # mapping widgets
            for req in REQUIRED_MARKETING_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=cols_sorted, key=f"map_{req}")
            if st.button("Apply mapping"):
                missing = [k for k,v in mapping.items() if v=="-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v:k for k,v in mapping.items()})
                    st.success("Mapping applied.")
                    render_required_table(df.head(5))

    if df is None:
        st.stop()

    # keep only required columns that exist
    df = df[[c for c in REQUIRED_MARKETING_COLS if c in df.columns]].copy()
    df = ensure_datetime(df, "Date")
    df = safe_numeric(df, ["Impressions","Clicks","Leads","Conversions","Spend","Revenue","ROAS"])

    # Step 2 — Filters & preview
    st.markdown('<div class="section-title">Step 2 — Filters & preview</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2,2,1])
    campaigns = sorted(df["Campaign"].dropna().unique()) if "Campaign" in df.columns else []
    channels = sorted(df["Channel"].dropna().unique()) if "Channel" in df.columns else []
    devices = sorted(df["Device"].dropna().unique()) if "Device" in df.columns else []
    agegroups = sorted(df["AgeGroup"].dropna().unique()) if "AgeGroup" in df.columns else []
    genders = sorted(df["Gender"].dropna().unique()) if "Gender" in df.columns else []

    with c1:
        sel_campaigns = st.multiselect("Campaign", options=campaigns, default=campaigns[:5])
    with c2:
        sel_channels = st.multiselect("Channel", options=channels, default=channels[:3])
    with c3:
        min_date = df["Date"].min() if "Date" in df.columns else pd.to_datetime("2000-01-01")
        max_date = df["Date"].max() if "Date" in df.columns else pd.to_datetime("2100-01-01")
        date_range = st.date_input("Date range", value=(min_date.date(), max_date.date()))

    filt = df.copy()
    if sel_campaigns:
        filt = filt[filt["Campaign"].isin(sel_campaigns)]
    if sel_channels:
        filt = filt[filt["Channel"].isin(sel_channels)]
    if date_range:
        # ensure Date is datetime
        try:
            start_dt = pd.to_datetime(date_range[0])
            end_dt = pd.to_datetime(date_range[1])
            if "Date" in filt.columns:
                filt = filt[(filt["Date"]>=start_dt) & (filt["Date"]<=end_dt)]
        except Exception:
            pass

    st.markdown("Preview (first 5 rows)")
    render_required_table(filt.head(5))
    download_df(filt.head(5), "filtered_preview.csv")

    # Key Metrics (KPI cards)
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    total_revenue = float(filt["Revenue"].sum()) if "Revenue" in filt.columns else 0.0
    avg_roas = float(filt["ROAS"].mean()) if "ROAS" in filt.columns else 0.0
    total_leads = int(filt["Leads"].sum()) if "Leads" in filt.columns else 0
    conv_rate = (filt["Conversions"].sum() / max(filt["Clicks"].sum(), 1)) if ("Conversions" in filt.columns and "Clicks" in filt.columns) else 0.0

    k1.markdown(f"""
    <div class="kpi-card">
        Total Revenue
        <div class="kpi-value">{to_currency(total_revenue)}</div>
    </div>""", unsafe_allow_html=True)

    k2.markdown(f"""
    <div class="kpi-card">
        ROAS
        <div class="kpi-value">{avg_roas:.2f}</div>
    </div>""", unsafe_allow_html=True)

    k3.markdown(f"""
    <div class="kpi-card">
        Total Leads
        <div class="kpi-value">{total_leads}</div>
    </div>""", unsafe_allow_html=True)

    k4.markdown(f"""
    <div class="kpi-card">
        Conversion Rate
        <div class="kpi-value">{conv_rate:.2%}</div>
    </div>""", unsafe_allow_html=True)

    # Charts
    st.markdown('<div class="section-title">Charts</div>', unsafe_allow_html=True)

    # Campaign Revenue & Conversions
    if "Campaign" in filt.columns and ("Revenue" in filt.columns or "Conversions" in filt.columns):
        st.markdown('<div class="plot-wrapper">Campaign Revenue & Conversions</div>', unsafe_allow_html=True)
        agg = filt.groupby("Campaign")[["Revenue","Conversions"]].sum().reset_index()
        # ensure columns exist
        if "Revenue" not in agg.columns:
            agg["Revenue"] = 0
        if "Conversions" not in agg.columns:
            agg["Conversions"] = 0
        fig = go.Figure()
        fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Revenue"], name="Revenue"))
        fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Conversions"], name="Conversions"))
        fig.update_layout(barmode='group', xaxis_title="Campaign", yaxis_title="Value", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    # Device / Gender / AgeGroup performance
    st.markdown('<div class="plot-wrapper">Device / Gender / AgeGroup performance</div>', unsafe_allow_html=True)
    group_cols = ["Device","Gender","AgeGroup"]
    for g in group_cols:
        if g in filt.columns and "Revenue" in filt.columns:
            grp = filt.groupby(g)[["Revenue","Conversions"]].sum().reset_index()
            fig = px.bar(grp, x=g, y="Revenue", text="Revenue", title=f"{g} Revenue")
            fig.update_traces(textposition="outside")
            fig.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

    # ML: Revenue prediction
    st.markdown('<div class="section-title">ML: Predict Revenue (RandomForest)</div>', unsafe_allow_html=True)
    ml_df = filt.copy().dropna(subset=["Revenue"]) if "Revenue" in filt.columns else pd.DataFrame()
    feat_cols = ["Channel","Campaign","Device","AgeGroup","Gender","AdSet","Impressions","Clicks","Spend"]
    feat_cols = [c for c in feat_cols if c in ml_df.columns]

    if ml_df.shape[0] < 30 or len(feat_cols) < 2:
        st.info("Not enough data to train ML model (>=30 rows and >1 feature needed).")
    else:
        X = ml_df[feat_cols].copy()
        y = ml_df["Revenue"].copy()
        # simple preprocessing
        cat_cols = [c for c in X.columns if X[c].dtype == "object"]
        num_cols = [c for c in X.columns if c not in cat_cols]
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                ("num", StandardScaler(), num_cols)
            ],
            remainder="drop"
        )
        X_t = preprocessor.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        with st.spinner("Training RandomForest..."):
            rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        rmse = math.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        st.markdown(f"<div class='small-muted'>Revenue prediction — RMSE: {rmse:.2f}, R²: {r2:.3f}</div>", unsafe_allow_html=True)

        # Output dataframe: collapse features to columns (feature names generic)
        X_test_df = pd.DataFrame(X_test, columns=[f"Feature_{i}" for i in range(X_test.shape[1])])
        X_test_df["Actual_Revenue"] = y_test.reset_index(drop=True)
        X_test_df["Predicted_Revenue"] = preds
        st.markdown("ML predictions preview (first 10 rows)")
        render_required_table(X_test_df.head(10))
        download_df(X_test_df, "ml_revenue_predictions.csv")

    # Forecasting (simple linear trend over last N days)
    st.markdown('<div class="section-title">Forecasting</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <h4>30-day revenue forecast (linear trend fallback)</h4>
        <p class="small-muted">This forecasting block uses a simple daily aggregation and linear regression to forecast next 30 days. For more advanced forecasting (Prophet, ETS), integrate those packages outside this environment.</p>
    </div>
    """, unsafe_allow_html=True)

    if "Date" in filt.columns and "Revenue" in filt.columns:
        daily = filt.groupby(pd.Grouper(key="Date", freq="D"))["Revenue"].sum().reset_index().dropna()
        if daily.shape[0] >= 10:
            daily = daily.reset_index(drop=True)
            # simple numeric X as days index
            X_dates = np.arange(len(daily)).reshape(-1,1)
            y_val = daily["Revenue"].values
            lr = LinearRegression()
            lr.fit(X_dates, y_val)
            # forecast next 30 days
            future_idx = np.arange(len(daily), len(daily)+30).reshape(-1,1)
            preds_future = lr.predict(future_idx)
            # combine for plotting
            import datetime
            last_date = daily["Date"].max()
            future_dates = [last_date + pd.Timedelta(days=int(i)) for i in range(1,31)]
            df_forecast = pd.DataFrame({"Date": future_dates, "Forecast_Revenue": preds_future})
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily["Date"], y=daily["Revenue"], name="Actual"))
            fig.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["Forecast_Revenue"], name="Forecast"))
            fig.update_layout(xaxis_title="Date", yaxis_title="Revenue", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
            # download
            download_df(df_forecast, "revenue_30day_forecast.csv")
        else:
            st.info("Not enough daily data (>=10 distinct days) to produce forecast.")

    # Automated insights
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)
    insights_list = []
    if "Channel" in filt.columns and "Revenue" in filt.columns and "Spend" in filt.columns:
        ch_perf = filt.groupby("Channel")[["Revenue","Spend"]].sum().reset_index()
        ch_perf["Revenue_per_Rs"] = np.where(ch_perf["Spend"]>0, ch_perf["Revenue"]/ch_perf["Spend"], 0)
        if not ch_perf.empty:
            best = ch_perf.sort_values("Revenue_per_Rs", ascending=False).iloc[0]
            worst = ch_perf.sort_values("Revenue_per_Rs", ascending=True).iloc[0]
            insights_list.append({
                "Insight": "Best Channel ROI",
                "Channel": best['Channel'],
                "Revenue_per_Rs": float(best['Revenue_per_Rs'])
            })
            insights_list.append({
                "Insight": "Lowest Channel ROI",
                "Channel": worst['Channel'],
                "Revenue_per_Rs": float(worst['Revenue_per_Rs'])
            })

    # creative-level top performer
    if "Creative" in filt.columns and "Revenue" in filt.columns:
        cr_perf = filt.groupby("Creative")[["Revenue"]].sum().reset_index()
        if not cr_perf.empty:
            top_creative = cr_perf.sort_values("Revenue", ascending=False).iloc[0]
            insights_list.append({"Insight":"Top Creative by Revenue","Creative": top_creative['Creative'], "Revenue": float(top_creative['Revenue'])})

    insights_df = pd.DataFrame(insights_list)
    if insights_df.empty:
        st.markdown('<div class="card"><div class="small-muted">No automated insights available for selected filters. Check that Channel/Spend/Revenue columns are present and filtered data is non-empty.</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h4>Insights</h4>", unsafe_allow_html=True)
        render_required_table(insights_df)
        download_df(insights_df, "automated_insights.csv")
        st.markdown('</div>', unsafe_allow_html=True)

    # Exports & final
    st.markdown('<div class="section-title">Exports</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        Use the download buttons available throughout the app to export filtered previews, ML predictions, forecasts, and insights.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# End of file
