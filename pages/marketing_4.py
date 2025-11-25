# ====================================================================================
# marketing_performance_app.py
# FULLY STANDARDIZED — Marketing Intelligence & Forecasting Lab UI SYSTEM
# PURE BLACK TEXT GLOBAL / BLUE TEXT IN CARDS + KPIs + VARIABLE BOXES
# STRICT 3-TAB STRUCTURE / HOVER LIFT / FADE-IN ANIMATION / INDEX-SAFE TABLE RENDERING
# ====================================================================================

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
st.set_page_config(page_title="Marketing Performance Analysis",
                   layout="wide",
                   initial_sidebar_state="collapsed")

# -------------------------
# Constants / UI tokens
# -------------------------
BLUE = "#064b86"
BLACK = "#000000"
BASE_FONT_SIZE_PX = 16  

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
# CSS — STRICT MASTER SPEC
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

.block-container {{
    padding-top: 20px !important;
}}

/* Fade-in animation */
.fade-in {{
  animation: fadeIn 0.45s ease-in-out;
}}
@keyframes fadeIn {{
  from {{ opacity: 0; transform: translateY(6px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}

/* Section Title */
.section-title {{
    color: {BLACK};
    font-size: 22px;
    font-weight: 500;
    margin: 20px 0 12px 0;
    position: relative;
    display: inline-block;
}}
.section-title:hover::after {{
    content:"";
    width: 40%;
    position:absolute;
    left:0;
    bottom:-6px;
    height:3px;
    background:{BLUE};
    border-radius:2px;
}}

/* Card */
.card {{
    background: #ffffff;
    border: 1px solid #e6e6e6;
    border-radius: 13px;
    padding: 18px;
    color: {BLUE};
    box-shadow:0 5px 18px rgba(0,0,0,0.06);
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}}
.card:hover {{
    transform: translateY(-6px);
    border-color: {BLUE};
    box-shadow:0 14px 34px rgba(6,75,134,0.14);
}}

/* KPI Card */
.kpi-card {{
    background:#ffffff;
    color:{BLUE};
    border-radius:12px;
    border:1px solid #e6e6e6;
    padding:15px;
    text-align:center;
    font-weight:600;
    transition: all .2s ease;
    box-shadow:0 4px 14px rgba(0,0,0,0.06);
}}
.kpi-card:hover {{
    transform:translateY(-6px);
    box-shadow:0 18px 40px rgba(6,75,134,0.18);
}}
.kpi-value {{
    font-size: 20px;
    margin-top:6px;
    color:{BLUE};
}}

/* Variable Box */
.variable-box {{
    background:#ffffff;
    color:{BLUE};
    border-radius:12px;
    padding:14px;
    border:1px solid #e6e6e6;
    text-align:center;
    font-weight:600;
    box-shadow:0 4px 12px rgba(0,0,0,0.05);
    transition:all .2s ease;
}}
.variable-box:hover {{
    transform:translateY(-6px);
    box-shadow:0 14px 34px rgba(6,75,134,0.14);
    border-color:{BLUE};
}}

/* Required Table */
.required-table {{
    width:100%;
    border-collapse:collapse;
    font-size:17px;
    color:{BLACK};
}}
.required-table thead th {{
    border-bottom:2px solid #000;
    padding:10px;
    font-weight:600;
}}
.required-table tbody td {{
    padding:10px;
    border-bottom:1px solid #efefef;
}}
.required-table tbody tr:hover {{
    background:#fafafa;
}}

/* Logo */
.logo-row {{
    display:flex;
    align-items:center;
    gap:12px;
    margin-bottom:6px;
}}
.app-title {{
    color:{BLUE};
    font-size:32px;
    font-weight:700;
}}
.app-subtitle {{
    font-size:14px;
    color:{BLACK};
}}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header (logo + title)
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

st.markdown(f"""
<div class="fade-in logo-row">
    <img src="{logo_url}" width="52"/>
    <div>
        <div class="app-title">Analytics Avenue & Advanced Analytics</div>
        <div class="app-subtitle">Marketing Performance Analysis</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# =======================================================================================
# TAB 1 — OVERVIEW
# =======================================================================================
with tab1:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        Provides campaign, channel, creative, demographic and device-level marketing insights. Predicts revenue & conversions using ML and generates automated insights for decision-making.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        • Multi-channel analytics<br>
        • Creative-level breakdowns<br>
        • Machine learning revenue forecasting<br>
        • Automated insights engine<br>
        • Executive-ready KPI system<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Impact</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        • Improves ROI<br>
        • Reduces wasted spend<br>
        • Enhances audience targeting<br>
        • Strengthens predictive decision-making<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =======================================================================================
# TAB 2 — IMPORTANT ATTRIBUTES
# =======================================================================================
with tab2:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Variables</div>', unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("""
            <div class="variable-box">
                Independent Variables<br><br>
                Campaign · Channel · Device · AgeGroup · Gender · Impressions · Clicks · Spend · AdSet · Creative
            </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown("""
            <div class="variable-box">
                Dependent Variables<br><br>
                Revenue · Conversions · ROAS · Leads
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Required Columns</div>', unsafe_allow_html=True)
    req_df = pd.DataFrame({"Column": REQUIRED_MARKETING_COLS})
    render_required_table(req_df)

    st.markdown('</div>', unsafe_allow_html=True)

# =======================================================================================
# TAB 3 — APPLICATION
# =======================================================================================
with tab3:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Application</div>', unsafe_allow_html=True)

    # ===========================
    # STEP 1: LOAD DATASET
    # ===========================
    st.markdown('<div class="section-title">Step 1 — Load dataset</div>', unsafe_allow_html=True)
    mode = st.radio("Dataset option:", 
                    ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], 
                    horizontal=True)

    df = None

    if mode == "Default dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/marketing.csv"
        df = pd.read_csv(URL)
        df.columns = df.columns.str.strip()
        df = auto_map_columns(df)
        st.success("Default dataset loaded")
        render_required_table(df.head(5))

    elif mode == "Upload CSV":
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up:
            df = pd.read_csv(up)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("File uploaded successfully")
            render_required_table(df.head(5))

    else:
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.markdown("Preview (first 5 rows):")
            render_required_table(raw.head(5))

            mapping = {}
            opts = ["-- Select --"] + list(raw.columns)
            for req in REQUIRED_MARKETING_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", opts)

            if st.button("Apply mapping"):
                missing = [k for k,v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Map all fields: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v:k for k,v in mapping.items()})
                    st.success("Mapping applied")
                    render_required_table(df.head(5))

    if df is None:
        st.stop()

    df = df[[c for c in REQUIRED_MARKETING_COLS if c in df.columns]].copy()
    df = ensure_datetime(df, "Date")
    df = safe_numeric(df, ["Impressions","Clicks","Leads","Conversions","Spend","Revenue","ROAS"])

    # ===========================
    # STEP 2: FILTERS
    # ===========================
    st.markdown('<div class="section-title">Step 2 — Filters & preview</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2,2,1])
    campaigns = sorted(df["Campaign"].dropna().unique()) if "Campaign" in df.columns else []
    channels  = sorted(df["Channel"].dropna().unique()) if "Channel" in df.columns else []

    with c1:
        sel_campaigns = st.multiselect("Campaign", campaigns, default=campaigns[:5])
    with c2:
        sel_channels = st.multiselect("Channel", channels, default=channels[:3])
    with c3:
        dmin = df["Date"].min().date()
        dmax = df["Date"].max().date()
        date_range = st.date_input("Date range", (dmin, dmax))

    filt = df.copy()
    if sel_campaigns:
        filt = filt[filt["Campaign"].isin(sel_campaigns)]
    if sel_channels:
        filt = filt[filt["Channel"].isin(sel_channels)]

    try:
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1])
        filt = filt[(filt["Date"]>=start_dt) & (filt["Date"]<=end_dt)]
    except:
        pass

    st.markdown("Preview (first 5 rows)")
    render_required_table(filt.head(5))
    download_df(filt.head(5), "filtered_preview.csv")

    # ===========================
    # KPIs
    # ===========================
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)

    total_rev = filt["Revenue"].sum() if "Revenue" in filt.columns else 0
    avg_roas  = filt["ROAS"].mean() if "ROAS" in filt.columns else 0
    total_leads = filt["Leads"].sum() if "Leads" in filt.columns else 0
    conv_rate = (filt["Conversions"].sum() / max(filt["Clicks"].sum(),1)) if ("Conversions" in filt.columns and "Clicks" in filt.columns) else 0

    k1.markdown(f"""<div class="kpi-card">Total Revenue<div class="kpi-value">{to_currency(total_rev)}</div></div>""", unsafe_allow_html=True)
    k2.markdown(f"""<div class="kpi-card">ROAS<div class="kpi-value">{avg_roas:.2f}</div></div>""", unsafe_allow_html=True)
    k3.markdown(f"""<div class="kpi-card">Total Leads<div class="kpi-value">{int(total_leads)}</div></div>""", unsafe_allow_html=True)
    k4.markdown(f"""<div class="kpi-card">Conversion Rate<div class="kpi-value">{conv_rate:.2%}</div></div>""", unsafe_allow_html=True)

    # ===========================
    # CHARTS
    # ===========================
    st.markdown('<div class="section-title">Charts</div>', unsafe_allow_html=True)

    # Campaign Revenue & Conversions
    if "Campaign" in filt.columns and ("Revenue" in filt.columns or "Conversions" in filt.columns):
        agg = filt.groupby("Campaign")[["Revenue","Conversions"]].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Revenue"], name="Revenue"))
        fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Conversions"], name="Conversions"))
        fig.update_layout(barmode='group', plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    # Device / Gender / AgeGroup performance
    group_cols = ["Device","Gender","AgeGroup"]
    for g in group_cols:
        if g in filt.columns and "Revenue" in filt.columns:
            grp = filt.groupby(g)["Revenue"].sum().reset_index()
            fig = px.bar(grp, x=g, y="Revenue", text="Revenue")
            fig.update_traces(textposition="outside")
            fig.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

    # ===========================
    # ML: Revenue Prediction
    # ===========================
    st.markdown('<div class="section-title">ML: Predict Revenue</div>', unsafe_allow_html=True)

    ml_df = filt.dropna(subset=["Revenue"]) if "Revenue" in filt.columns else pd.DataFrame()
    feat_cols = ["Channel","Campaign","Device","AgeGroup","Gender","AdSet","Impressions","Clicks","Spend"]
    feat_cols = [c for c in feat_cols if c in ml_df.columns]

    if ml_df.shape[0] < 30 or len(feat_cols) < 2:
        st.info("Not enough data for ML model (>=30 rows).")
    else:
        X = ml_df[feat_cols]
        y = ml_df["Revenue"]
        cat_cols = [c for c in X.columns if X[c].dtype=="object"]
        num_cols = [c for c in X.columns if c not in cat_cols]

        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols)
        ])
        X_t = preprocessor.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = math.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        st.markdown(f"<div class='card'>RMSE: {rmse:.2f}  |  R²: {r2:.3f}</div>", unsafe_allow_html=True)

        out = pd.DataFrame({"Actual": y_test, "Predicted": preds})
        render_required_table(out.head(10))
        download_df(out, "ml_revenue_predictions.csv")

    # ===========================
    # FORECASTING
    # ===========================
    st.markdown('<div class="section-title">Forecasting</div>', unsafe_allow_html=True)

    if "Date" in filt.columns and "Revenue" in filt.columns:
        daily = filt.groupby(pd.Grouper(key="Date", freq="D"))["Revenue"].sum().reset_index()
        if daily.shape[0] >= 10:
            X_d = np.arange(len(daily)).reshape(-1,1)
            y_d = daily["Revenue"].values
            lr = LinearRegression()
            lr.fit(X_d, y_d)
            future_idx = np.arange(len(daily), len(daily)+30).reshape(-1,1)
            preds = lr.predict(future_idx)
            future_dates = pd.date_range(daily["Date"].max()+pd.Timedelta(days=1), periods=30)
            df_forecast = pd.DataFrame({"Date": future_dates, "Forecast_Revenue": preds})

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily["Date"], y=daily["Revenue"], name="Actual"))
            fig.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["Forecast_Revenue"], name="Forecast"))
            fig.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

            download_df(df_forecast, "revenue_30day_forecast.csv")
        else:
            st.info("Not enough days for forecast (>=10).")

    # ===========================
    # AUTOMATED INSIGHTS
    # ===========================
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)

    insights = []
    if "Channel" in filt.columns and "Revenue" in filt.columns and "Spend" in filt.columns:
        ch = filt.groupby("Channel")[["Revenue","Spend"]].sum().reset_index()
        ch["ROI"] = np.where(ch["Spend"]>0, ch["Revenue"]/ch["Spend"], 0)
        if not ch.empty:
            best = ch.loc[ch["ROI"].idxmax()]
            worst = ch.loc[ch["ROI"].idxmin()]
            insights.append({"Insight":"Best Channel ROI","Channel":best["Channel"],"ROI":best["ROI"]})
            insights.append({"Insight":"Lowest Channel ROI","Channel":worst["Channel"],"ROI":worst["ROI"]})

    if "Creative" in filt.columns and "Revenue" in filt.columns:
        cr = filt.groupby("Creative")["Revenue"].sum().reset_index()
        if not cr.empty:
            top = cr.iloc[cr["Revenue"].idxmax()]
            insights.append({"Insight":"Top Creative","Creative":top["Creative"],"Revenue":top["Revenue"]})

    if insights:
        df_ins = pd.DataFrame(insights)
        render_required_table(df_ins)
        download_df(df_ins, "automated_insights.csv")
    else:
        st.markdown('<div class="card">No insights available.</div>', unsafe_allow_html=True)

    # ===========================
    # EXPORTS
    # ===========================
    st.markdown('<div class="section-title">Exports</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">Use download buttons above to export: filtered previews, ML outputs, forecasting, insights.</div>
    """, unsafe_allow_html=True)

# END OF FILE
