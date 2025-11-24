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

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Click & Conversion Analytics", layout="wide")

# Hide default sidebar navigation (optional)
st.markdown("""<style>[data-testid="stSidebarNav"]{display:none;}</style>""", unsafe_allow_html=True)

# -------------------------
# Company Logo + Name
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display: flex; align-items: center; margin-bottom:10px;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:34px; font-weight:700; margin:0; padding:0;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:34px; font-weight:700; margin:0; padding:0;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Utility functions & mappings
# -----------------------------
REQUIRED_MARKETING_COLS = ["Campaign","Channel","Date","Impressions","Clicks","Leads","Conversions","Spend"]

AUTO_MAPS = {
    "Campaign":["campaign name","campaign_name","campaign","Campaign name","Campaign Name"],
    "Channel":["page name","page","channel","source","platform","adset","adset name","placement","medium"],
    "Date":["date","day","reporting starts","reporting ends","start date","end date"],
    "Impressions":["impressions","Impression","Impressions"],
    "Clicks":["link clicks","clicks","all clicks","total clicks"],
    "Leads":["results","leads","lead","cpl results"],
    "Conversions":["conversions","website conversions","purchase","add to cart","complete registration"],
    "Spend":["amount spent (inr)","amount spent","spend","cost","ad spend","budget used"]
}

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

def download_df(df, filename, label="Download CSV"):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")

def to_currency(x):
    try: return "₹ "+f"{float(x):,.2f}"
    except: return x

def ensure_datetime(df, col="Date"):
    try: df[col] = pd.to_datetime(df[col], errors="coerce")
    except: pass
    return df

# -----------------------------
# CSS (consistent with previous app)
# -----------------------------
st.markdown("""
<style>
* { font-family: 'Inter', sans-serif; }

/* Main header */
h1 { font-size: 28px; font-weight:700; margin:0 0 6px 0; color:#000; }

/* Section title (applied across tabs) */
.section-title {
    font-size: 20px !important;
    font-weight: 600 !important;
    margin-top:20px;
    margin-bottom:10px;
    color:#000 !important;
    position:relative;
}
.section-title:after {
    content:"";
    position:absolute;
    bottom:-6px;
    left:0;
    height:3px;
    width:0%;
    background:#064b86;
    transition: width 0.35s ease;
}
.section-title:hover:after { width:45%; }

/* Card */
.card {
    background: #ffffff;
    padding:18px;
    border-radius:12px;
    border:1px solid #e6e6e6;
    font-size:15px;
    color:#064b86 !important;
    font-weight:500;
    box-shadow:0 3px 10px rgba(0,0,0,0.06);
    transition: all 0.22s ease;
}
.card:hover { transform: translateY(-4px); box-shadow:0 10px 22px rgba(6,75,134,0.12); border-color:#064b86; }

/* KPI (blue) */
.metric-card {
    padding:16px;
    border-radius:12px;
    text-align:center;
    font-weight:700;
    font-size:16px;
    color:#064b86 !important;
    border:1px solid #e6e6e6;
    background:#ffffff;
    box-shadow:0 3px 10px rgba(0,0,0,0.06);
}
.metric-card:hover { transform: translateY(-3px); box-shadow:0 12px 22px rgba(6,75,134,0.12); }

/* Variable box (blue) */
.variable-box {
    padding:14px;
    border-radius:12px;
    background:#ffffff;
    border:1px solid #e6e6e6;
    box-shadow:0 2px 8px rgba(0,0,0,0.06);
    transition: 0.2s ease;
    text-align:center;
    font-size:16px !important;
    font-weight:600;
    color:#064b86 !important;
    margin-bottom:12px;
}
.variable-box:hover { transform: translateY(-4px); box-shadow:0 12px 22px rgba(6,75,134,0.12); border-color:#064b86; }

/* Dataframe override for required table (pure black, larger) */
.required-table th {
    background: #ffffff !important;
    color: #000 !important;
    font-size:18px !important;
    border-bottom:2px solid #000 !important;
}
.required-table td {
    color:#000 !important;
    font-size:16px !important;
    padding:8px !important;
    border-bottom:1px solid #efefef !important;
}
.required-table tbody tr:hover td { background:#fbfbfb !important; }

/* Dataframe standard */
.dataframe th { background:#064b86 !important; color:#fff !important; font-size:14px !important; }
.dataframe td { font-size:14px !important; color:#000 !important; }

/* Buttons */
.stButton>button, .stDownloadButton>button {
    background:#064b86 !important;
    color:#fff !important;
    border-radius:8px !important;
    padding:8px 18px !important;
    font-weight:600 !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Tabs: Overview, Important Attributes, Application
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Overview","Important Attributes","Application"])

# -----------------------------
# Overview Tab
# -----------------------------
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("<div class='card'><b>Purpose:</b><br>End-to-end click & conversion analytics to measure campaign performance, forecast key metrics, and run ML experiments for decisions that actually move the needle.</div>", unsafe_allow_html=True)

    colL, colR = st.columns(2)
    with colL:
        st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
        st.markdown("<div class='card'>• Multi-channel campaign aggregation<br>• Time-series forecasting & ML predictions<br>• Automated insights & KPI monitoring<br>• Export-ready outputs for BI & ops</div>", unsafe_allow_html=True)
    with colR:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("<div class='card'>• Lower wasted ad spend<br>• Improve CPL & conversion efficiency<br>• Faster, data-driven budget allocation</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    kk1, kk2, kk3, kk4 = st.columns(4)
    kk1.markdown("<div class='metric-card'>Total Clicks</div>", unsafe_allow_html=True)
    kk2.markdown("<div class='metric-card'>Total Leads</div>", unsafe_allow_html=True)
    kk3.markdown("<div class='metric-card'>Conversion Rate</div>", unsafe_allow_html=True)
    kk4.markdown("<div class='metric-card'>ROAS</div>", unsafe_allow_html=True)

# -----------------------------
# Important Attributes Tab
# -----------------------------
with tab2:
    # Required columns table (pure black, larger font)
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)
    required_dict = {
        "Campaign": "Name of the marketing campaign.",
        "Channel": "Source platform (Facebook, Google, Email etc.).",
        "Date": "Date of the record / activity.",
        "Impressions": "Number of times ad was shown.",
        "Clicks": "Number of clicks on the ad.",
        "Leads": "Number of signups/interest captures.",
        "Conversions": "Desired outcomes (purchase, signup etc.).",
        "Spend": "Amount spent on the campaign (INR)."
    }
    req_df = pd.DataFrame([{"Attribute":k,"Description":v} for k,v in required_dict.items()])

    # inject style then show as styled table
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.dataframe(req_df.style.set_table_attributes('class="required-table"'), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Independent variables (left) and dependent variables (right)
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        indep_vars = ["Campaign","Channel","Date","Impressions","Clicks","Spend"]
        for v in indep_vars:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        dep_vars = ["Leads","Conversions"]
        for v in dep_vars:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# -----------------------------
# Application Tab
# -----------------------------
with tab3:
    st.markdown('<div class="section-title">Application</div>', unsafe_allow_html=True)

    mode = st.radio("Dataset option:", ["Default dataset","Upload CSV","Upload CSV + Column mapping"], horizontal=True)
    df = None

    # --- Default dataset ---
    if mode=="Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/marketing.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("Default dataset loaded")
        except Exception as e:
            st.error(f"Failed to load default dataset: {e}")
            st.stop()

    # --- Upload CSV ---
    elif mode=="Upload CSV":
        try:
            sample_df = pd.read_csv(DEFAULT_URL).head(5)
            st.download_button("Download Sample CSV", sample_df.to_csv(index=False), "sample_dataset.csv","text/csv")
        except: pass

        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("File uploaded successfully")
            download_df(df.head(5), "sample_uploaded_5rows.csv")

    # --- Upload + mapping ---
    else:
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.write("Preview (first 5 rows)")
            st.dataframe(raw.head())
            st.markdown("Map your columns to required fields:")
            mapping = {}
            for req in REQUIRED_MARKETING_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + list(raw.columns))
            if st.button("Apply mapping"):
                missing = [k for k,v in mapping.items() if v=="-- Select --"]
                if missing:
                    st.error("Map all required columns: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v:k for k,v in mapping.items()})
                    st.success("Mapping applied")
                    download_df(df.head(5), "mapped_sample_5rows.csv")

    # Stop if no df
    if df is None:
        st.stop()

    # -------------------------
    # Data prep & derived metrics
    # -------------------------
    df.columns = df.columns.str.strip()
    df = ensure_datetime(df, "Date")
    for col in ["Impressions","Clicks","Leads","Conversions","Spend"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["CTR"] = np.where(df["Impressions"]>0, df["Clicks"]/df["Impressions"], 0)
    df["CPC"] = np.where(df["Clicks"]>0, df["Spend"]/df["Clicks"], 0)
    df["Cost_per_Lead"] = np.where(df["Leads"]>0, df["Spend"]/df["Leads"], np.nan)
    df["Conversion_Rate"] = np.where(df["Clicks"]>0, df["Conversions"]/df["Clicks"], 0)

    st.markdown('<div class="section-title">Filters & Preview</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns([2,2,1])
    campaigns = sorted(df["Campaign"].dropna().unique()) if "Campaign" in df.columns else []
    channels = sorted(df["Channel"].dropna().unique()) if "Channel" in df.columns else []
    with c1:
        sel_campaigns = st.multiselect("Campaign", campaigns, default=campaigns[:5])
    with c2:
        sel_channels = st.multiselect("Channel", channels, default=channels[:3])
    with c3:
        if df["Date"].notna().any():
            date_range = st.date_input("Date range", (df["Date"].min().date(), df["Date"].max().date()))
        else:
            date_range = None

    filt = df.copy()
    if sel_campaigns:
        filt = filt[filt["Campaign"].isin(sel_campaigns)]
    if sel_channels:
        filt = filt[filt["Channel"].isin(sel_channels)]
    if date_range:
        start,end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filt = filt[(filt["Date"]>=start)&(filt["Date"]<=end)]

    st.markdown('<div class="section-title">Filtered Preview</div>', unsafe_allow_html=True)
    st.dataframe(filt.head(), use_container_width=True)
    download_df(filt.head(5), "filtered_preview.csv")

    # -------------------------
    # KPIs
    # -------------------------
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    kk1, kk2, kk3, kk4 = st.columns(4)
    kk1.metric("Total Impressions", f"{int(filt['Impressions'].sum()):,}")
    kk2.metric("Total Clicks", f"{int(filt['Clicks'].sum()):,}")
    kk3.metric("Total Leads", f"{int(filt['Leads'].sum()):,}")
    kk4.metric("Total Spend", to_currency(filt['Spend'].sum()))

    # -------------------------
    # Charts
    # -------------------------
    st.markdown('<div class="section-title">Campaign Performance</div>', unsafe_allow_html=True)
    agg = filt.groupby("Campaign")[["Impressions","Clicks","Leads","Conversions","Spend"]].sum().reset_index().sort_values("Clicks", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Clicks"], name="Clicks", text=agg["Clicks"], textposition="outside"))
    fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Conversions"], name="Conversions", text=agg["Conversions"], textposition="outside"))
    fig.update_layout(barmode='group', xaxis_title="<b>Campaign</b>", yaxis_title="<b>Count</b>", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Channel ROI</div>', unsafe_allow_html=True)
    roi_df = filt.groupby("Channel").agg({"Leads":"sum","Spend":"sum"}).reset_index()
    if not roi_df.empty:
        roi_df["Leads_per_Rs"] = np.where(roi_df["Spend"]>0, roi_df["Leads"]/roi_df["Spend"], 0)
        fig2 = px.bar(roi_df.sort_values("Leads_per_Rs", ascending=False), x="Channel", y="Leads_per_Rs", text=roi_df["Leads_per_Rs"].round(4))
        fig2.update_traces(textposition="outside")
        fig2.update_layout(yaxis_title="Leads per ₹", xaxis_title="<b>Channel</b>", template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Not enough Channel data for ROI chart.")

    st.markdown('<div class="section-title">Time-series: Clicks over Time</div>', unsafe_allow_html=True)
    ts = filt.groupby("Date")["Clicks"].sum().reset_index().sort_values("Date")
    if not ts.empty:
        ts["MA_7"] = ts["Clicks"].rolling(7,min_periods=1).mean()
        fig3 = px.line(ts, x="Date", y=["Clicks","MA_7"], labels={"value":"Clicks","variable":"Series"}, template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Not enough date data.")

    # -------------------------
    # ML: Clicks Regression + Download
    # -------------------------
    st.markdown('<div class="section-title">ML: Clicks Regression</div>', unsafe_allow_html=True)
    ml_df = filt.copy().dropna(subset=["Clicks","Impressions","Spend"])
    if "Date" in ml_df.columns:
        ml_df["dayofweek"] = ml_df["Date"].dt.dayofweek
        ml_df["month"] = ml_df["Date"].dt.month
    feat_cols = [c for c in ["Channel","Campaign","Impressions","Spend","dayofweek","month"] if c in ml_df.columns]
    if len(ml_df)>=40 and len(feat_cols)>=2:
        X = ml_df[feat_cols].copy()
        y = ml_df["Clicks"].astype(float)
        cat_cols = [c for c in X.columns if X[c].dtype=="object"]
        num_cols = [c for c in X.columns if c not in cat_cols]
        preprocessor = ColumnTransformer(transformers=[("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),cat_cols),
                                                       ("num",StandardScaler(),num_cols)], remainder="drop")
        X_t = preprocessor.fit_transform(X)
        X_train,X_test,y_train,y_test = train_test_split(X_t,y,test_size=0.2,random_state=42)
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        with st.spinner("Training RF model..."):
            rf.fit(X_train,y_train)
        preds = rf.predict(X_test)
        st.write(f"RMSE: {math.sqrt(mean_squared_error(y_test,preds)):.2f}, R²: {r2_score(y_test,preds):.3f}")

        X_test_df = pd.DataFrame(X_test, columns=[f"Feature_{i}" for i in range(X_test.shape[1])])
        X_test_df["Actual_Clicks"] = y_test.reset_index(drop=True)
        X_test_df["Predicted_Clicks"] = preds
        st.dataframe(X_test_df.head())
        download_df(X_test_df, "ml_clicks_predictions.csv", label="Download ML predictions")
    else:
        st.info("Not enough rows or features to train ML (need >=40 rows).")

    # -------------------------
    # Forecasting (linear fallback if Prophet not installed)
    # -------------------------
    st.markdown('<div class="section-title">Forecasting Clicks</div>', unsafe_allow_html=True)
    ts_agg = filt.groupby("Date")["Clicks"].sum().reset_index().sort_values("Date")
    if len(ts_agg)>=6 and ts_agg["Clicks"].sum()>0:
        try:
            from prophet import Prophet
            prophet_ok = True
        except:
            prophet_ok = False

        if prophet_ok:
            dfp = ts_agg.rename(columns={"Date":"ds","Clicks":"y"}).set_index("ds").asfreq("D").fillna(0).reset_index()
            m = Prophet(daily_seasonality=False,weekly_seasonality=True,yearly_seasonality=True)
            with st.spinner("Training Prophet..."):
                m.fit(dfp)
            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)
            figf = px.line(forecast, x="ds", y=["yhat","yhat_lower","yhat_upper"], title="Forecast (Prophet)")
            st.plotly_chart(figf, use_container_width=True)
        else:
            ts_lr = ts_agg.set_index("Date").resample("D").sum().fillna(0).reset_index()
            ts_lr["t"] = np.arange(len(ts_lr))
            lr = LinearRegression(); lr.fit(ts_lr[["t"]], ts_lr["Clicks"])
            fut_dates = pd.date_range(ts_lr["Date"].max()+pd.Timedelta(days=1), periods=30)
            preds = lr.predict(np.arange(len(ts_lr), len(ts_lr)+30).reshape(-1,1))
            figf = go.Figure([go.Scatter(x=ts_lr["Date"], y=ts_lr["Clicks"], name="Actual"),
                              go.Scatter(x=fut_dates, y=preds, name="Forecast")])
            st.plotly_chart(figf, use_container_width=True)
    else:
        st.info("Not enough historical data for reliable forecasting (need at least ~6 daily points).")

    # -------------------------
    # Automated Insights (Table + Download)
    # -------------------------
    st.markdown('<div class="section-title">Automated Insights (Table + Download)</div>', unsafe_allow_html=True)
    insights_list = []
    if "Channel" in filt.columns and "Leads" in filt.columns and "Spend" in filt.columns:
        ch_perf = filt.groupby("Channel").agg({"Leads":"sum","Spend":"sum"}).reset_index()
        if not ch_perf.empty:
            ch_perf["Leads_per_Rs"] = np.where(ch_perf["Spend"]>0,ch_perf["Leads"]/ch_perf["Spend"],0)
            best = ch_perf.sort_values("Leads_per_Rs",ascending=False).iloc[0]
            worst = ch_perf.sort_values("Leads_per_Rs",ascending=True).iloc[0]
            insights_list.append({"Insight":"Best Channel", "Channel":best['Channel'], "Leads_per_Rs":best['Leads_per_Rs']})
            insights_list.append({"Insight":"Worst Channel", "Channel":worst['Channel'], "Leads_per_Rs":worst['Leads_per_Rs']})
    if insights_list:
        insights_df = pd.DataFrame(insights_list)
        st.dataframe(insights_df, use_container_width=True)
        download_df(insights_df, "automated_insights.csv", label="Download insights")
    else:
        st.info("No automated insights available for the selected filter.")
