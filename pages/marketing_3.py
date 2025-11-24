# click_conversion_analytics_ui.py
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

# -----------------------------
# Helper functions
# -----------------------------
def download_df(df, filename):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button("Download CSV", b, file_name=filename, mime="text/csv")

def to_currency(x):
    try:
        return "₹ " + f"{float(x):,.2f}"
    except:
        return x

def ensure_datetime(df, col="Date"):
    try:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    except:
        pass
    return df

# -----------------------------
# Auto-mapping logic (same as provided)
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

# -----------------------------
# CSS – follow your master spec
# -----------------------------
st.markdown("""
<style>
/* Global font and base */
* { font-family: 'Inter', sans-serif; }

/* Main header */
.header-wrap { display:flex; align-items:center; gap:12px; margin-bottom:6px; }
.app-title { font-size:34px !important; font-weight:700 !important; color:#000 !important; margin:0; }
.app-sub { color:#222; margin-top:2px; margin-bottom:12px; font-size:14.5px; }

/* Section title (uniform across tabs) */
.section-title {
    font-size:22px !important;
    font-weight:600 !important;
    color:#000 !important;
    margin-top:22px;
    margin-bottom:10px;
    position: relative;
}
.section-title:after {
    content: "";
    position: absolute;
    bottom: -6px;
    left: 0;
    height: 3px;
    width: 0%;
    background: #064b86;
    transition: width 0.35s ease;
}
.section-title:hover:after { width:40%; }

/* Card (glass-style inside) */
.card {
    background: #ffffff;
    padding:18px;
    border-radius:12px;
    border:1px solid #e6e6e6;
    color:#064b86 !important;      /* blue text inside cards */
    box-shadow: 0 6px 22px rgba(0,0,0,0.06);
    transition: transform 0.22s ease, box-shadow 0.22s ease;
}
.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 18px 44px rgba(6,75,134,0.14);
    border-color: #064b86;
}

/* Metric / KPI card */
.metric-card {
    background: #ffffff;
    padding:18px;
    border-radius:12px;
    border:1px solid #e6e6e6;
    text-align:center;
    font-weight:600;
    font-size:16px;
    color:#064b86 !important;
    box-shadow:0 6px 18px rgba(0,0,0,0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow:0 16px 36px rgba(6,75,134,0.14);
}

/* Variable card */
.variable-box {
    padding:14px;
    border-radius:12px;
    background: #ffffff;
    border: 1px solid #e6e6e6;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    font-size:16px !important;
    font-weight:600 !important;
    color:#064b86 !important;
    text-align:center;
    margin-bottom:12px;
}
.variable-box:hover {
    transform: translateY(-6px);
    box-shadow:0 18px 40px rgba(6,75,134,0.12);
    border-color:#064b86;
}

/* Table (Required Columns) - pure black, large font */
.required-table thead th {
    background: #fff !important;
    color: #000 !important;
    font-size:18px !important;
    border-bottom: 2px solid #000 !important;
    padding:10px !important;
}
.required-table tbody td {
    color: #000 !important;
    font-size:17px !important;
    padding:10px !important;
    border-bottom: 1px solid #e6e6e6 !important;
}
.required-table tbody tr:hover td {
    background: #fafafa !important;
}

/* Streamlit elements tweaks (buttons) */
.stButton>button, .stDownloadButton>button {
    background:#064b86 !important;
    color:white !important;
    border-radius:8px !important;
    padding:8px 18px !important;
    font-weight:600 !important;
    border:none !important;
}

/* Fade-in for main container (subtle) */
.block-container { animation: fadeIn 0.45s ease; }
@keyframes fadeIn { from {opacity:0; transform: translateY(6px);} to {opacity:1; transform:translateY(0);} }

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header area
# -----------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div class="header-wrap">
    <img src="{logo_url}" width="60" />
    <div>
        <div class="app-title">Click & Conversion Analytics</div>
        <div class="app-sub">Enterprise-capable marketing analytics: predictions, forecasts, and actionable insights.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Tabs layout (Overview, Important Attributes, Application)
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# -----------------------------
# Overview tab
# -----------------------------
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("<div class='card'><b>Purpose:</b> Centralize campaign signals so teams can stop guessing and start scaling profitably. Consolidates impressions → clicks → leads → conversions, and supports ML forecasts and recommendations.</div>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="card">
            • Multi-channel performance aggregation<br>
            • Full-funnel KPIs & derived metrics (CTR, CPC, CPL, Conversion Rate)<br>
            • Campaign-level and channel-level forecasting<br>
            • ML model for Click prediction + downloadable predictions<br>
            • Automated, exportable insights for quick ops actions
            </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="card">
            • Reduce wasted spend and improve CPL<br>
            • Identify high-performing channels and scale them<br>
            • Faster decisions with automated insights and forecasts<br>
            • Reproducible measurement for planning and budgeting
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Key Performance Indicators</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown('<div class="metric-card">Total Clicks</div>', unsafe_allow_html=True)
    k2.markdown('<div class="metric-card">Total Leads</div>', unsafe_allow_html=True)
    k3.markdown('<div class="metric-card">Conversion Rate</div>', unsafe_allow_html=True)
    k4.markdown('<div class="metric-card">ROAS</div>', unsafe_allow_html=True)

# -----------------------------
# Important Attributes tab (Option 1: Simple)
# -----------------------------
with tab2:
    # Required Columns table (pure black, big font)
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    required_dict = {
        "Campaign": "Name of the marketing campaign.",
        "Channel": "Source platform (Facebook, Google, Email, etc.).",
        "Date": "Date of each campaign activity.",
        "Impressions": "Total number of times ads were shown.",
        "Clicks": "Number of times users clicked the ads.",
        "Leads": "Users who expressed interest (lead capture).",
        "Conversions": "Users who completed the desired final action (purchase, signup, etc.).",
        "Spend": "Amount spent on advertising (INR)."
    }

    req_df = pd.DataFrame([{"Attribute": k, "Description": v} for k, v in required_dict.items()])

    # Put the dataframe with the required-table class
    # Use .to_html via styler to apply class attribute (Streamlit will render HTML)
    try:
        styled = req_df.style.set_table_attributes('class="required-table"').hide_index()
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write(styled.to_html(), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception:
        # fallback to st.dataframe if styler fails in some environments
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.dataframe(req_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Independent (LEFT) and Dependent (RIGHT)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        indep_vars = ["Campaign", "Channel", "Date", "Impressions", "Clicks", "Spend"]
        for v in indep_vars:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        dep_vars = ["Leads", "Conversions"]
        for v in dep_vars:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# -----------------------------
# Application tab (full app functionality)
# -----------------------------
with tab3:
    st.markdown('<div class="section-title">Application</div>', unsafe_allow_html=True)
    st.markdown("<div class='card'>Load data, filter, inspect KPIs, run ML predictions, and export results.</div>", unsafe_allow_html=True)

    mode = st.radio("Dataset option:", ["Default dataset","Upload CSV","Upload CSV + Column mapping"], horizontal=True)
    df = None

    # --- Default dataset ---
    if mode == "Default dataset":
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
    elif mode == "Upload CSV":
        SAMPLE_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/Marketing_Analytics.csv"
        try:
            sample_df = pd.read_csv(SAMPLE_URL).head(5)
            st.download_button("Download Sample CSV", sample_df.to_csv(index=False), "sample_dataset.csv","text/csv")
        except:
            pass

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
                    # reverse mapping: user_col -> required name
                    rename_map = {v:k for k,v in mapping.items()}
                    df = raw.rename(columns=rename_map)
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

    # derived metrics
    df["CTR"] = np.where(df["Impressions"]>0, df["Clicks"]/df["Impressions"], 0)
    df["CPC"] = np.where(df["Clicks"]>0, df["Spend"]/df["Clicks"], 0)
    df["Cost_per_Lead"] = np.where(df["Leads"]>0, df["Spend"]/df["Leads"], np.nan)
    df["Conversion_Rate"] = np.where(df["Clicks"]>0, df["Conversions"]/df["Clicks"], 0)

    # -------------------------
    # Filters & preview
    # -------------------------
    st.markdown('<div class="section-title">Filters & Preview</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2,2,1])
    campaigns = sorted(df["Campaign"].dropna().unique()) if "Campaign" in df.columns else []
    channels = sorted(df["Channel"].dropna().unique()) if "Channel" in df.columns else []

    with c1:
        sel_campaigns = st.multiselect("Campaign", campaigns, default=campaigns[:5] if campaigns else [])
    with c2:
        sel_channels = st.multiselect("Channel", channels, default=channels[:3] if channels else [])
    with c3:
        # date range input: safe fallback if missing dates
        if "Date" in df.columns and not df["Date"].isna().all():
            min_d = df["Date"].min().date()
            max_d = df["Date"].max().date()
            date_range = st.date_input("Date range", value=(min_d, max_d))
        else:
            date_range = None

    filt = df.copy()
    if sel_campaigns:
        filt = filt[filt["Campaign"].isin(sel_campaigns)]
    if sel_channels:
        filt = filt[filt["Channel"].isin(sel_channels)]
    if date_range and date_range[0] is not None:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filt = filt[(filt["Date"]>=start) & (filt["Date"]<=end)]

    st.dataframe(filt.head(), use_container_width=True)
    download_df(filt.head(5), "filtered_preview.csv")

    # -------------------------
    # KPIs
    # -------------------------
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Impressions", f"{int(filt['Impressions'].sum()):,}")
    k2.metric("Total Clicks", f"{int(filt['Clicks'].sum()):,}")
    k3.metric("Total Leads", f"{int(filt['Leads'].sum()):,}")
    k4.metric("Total Spend", to_currency(filt['Spend'].sum()))

    # -------------------------
    # Charts
    # -------------------------
    st.markdown('<div class="section-title">Campaign Performance</div>', unsafe_allow_html=True)
    if not filt.empty:
        agg = filt.groupby("Campaign")[["Impressions","Clicks","Leads","Conversions","Spend"]].sum().reset_index().sort_values("Clicks", ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Clicks"], name="Clicks", text=agg["Clicks"], textposition="outside"))
        fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Conversions"], name="Conversions", text=agg["Conversions"], textposition="outside"))
        fig.update_layout(barmode='group', xaxis_title="<b>Campaign</b>", yaxis_title="<b>Count</b>", template="plotly_white")
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data to chart for selected filters.")

    st.markdown('<div class="section-title">Channel ROI</div>', unsafe_allow_html=True)
    if "Channel" in filt.columns and not filt.empty:
        roi_df = filt.groupby("Channel").agg({"Leads":"sum","Spend":"sum"}).reset_index()
        roi_df["Leads_per_Rs"] = np.where(roi_df["Spend"]>0, roi_df["Leads"]/roi_df["Spend"], 0)
        fig2 = px.bar(roi_df.sort_values("Leads_per_Rs", ascending=False), x="Channel", y="Leads_per_Rs", text=roi_df["Leads_per_Rs"].round(4))
        fig2.update_traces(textposition="outside")
        fig2.update_layout(yaxis_title="Leads per ₹", xaxis_title="<b>Channel</b>", template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Time-series: Clicks over Time</div>', unsafe_allow_html=True)
    if "Date" in filt.columns and not filt["Date"].isna().all():
        ts = filt.groupby("Date")["Clicks"].sum().reset_index().sort_values("Date")
        if not ts.empty:
            ts["MA_7"] = ts["Clicks"].rolling(7,min_periods=1).mean()
            fig3 = px.line(ts, x="Date", y=["Clicks","MA_7"], labels={"value":"Clicks","variable":"Series"}, template="plotly_white")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Not enough date data.")
    else:
        st.info("Date column not found or empty; time-series unavailable.")

    # -------------------------
    # ML: Clicks Regression
    # -------------------------
    st.markdown('<div class="section-title">ML: Clicks Regression</div>', unsafe_allow_html=True)
    ml_df = filt.copy()
    if ml_df.shape[0] < 40:
        st.info("Not enough rows to train ML model reliably (need >=40). You can still download the data or try smaller experiments.")
    else:
        # create date features if possible
        if "Date" in ml_df.columns:
            ml_df["dayofweek"] = ml_df["Date"].dt.dayofweek
            ml_df["month"] = ml_df["Date"].dt.month
        feat_cols = [c for c in ["Channel","Campaign","Impressions","Spend","dayofweek","month"] if c in ml_df.columns]
        if len(feat_cols) < 2:
            st.info("Not enough feature columns available for ML.")
        else:
            X = ml_df[feat_cols].copy()
            y = ml_df["Clicks"].astype(float)
            cat_cols = [c for c in X.columns if X[c].dtype == "object"]
            num_cols = [c for c in X.columns if c not in cat_cols]
            # build preprocessor
            preprocessor = ColumnTransformer(transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                ("num", StandardScaler(), num_cols)
            ], remainder="drop")
            try:
                X_t = preprocessor.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)
                rf = RandomForestRegressor(n_estimators=200, random_state=42)
                with st.spinner("Training RandomForest for Clicks..."):
                    rf.fit(X_train, y_train)
                preds = rf.predict(X_test)
                rmse = math.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)
                st.write(f"Clicks regression — RMSE: {rmse:.2f}, R²: {r2:.3f}")

                # show sample of predictions + allow download
                X_test_df = pd.DataFrame(X_test, columns=[f"Feature_{i}" for i in range(X_test.shape[1])])
                X_test_df["Actual_Clicks"] = y_test.reset_index(drop=True)
                X_test_df["Predicted_Clicks"] = preds
                st.dataframe(X_test_df.head())
                download_df(X_test_df, "ml_clicks_predictions.csv")
            except Exception as e:
                st.error("ML model failed: " + str(e))

    # -------------------------
    # Forecasting: clicks (linear fallback)
    # -------------------------
    st.markdown('<div class="section-title">Forecasting Clicks (30 days)</div>', unsafe_allow_html=True)
    if "Date" in filt.columns and not filt["Date"].isna().all():
        ts_agg = filt.groupby("Date")["Clicks"].sum().reset_index().sort_values("Date")
        if len(ts_agg) < 6:
            st.info("Not enough historical daily points (need ~6+) for decent forecast.")
        else:
            try:
                from prophet import Prophet
                prophet_ok = True
            except Exception:
                prophet_ok = False

            if prophet_ok:
                dfp = ts_agg.rename(columns={"Date":"ds","Clicks":"y"}).set_index("ds").asfreq("D").fillna(0).reset_index()
                m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
                with st.spinner("Training Prophet..."):
                    m.fit(dfp)
                future = m.make_future_dataframe(periods=30)
                forecast = m.predict(future)
                figf = px.line(forecast, x="ds", y=["yhat","yhat_lower","yhat_upper"], labels={"ds":"Date"})
                st.plotly_chart(figf, use_container_width=True)
            else:
                ts_lr = ts_agg.set_index("Date").resample("D").sum().fillna(0).reset_index()
                ts_lr["t"] = np.arange(len(ts_lr))
                lr = LinearRegression(); lr.fit(ts_lr[["t"]], ts_lr["Clicks"])
                fut_dates = pd.date_range(ts_lr["Date"].max() + pd.Timedelta(days=1), periods=30)
                preds = lr.predict(np.arange(len(ts_lr), len(ts_lr) + 30).reshape(-1,1))
                figf = go.Figure([go.Scatter(x=ts_lr["Date"], y=ts_lr["Clicks"], name="Actual"),
                                  go.Scatter(x=fut_dates, y=preds, name="Forecast")])
                st.plotly_chart(figf, use_container_width=True)

    else:
        st.info("Date column missing or invalid; forecasting unavailable.")

    # -------------------------
    # Automated Insights
    # -------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)
    insights = []
    if "Channel" in filt.columns and "Leads" in filt.columns and "Spend" in filt.columns:
        ch_perf = filt.groupby("Channel").agg({"Leads":"sum","Spend":"sum"}).reset_index()
        ch_perf["Leads_per_Rs"] = np.where(ch_perf["Spend"]>0, ch_perf["Leads"]/ch_perf["Spend"], 0)
        if not ch_perf.empty:
            best = ch_perf.sort_values("Leads_per_Rs", ascending=False).iloc[0]
            worst = ch_perf.sort_values("Leads_per_Rs", ascending=True).iloc[0]
            insights.append({"Type":"Best Channel", "Channel":best["Channel"], "Value":round(float(best["Leads_per_Rs"]),6)})
            insights.append({"Type":"Worst Channel", "Channel":worst["Channel"], "Value":round(float(worst["Leads_per_Rs"]),6)})

    if insights:
        ins_df = pd.DataFrame(insights)
        st.dataframe(ins_df)
        download_df(ins_df, "automated_insights.csv")
    else:
        st.info("No automated insights available for the current filters.")

# End of app
