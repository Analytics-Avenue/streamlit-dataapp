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

# -------------------------
# Company Logo + Name
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display:flex; align-items:center; margin-bottom:20px;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

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
# Required columns
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
# CSS – Updated (PURE BLACK everywhere except KPI + variable boxes)
# -----------------------------
st.markdown("""
<style>
* { font-family: 'Inter', sans-serif; }

/* Main header */
.header-wrap { display:flex; align-items:center; gap:12px; margin-bottom:6px; }
.app-title { font-size:34px !important; font-weight:700 !important; color:#000 !important; margin:0; }
.app-sub { color:#000 !important; margin-top:2px; margin-bottom:12px; font-size:14.5px; }

/* Section title */
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

/* CARD — NOW PURE BLACK TEXT */
.card {
    background: #ffffff;
    padding:18px;
    border-radius:12px;
    border:1px solid #e6e6e6;
    color:#000 !important;    /* FIXED: pure black */
    box-shadow: 0 6px 22px rgba(0,0,0,0.06);
    transition: transform 0.22s ease, box-shadow 0.22s ease;
}
.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 18px 44px rgba(6,75,134,0.14);
    border-color: #064b86;
}

/* KPI CARD — BLUE TEXT (unchanged) */
.metric-card {
    background: #ffffff;
    padding:18px;
    border-radius:12px;
    border:1px solid #e6e6e6;
    text-align:center;
    font-weight:600;
    font-size:16px;
    color:#064b86 !important;   /* BLUE TEXT */
    box-shadow:0 6px 18px rgba(0,0,0,0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow:0 16px 36px rgba(6,75,134,0.14);
}

/* VARIABLE BOX — BLUE TEXT */
.variable-box {
    padding:14px;
    border-radius:12px;
    background: #ffffff;
    border: 1px solid #e6e6e6;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    font-size:16px !important;
    font-weight:600 !important;
    color:#064b86 !important;   /* BLUE TEXT */
    text-align:center;
    margin-bottom:12px;
}
.variable-box:hover {
    transform: translateY(-6px);
    box-shadow:0 18px 40px rgba(6,75,134,0.12);
    border-color:#064b86;
}

/* Required Table — pure black */
.required-table thead th {
    background:#fff !important;
    color:#000 !important;
    font-size:18px !important;
    border-bottom:2px solid #000 !important;
    padding:10px !important;
}
.required-table tbody td {
    color:#000 !important;
    font-size:17px !important;
    padding:10px !important;
    border-bottom:1px solid #e6e6e6 !important;
}
.required-table tbody tr:hover td { background:#fafafa !important; }

/* Buttons */
.stButton>button, .stDownloadButton>button {
    background:#064b86 !important;
    color:white !important;
    border-radius:8px !important;
    padding:8px 18px !important;
    font-weight:600 !important;
    border:none !important;
}

/* Fade-in */
.block-container { animation: fadeIn 0.45s ease; }
@keyframes fadeIn { from {opacity:0; transform: translateY(6px);} to {opacity:1; transform:translateY(0);} }

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("<h1 style='margin-bottom:0.2rem'>Click & Conversion Analytics</h1>", unsafe_allow_html=True)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# -----------------------------
# OVERVIEW TAB
# -----------------------------
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="card">
        <b>Purpose:</b> Centralize marketing signals and enable performance optimization,
        forecasting, and ML-based predictions for decision-makers.
        </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="card">
            • Multi-channel aggregation<br>
            • Funnel metrics with CTR, CPC, CPL<br>
            • ML-driven Click prediction<br>
            • Forecasting using Prophet/Linear models<br>
            • Automated insights table<br>
            </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="card">
            • Reduce wasted ad spend<br>
            • Improve lead quality<br>
            • Build predictable growth systems<br>
            • Strengthen budget planning<br>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Key Performance Indicators</div>', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown('<div class="metric-card">Total Clicks</div>', unsafe_allow_html=True)
    k2.markdown('<div class="metric-card">Total Leads</div>', unsafe_allow_html=True)
    k3.markdown('<div class="metric-card">Conversion Rate</div>', unsafe_allow_html=True)
    k4.markdown('<div class="metric-card">ROAS</div>', unsafe_allow_html=True)

# -----------------------------
# IMPORTANT ATTRIBUTES TAB
# -----------------------------
with tab2:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    required_dict = {
        "Campaign": "Name of the marketing campaign.",
        "Channel": "Traffic source or platform.",
        "Date": "Daily campaign activity date.",
        "Impressions": "Ad views count.",
        "Clicks": "User clicks on ads.",
        "Leads": "Converted interested users.",
        "Conversions": "Final successful actions.",
        "Spend": "Total ad cost."
    }

    req_df = pd.DataFrame([{"Attribute": k, "Description": v} for k, v in required_dict.items()])
    styled = req_df.style.set_table_attributes('class="required-table"')
    html = styled.to_html()
    html = html.replace("<th></th>", "")   # remove index header
    html = html.replace("<td></td>", "")   # remove index column
    st.write(html, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        for v in ["Campaign","Channel","Date","Impressions","Clicks","Spend"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        for v in ["Leads","Conversions"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# -----------------------------
# APPLICATION TAB
# -----------------------------
with tab3:
    st.markdown('<div class="section-title">Application</div>', unsafe_allow_html=True)
    st.markdown("<div class='card'>Load dataset, filter, analyze, generate automations, ML predictions & forecasting.</div>", unsafe_allow_html=True)

    mode = st.radio("Dataset option:", ["Default dataset","Upload CSV","Upload CSV + Column mapping"], horizontal=True)
    df = None

    # DEFAULT DATASET
    if mode == "Default dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/marketing.csv"
        try:
            df = pd.read_csv(URL)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("Default dataset loaded ✔")
        except:
            st.error("Failed to load default dataset.")
            st.stop()

    # UPLOAD CSV
    elif mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("File uploaded successfully ✔")
            download_df(df.head(5), "sample_uploaded.csv")

    # COLUMN MAPPING
    else:
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.write("Preview:")
            st.dataframe(raw.head())
            mapping = {}
            for req in REQUIRED_MARKETING_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", ["-- Select --"] + list(raw.columns))
            if st.button("Apply mapping"):
                missing = [k for k,v in mapping.items() if v=="-- Select --"]
                if missing:
                    st.error("Map all required columns.")
                else:
                    df = raw.rename(columns={v:k for k,v in mapping.items()})
                    st.success("Mapping applied ✔")
                    download_df(df.head(5), "mapped_sample.csv")

    if df is None:
        st.stop()

    # DATA PROCESSING
    df = ensure_datetime(df, "Date")
    for col in ["Impressions","Clicks","Leads","Conversions","Spend"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["CTR"] = np.where(df["Impressions"]>0, df["Clicks"]/df["Impressions"], 0)
    df["CPC"] = np.where(df["Clicks"]>0, df["Spend"]/df["Clicks"], 0)
    df["Cost_per_Lead"] = np.where(df["Leads"]>0, df["Spend"]/df["Leads"], np.nan)
    df["Conversion_Rate"] = np.where(df["Clicks"]>0, df["Conversions"]/df["Clicks"], 0)

    # FILTERS
    st.markdown('<div class="section-title">Filters & Preview</div>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns([2,2,1])
    campaigns = sorted(df["Campaign"].unique())
    channels = sorted(df["Channel"].unique())

    with c1:
        sel_campaigns = st.multiselect("Campaign", campaigns, default=campaigns[:5])
    with c2:
        sel_channels = st.multiselect("Channel", channels, default=channels[:3])
    with c3:
        min_d, max_d = df["Date"].min().date(), df["Date"].max().date()
        date_range = st.date_input("Date range", value=(min_d, max_d))

    filt = df.copy()
    if sel_campaigns:
        filt = filt[filt["Campaign"].isin(sel_campaigns)]
    if sel_channels:
        filt = filt[filt["Channel"].isin(sel_channels)]
    
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filt = filt[(filt["Date"]>=start) & (filt["Date"]<=end)]

    st.dataframe(filt.head(), use_container_width=True)
    download_df(filt.head(), "filtered_preview.csv")

    # KPIs
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Impressions", f"{int(filt['Impressions'].sum()):,}")
    k2.metric("Total Clicks", f"{int(filt['Clicks'].sum()):,}")
    k3.metric("Total Leads", f"{int(filt['Leads'].sum()):,}")
    k4.metric("Total Spend", to_currency(filt['Spend'].sum()))

    # CHARTS
    st.markdown('<div class="section-title">Campaign Performance</div>', unsafe_allow_html=True)
    if not filt.empty:
        agg = filt.groupby("Campaign")[["Clicks","Conversions"]].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Clicks"], name="Clicks", text=agg["Clicks"], textposition="outside"))
        fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Conversions"], name="Conversions", text=agg["Conversions"], textposition="outside"))
        fig.update_layout(barmode='group', template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Channel ROI</div>', unsafe_allow_html=True)
    if "Channel" in filt.columns:
        roi_df = filt.groupby("Channel").agg({"Leads":"sum","Spend":"sum"}).reset_index()
        roi_df["Leads_per_Rs"] = np.where(roi_df["Spend"]>0, roi_df["Leads"]/roi_df["Spend"], 0)
        fig2 = px.bar(roi_df, x="Channel", y="Leads_per_Rs", text=roi_df["Leads_per_Rs"].round(4), template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Time-Series: Clicks</div>', unsafe_allow_html=True)
    ts = filt.groupby("Date")["Clicks"].sum().reset_index()
    if not ts.empty:
        ts["MA_7"] = ts["Clicks"].rolling(7, min_periods=1).mean()
        fig3 = px.line(ts, x="Date", y=["Clicks","MA_7"], template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)

    # ML
    st.markdown('<div class="section-title">ML: Clicks Regression</div>', unsafe_allow_html=True)
    ml_df = filt.copy()
    if ml_df.shape[0] >= 40:
        ml_df["dayofweek"] = ml_df["Date"].dt.dayofweek
        ml_df["month"] = ml_df["Date"].dt.month

        feat_cols = ["Channel","Campaign","Impressions","Spend","dayofweek","month"]
        feat_cols = [c for c in feat_cols if c in ml_df.columns]

        X = ml_df[feat_cols]
        y = ml_df["Clicks"]

        cat_cols = [c for c in X.columns if X[c].dtype == "object"]
        num_cols = [c for c in X.columns if c not in cat_cols]

        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols)
        ])

        X_t = preprocessor.fit_transform(X)
        X_train,X_test,y_train,y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=200)
        model.fit(X_train,y_train)

        preds = model.predict(X_test)
        st.write(f"RMSE: {math.sqrt(mean_squared_error(y_test,preds)):.2f} | R²: {r2_score(y_test,preds):.3f}")

        pred_out = pd.DataFrame({"Actual": y_test, "Predicted": preds})
        st.dataframe(pred_out.head())
        download_df(pred_out, "ml_click_predictions.csv")

    # FORECASTING
    st.markdown('<div class="section-title">Forecasting: 30-Day Clicks</div>', unsafe_allow_html=True)

    ts2 = filt.groupby("Date")["Clicks"].sum().reset_index()
    if ts2.shape[0] >= 6:
        try:
            from prophet import Prophet
            dfp = ts2.rename(columns={"Date":"ds","Clicks":"y"}).set_index("ds").asfreq("D").fillna(0).reset_index()

            model = Prophet()
            model.fit(dfp)

            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            figf = px.line(forecast, x="ds", y=["yhat","yhat_lower","yhat_upper"], template="plotly_white")
            st.plotly_chart(figf, use_container_width=True)

        except:
            ts2["t"] = np.arange(len(ts2))
            lr = LinearRegression()
            lr.fit(ts2[["t"]], ts2["Clicks"])

            fut = np.arange(len(ts2), len(ts2)+30).reshape(-1,1)
            preds = lr.predict(fut)

            fut_dates = pd.date_range(ts2["Date"].max()+pd.Timedelta(days=1), periods=30)

            figf = go.Figure([
                go.Scatter(x=ts2["Date"], y=ts2["Clicks"], name="Actual"),
                go.Scatter(x=fut_dates, y=preds, name="Forecast")
            ])
            st.plotly_chart(figf, use_container_width=True)

    # INSIGHTS
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)

    insights = []
    ch = filt.groupby("Channel").agg({"Leads":"sum","Spend":"sum"}).reset_index()
    if not ch.empty:
        ch["LPR"] = np.where(ch["Spend"]>0,ch["Leads"]/ch["Spend"],0)
        insights.append({"Insight":"Best Channel", "Channel":ch.sort_values("LPR",ascending=False).iloc[0]["Channel"]})
        insights.append({"Insight":"Weakest Channel", "Channel":ch.sort_values("LPR",ascending=True).iloc[0]["Channel"]})

    if insights:
        idf = pd.DataFrame(insights)
        st.dataframe(idf)
        download_df(idf,"automated_insights.csv")
    else:
        st.info("No insights available.")
