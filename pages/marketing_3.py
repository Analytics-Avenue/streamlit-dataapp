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
st.set_page_config(page_title="Click & Convertion Analytics", layout="wide")

# Hide default sidebar navigation (optional)
st.markdown("""<style>[data-testid="stSidebarNav"]{display:none;}</style>""", unsafe_allow_html=True)



# -------------------------
# Company Logo + Name
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display: flex; align-items: center;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0; padding:0;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0; padding:0;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)


# Page header
st.markdown("<h1 style='margin-bottom:0.2rem'>Click & Convertion Analytics</h1>", unsafe_allow_html=True)
st.markdown("Enterprise-capable marketing analytics: ML predictions, forecasting, and actionable insights from campaigns. No fluff — just results.")

# -----------------------------
# Required columns & auto-mapping
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

def download_df(df, filename):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button("Download CSV", b, file_name=filename, mime="text/csv")

def to_currency(x):
    try: return "₹ "+f"{float(x):,.2f}"
    except: return x

def ensure_datetime(df, col="Date"):
    try: df[col] = pd.to_datetime(df[col], errors="coerce")
    except: pass
    return df

# -----------------------------
# CSS for cards & metrics
# -----------------------------
st.markdown("""
<style>
.card {background: rgba(255,255,255,0.08); padding:18px; border-radius:14px; margin-bottom:15px;
       border:1px solid rgba(255,255,255,0.25); box-shadow:0 4px 18px rgba(0,0,0,0.25); transition:all .25s ease;}
.card:hover {background: rgba(255,255,255,0.18); border:1px solid rgba(255,255,255,0.55); box-shadow:0 0 18px rgba(255,255,255,0.4); transform:scale(1.03); cursor:pointer;}
.metric-card {background: rgba(255,255,255,0.10); padding:20px; border-radius:14px; text-align:center; font-weight:600; font-size:16px;
             border:1px solid rgba(255,255,255,0.30); box-shadow:0 2px 10px rgba(0,0,0,0.18); transition:all 0.25s ease;}
.metric-card:hover {background: rgba(255,255,255,0.20); border:1px solid rgba(255,255,255,0.55); box-shadow:0 0 18px rgba(255,255,255,0.4); transform:scale(1.04);}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Tabs: Overview & Application
# -----------------------------
tabs = st.tabs(["Overview","Application"])

# -----------------------------
# Overview
# -----------------------------
with tabs[0]:
    st.markdown("## Overview")
    st.markdown("<div class='card'>App 3 provides end-to-end marketing analytics, predictive insights, and automated KPIs tracking — enterprise-ready.</div>", unsafe_allow_html=True)
    st.markdown("### Capabilities")
    st.markdown("<div class='card'>• Multi-channel tracking<br>• ML predictions for Clicks, Leads & Conversions<br>• Forecasting & trend analysis<br>• Campaign optimization insights<br>• Budget allocation guidance</div>", unsafe_allow_html=True)
    st.markdown("### Impact")
    st.markdown("<div class='card'>• Faster decision-making<br>• Reduced wasted spend<br>• Data-driven prioritization<br>• Improved conversion efficiency</div>", unsafe_allow_html=True)
    st.markdown("### KPIs")
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Total Clicks</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Total Leads</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Conversion Rate</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>ROAS</div>", unsafe_allow_html=True)

# -----------------------------
# Application
# -----------------------------
with tabs[1]:
    st.markdown("## Application")
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
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/Marketing_Analytics.csv"
        try:
            sample_df = pd.read_csv(URL).head(5)
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
    if df is None: st.stop()

    # -------------------------
    # Data prep & KPIs
    # -------------------------
    df.columns = df.columns.str.strip()
    df = ensure_datetime(df, "Date")
    for col in ["Impressions","Clicks","Leads","Conversions","Spend"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["CTR"] = np.where(df["Impressions"]>0, df["Clicks"]/df["Impressions"], 0)
    df["CPC"] = np.where(df["Clicks"]>0, df["Spend"]/df["Clicks"], 0)
    df["Cost_per_Lead"] = np.where(df["Leads"]>0, df["Spend"]/df["Leads"], np.nan)
    df["Conversion_Rate"] = np.where(df["Clicks"]>0, df["Conversions"]/df["Clicks"], 0)

    st.markdown("### Filters & Preview")
    c1,c2,c3 = st.columns([2,2,1])
    campaigns = sorted(df["Campaign"].dropna().unique())
    channels = sorted(df["Channel"].dropna().unique())
    with c1: sel_campaigns = st.multiselect("Campaign", campaigns, default=campaigns[:5])
    with c2: sel_channels = st.multiselect("Channel", channels, default=channels[:3])
    with c3: date_range = st.date_input("Date range", (df["Date"].min().date(), df["Date"].max().date()))
    filt = df.copy()
    if sel_campaigns: filt = filt[filt["Campaign"].isin(sel_campaigns)]
    if sel_channels: filt = filt[filt["Channel"].isin(sel_channels)]
    start,end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filt = filt[(filt["Date"]>=start)&(filt["Date"]<=end)]
    st.dataframe(filt.head())
    download_df(filt.head(5), "filtered_preview.csv")

    # -------------------------
    # KPIs
    # -------------------------
    st.markdown("### Key Metrics")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Impressions", f"{int(filt['Impressions'].sum()):,}")
    k2.metric("Total Clicks", f"{int(filt['Clicks'].sum()):,}")
    k3.metric("Total Leads", f"{int(filt['Leads'].sum()):,}")
    k4.metric("Total Spend", to_currency(filt['Spend'].sum()))

    # -------------------------
    # Charts
    # -------------------------
    st.markdown("### Campaign Performance")
    agg = filt.groupby("Campaign")[["Impressions","Clicks","Leads","Conversions","Spend"]].sum().reset_index().sort_values("Clicks", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Clicks"], name="Clicks", text=agg["Clicks"], textposition="outside"))
    fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Conversions"], name="Conversions", text=agg["Conversions"], textposition="outside"))
    fig.update_layout(barmode='group', xaxis_title="<b>Campaign</b>", yaxis_title="<b>Count</b>", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Channel ROI")
    roi_df = filt.groupby("Channel").agg({"Leads":"sum","Spend":"sum"}).reset_index()
    roi_df["Leads_per_Rs"] = np.where(roi_df["Spend"]>0, roi_df["Leads"]/roi_df["Spend"], 0)
    fig2 = px.bar(roi_df.sort_values("Leads_per_Rs", ascending=False), x="Channel", y="Leads_per_Rs", text=roi_df["Leads_per_Rs"].round(4))
    fig2.update_traces(textposition="outside")
    fig2.update_layout(yaxis_title="Leads per ₹", xaxis_title="<b>Channel</b>", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Time-series: Clicks over Time")
    ts = filt.groupby("Date")["Clicks"].sum().reset_index()
    if not ts.empty:
        ts["MA_7"] = ts["Clicks"].rolling(7,min_periods=1).mean()
        fig3 = px.line(ts, x="Date", y=["Clicks","MA_7"], labels={"value":"Clicks","variable":"Series"}, template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)
    else: st.info("Not enough date data.")


    # -------------------------
    # ML: Clicks Regression + Download
    # -------------------------
    st.markdown("### ML: Clicks Regression")
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
    
        # Combine predictions with original input features for download
        X_test_df = pd.DataFrame(X_test, columns=[f"Feature_{i}" for i in range(X_test.shape[1])])
        X_test_df["Actual_Clicks"] = y_test.reset_index(drop=True)
        X_test_df["Predicted_Clicks"] = preds
        st.dataframe(X_test_df.head())
        download_df(X_test_df, "ml_clicks_predictions.csv")

    # -------------------------
    # Forecasting (linear fallback if Prophet not installed)
    # -------------------------
    st.markdown("### Forecasting Clicks")
    ts_agg = filt.groupby("Date")["Clicks"].sum().reset_index().sort_values("Date")
    if len(ts_agg)>=6:
        try:
            from prophet import Prophet
            m = Prophet(daily_seasonality=False,weekly_seasonality=True,yearly_seasonality=True)
            prophet_ok = True
        except:
            prophet_ok = False
        if prophet_ok:
            dfp = ts_agg.rename(columns={"Date":"ds","Clicks":"y"}).set_index("ds").asfreq("D").fillna(0).reset_index()
            m.fit(dfp)
            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)
            figf = px.line(forecast, x="ds", y=["yhat","yhat_lower","yhat_upper"])
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


    
        # -------------------------
        # Automated Insights (Table + Download)
        # -------------------------
        st.markdown("### Automated Insights (Table + Download)")
        insights_list = []
        if "Channel" in filt.columns and "Leads" in filt.columns and "Spend" in filt.columns:
            ch_perf = filt.groupby("Channel").agg({"Leads":"sum","Spend":"sum"}).reset_index()
            ch_perf["Leads_per_Rs"] = np.where(ch_perf["Spend"]>0,ch_perf["Leads"]/ch_perf["Spend"],0)
            best = ch_perf.sort_values("Leads_per_Rs",ascending=False).iloc[0]
            worst = ch_perf.sort_values("Leads_per_Rs",ascending=True).iloc[0]
            insights_list.append({"Insight":"Best Channel", "Channel":best['Channel'], "Leads_per_Rs":best['Leads_per_Rs']})
            insights_list.append({"Insight":"Worst Channel", "Channel":worst['Channel'], "Leads_per_Rs":worst['Leads_per_Rs']})
        
        insights_df = pd.DataFrame(insights_list)
        st.dataframe(insights_df)
        download_df(insights_df, "automated_insights.csv")
    
        
