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
from sklearn.metrics import mean_squared_error, r2_score
import math
import warnings
from prophet import Prophet
warnings.filterwarnings("ignore")

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="Email & WhatsApp Marketing Forecast Lab", layout="wide")

# Hide sidebar
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

# -------------------------
# Custom CSS for left-aligned cards
# -------------------------
st.markdown("""
<style>
div[data-testid="stMarkdownContainer"] .metric-card {
    background: rgba(255,255,255,0.10);
    padding: 20px;
    border-radius: 14px;
    text-align: left !important;
    border: 1px solid rgba(255,255,255,0.30);
    font-weight: 600;
    font-size: 16px;
    transition: all 0.25s ease;
    box-shadow: 0 2px 10px rgba(0,0,0,0.18);
    backdrop-filter: blur(4px);
}
div[data-testid="stMarkdownContainer"] .metric-card:hover {
    background: rgba(255,255,255,0.20);
    border: 1px solid rgba(255,255,255,0.55);
    box-shadow: 0 0 18px rgba(255,255,255,0.4);
    transform: scale(1.04);
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Constants
# -------------------------
REQUIRED_COLS = [
    "Date","Campaign_Type","Campaign_Name","Country","Device","Recipients",
    "Delivered","Failed","Opened","Clicked","Bounce_Rate","Unsubscribed",
    "Replies","Open_Rate","Click_Rate","Conversion_Rate","Revenue","ThruPlay_Rate"
]

# -------------------------
# Load dataset
# -------------------------
mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)
df = None

if mode == "Default dataset":
    DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/email_whatsapp_marketing.csv"
    try:
        df = pd.read_csv(DEFAULT_URL)
        st.success("Default dataset loaded")
        st.dataframe(df.head())
    except Exception as e:
        st.error("Failed to load default dataset: " + str(e))
        st.stop()

elif mode == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success("File uploaded.")
        st.dataframe(df.head())

else:  # Upload + mapping
    uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        st.write("Preview (first 5 rows):")
        st.dataframe(raw.head())
        st.markdown("Map your columns to required fields (only required fields shown).")
        mapping = {}
        for req in REQUIRED_COLS:
            mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + list(raw.columns))
        if st.button("Apply mapping"):
            missing = [k for k,v in mapping.items() if v == "-- Select --"]
            if missing:
                st.error("Please map all required columns: " + ", ".join(missing))
            else:
                df = raw.rename(columns={v:k for k,v in mapping.items()})
                st.success("Mapping applied.")
                st.dataframe(df.head())

if df is None:
    st.stop()

# -------------------------
# Convert Date
# -------------------------
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["Overview", "Application"])

# -------------------------
# Overview tab
# -------------------------
with tabs[0]:
    st.markdown("### Overview")
    st.markdown("""
    <div class='metric-card'>
    Analyze Email & WhatsApp campaigns, track open/click/conversion rates, bounce & reply rates, revenue, engagement over time, and forecast future trends.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Key Metrics")
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown(f"<div class='metric-card'>Total Recipients: {df['Recipients'].sum():,}</div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='metric-card'>Total Delivered: {df['Delivered'].sum():,}</div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='metric-card'>Total Opens: {df['Opened'].sum():,}</div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='metric-card'>Total Clicks: {df['Clicked'].sum():,}</div>", unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown(f"<div class='metric-card'>Open Rate (%): {df['Open_Rate'].mean():.2f}</div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='metric-card'>Click Rate (%): {df['Click_Rate'].mean():.2f}</div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='metric-card'>Conversion Rate (%): {df['Conversion_Rate'].mean():.2f}</div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='metric-card'>Revenue: ₹ {df['Revenue'].sum():,.2f}</div>", unsafe_allow_html=True)

    # Forecast
    st.markdown("### Forecast Daily Opens, Clicks, Revenue (Next 30 Days)")
    ts_cols = ["Opened","Clicked","Revenue"]
    ts_metric = st.selectbox("Metric", ts_cols)
    ts_df = df.groupby("Date")[ts_metric].sum().reset_index().rename(columns={"Date":"ds", ts_metric:"y"})
    m = Prophet(daily_seasonality=True)
    m.fit(ts_df)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    fig = px.line(forecast, x='ds', y='yhat', title=f"{ts_metric} Forecast", labels={"ds":"Date","yhat":ts_metric})
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Application tab
# -------------------------
with tabs[1]:
    st.header("Application")
    
    # Filters
    st.markdown("### Filters")
    c1,c2,c3 = st.columns(3)
    campaign_types = sorted(df["Campaign_Type"].dropna().unique().tolist())
    countries = sorted(df["Country"].dropna().unique().tolist())
    devices = sorted(df["Device"].dropna().unique().tolist())
    with c1: sel_campaign = st.multiselect("Campaign Type", options=campaign_types, default=campaign_types)
    with c2: sel_country = st.multiselect("Country", options=countries, default=countries)
    with c3: sel_device = st.multiselect("Device", options=devices, default=devices)
    
    filt = df.copy()
    filt = filt[filt["Campaign_Type"].isin(sel_campaign)]
    filt = filt[filt["Country"].isin(sel_country)]
    filt = filt[filt["Device"].isin(sel_device)]
    
    st.markdown("Filtered preview")
    st.dataframe(filt.head(10), use_container_width=True)
    
    # Charts
    st.markdown("### Campaign Performance")
    agg = filt.groupby("Campaign_Name").agg({
        "Recipients":"sum","Delivered":"sum","Opened":"sum","Clicked":"sum","Revenue":"sum"
    }).reset_index().sort_values("Opened", ascending=False)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["Campaign_Name"], y=agg["Opened"], name="Opened"))
    fig.add_trace(go.Bar(x=agg["Campaign_Name"], y=agg["Clicked"], name="Clicked"))
    fig.update_layout(barmode='group', xaxis_title="<b>Campaign Name</b>", yaxis_title="<b>Count</b>")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Revenue & Conversion Rate per Campaign")
    conv_rate = filt.groupby('Campaign_Name')['Conversion_Rate'].mean().reindex(agg['Campaign_Name'])
    fig2 = px.scatter(agg, x="Revenue", y=conv_rate, size="Clicked", hover_name="Campaign_Name", color="Campaign_Name", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)
    
    # ML Revenue Predictions
    st.markdown("### ML Revenue Predictions")
    ml_df = filt.dropna(subset=["Revenue","Recipients","Delivered"])
    if len(ml_df) < 30:
        st.info("Not enough data for ML model (need at least 30 rows).")
    else:
        feat_cols = ["Campaign_Type","Country","Device","Recipients","Delivered"]
        target_col = "Revenue"
        X = ml_df[feat_cols].copy()
        y = ml_df[target_col].copy()
        cat_cols = ["Campaign_Type","Country","Device"]
        num_cols = ["Recipients","Delivered"]
        preprocessor = ColumnTransformer(transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols)
        ])
        X_t = preprocessor.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        with st.spinner("Training RandomForest for Revenue..."):
            rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        ml_result_df = pd.DataFrame(X_test, columns=preprocessor.get_feature_names_out())
        ml_result_df[target_col+"_Actual"] = y_test.values
        ml_result_df[target_col+"_Predicted"] = preds
        ml_result_df = ml_result_df[[*ml_result_df.columns[-2:], *ml_result_df.columns[:-2]]]
        st.dataframe(ml_result_df.head(), use_container_width=True)
        b = BytesIO()
        b.write(ml_result_df.to_csv(index=False).encode("utf-8"))
        b.seek(0)
        st.download_button("Download ML Revenue Predictions", b, file_name="ml_revenue_predictions.csv", mime="text/csv")

    # Automated Insights
    st.markdown("### Automated Insights")
    insights_df = filt.groupby(["Campaign_Name","Campaign_Type"]).agg({
        "Revenue":"sum","Opened":"sum","Clicked":"sum","Conversion_Rate":"mean"
    }).reset_index().sort_values("Revenue", ascending=False)
    st.dataframe(insights_df, use_container_width=True)
    b2 = BytesIO()
    b2.write(insights_df.to_csv(index=False).encode("utf-8"))
    b2.seek(0)
    st.download_button("Download Automated Insights", b2, file_name="automated_insights.csv", mime="text/csv")
