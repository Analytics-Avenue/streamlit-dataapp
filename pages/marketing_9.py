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
st.markdown("<h1>Email & WhatsApp Marketing Forecast Lab</h1>", unsafe_allow_html=True)
st.markdown("Analyze campaign performance, predict future engagement and revenue, and forecast trends for better marketing strategy.")

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

# -------------------------
# Stop if no data
# -------------------------
if df is None:
    st.stop()

# -------------------------
# Convert Date column
# -------------------------
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

# -------------------------
# Tabs: Overview & Application
# -------------------------
tabs = st.tabs(["Overview", "Application"])

with tabs[0]:
    st.markdown("### Overview")
    st.markdown("""
    Analyze Email & WhatsApp campaigns, track open/click/conversion rates, bounce and reply rates, revenue, engagement over time, and forecast future trends.
    """)
    
    st.markdown("### Key Metrics")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Recipients", f"{df['Recipients'].sum():,}")
    k2.metric("Total Delivered", f"{df['Delivered'].sum():,}")
    k3.metric("Total Opens", f"{df['Opened'].sum():,}")
    k4.metric("Total Clicks", f"{df['Clicked'].sum():,}")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Open Rate (%)", f"{df['Open_Rate'].mean():.2f}")
    k2.metric("Click Rate (%)", f"{df['Click_Rate'].mean():.2f}")
    k3.metric("Conversion Rate (%)", f"{df['Conversion_Rate'].mean():.2f}")
    k4.metric("Revenue", f"₹ {df['Revenue'].sum():,.2f}")

    # -------------------------
    # Time-series Forecast
    # -------------------------
    st.markdown("### Forecast Daily Opens, Clicks, Revenue (Next 30 Days)")
    ts_cols = ["Opened","Clicked","Revenue"]
    st.write("Select metric to forecast:")
    ts_metric = st.selectbox("Metric", ts_cols)

    ts_df = df.groupby("Date")[ts_metric].sum().reset_index().rename(columns={"Date":"ds", ts_metric:"y"})
    m = Prophet(daily_seasonality=True)
    m.fit(ts_df)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)

    fig = px.line(forecast, x='ds', y='yhat', title=f"{ts_metric} Forecast", labels={"ds":"Date","yhat":ts_metric})
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.header("Application")
    
    # -------------------------
    # Filters
    # -------------------------
    st.markdown("### Filters")
    c1,c2,c3 = st.columns(3)
    campaign_types = sorted(df["Campaign_Type"].dropna().unique().tolist())
    countries = sorted(df["Country"].dropna().unique().tolist())
    devices = sorted(df["Device"].dropna().unique().tolist())
    with c1:
        sel_campaign = st.multiselect("Campaign Type", options=campaign_types, default=campaign_types)
    with c2:
        sel_country = st.multiselect("Country", options=countries, default=countries)
    with c3:
        sel_device = st.multiselect("Device", options=devices, default=devices)
    
    filt = df.copy()
    filt = filt[filt["Campaign_Type"].isin(sel_campaign)]
    filt = filt[filt["Country"].isin(sel_country)]
    filt = filt[filt["Device"].isin(sel_device)]
    
    st.markdown("Filtered preview")
    st.dataframe(filt.head(10), use_container_width=True)
    
    # -------------------------
    # Charts
    # -------------------------
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
    fig2 = px.scatter(agg, x="Revenue", y=filt.groupby('Campaign_Name')['Conversion_Rate'].mean().reindex(agg['Campaign_Name']),
                      size="Clicked", hover_name="Campaign_Name", color="Campaign_Name", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)
    
    # -------------------------
    # ML: Predict future Opens, Clicks, Revenue
    # -------------------------
    st.markdown("### Predictive Modeling")
    ml_df = filt.copy()
    ml_df = ml_df.dropna(subset=["Recipients","Delivered","Opened","Clicked","Revenue"])
    
    if len(ml_df) < 30:
        st.info("Not enough data to train ML model (need at least 30 rows).")
    else:
        feat_cols = ["Campaign_Type","Country","Device","Recipients","Delivered"]
        target_cols = ["Opened","Clicked","Revenue"]
        
        X = ml_df[feat_cols].copy()
        y = ml_df[target_cols].copy()
        
        cat_cols = ["Campaign_Type","Country","Device"]
        num_cols = ["Recipients","Delivered"]
        
        preprocessor = ColumnTransformer(transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols)
        ])
        
        X_t = preprocessor.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)
        
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        with st.spinner("Training RandomForest models..."):
            rf.fit(X_train, y_train)
        
        preds = rf.predict(X_test)
        for i, target in enumerate(target_cols):
            rmse = math.sqrt(mean_squared_error(y_test.iloc[:,i], preds[:,i]))
            r2 = r2_score(y_test.iloc[:,i], preds[:,i])
            st.write(f"{target} — RMSE: {rmse:.2f}, R²: {r2:.3f}")
        
        st.markdown("### Quick Prediction for a Campaign")
        sel_type = st.selectbox("Campaign Type", options=ml_df["Campaign_Type"].unique())
        sel_country = st.selectbox("Country", options=ml_df["Country"].unique())
        sel_device = st.selectbox("Device", options=ml_df["Device"].unique())
        inp_rec = st.number_input("Recipients", value=int(ml_df["Recipients"].median()))
        inp_del = st.number_input("Delivered", value=int(ml_df["Delivered"].median()))
        
        if st.button("Predict"):
            row = pd.DataFrame([{
                "Campaign_Type": sel_type,
                "Country": sel_country,
                "Device": sel_device,
                "Recipients": inp_rec,
                "Delivered": inp_del
            }])
            row_t = preprocessor.transform(row)
            pred_vals = rf.predict(row_t)[0]
            st.success(f"Predicted Opens: {int(pred_vals[0])}, Clicks: {int(pred_vals[1])}, Revenue: ₹ {pred_vals[2]:.2f}")

