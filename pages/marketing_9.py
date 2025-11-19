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
# Tabs
# -------------------------
tabs = st.tabs(["Overview", "Application"])

# -------------------------
# Overview Tab - Informational Only
# -------------------------
with tabs[0]:
    st.markdown("<h1>Email & WhatsApp Marketing Forecast Lab</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='metric-card'>This application helps businesses analyze email and WhatsApp marketing campaigns to improve engagement, conversions, and revenue.</div>", unsafe_allow_html=True)
    
    st.markdown("### Capabilities")
    st.markdown("""
    <div class='metric-card'>
    • Analyze campaign performance: opens, clicks, conversions<br>
    • Track bounce rate, unsubscribes, replies<br>
    • Forecast future performance trends<br>
    • Identify high ROI campaigns and optimal target segments
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Business Impact")
    st.markdown("""
    <div class='metric-card'>
    • Improve marketing ROI by focusing on high-performing campaigns<br>
    • Reduce wastage on low-engagement campaigns<br>
    • Make data-driven decisions for content, timing, and audience targeting
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Key KPIs")
    st.markdown("""
    <div class='metric-card'>
    • Open Rate<br>
    • Click Rate<br>
    • Conversion Rate<br>
    • Revenue<br>
    • Bounce Rate
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Use of App")
    st.markdown("""
    <div class='metric-card'>
    • Monitor marketing campaign effectiveness<br>
    • Forecast future engagement and revenue<br>
    • Compare different campaigns, devices, countries
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Who Should Use")
    st.markdown("""
    <div class='metric-card'>
    • Marketing managers and analysts<br>
    • Digital marketing teams<br>
    • Growth and strategy teams<br>
    • Agencies managing multiple campaigns
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# Application Tab - Dataset, Filters, Charts, ML, Insights
# -------------------------
with tabs[1]:
    st.header("Application")
    
    # Dataset input
    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)
    df = None

    if mode == "Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/email_whatsapp_marketing.csv"
        df = pd.read_csv(DEFAULT_URL)
        st.success("Default dataset loaded")
    elif mode == "Upload CSV":
        st.markdown("#### Download Sample CSV for Reference")
        try:
            # Load default dataset
            sample_df = pd.read_csv(URL).head(5)  # Take first 5 rows
            sample_csv = sample_df.to_csv(index=False)
            st.download_button("Download Sample CSV", sample_csv, "sample_dataset.csv", "text/csv")
        except Exception as e:
            st.info(f"Sample CSV unavailable: {e}")
    
        # Upload actual CSV
        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)

    else:
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded:
            raw = pd.read_csv(uploaded)
            mapping = {}
            for req in REQUIRED_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + list(raw.columns))
            if st.button("Apply mapping"):
                df = raw.rename(columns={v:k for k,v in mapping.items() if v != "-- Select --"})
                st.success("Mapping applied")

    if df is None: st.stop()
    
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    
    # Filters
    st.markdown("### Filters & Preview")
    c1,c2,c3 = st.columns(3)
    campaign_types = sorted(df["Campaign_Type"].dropna().unique())
    countries = sorted(df["Country"].dropna().unique())
    devices = sorted(df["Device"].dropna().unique())
    with c1: sel_campaign = st.multiselect("Campaign Type", campaign_types, default=campaign_types)
    with c2: sel_country = st.multiselect("Country", countries, default=countries)
    with c3: sel_device = st.multiselect("Device", devices, default=devices)
    
    filt = df[(df["Campaign_Type"].isin(sel_campaign)) & 
              (df["Country"].isin(sel_country)) & 
              (df["Device"].isin(sel_device))]
    
    st.dataframe(filt.head(10), use_container_width=True)
    
    # Dynamic KPI Cards
    st.markdown("### Key Metrics")
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.markdown(f"<div class='metric-card'>Recipients<br><b>{filt['Recipients'].sum():,}</b></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='metric-card'>Delivered<br><b>{filt['Delivered'].sum():,}</b></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='metric-card'>Opens<br><b>{filt['Opened'].sum():,}</b></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='metric-card'>Clicks<br><b>{filt['Clicked'].sum():,}</b></div>", unsafe_allow_html=True)
    k5.markdown(f"<div class='metric-card'>Open Rate (%)<br><b>{filt['Open_Rate'].mean():.2f}</b></div>", unsafe_allow_html=True)
    k6.markdown(f"<div class='metric-card'>Click Rate (%)<br><b>{filt['Click_Rate'].mean():.2f}</b></div>", unsafe_allow_html=True)
    
    # Charts
    st.markdown("### Campaign Performance Charts")
    agg = filt.groupby("Campaign_Name").agg({"Opened":"sum","Clicked":"sum","Revenue":"sum"}).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["Campaign_Name"], y=agg["Opened"], name="Opened"))
    fig.add_trace(go.Bar(x=agg["Campaign_Name"], y=agg["Clicked"], name="Clicked"))
    st.plotly_chart(fig, use_container_width=True)
    
    fig2 = px.scatter(agg, x="Revenue", y=filt.groupby("Campaign_Name")["Conversion_Rate"].mean().reindex(agg["Campaign_Name"]),
                      size="Clicked", hover_name="Campaign_Name", color="Campaign_Name", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)
    
    # ML Revenue Predictions
    st.markdown("### ML Revenue Predictions (Actual vs Predicted)")
    ml_df = filt.dropna(subset=["Revenue","Recipients","Delivered"])
    if len(ml_df) >= 30:
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
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        ml_result_df = pd.DataFrame(X_test, columns=preprocessor.get_feature_names_out())
        ml_result_df[target_col+"_Actual"] = y_test.values
        ml_result_df[target_col+"_Predicted"] = preds
        ml_result_df = ml_result_df[[target_col+"_Actual", target_col+"_Predicted"] + list(ml_result_df.columns[:-2])]
        st.dataframe(ml_result_df.head(), use_container_width=True)
        b = BytesIO()
        b.write(ml_result_df.to_csv(index=False).encode("utf-8"))
        b.seek(0)
        st.download_button("Download ML Predictions CSV", b, file_name="ml_revenue_predictions.csv", mime="text/csv")
    
    # Automated Insights
    st.markdown("### Automated Insights")
    insights_df = filt.groupby(["Campaign_Name","Campaign_Type"]).agg({
        "Revenue":"sum","Opened":"sum","Clicked":"sum","Conversion_Rate":"mean"
    }).reset_index().sort_values("Revenue", ascending=False)
    st.dataframe(insights_df, use_container_width=True)
    b2 = BytesIO()
    b2.write(insights_df.to_csv(index=False).encode("utf-8"))
    b2.seek(0)
    st.download_button("Download Automated Insights CSV", b2, file_name="automated_insights.csv", mime="text/csv")
