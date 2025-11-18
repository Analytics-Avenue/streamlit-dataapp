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
import math
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Google Ads & SEO Performance Lab", layout="wide")

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


# -------------------------
# Helper utils
# -------------------------
REQUIRED_COLS = [
    "Date","Campaign","AdGroup","Keyword","Device","Country",
    "Impressions","Clicks","CTR","CPC","Cost","Conversions","ConversionRate","Revenue","ROAS",
    "Organic_Impressions","Organic_Clicks","Organic_CTR","Bounce_Rate","Avg_Time_on_Page_sec",
    "Page_Position","Backlinks"
]

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

def download_df(df, filename):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button("Download CSV", b, file_name=filename, mime="text/csv")

# -------------------------
# Custom CSS for cards
# -------------------------
st.markdown("""
<style>
div[data-testid="stMarkdownContainer"] .metric-card {
    background: rgba(255,255,255,0.10);
    padding: 20px;
    border-radius: 14px;
    text-align: center;
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
# Tabs
# -------------------------
tabs = st.tabs(["Overview", "Application"])

with tabs[0]:
    st.markdown("<h1 style='margin-bottom:0.2rem'>Google Ads & SEO Performance Lab</h1>", unsafe_allow_html=True)
    st.markdown("Analyze paid search campaigns along with SEO performance to optimize clicks, conversions, and revenue.")

    st.markdown("### Key Metrics")
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown("<div class='metric-card' title='Total clicks on paid campaigns'>Paid Clicks</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card' title='Total clicks on organic content'>Organic Clicks</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card' title='Total revenue from all campaigns'>Revenue</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card' title='Overall ROAS (Revenue/Cost)'>ROAS</div>", unsafe_allow_html=True)

    st.markdown("### Purpose")
    st.markdown("""
    <div class='metric-card'>
    • Compare Google Ads and SEO performance<br>
    • Identify high-ROI keywords & campaigns<br>
    • Track bounce rate and time on page<br>
    • Optimize ad spend and content strategy<br>
    • Monitor SERP positions and backlinks
    </div>
    """, unsafe_allow_html=True)

with tabs[1]:
    st.header("Application")

    # -------------------------
    # Dataset input
    # -------------------------
    st.markdown("### Step 1 — Load dataset")
    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)
    df = None

    if mode == "Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/google_ads_seo_performance.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            st.success("Default dataset loaded")
            st.dataframe(df.head())
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    elif mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            st.success("File uploaded.")
            st.dataframe(df.head())

    else:  # Upload + mapping
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head())
            st.markdown("Map your columns to required fields (only required fields shown).")
            mapping = {}
            for req in REQUIRED_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + list(raw.columns))
            if st.button("Apply mapping"):
                missing = [k for k,v in mapping.items() if v=="-- Select --"]
                if missing:
                    st.error("Please map all required columns: "+", ".join(missing))
                else:
                    df = raw.rename(columns={v:k for k,v in mapping.items()})
                    st.success("Mapping applied.")
                    st.dataframe(df.head())

    if df is None:
        st.stop()

    # -------------------------
    # Type conversions
    # -------------------------
    df = ensure_datetime(df, "Date")
    numeric_cols = [
        "Impressions","Clicks","CTR","CPC","Cost","Conversions","ConversionRate","Revenue","ROAS",
        "Organic_Impressions","Organic_Clicks","Organic_CTR","Bounce_Rate","Avg_Time_on_Page_sec",
        "Page_Position","Backlinks"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # -------------------------
    # Filters
    # -------------------------
    st.markdown("### Step 2 — Filters & preview")
    c1,c2,c3 = st.columns([2,2,1])
    campaigns = sorted(df["Campaign"].dropna().unique().tolist())
    adgroups = sorted(df["AdGroup"].dropna().unique().tolist())
    keywords = sorted(df["Keyword"].dropna().unique().tolist())
    devices = sorted(df["Device"].dropna().unique().tolist())
    countries = sorted(df["Country"].dropna().unique().tolist())

    with c1:
        sel_campaigns = st.multiselect("Campaign", options=campaigns, default=campaigns[:3])
    with c2:
        sel_adgroups = st.multiselect("AdGroup", options=adgroups, default=adgroups[:3])
    with c3:
        date_range = st.date_input("Date range", value=(df["Date"].min().date(), df["Date"].max().date()))

    filt = df.copy()
    if sel_campaigns:
        filt = filt[filt["Campaign"].isin(sel_campaigns)]
    if sel_adgroups:
        filt = filt[filt["AdGroup"].isin(sel_adgroups)]
    if date_range:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filt = filt[(filt["Date"]>=start) & (filt["Date"]<=end)]

    st.markdown("Filtered preview")
    st.dataframe(filt.head(5), use_container_width=True)
    download_df(filt.head(5), "filtered_preview.csv")

    # -------------------------
    # KPIs
    # -------------------------
    st.markdown("### Key Metrics")
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Paid Clicks", int(filt["Clicks"].sum()))
    k2.metric("Organic Clicks", int(filt["Organic_Clicks"].sum()))
    k3.metric("Revenue", to_currency(filt["Revenue"].sum()))
    k4.metric("ROAS", round(filt["ROAS"].mean(),2))
    k5.metric("Bounce Rate", f"{round(filt['Bounce_Rate'].mean()*100,2)}%")
    k6.metric("Avg Time/Page (s)", round(filt['Avg_Time_on_Page_sec'].mean(),2))

    # -------------------------
    # Charts
    # -------------------------
    st.markdown("### Paid vs Organic Clicks per Campaign")
    agg = filt.groupby("Campaign").agg({
        "Clicks":"sum","Organic_Clicks":"sum","Revenue":"sum"
    }).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Clicks"], name="Paid Clicks"))
    fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Organic_Clicks"], name="Organic Clicks"))
    fig.update_layout(barmode='group', xaxis_title="Campaign", yaxis_title="Clicks", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### CTR Trends over Time")
    ts = filt.groupby("Date").agg({"CTR":"mean","Organic_CTR":"mean"}).reset_index()
    fig2 = px.line(ts, x="Date", y=["CTR","Organic_CTR"], labels={"value":"CTR","variable":"Type"}, template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Revenue & ROAS by Keyword")
    agg2 = filt.groupby("Keyword").agg({"Revenue":"sum","ROAS":"mean"}).reset_index()
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=agg2["Keyword"], y=agg2["Revenue"], name="Revenue"))
    fig3.add_trace(go.Scatter(x=agg2["Keyword"], y=agg2["ROAS"], name="ROAS", yaxis="y2", mode="markers+lines"))
    fig3.update_layout(
        xaxis_title="Keyword",
        yaxis_title="Revenue",
        yaxis2=dict(title="ROAS", overlaying="y", side="right"),
        template="plotly_white"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # -------------------------
    # Download filtered data
    # -------------------------
    st.markdown("### Export filtered data")
    download_df(filt, "google_ads_seo_filtered.csv")

