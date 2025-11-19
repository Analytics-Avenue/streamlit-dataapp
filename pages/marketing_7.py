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

st.set_page_config(page_title="Customer Journey & Funnel Analytics", layout="wide")

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
# Required columns
# -------------------------
REQUIRED_COLS = [
    "Date","Campaign","Channel","Stage","Conversion_Flag","Revenue",
    "Impressions","Clicks","Leads","CTR","CPC","CPA",
    "Video_50%","Video_75%","Video_100%","ThruPlay_Rate",
    "Country","Device","AgeGroup","Gender"
]

# -------------------------
# Helper functions
# -------------------------
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

def ensure_datetime(df, col="Date"):
    try:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    except:
        pass
    return df

# -------------------------
# Page header + overview
# -------------------------
st.markdown("<h1 style='margin-bottom:0.2rem'>Customer Journey & Funnel Analytics</h1>", unsafe_allow_html=True)
st.markdown("Track marketing funnels, conversions, video engagement, and predictive insights in one enterprise-ready dashboard.")

# -------------------------
# CSS for proper alignment
# -------------------------
st.markdown("""
<style>
/* Left-align main container (adjust margin as needed) */
.css-1d391kg {  
    margin-left: 2rem;
    margin-right: 2rem;
    max-width: 1200px;
}

/* Left-align all cards inside markdown */
div[data-testid="stMarkdownContainer"] .card {
    text-align: left !important;
    margin-bottom: 20px;
}

/* Center metrics */
div[data-testid="stMarkdownContainer"] .metric-card {
    margin: auto;
    text-align: center !important;
}

/* Optional: larger card titles */
div[data-testid="stMarkdownContainer"] .card h3 {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Application"])

# -------------------------
# Overview Tab
# -------------------------
with tabs[0]:
    # -------------------------
    # Overview Section
    # -------------------------
    st.markdown("### Overview")
    st.markdown("""
    <div class='card'>
        This app delivers <b>end-to-end marketing performance tracking</b>, across campaigns, channels, creatives, and audience segments. 
        It aggregates campaign data, measures effectiveness, predicts revenue and conversions using <b>machine learning</b>, 
        and provides <b>forecasting</b> for short- and medium-term decision-making. 
        Built for <b>data-driven marketing teams</b>, the app gives actionable insights at a glance.
    </div>
    """, unsafe_allow_html=True)

    # -------------------------
    # Capabilities
    # -------------------------
    st.markdown("### Capabilities")
    st.markdown("""
    <div class='card'>
        • Multi-channel campaign tracking with breakdowns by channel, device, audience segment<br>
        • Audience analysis by Age, Gender, Device, and other demographic segments<br>
        • Creative performance insights: AdSet & Creative level ROI<br>
        • Predictive analytics: Revenue & Conversion forecasting using <b>RandomForest</b> & <b>Linear Regression</b><br>
        • Campaign optimization suggestions & ROI comparisons<br>
        • Automated insights highlighting best and worst-performing segments
    </div>
    """, unsafe_allow_html=True)

    # -------------------------
    # Impact
    # -------------------------
    st.markdown("### Impact")
    st.markdown("""
    <div class='card'>
        • Make <b>data-driven marketing decisions</b> faster<br>
        • Identify high-ROI campaigns & avoid wasted spend<br>
        • Prioritize channels, creatives, and audience segments based on predicted performance<br>
        • Improve conversion efficiency and revenue per spend unit<br>
        • Align marketing strategy with real-time insights and predictive trends
    </div>
    """, unsafe_allow_html=True)

    # -------------------------
    # Key Metrics
    # -------------------------
    st.markdown("### Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card' title='Total revenue generated across campaigns and channels'>Total Revenue</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card' title='Average return on ad spend across campaigns'>ROAS</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card' title='Total leads generated'>Total Leads</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card' title='Overall conversion rate (Conversions / Clicks)'>Conversion Rate</div>", unsafe_allow_html=True)

    # -------------------------
    # Forecasting & ML Capabilities
    # -------------------------
    st.markdown("### Forecasting & ML Capabilities")
    st.markdown("""
    <div class='card'>
        • Revenue & Conversion predictions using <b>RandomForest Regression</b><br>
        • Trend forecasting for next 30 days with <b>linear regression fallback</b> if Prophet is unavailable<br>
        • Automatic identification of top-performing campaigns, channels, and audience segments<br>
        • Model performance metrics (R², RMSE) displayed for transparency and trust<br>
        • Downloadable ML predictions (Actual vs Predicted + features) for further analysis
    </div>
    """, unsafe_allow_html=True)

    # -------------------------
    # Automated Insights
    # -------------------------
    st.markdown("### Automated Insights")
    st.markdown("""
    <div class='card'>
        • Channel-level ROI comparisons<br>
        • Identification of best and worst performing channels, creatives, and segments<br>
        • Downloadable insights tables for executive reporting<br>
        • Supports multi-dimensional filtering for campaigns, channels, device types, age-groups, and gender
    </div>
    """, unsafe_allow_html=True)

    # -------------------------
    # Who Should Use
    # -------------------------
    st.markdown("### Who Should Use This App?")
    st.markdown("""
    <div class='card'>
        • <b>Marketing Analysts</b> who want predictive insights and campaign breakdowns<br>
        • <b>CMOs / Marketing Heads</b> needing executive-ready dashboards<br>
        • <b>Digital Marketing Teams</b> optimizing ad spend across channels<br>
        • <b>Growth Teams</b> tracking conversion efficiency and revenue trends
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# Application tab
# -------------------------
with tabs[1]:
    st.header("Application")
    
    # -------------------------
    # Dataset input
    # -------------------------
    st.markdown("### Step 1 — Load dataset")
    mode = st.radio("Dataset option:", ["Default dataset","Upload CSV","Upload CSV + Mapping"], horizontal=True)
    df = None

    if mode=="Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/customer_journey_funnel_video.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            st.success("Default dataset loaded")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Failed to load default dataset: {e}")
            st.stop()
    
    elif mode=="Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            st.success("File uploaded")
            st.dataframe(df.head())
    
    else:
        uploaded = st.file_uploader("Upload CSV for mapping", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.write("Preview:")
            st.dataframe(raw.head())
            mapping = {}
            for req in REQUIRED_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + list(raw.columns))
            if st.button("Apply mapping"):
                missing = [k for k,v in mapping.items() if v=="-- Select --"]
                if missing:
                    st.error("Map all required columns: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v:k for k,v in mapping.items()})
                    st.success("Mapping applied")
                    st.dataframe(df.head())

    if df is None:
        st.stop()

    # -------------------------
    # Type conversion & derived metrics
    # -------------------------
    df = ensure_datetime(df,"Date")
    for col in ["Impressions","Clicks","Leads","Revenue","Video_50%","Video_75%","Video_100%","ThruPlay_Rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col],errors="coerce").fillna(0)
    
    # Derived metrics
    df["Conversion_Rate"] = np.where(df["Clicks"]>0, df["Leads"]/df["Clicks"],0)

    # -------------------------
    # Filters
    # -------------------------
    st.markdown("### Step 2 — Filters")
    c1,c2,c3 = st.columns([2,2,1])
    campaigns = sorted(df["Campaign"].dropna().unique().tolist())
    channels = sorted(df["Channel"].dropna().unique().tolist())
    
    with c1: sel_campaigns = st.multiselect("Campaign", options=campaigns, default=campaigns[:5])
    with c2: sel_channels = st.multiselect("Channel", options=channels, default=channels[:3])
    with c3: date_range = st.date_input("Date range", value=(df["Date"].min().date(), df["Date"].max().date()))
    
    filt = df.copy()
    if sel_campaigns: filt = filt[filt["Campaign"].isin(sel_campaigns)]
    if sel_channels: filt = filt[filt["Channel"].isin(sel_channels)]
    start,end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filt = filt[(filt["Date"]>=start) & (filt["Date"]<=end)]

    st.markdown("Filtered preview")
    st.dataframe(filt.head(5), use_container_width=True)
    download_df(filt.head(5),"filtered_customer_journey.csv")

    # -------------------------
    # KPIs
    # -------------------------
    st.markdown("### Key Metrics")
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Total Impressions",f"{int(filt['Impressions'].sum()):,}")
    k2.metric("Total Clicks",f"{int(filt['Clicks'].sum()):,}")
    k3.metric("Total Leads",f"{int(filt['Leads'].sum()):,}")
    k4.metric("Conversion Rate",f"{filt['Conversion_Rate'].mean():.2%}")
    k5.metric("Revenue",to_currency(filt['Revenue'].sum()))
    k6.metric("ThruPlay Rate",f"{filt['ThruPlay_Rate'].mean():.2%}")

    # -------------------------
    # Funnel chart
    # -------------------------
    st.markdown("### Funnel Stage Distribution")
    funnel = filt.groupby("Stage").agg({"Leads":"sum"}).reset_index()
    fig_funnel = px.funnel(funnel, x="Leads", y="Stage", text="Leads")
    st.plotly_chart(fig_funnel,use_container_width=True)

    # -------------------------
    # Video engagement chart
    # -------------------------
    st.markdown("### Video Engagement Metrics")
    video_df = filt[["Video_50%","Video_75%","Video_100%"]].mean().reset_index()
    video_df.columns = ["Metric","Average"]
    fig_video = px.bar(video_df, x="Metric", y="Average", text=video_df["Average"].round(2))
    st.plotly_chart(fig_video,use_container_width=True)

    # -------------------------
    # Predictive ML (Revenue)
    # -------------------------
    st.markdown("### ML: Predict Revenue")
    ml_df = filt.copy().dropna(subset=["Revenue"])
    if len(ml_df)<30:
        st.info("Not enough data for ML model.")
    else:
        feat_cols = ["Channel","Campaign","Impressions","Clicks","Leads","Video_50%","Video_75%","Video_100%"]
        X = ml_df[feat_cols]
        y = ml_df["Revenue"]
        cat_cols = [c for c in X.columns if X[c].dtype=="object"]
        num_cols = [c for c in X.columns if c not in cat_cols]
        preprocessor = ColumnTransformer([
            ("cat",OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num",StandardScaler(), num_cols)
        ])
        X_t = preprocessor.fit_transform(X)
        X_train,X_test,y_train,y_test = train_test_split(X_t,y,test_size=0.2,random_state=42)
        rf = RandomForestRegressor(n_estimators=200,random_state=42)
        with st.spinner("Training RandomForest for Revenue..."):
            rf.fit(X_train,y_train)
        preds = rf.predict(X_test)
        rmse = math.sqrt(np.mean((y_test-preds)**2))
        r2 = rf.score(X_test,y_test)
        st.write(f"Revenue Prediction — RMSE: {rmse:.2f}, R²: {r2:.3f}")
