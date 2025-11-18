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
st.set_page_config(page_title="Marketing Performance Analysis", layout="wide")

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
REQUIRED_MARKETING_COLS = [
    "Campaign", "Channel", "Date", "Impressions", "Clicks", "Leads", 
    "Conversions", "Spend", "Revenue", "ROAS", "Device", "AgeGroup", 
    "Gender", "AdSet", "Creative"
]

# Auto mapping for common variants
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
div[data-testid="stMarkdownContainer"] .card {
    background: rgba(255,255,255,0.07);
    padding: 18px 20px;
    border-radius: 14px;
    margin-bottom: 15px;
    border: 1px solid rgba(255,255,255,0.25);
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    backdrop-filter: blur(4px);
}
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
.metric-card[title]:hover:after {
    content: attr(title);
    position: absolute;
    bottom: -40px;
    left: 50%;
    transform: translateX(-50%);
    background: #222;
    padding: 6px 10px;
    color: #fff;
    border-radius: 6px;
    font-size: 12px;
    white-space: nowrap;
    box-shadow: 0 2px 10px rgba(0,0,0,0.35);
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Page header + tabs
# -------------------------
st.markdown("<h1 style='margin-bottom:0.2rem'>Marketing Performance Analysis</h1>", unsafe_allow_html=True)
st.markdown("Analyze campaign ROI, creative & audience insights, predict revenue & conversions. Smart, actionable metrics.")

tabs = st.tabs(["Overview", "Application"])

with tabs[0]:
    st.markdown("### Overview")
    st.markdown("""
    <div class='card'>
        This app tracks marketing performance across campaigns, channels, creatives, and audience segments.
        It predicts revenue & conversions using ML, provides forecasts and highlights high-ROI strategies.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Capabilities")
    st.markdown("""
    <div class='card'>
        • Analyze multi-channel campaign performance<br>
        • Measure audience & creative effectiveness<br>
        • Forecast revenue trends<br>
        • Predict conversion outcomes<br>
        • Identify top-performing campaigns and segments
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card' title='Total revenue generated'>Total Revenue</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card' title='Return on Ad Spend'>ROAS</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card' title='Total leads captured'>Total Leads</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card' title='Overall conversion rate'>Conversion Rate</div>", unsafe_allow_html=True)

with tabs[1]:
    st.header("Application")

    # -------------------------
    # Dataset input: all 3 modes
    # -------------------------
    st.markdown("### Step 1 — Load dataset")
    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)
    df = None

    if mode == "Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/marketing.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
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
            df = auto_map_columns(df)
            st.success("File uploaded.")
            st.dataframe(df.head())

    else:  # Upload + mapping
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head())
            st.markdown("Map your columns to required fields.")
            mapping = {}
            for req in REQUIRED_MARKETING_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + list(raw.columns))
            if st.button("Apply mapping"):
                missing = [k for k,v in mapping.items() if v=="-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v:k for k,v in mapping.items()})
                    st.success("Mapping applied.")
                    st.dataframe(df.head())

    if df is None:
        st.stop()

    # Strip spaces, keep only required columns
    df = df[[c for c in REQUIRED_MARKETING_COLS if c in df.columns]].copy()
    df = ensure_datetime(df, "Date")
    for col in ["Impressions","Clicks","Leads","Conversions","Spend","Revenue","ROAS"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # -------------------------
    # Filters
    # -------------------------
    st.markdown("### Step 2 — Filters & preview")
    c1, c2, c3 = st.columns([2,2,1])
    campaigns = sorted(df["Campaign"].dropna().unique())
    channels = sorted(df["Channel"].dropna().unique())
    devices = sorted(df["Device"].dropna().unique())
    agegroups = sorted(df["AgeGroup"].dropna().unique())
    genders = sorted(df["Gender"].dropna().unique())

    with c1:
        sel_campaigns = st.multiselect("Campaign", options=campaigns, default=campaigns[:5])
    with c2:
        sel_channels = st.multiselect("Channel", options=channels, default=channels[:3])
    with c3:
        date_range = st.date_input("Date range", value=(df["Date"].min().date(), df["Date"].max().date()))

    filt = df.copy()
    if sel_campaigns: filt = filt[filt["Campaign"].isin(sel_campaigns)]
    if sel_channels: filt = filt[filt["Channel"].isin(sel_channels)]
    if date_range: filt = filt[(filt["Date"]>=pd.to_datetime(date_range[0])) & (filt["Date"]<=pd.to_datetime(date_range[1]))]

    st.dataframe(filt.head(5), use_container_width=True)
    download_df(filt.head(5), "filtered_preview.csv")

    # -------------------------
    # Key metrics cards
    # -------------------------
    st.markdown("### Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Revenue", to_currency(filt["Revenue"].sum()))
    k2.metric("ROAS", f"{filt['ROAS'].mean():.2f}")
    k3.metric("Total Leads", int(filt["Leads"].sum()))
    k4.metric("Conversion Rate", f"{filt['Conversions'].sum()/max(filt['Clicks'].sum(),1):.2%}")

    # -------------------------
    # Charts
    # -------------------------
    st.markdown("### Campaign Revenue & Conversions")
    agg = filt.groupby("Campaign")[["Revenue","Conversions"]].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Revenue"], name="Revenue"))
    fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Conversions"], name="Conversions"))
    fig.update_layout(barmode='group', xaxis_title="Campaign", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Device / Gender / AgeGroup performance")
    group_cols = ["Device","Gender","AgeGroup"]
    for g in group_cols:
        if g in filt.columns:
            grp = filt.groupby(g)[["Revenue","Conversions"]].sum().reset_index()
            fig = px.bar(grp, x=g, y="Revenue", text="Revenue", title=f"{g} Revenue")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)


    # -------------------------
    # ML: Revenue prediction + Download
    # -------------------------
    st.markdown("### ML: Predict Revenue (RandomForest)")
    ml_df = filt.copy().dropna(subset=["Revenue"])
    feat_cols = ["Channel","Campaign","Device","AgeGroup","Gender","AdSet","Impressions","Clicks","Spend"]
    feat_cols = [c for c in feat_cols if c in ml_df.columns]
    if len(ml_df) < 30 or len(feat_cols)<2:
        st.info("Not enough data to train ML model (>=30 rows needed)")
    else:
        X = ml_df[feat_cols]
        y = ml_df["Revenue"]
        cat_cols = [c for c in X.columns if X[c].dtype=="object"]
        num_cols = [c for c in X.columns if c not in cat_cols]
        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols)
        ], remainder="drop")
        X_t = preprocessor.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        with st.spinner("Training RandomForest..."):
            rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        rmse = math.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        st.write(f"Revenue prediction — RMSE: {rmse:.2f}, R²: {r2:.3f}")
    
        # Create downloadable dataframe
        X_test_df = pd.DataFrame(X_test, columns=[f"Feature_{i}" for i in range(X_test.shape[1])])
        X_test_df["Actual_Revenue"] = y_test.reset_index(drop=True)
        X_test_df["Predicted_Revenue"] = preds
        st.dataframe(X_test_df.head())
        download_df(X_test_df, "ml_revenue_predictions.csv")
    
    # -------------------------
    # Automated Insights (Table + Download)
    # -------------------------
    st.markdown("### Automated Insights")
    insights_list = []
    if "Channel" in filt.columns and "Revenue" in filt.columns and "Spend" in filt.columns:
        ch_perf = filt.groupby("Channel")[["Revenue","Spend"]].sum().reset_index()
        ch_perf["Revenue_per_Rs"] = np.where(ch_perf["Spend"]>0, ch_perf["Revenue"]/ch_perf["Spend"],0)
        best = ch_perf.sort_values("Revenue_per_Rs", ascending=False).iloc[0]
        worst = ch_perf.sort_values("Revenue_per_Rs", ascending=True).iloc[0]
        insights_list.append({"Insight":"Best Channel ROI","Channel":best['Channel'], "Revenue_per_Rs":best['Revenue_per_Rs']})
        insights_list.append({"Insight":"Lowest Channel ROI","Channel":worst['Channel'], "Revenue_per_Rs":worst['Revenue_per_Rs']})
    
    insights_df = pd.DataFrame(insights_list)
    st.dataframe(insights_df)
    download_df(insights_df, "automated_insights.csv")
