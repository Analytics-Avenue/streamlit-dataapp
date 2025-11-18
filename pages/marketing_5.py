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
from sklearn.metrics import mean_squared_error, r2_score
import math
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# App Config
# -------------------------
st.set_page_config(page_title="Content & SEO Dashboard", layout="wide")

# -------------------------
# Company Logo + Name
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"  # Replace with your logo

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
# Header
# -------------------------
st.markdown("<h1 style='margin-top:0.5rem; margin-bottom:0.2rem'>Content & SEO Performance Dashboard</h1>", unsafe_allow_html=True)
st.markdown("Track page performance, keyword ROI, audience engagement & conversions for SEO & content strategy.")

# -------------------------
# Required columns
# -------------------------
REQUIRED_CONTENT_COLS = [
    "Date","Page","Content_Type","Keyword","Device","Country",
    "Impressions","Clicks","CTR","Bounce_Rate","Time_on_Page_sec",
    "Backlinks","Conversions","Revenue"
]

AUTO_MAPS = {
    "Date": ["date"],
    "Page": ["page","url","landing page"],
    "Content_Type": ["content_type","type","format"],
    "Keyword": ["keyword","search term","query"],
    "Device": ["device","platform"],
    "Country": ["country","region"],
    "Impressions": ["impressions","impression"],
    "Clicks": ["clicks","click"],
    "CTR": ["ctr","click through rate"],
    "Bounce_Rate": ["bounce","bouncerate"],
    "Time_on_Page_sec": ["time_on_page","time_sec","duration"],
    "Backlinks": ["backlinks","links"],
    "Conversions": ["conversions","leads","goals"],
    "Revenue": ["revenue","earnings"]
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
# CSS for cards
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
</style>
""", unsafe_allow_html=True)

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["Overview", "Application"])

# -------------------------
# Overview Tab
# -------------------------
with tabs[0]:
    st.markdown("### Overview")
    st.markdown("""
    <div class='card'>
        It tracks content & SEO performance across pages, keywords, devices, and countries.
        Provides engagement, conversion, and revenue insights, along with ML-driven predictions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Capabilities")
    st.markdown("""
    <div class='card'>
        • Page & keyword performance analysis<br>
        • Bounce rate & time-on-page insights<br>
        • Device / country engagement trends<br>
        • Revenue & conversion prediction<br>
        • SEO & content optimization recommendations
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Total Revenue</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Total Conversions</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Average CTR</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Average Bounce Rate</div>", unsafe_allow_html=True)

# -------------------------
# Application Tab
# -------------------------
with tabs[1]:
    st.header("Application")

    # Step 1 — Load Dataset
    st.markdown("### Step 1 — Load dataset")
    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)
    df = None

    if mode == "Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/content_seo_dataset.csv"
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
        st.markdown("#### Download Sample CSV for Reference")
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/content_seo_dataset.csv"
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
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head())
            st.markdown("Map your columns to required fields.")
            mapping = {}
            for req in REQUIRED_CONTENT_COLS:
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

    df = df[[c for c in REQUIRED_CONTENT_COLS if c in df.columns]].copy()
    df = ensure_datetime(df, "Date")
    for col in ["Impressions","Clicks","CTR","Bounce_Rate","Time_on_Page_sec","Backlinks","Conversions","Revenue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Step 2 — Filters
    st.markdown("### Step 2 — Filters & preview")
    c1,c2,c3 = st.columns([2,2,1])
    pages = sorted(df["Page"].dropna().unique())
    devices = sorted(df["Device"].dropna().unique())
    countries = sorted(df["Country"].dropna().unique())
    content_types = sorted(df["Content_Type"].dropna().unique())

    with c1:
        sel_pages = st.multiselect("Page", options=pages, default=pages[:5])
    with c2:
        sel_devices = st.multiselect("Device", options=devices, default=devices[:3])
    with c3:
        date_range = st.date_input("Date range", value=(df["Date"].min().date(), df["Date"].max().date()))

    filt = df.copy()
    if sel_pages: filt = filt[filt["Page"].isin(sel_pages)]
    if sel_devices: filt = filt[filt["Device"].isin(sel_devices)]
    if date_range: filt = filt[(filt["Date"]>=pd.to_datetime(date_range[0])) & (filt["Date"]<=pd.to_datetime(date_range[1]))]

    st.dataframe(filt.head(5), use_container_width=True)
    download_df(filt.head(5), "filtered_preview_content.csv")

    # Key metrics
    st.markdown("### Key Metrics")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Revenue", to_currency(filt["Revenue"].sum()))
    k2.metric("Total Conversions", int(filt["Conversions"].sum()))
    k3.metric("Average CTR", f"{filt['CTR'].mean():.2%}")
    k4.metric("Avg Bounce Rate", f"{filt['Bounce_Rate'].mean():.2%}")

    # Charts
    # Revenue & Conversions per Page
    st.markdown("### Revenue & Conversions per Page")
    page_agg = filt.groupby("Page")[["Revenue","Conversions"]].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=page_agg["Page"], 
        y=page_agg["Revenue"], 
        name="Revenue",
        text=page_agg["Revenue"],
        textposition="outside"
    ))
    fig.add_trace(go.Bar(
        x=page_agg["Page"], 
        y=page_agg["Conversions"], 
        name="Conversions",
        text=page_agg["Conversions"],
        textposition="outside"
    ))
    fig.update_layout(
        barmode="group", 
        xaxis_title="Page", 
        yaxis_title="Value",
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Device / Country Performance")
    group_cols = ["Device","Country","Content_Type"]
    for g in group_cols:
        if g in filt.columns:
            grp = filt.groupby(g)[["Revenue","Conversions"]].sum().reset_index()
            fig = px.bar(grp, x=g, y="Revenue", text="Revenue", title=f"{g} Revenue")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

# ML Prediction
st.markdown("### ML: Predict Revenue (RandomForest)")
ml_df = filt.copy().dropna(subset=["Revenue"])
feat_cols = ["Page","Content_Type","Device","Country","Impressions","Clicks","Time_on_Page_sec","Backlinks"]
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

    # Combine predictions with input features for download
    X_test_df = pd.DataFrame(X_test, columns=[f"Feature_{i}" for i in range(X_test.shape[1])])
    X_test_df["Actual_Revenue"] = y_test.reset_index(drop=True)
    X_test_df["Predicted_Revenue"] = preds
    st.dataframe(X_test_df.head())
    download_df(X_test_df, "ml_revenue_predictions.csv")

    # Automated Insights (Table + Download)
    st.markdown("### Automated Insights")

    insights = []
    country_perf = filt.groupby("Country")[["Revenue","Conversions"]].sum().reset_index()
    country_perf["Revenue_per_Conversion"] = np.where(country_perf["Conversions"]>0,
                                                     country_perf["Revenue"]/country_perf["Conversions"],0)
    if not country_perf.empty:
        best = country_perf.sort_values("Revenue_per_Conversion", ascending=False).iloc[0]
        worst = country_perf.sort_values("Revenue_per_Conversion").iloc[0]
        insights.append({"Insight":"Best Country ROI", "Country":best['Country'], "Revenue_per_Conversion":best['Revenue_per_Conversion']})
        insights.append({"Insight":"Lowest Country ROI", "Country":worst['Country'], "Revenue_per_Conversion":worst['Revenue_per_Conversion']})

    insights_df = pd.DataFrame(insights)
    st.dataframe(insights_df)
    download_df(insights_df, "automated_insights.csv")
