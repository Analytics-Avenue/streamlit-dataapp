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

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Google Ads & SEO Performance Lab", layout="wide")
st.markdown("""<style>[data-testid="stSidebarNav"]{display:none;}</style>""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display:flex; align-items:center; margin-bottom:16px;">
    <img src="{logo_url}" width="60" style="margin-right:12px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:700;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:700;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# GLOBAL CSS (Your Master Spec)
# ============================================================
st.markdown("""
<style>
* { font-family:'Inter', sans-serif; }

/* GLOBAL TEXT */
body, [class*="css"] { color:#000 !important; font-size:17px; }

/* SECTION TITLE */
.section-title {
    font-size:24px !important;
    font-weight:600 !important;
    margin-top:30px;
    margin-bottom:12px;
    color:#000 !important;
    position:relative;
}
.section-title:after {
    content:"";
    position:absolute;
    bottom:-5px;
    left:0;
    height:2px;
    width:0%;
    background:#064b86;
    transition:width 0.35s ease;
}
.section-title:hover:after { width:40%; }

/* CARDS */
.card {
    background:white;
    padding:22px;
    border-radius:14px;
    border:1px solid #e6e6e6;
    color:#000 !important;
    font-size:16.5px;
    font-weight:500;
    box-shadow:0 3px 14px rgba(0,0,0,0.08);
    transition:0.25s ease;
}
.card:hover {
    transform:translateY(-4px);
    box-shadow:0 12px 24px rgba(6,75,134,0.18);
    border-color:#064b86;
}

/* KPI */
.kpi {
    background:white;
    padding:20px;
    border-radius:14px;
    border:1px solid #e2e2e2;
    text-align:center;
    color:#064b86 !important;
    font-size:20px;
    font-weight:600;
    box-shadow:0 3px 12px rgba(0,0,0,0.07);
    transition:0.25s ease;
}
.kpi:hover {
    transform:translateY(-4px);
    box-shadow:0 12px 24px rgba(6,75,134,0.20);
    border-color:#064b86;
}

/* TABLE */
.dataframe th {
    background:#064b86 !important;
    color:white !important;
    padding:10px !important;
    font-size:15px !important;
}
.dataframe td {
    padding:9px !important;
    color:#000 !important;
    font-size:15px !important;
    border-bottom:1px solid #eaeaea !important;
}
.dataframe tbody tr:hover { background:#f4f9ff !important; }

/* BUTTONS */
.stButton>button, .stDownloadButton>button {
    background:#064b86 !important;
    color:white !important;
    padding:10px 22px;
    border-radius:8px !important;
    border:none;
    font-weight:600;
    transition:0.25s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    transform:translateY(-3px);
    background:#0a6eb3 !important;
}

/* FADE-IN */
.block-container { animation: fadeIn 0.5s ease; }
@keyframes fadeIn {
    from {opacity:0; transform:translateY(10px);}
    to {opacity:1; transform:translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================
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

# ============================================================
# TABS
# ============================================================
tab1, tab2 = st.tabs(["Overview", "Application"])

# ============================================================
# OVERVIEW TAB
# ============================================================
with tab1:
    st.markdown("<div class='section-title'>Google Ads & SEO Performance Lab</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    Analyze paid search campaigns alongside SEO metrics to optimize your keyword strategy,
    campaign spend, and full-funnel conversions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        • Unified paid + organic performance tracking<br>
        • Keyword-level ROI & conversion analysis<br>
        • SERP position, time-on-page, bounce rate insights<br>
        • ML-based revenue prediction<br>
        • Exportable performance tables
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Impact</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        • Improve ROAS & keyword performance<br>
        • Optimize landing pages & SEO focus<br>
        • Identify high-intent organic keywords<br>
        • Reduce wasted spend<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("<div class='kpi'>Paid Clicks</div>", unsafe_allow_html=True)
    c2.markdown("<div class='kpi'>Organic Clicks</div>", unsafe_allow_html=True)
    c3.markdown("<div class='kpi'>Revenue</div>", unsafe_allow_html=True)
    c4.markdown("<div class='kpi'>ROAS</div>", unsafe_allow_html=True)

# ============================================================
# APPLICATION TAB
# ============================================================
with tab2:
    st.markdown('<div class="section-title">Step 1 — Load Dataset</div>', unsafe_allow_html=True)

    mode = st.radio("Dataset option:", 
                    ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], 
                    horizontal=True)

    df = None

    if mode == "Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/google_ads_seo_performance.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            st.success("Default dataset loaded")
            st.dataframe(df.head(), use_container_width=True)
        except:
            st.error("Could not load default dataset.")
            st.stop()

    elif mode == "Upload CSV":
        st.markdown("Download sample CSV (optional)")
        SAMPLE_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/google_ads_seo_performance.csv"
        try:
            smp = pd.read_csv(SAMPLE_URL).head(5)
            st.download_button("Download Sample CSV", smp.to_csv(index=False), mime="text/csv")
        except:
            pass

        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)

    else:
        uploaded = st.file_uploader("Upload CSV for column mapping", type=["csv"])
        if uploaded:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.dataframe(raw.head(), use_container_width=True)

            mapping = {}
            for req in REQUIRED_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", 
                                            ["-- Select --"] + list(raw.columns))

            if st.button("Apply mapping"):
                missing = [k for k, v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Map all required columns: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v: k for k, v in mapping.items()})
                    st.success("Mapping applied")

    if df is None:
        st.stop()

    # ============================================================
    # CLEANING
    # ============================================================
    df = ensure_datetime(df, "Date")

    numeric_cols = [
        "Impressions","Clicks","CTR","CPC","Cost","Conversions","ConversionRate",
        "Revenue","ROAS","Organic_Impressions","Organic_Clicks","Organic_CTR",
        "Bounce_Rate","Avg_Time_on_Page_sec","Page_Position","Backlinks"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ============================================================
    # FILTERS
    # ============================================================
    st.markdown('<div class="section-title">Step 2 — Filters</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 2, 1])

    campaigns = sorted(df["Campaign"].dropna().unique())
    adgroups = sorted(df["AdGroup"].dropna().unique())

    with c1:
        sel_campaigns = st.multiselect("Campaign", campaigns, campaigns[:3])

    with c2:
        sel_adgroups = st.multiselect("AdGroup", adgroups, adgroups[:3])

    with c3:
        date_range = st.date_input("Date range",
                                   (df["Date"].min().date(), df["Date"].max().date()))

    filt = df.copy()
    if sel_campaigns:
        filt = filt[filt["Campaign"].isin(sel_campaigns)]
    if sel_adgroups:
        filt = filt[filt["AdGroup"].isin(sel_adgroups)]

    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filt = filt[(filt["Date"] >= start) & (filt["Date"] <= end)]

    st.dataframe(filt.head(), use_container_width=True)
    download_df(filt.head(10), "filtered_preview.csv")

    # ============================================================
    # KPIs
    # ============================================================
    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)

    kk1, kk2, kk3, kk4, kk5, kk6 = st.columns(6)

    kk1.metric("Paid Clicks", int(filt["Clicks"].sum()))
    kk2.metric("Organic Clicks", int(filt["Organic_Clicks"].sum()))
    kk3.metric("Revenue", to_currency(filt["Revenue"].sum()))
    kk4.metric("ROAS", round(filt["ROAS"].mean(), 2))
    kk5.metric("Bounce Rate", f"{round(filt['Bounce_Rate'].mean()*100,2)}%")
    kk6.metric("Avg Time/Page (s)", round(filt['Avg_Time_on_Page_sec'].mean(), 2))

    # ============================================================
    # CHARTS
    # ============================================================
    st.markdown('<div class="section-title">Paid vs Organic Clicks per Campaign</div>', unsafe_allow_html=True)

    agg = filt.groupby("Campaign").agg({
        "Clicks":"sum",
        "Organic_Clicks":"sum"
    }).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Clicks"], name="Paid Clicks"))
    fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Organic_Clicks"], name="Organic Clicks"))
    fig.update_layout(barmode="group", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">CTR Trends over Time</div>', unsafe_allow_html=True)

    ts = filt.groupby("Date").agg({"CTR":"mean","Organic_CTR":"mean"}).reset_index()
    fig2 = px.line(ts, x="Date", y=["CTR","Organic_CTR"], template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Revenue & ROAS by Keyword</div>', unsafe_allow_html=True)

    agg2 = filt.groupby("Keyword").agg({
        "Revenue":"sum",
        "ROAS":"mean"
    }).reset_index()

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=agg2["Keyword"], y=agg2["Revenue"], name="Revenue"))
    fig3.add_trace(go.Scatter(
        x=agg2["Keyword"], y=agg2["ROAS"], name="ROAS", mode="markers+lines", yaxis="y2"
    ))

    fig3.update_layout(
        template="plotly_white",
        yaxis=dict(title="Revenue"),
        yaxis2=dict(title="ROAS", overlaying="y", side="right")
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ============================================================
    # ML
    # ============================================================
    st.markdown('<div class="section-title">ML — Predict Revenue</div>', unsafe_allow_html=True)

    ml_df = filt.dropna(subset=["Revenue"])
    if len(ml_df) < 30:
        st.info("Not enough data for ML model.")
    else:
        feat_cols = ["Clicks","Impressions","Organic_Clicks","Conversions","CPC","CTR"]
        X = ml_df[feat_cols]
        y = ml_df["Revenue"]

        X_t = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)

        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        with st.spinner("Training RandomForest…"):
            rf.fit(X_train, y_train)

        preds = rf.predict(X_test)
        rmse = math.sqrt(((y_test - preds)**2).mean())
        r2 = rf.score(X_test, y_test)

        st.markdown(f"""
        <div class='card'>
        <b>RMSE:</b> {rmse:.2f}<br>
        <b>R² Score:</b> {r2:.3f}
        </div>
        """, unsafe_allow_html=True)

        out = pd.DataFrame({
            "Actual Revenue": y_test.values,
            "Predicted Revenue": preds
        })
        st.dataframe(out.head(), use_container_width=True)
        download_df(out, "ml_revenue_predictions.csv")

    # ============================================================
    # AUTOMATED INSIGHTS
    # ============================================================
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)

    ins = filt.groupby(["Campaign","Keyword"]).agg({
        "Revenue":"sum",
        "Clicks":"sum",
        "Conversions":"sum",
        "ROAS":"mean"
    }).reset_index().sort_values("Revenue", ascending=False)

    st.dataframe(ins, use_container_width=True)
    download_df(ins, "automated_insights.csv")

    # ============================================================
    # EXPORT
    # ============================================================
    st.markdown('<div class="section-title">Export Filtered Data</div>', unsafe_allow_html=True)
    download_df(filt, "google_ads_seo_filtered.csv")

# End
