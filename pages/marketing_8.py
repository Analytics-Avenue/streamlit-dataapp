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
# COMPANY HEADER
# ============================================================
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display:flex; align-items:center; margin-bottom:18px;">
    <img src="{logo_url}" width="60" style="margin-right:12px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:700;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:700;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# GLOBAL MI&FL UI CSS
# ============================================================
st.markdown("""
<style>
* { font-family: 'Inter', sans-serif; }

/* pure black text */
body, [class*="css"] { color:#000 !important; font-size:17px; }

/* section title */
.section-title {
    font-size:24px;
    font-weight:600;
    margin-top:30px;
    margin-bottom:12px;
    position:relative;
}
.section-title:after {
    content:"";
    position:absolute;
    bottom:-6px;
    left:0;
    width:0%;
    height:2px;
    background:#064b86;
    transition:0.35s ease;
}
.section-title:hover:after { width:40%; }

/* card */
.card {
    background:#fff;
    padding:22px;
    border-radius:14px;
    border:1px solid #e6e6e6;
    font-size:16.5px;
    font-weight:500;
    box-shadow:0 4px 14px rgba(0,0,0,0.07);
    transition:0.25s ease;
}
.card:hover {
    transform:translateY(-4px);
    border-color:#064b86;
    box-shadow:0 12px 25px rgba(6,75,134,0.18);
}

/* KPI */
.kpi {
    background:#fff;
    padding:22px;
    text-align:center;
    border-radius:14px;
    border:1px solid #dedede;
    color:#064b86 !important;
    font-size:20px;
    font-weight:600;
    box-shadow:0 3px 14px rgba(0,0,0,0.08);
    transition:0.25s ease;
}
.kpi:hover {
    transform:translateY(-4px);
    border-color:#064b86;
}

/* Buttons */
.stButton>button,
.stDownloadButton>button {
    background:#064b86 !important;
    color:white !important;
    padding:10px 22px;
    border-radius:8px;
    border:none;
    font-weight:600;
    transition:0.25s ease;
}
.stButton>button:hover,
.stDownloadButton>button:hover {
    background:#0a6eb3 !important;
    transform:translateY(-3px);
}

/* Fade */
.block-container { animation:fadeIn 0.5s ease; }
@keyframes fadeIn {
    from { opacity:0; transform:translateY(10px); }
    to { opacity:1; transform:translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# REQUIRED COLUMNS
# ============================================================
REQUIRED_COLS = [
    "Date","Campaign","AdGroup","Keyword","Device","Country",
    "Impressions","Clicks","CTR","CPC","Cost","Conversions","ConversionRate","Revenue","ROAS",
    "Organic_Impressions","Organic_Clicks","Organic_CTR","Bounce_Rate","Avg_Time_on_Page_sec",
    "Page_Position","Backlinks"
]

def to_currency(x):
    try: return "₹ "+f"{float(x):,.2f}"
    except: return x

def ensure_datetime(df, col="Date"):
    try: df[col] = pd.to_datetime(df[col], errors="coerce")
    except: pass
    return df

def download_df(df, filename):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button("Download CSV", b, file_name=filename, mime="text/csv")

# ============================================================
# --- TABS: Overview / Important Attributes / Application ---
# ============================================================
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    Analyze Google Ads and SEO performance together for unified ROI, CTR, revenue and keyword insights. 
    Supports paid + organic metrics, conversion funnel behavior, ML revenue prediction, and automated insights.
    </div>
    """, unsafe_allow_html=True)

    L, R = st.columns(2)

    with L:
        st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Paid + organic search performance<br>
        • Keyword-level ROI breakdown<br>
        • Bounce rate, time on page & organic CTR insights<br>
        • ML-driven revenue prediction<br>
        • Campaign and AdGroup analysis<br>
        • Executive-ready reporting exports
        </div>
        """, unsafe_allow_html=True)

    with R:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Improve ROAS by combining SEO + ads<br>
        • Identify high-ROI keywords<br>
        • Reduce wasted paid spend<br>
        • Strengthen content & bidding strategy<br>
        • Forecast performance & revenue
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Paid Clicks</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Organic Clicks</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Revenue</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>ROAS</div>", unsafe_allow_html=True)

# ============================================================
# TAB 2 — IMPORTANT ATTRIBUTES
# ============================================================
with tab2:
    st.markdown('<div class="section-title">Required Column Dictionary</div>', unsafe_allow_html=True)
    req_dict = {
        "Campaign":"Google Ads campaign name",
        "AdGroup":"Ad group name",
        "Keyword":"Keyword text",
        "Impressions":"Total paid impressions",
        "Organic_Impressions":"Organic impressions",
        "Clicks":"Paid clicks",
        "Organic_Clicks":"Clicks from organic search",
        "CTR":"Paid Click-through Rate",
        "CPC":"Cost per Click",
        "Cost":"Total cost",
        "Conversions":"Paid conversions",
        "Revenue":"Total revenue",
        "ROAS":"Return on Ad Spend",
        "Bounce_Rate":"SEO bounce rate",
        "Avg_Time_on_Page_sec":"Average time on page",
        "Page_Position":"SERP position",
        "Backlinks":"Backlink count"
    }

    df_attr = pd.DataFrame(
        [{"Attribute":k, "Description":v} for k,v in req_dict.items()]
    )
    st.dataframe(df_attr, use_container_width=True)

# ============================================================
# TAB 3 — APPLICATION
# ============================================================
with tab3:
    st.markdown('<div class="section-title">Step 1 — Load Dataset</div>', unsafe_allow_html=True)
    mode = st.radio("Dataset option:", ["Default dataset","Upload CSV","Upload CSV + Column mapping"], horizontal=True)
    df = None

    # ----------------- DEFAULT
    if mode == "Default dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/google_ads_seo_performance.csv"
        try:
            df = pd.read_csv(URL)
            df.columns = df.columns.str.strip()
            st.success("Default dataset loaded.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Could not load dataset: {e}")
            st.stop()

    # ----------------- SIMPLE UPLOAD
    elif mode == "Upload CSV":
        sample_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/google_ads_seo_performance.csv"
        try:
            smp = pd.read_csv(sample_url).head(5)
            st.download_button("Download Sample CSV", smp.to_csv(index=False), "sample.csv", "text/csv")
        except:
            pass

        f = st.file_uploader("Upload CSV file", type=["csv"])
        if f:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            st.dataframe(df.head())

    # ----------------- MAPPING MODE
    else:
        up = st.file_uploader("Upload CSV for mapping", type=["csv"])
        if up:
            raw = pd.read_csv(up)
            raw.columns = raw.columns.str.strip()
            st.write("Preview:")
            st.dataframe(raw.head())
            mapping = {}
            for req in REQUIRED_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", ["-- Select --"] + list(raw.columns))
            if st.button("Apply Mapping"):
                missing = [k for k,v in mapping.items() if v=="-- Select --"]
                if missing:
                    st.error("Map all required columns: "+", ".join(missing))
                else:
                    df = raw.rename(columns={v:k for k,v in mapping.items()})
                    st.success("Mapping applied.")
                    st.dataframe(df.head())

    if df is None:
        st.stop()

    # ============================================================
    # CLEANING
    # ============================================================
    df = ensure_datetime(df, "Date")
    num_cols = [
        "Impressions","Clicks","Organic_Clicks","CTR","CPC","Cost","Conversions",
        "ConversionRate","Revenue","ROAS","Organic_Impressions","Organic_CTR",
        "Bounce_Rate","Avg_Time_on_Page_sec","Page_Position","Backlinks"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # ============================================================
    # FILTERS
    # ============================================================
    st.markdown('<div class="section-title">Step 2 — Filters & Preview</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns([2,2,1])

    campaigns = sorted(df["Campaign"].dropna().unique().tolist())
    adgroups  = sorted(df["AdGroup"].dropna().unique().tolist())

    with c1:
        sel_campaigns = st.multiselect("Campaign", campaigns, default=campaigns[:3])
    with c2:
        sel_ag = st.multiselect("AdGroup", adgroups, default=adgroups[:3])
    with c3:
        date_range = st.date_input("Date range", (df["Date"].min().date(), df["Date"].max().date()))

    filt = df.copy()
    if sel_campaigns:
        filt = filt[filt["Campaign"].isin(sel_campaigns)]
    if sel_ag:
        filt = filt[filt["AdGroup"].isin(sel_ag)]
    start,end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filt = filt[(filt["Date"]>=start) & (filt["Date"]<=end)]

    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(300), "filtered_preview.csv")

    # ============================================================
    # KPIs
    # ============================================================
    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Paid Clicks", int(filt["Clicks"].sum()))
    k2.metric("Organic Clicks", int(filt["Organic_Clicks"].sum()))
    k3.metric("Revenue", to_currency(filt["Revenue"].sum()))
    k4.metric("ROAS", round(filt["ROAS"].mean(),2))
    k5.metric("Bounce Rate", f"{round(filt['Bounce_Rate'].mean()*100,2)}%")
    k6.metric("Avg Time Page (s)", round(filt["Avg_Time_on_Page_sec"].mean(),2))

    # ============================================================
    # CHARTS
    # ============================================================
    st.markdown('<div class="section-title">Charts</div>', unsafe_allow_html=True)

    agg = filt.groupby("Campaign").agg({"Clicks":"sum","Organic_Clicks":"sum"}).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Clicks"], name="Paid Clicks"))
    fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Organic_Clicks"], name="Organic Clicks"))
    fig.update_layout(barmode='group', template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    ts = filt.groupby("Date").agg({"CTR":"mean","Organic_CTR":"mean"}).reset_index()
    fig2 = px.line(ts, x="Date", y=["CTR","Organic_CTR"], template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    kw = filt.groupby("Keyword").agg({"Revenue":"sum","ROAS":"mean"}).reset_index()
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=kw["Keyword"], y=kw["Revenue"], name="Revenue"))
    fig3.add_trace(go.Scatter(x=kw["Keyword"], y=kw["ROAS"], name="ROAS", yaxis="y2"))
    fig3.update_layout(
        template="plotly_white",
        yaxis2=dict(title="ROAS", overlaying="y", side="right")
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ============================================================
    # ML
    # ============================================================
    st.markdown('<div class="section-title">ML — Predict Revenue</div>', unsafe_allow_html=True)
    ml_df = filt.dropna(subset=["Revenue"])
    if len(ml_df)<30:
        st.info("Not enough data for ML.")
    else:
        feat = ["Clicks","Impressions","Organic_Clicks","Conversions","CPC","CTR"]
        X = ml_df[feat]
        y = ml_df["Revenue"]

        scaler = StandardScaler()
        X_t = scaler.fit_transform(X)

        X_train,X_test,y_train,y_test = train_test_split(X_t,y,test_size=0.2,random_state=42)
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        with st.spinner("Training RandomForest..."):
            rf.fit(X_train,y_train)
        preds = rf.predict(X_test)
        rmse = math.sqrt(np.mean((y_test - preds)**2))
        r2 = rf.score(X_test,y_test)
        st.write(f"RMSE: {rmse:.2f} | R²: {r2:.3f}")

        results = pd.DataFrame({"Actual":y_test, "Predicted":preds})
        st.dataframe(results.head(), use_container_width=True)
        download_df(results, "ml_revenue_predictions.csv")

    # ============================================================
    # INSIGHTS
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
    st.markdown('<div class="section-title">Export Full Filtered Dataset</div>', unsafe_allow_html=True)
    download_df(filt, "google_ads_seo_filtered.csv")
