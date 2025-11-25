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
# -------------------------
# Helper: download DataFrame as CSV
# Must be defined before any call to download_df(...)
# -------------------------
from io import BytesIO
# ------------------------------------------------
# Auto column mapper (must be defined BEFORE usage)
# ------------------------------------------------
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
    "Device": ["device"],
    "AgeGroup": ["age group", "age"],
    "Gender": ["gender", "sex"],
    "AdSet": ["adset", "ad set"],
    "Creative": ["creative"]
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

def download_df(df: pd.DataFrame, filename: str):
    """
    Provide a safe CSV download button for dataframe `df`.
    Keeps no index and uses UTF-8. Call anywhere after this definition.
    """
    if df is None or df.empty:
        st.info("No data to download.")
        return

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    b = BytesIO()
    b.write(csv_bytes)
    b.seek(0)
    st.download_button(label="Download CSV", data=b, file_name=filename, mime="text/csv")


import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Marketing Performance Analysis", layout="wide")

# -------------------------
# Header & Logo
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display: flex; align-items: center; margin-bottom:16px;">
    <img src="{logo_url}" width="60" style="margin-right:12px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:700;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:700;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Global CSS — EXACT from your sample
# -------------------------
st.markdown("""
<style>
* { font-family: 'Inter', sans-serif; }

/* PURE BLACK global text */
body, [class*="css"] { color:#000 !important; font-size:17px; }

/* Section Title */
.section-title {
    font-size: 24px !important;
    font-weight: 600 !important;
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
    transition: width 0.35s ease;
}
.section-title:hover:after { width:40%; }

/* Cards (pure black text) */
.card {
    background:#ffffff;
    padding:22px;
    border-radius:14px;
    border:1px solid #e6e6e6;
    font-size:16.5px;
    color:#000 !important;
    font-weight:500;
    box-shadow:0 3px 14px rgba(0,0,0,0.08);
    transition:0.25s ease;
}
.card:hover {
    transform:translateY(-4px);
    box-shadow:0 12px 25px rgba(6,75,134,0.18);
    border-color:#064b86;
}

/* KPI CARDS (blue text) */
.kpi {
    background:#ffffff;
    padding:22px;
    border-radius:14px;
    border:1px solid #e2e2e2;
    font-size:20px !important;
    font-weight:600 !important;
    text-align:center;
    color:#064b86 !important;
    box-shadow:0 3px 14px rgba(0,0,0,0.07);
    transition:0.25s ease;
}
.kpi:hover {
    transform:translateY(-4px);
    box-shadow:0 13px 26px rgba(6,75,134,0.20);
    border-color:#064b86;
}

/* Variable boxes (blue text) */
.variable-box {
    padding:18px;
    border-radius:14px;
    background:white;
    border:1px solid #e5e5e5;
    box-shadow:0 2px 10px rgba(0,0,0,0.10);
    transition:0.25s ease;
    text-align:center;
    font-size:17.5px !important;
    font-weight:500 !important;
    color:#064b86 !important;
    margin-bottom:12px;
}
.variable-box:hover {
    transform:translateY(-5px);
    box-shadow:0 12px 22px rgba(6,75,134,0.18);
    border-color:#064b86;
}

/* Table styling */
.dataframe th {
    background:#064b86 !important;
    color:#fff !important;
    padding:11px !important;
    font-size:15.5px !important;
}
.dataframe td {
    font-size:15.5px !important;
    color:#000 !important;
    padding:9px !important;
    border-bottom:1px solid #efefef !important;
}
.dataframe tbody tr:hover { background:#f4f9ff !important; }

/* Buttons */
.stButton>button, .stDownloadButton>button {
    background:#064b86 !important;
    color:white !important;
    border:none;
    padding:10px 22px;
    border-radius:8px !important;
    font-size:15.5px !important;
    font-weight:600 !important;
    transition:0.25s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    transform:translateY(-3px);
    background:#0a6eb3 !important;
}

/* Fade-in animation */
.block-container { animation: fadeIn 0.5s ease; }
@keyframes fadeIn { 
    from {opacity:0; transform:translateY(10px);} 
    to {opacity:1; transform:translateY(0);} 
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Helpers
# -------------------------
REQUIRED_COLS = [
    "Campaign", "Channel", "Date", "Impressions", "Clicks",
    "Leads", "Conversions", "Spend", "Revenue", "ROAS"
]

def auto_map(df):
    mapping = {}
    lower = {c.lower(): c for c in df.columns}
    for req in REQUIRED_COLS:
        if req.lower() in lower:
            mapping[lower[req.lower()]] = req
    df = df.rename(columns=mapping)
    return df

def ensure_numeric(df):
    for c in ["Impressions","Clicks","Leads","Conversions","Spend","Revenue","ROAS"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def ensure_date(df):
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

def download(df, name):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button("Download CSV", b, name)

def to_money(v):
    try: return "₹ " + format(float(v), ",.2f")
    except: return "₹ 0.00"

# -------------------------
# Tabs
# -------------------------
t1, t2, t3 = st.tabs(["Overview", "Important Attributes", "Application"])

# =======================================================================================
# TAB 1 — OVERVIEW (Exact layout)
# =======================================================================================
with t1:

    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    A complete marketing performance intelligence system that unifies channel data, 
    produces insights, forecasts revenue, and provides ML-based predictions.
    </div>
    """, unsafe_allow_html=True)

    L, R = st.columns(2)

    with L:
        st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Multi-channel analytics<br>
        • KPI-driven performance benchmarking<br>
        • Machine Learning revenue prediction<br>
        • Spend optimization recommendations<br>
        • Automated insights engine<br>
        </div>
        """, unsafe_allow_html=True)

    with R:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Improve ROAS<br>
        • Reduce wasted spend<br>
        • Identify top converting audiences<br>
        • Strengthen budgeting & forecasting<br>
        • Enhance cross-channel visibility<br>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">High-Level KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Total Revenue</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>ROAS</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Leads</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Conversions</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Who Should Use This</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    Performance marketers, growth leads, CMOs, marketing analysts, 
    and teams needing unified revenue-driven marketing intelligence.
    </div>
    """, unsafe_allow_html=True)

# =======================================================================================
# TAB 2 — IMPORTANT ATTRIBUTES (Exact layout)
# =======================================================================================
with t2:

    # --------------------------------------------
    # Required Column Data Dictionary (Marketing Performance)
    # --------------------------------------------
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)
    
    required_dict = {
        "Campaign": "Name of the marketing campaign or initiative.",
        "Channel": "Acquisition platform (Facebook, Google, Meta Ads, Email, etc.).",
        "Date": "Date of the campaign record. Must be parseable as datetime.",
        "Impressions": "Total number of times the ad was shown to users.",
        "Clicks": "Total number of user clicks recorded for the ad.",
        "Leads": "Number of users who submitted a form / expressed intent.",
        "Conversions": "Number of users who completed the final business goal (purchase / signup / booking).",
        "Spend": "Total marketing cost for that row/period.",
        "Revenue": "Revenue generated from the campaign activity.",
        "ROAS": "Return on Ad Spend = Revenue / Spend."
    }
    
    dict_df = pd.DataFrame(
        [{"Column": k, "Description": v} for k, v in required_dict.items()]
    )
    
    # PURE BLACK table styling override (same as your sample)
    st.markdown("""
        <style>
            .req-table th {
                background:#ffffff !important;
                color:#000 !important;
                font-size:22px !important;
                border-bottom:2px solid #000 !important;
            }
            .req-table td {
                color:#000 !important;
                font-size:20px !important;
                padding:10px !important;
                border-bottom:1px solid #dcdcdc !important;
            }
            .req-table tr:hover td {
                background:#f8f8f8 !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.dataframe(dict_df.style.set_table_attributes('class="req-table"'), use_container_width=True)


    L, R = st.columns(2)

    with L:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        for v in ["Campaign","Channel","Date","Impressions","Clicks","Spend"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with R:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        for v in ["Leads","Conversions","Revenue","ROAS"]:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# =======================================================================================
# TAB 3 — APPLICATION (Exact layout)
# =======================================================================================
with t3:

    # -------------------------
    # Step 1 — Load dataset
    # -------------------------
    st.markdown('<div class="section-title">Step 1 — Load dataset</div>', unsafe_allow_html=True)
    
    mode = st.radio(
        "Dataset option:",
        ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"],
        horizontal=True
    )
    
    df = None
    
    # ----------------------------------------------------
    # DEFAULT DATASET
    # ----------------------------------------------------
    if mode == "Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/marketing.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("Default dataset loaded successfully.")
            render_required_table(df.head(5))
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()
    
    # ----------------------------------------------------
    # UPLOAD SIMPLE CSV (AUTO MAP)
    # ----------------------------------------------------
    elif mode == "Upload CSV":
        st.markdown("#### Download sample CSV (optional)")
        SAMPLE_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/marketing.csv"
        try:
            smp = pd.read_csv(SAMPLE_URL).head(5)
            st.download_button(
                "Download Sample CSV",
                smp.to_csv(index=False),
                "sample_marketing.csv",
                "text/csv"
            )
        except:
            pass
    
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("File uploaded.")
            render_required_table(df.head(5))
    
    
    # ----------------------------------------------------
    # UPLOAD + FULL COLUMN MAPPING
    # ----------------------------------------------------
    else:  # Upload CSV + mapping
        uploaded = st.file_uploader("Upload CSV for manual mapping", type=["csv"])
        if uploaded:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
    
            st.markdown("### Preview (first 5 rows)")
            render_required_table(raw.head(5))
    
            st.markdown("### Map your columns to required fields")
            mapping = {}
            options = ["-- Select --"] + list(raw.columns)
    
            for req_col in REQUIRED_MARKETING_COLS:
                mapping[req_col] = st.selectbox(
                    f"Map → {req_col}",
                    options=options,
                    key=f"map_{req_col}"
                )
    
            if st.button("Apply Mapping"):
                missing = [k for k,v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v: k for k, v in mapping.items()})
                    st.success("Mapping applied successfully.")
                    render_required_table(df.head(5))
    
    # STOP IF NO DATAFRAME
    if df is None:
        st.stop()

    df = ensure_date(df)
    df = ensure_numeric(df)

    st.markdown("Preview:")
    st.dataframe(df.head(10), use_container_width=True)

    # -------------------------
    # Step 2 — Filters
    # -------------------------
    st.markdown('<div class="section-title">Step 2 — Filters</div>', unsafe_allow_html=True)

    C1, C2, C3 = st.columns([2,2,1])

    with C1:
        camp = sorted(df["Campaign"].dropna().unique())
        s_camp = st.multiselect("Campaign", camp, default=camp[:5])

    with C2:
        ch = sorted(df["Channel"].dropna().unique())
        s_ch = st.multiselect("Channel", ch, default=ch[:5])

    with C3:
        min_d = df["Date"].min()
        max_d = df["Date"].max()
        dr = st.date_input("Date Range", (min_d, max_d))

    filt = df.copy()
    if s_camp:
        filt = filt[filt["Campaign"].isin(s_camp)]
    if s_ch:
        filt = filt[filt["Channel"].isin(s_ch)]
    try:
        filt = filt[(filt["Date"] >= pd.to_datetime(dr[0])) &
                    (filt["Date"] <= pd.to_datetime(dr[1]))]
    except:
        pass

    st.markdown("Filtered preview:")
    st.dataframe(filt.head(10), use_container_width=True)
    download(filt.head(500), "filtered_preview.csv")

    # -------------------------
    # KPIs
    # -------------------------
    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    total_rev = filt["Revenue"].sum()
    avg_roas  = filt["ROAS"].mean()
    total_leads = filt["Leads"].sum()
    total_conv = filt["Conversions"].sum()

    k1.markdown(f"<div class='kpi'>Revenue<br>{to_money(total_rev)}</div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'>ROAS<br>{avg_roas:.2f}</div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'>Leads<br>{int(total_leads)}</div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'>Conversions<br>{int(total_conv)}</div>", unsafe_allow_html=True)

    # -------------------------
    # Charts & EDA
    # -------------------------
    st.markdown('<div class="section-title">Charts & EDA</div>', unsafe_allow_html=True)

    ag = filt.groupby("Campaign")[["Revenue","Conversions"]].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ag["Campaign"], y=ag["Revenue"], name="Revenue"))
    fig.add_trace(go.Bar(x=ag["Campaign"], y=ag["Conversions"], name="Conversions"))
    fig.update_layout(barmode="group", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # ML prediction (Revenue)
    
    st.markdown('<div class="section-title">ML — Predict Revenue (RandomForest)</div>', unsafe_allow_html=True)
    
    ml_df = filt.copy().dropna(subset=["Revenue"]) if "Revenue" in filt.columns else pd.DataFrame()
    feat_cols = ["Channel","Campaign","Device","AgeGroup","Gender","AdSet",
                 "Impressions","Clicks","Spend"]
    feat_cols = [c for c in feat_cols if c in ml_df.columns]
    
    if ml_df.shape[0] < 30 or len(feat_cols) < 2:
        st.info("Not enough data to train ML model (>=30 rows and >1 feature required).")
    else:
        X = ml_df[feat_cols]
        y = ml_df["Revenue"]
    
        cat_cols = [c for c in X.columns if X[c].dtype == "object"]
        num_cols = [c for c in X.columns if c not in cat_cols]
    
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                ("num", StandardScaler(), num_cols)
            ],
            remainder="drop"
        )
    
        X_t = preprocessor.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_t, y, test_size=0.2, random_state=42
        )
    
        rf = RandomForestRegressor(n_estimators=220, random_state=42)
        with st.spinner("Training RandomForest…"):
            rf.fit(X_train, y_train)
    
        preds = rf.predict(X_test)
        rmse = math.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
    
        # Performance summary card
        st.markdown(
            f"""
            <div class='card'>
                <h4>Model Performance</h4>
                RMSE: {rmse:.2f}<br>
                R² Score: {r2:.3f}
            </div>
            """,
            unsafe_allow_html=True,
        )
    
        # Build output table
        out_df = pd.DataFrame({
            "Actual_Revenue": y_test.reset_index(drop=True),
            "Predicted_Revenue": preds
        })
    
        # after training and producing out_df (Actual_Revenue, Predicted_Revenue)
        st.markdown("<div class='section-title'>ML Predictions Preview</div>", unsafe_allow_html=True)
        preview_df = out_df.head(10)
        
        # render preview with required-table style (no index shown)
        styled_preview = preview_df.style.set_table_attributes('class="required-table"')
        html_preview = styled_preview.to_html()
        html_preview = html_preview.replace("<th></th>", "").replace("<td></td>", "")
        st.write(html_preview, unsafe_allow_html=True)
        
        # Download full predictions (calls download_df defined above)
        download_df(out_df, "ml_revenue_predictions.csv")



    
    # -------------------------
    # Forecasting
    # -------------------------
    st.markdown('<div class="section-title">Forecasting</div>', unsafe_allow_html=True)

    if "Date" in filt.columns and "Revenue" in filt.columns:
        daily = filt.groupby(pd.Grouper(key="Date", freq="D"))["Revenue"].sum().reset_index()
        if len(daily) >= 10:
            daily["t"] = np.arange(len(daily))
            lr = LinearRegression()
            lr.fit(daily[["t"]], daily["Revenue"])

            future_t = np.arange(len(daily), len(daily)+30)
            preds = lr.predict(future_t.reshape(-1,1))
            future_dates = pd.date_range(daily["Date"].max() + pd.Timedelta(days=1), periods=30)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=daily["Date"], y=daily["Revenue"], name="Actual"))
            fig2.add_trace(go.Scatter(x=future_dates, y=preds, name="Forecast"))
            fig2.update_layout(template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Not enough daily data (need 10+).")

    # -------------------------
    # Insights
    # -------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)

    insights = []

    if "Channel" in filt.columns:
        ch = filt.groupby("Channel")[["Revenue","Spend"]].sum().reset_index()
        ch["ROI"] = np.where(ch["Spend"]>0, ch["Revenue"]/ch["Spend"], 0)
        if not ch.empty:
            best = ch.loc[ch["ROI"].idxmax()]
            worst = ch.loc[ch["ROI"].idxmin()]
            insights.append(f"Best ROI channel: {best['Channel']} — {best['ROI']:.2f}")
            insights.append(f"Worst ROI channel: {worst['Channel']} — {worst['ROI']:.2f}")

    if insights:
        for i,v in enumerate(insights):
            st.markdown(f"<div class='card'><b>Insight {i+1}:</b> {v}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card'>No insights could be generated.</div>", unsafe_allow_html=True)

    # -------------------------
    # Export
    # -------------------------
    st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
    st.download_button("Download full filtered data", filt.to_csv(index=False), "marketing_filtered.csv")

# END FILE
