import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import math
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Marketing Intelligence & Forecasting Lab", layout="wide")

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
# Global CSS (Option A sizes, pure-black text except KPI & variable cards)
# -------------------------
st.markdown("""
<style>
* { font-family: 'Inter', sans-serif; }

/* GLOBAL TEXT */
body, [class*="css"] { color:#000 !important; font-size:17px; }

/* MAIN HEADER */
.big-header {
    font-size: 36px !important;
    font-weight: 700 !important;
    color:#000 !important;
    margin-bottom:12px;
}

/* SECTION TITLE */
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

/* CARD (pure black text) */
.card {
    background:#ffffff;
    padding:22px;
    border-radius:14px;
    border:1px solid #e6e6e6;
    font-size:16.5px;
    color:#000 !important;
    font-weight:500;
    box-shadow:0 3px 14px rgba(0,0,0,0.08);
    transition: all 0.25s ease;
}
.card:hover {
    transform:translateY(-4px);
    box-shadow:0 12px 25px rgba(6,75,134,0.18);
    border-color:#064b86;
}

/* KPI CARDS - blue text */
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

/* VARIABLE BOXES - blue text */
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

/* Table */
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
.stButton>button,
.stDownloadButton>button {
    background:#064b86 !important;
    color:white !important;
    border:none;
    padding:10px 22px;
    border-radius:8px !important;
    font-size:15.5px !important;
    font-weight:600 !important;
    transition:0.25s ease;
}
.stButton>button:hover,
.stDownloadButton>button:hover {
    transform:translateY(-3px);
    background:#0a6eb3 !important;
}

/* Page fade */
.block-container { animation: fadeIn 0.5s ease; }
@keyframes fadeIn { from {opacity:0; transform:translateY(10px);} to {opacity:1; transform:translateY(0);} }

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-header'>Marketing Intelligence & Forecasting Lab</div>", unsafe_allow_html=True)

# -------------------------
# Helpers
# -------------------------
REQUIRED_MARKETING_COLS = [
    "Campaign", "Channel", "Date", "Impressions", "Clicks", "Leads", "Conversions", "Spend"
]

AUTO_MAPS = {
    "Campaign": ["campaign name", "campaign_name", "campaign", "Campaign name", "Campaign Name"],
    "Date": ["date", "day", "start date", "end date", "reporting starts", "reporting ends"],
    "Impressions": ["impressions", "Impression", "Impressions"],
    "Channel": ["page name", "page", "channel", "source", "platform", "adset", "placement", "medium"],
    "Clicks": ["link clicks", "clicks", "all clicks", "total clicks"],
    "Leads": ["results", "leads", "lead", "cpl results"],
    "Conversions": ["conversions", "website conversions", "purchase", "complete registration"],
    "Spend": ["amount spent (inr)", "amount spent", "spend", "cost", "ad spend", "budget used"]
}

def auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    cols = [c.strip() for c in df.columns]
    for req, candidates in AUTO_MAPS.items():
        for c in cols:
            low = c.lower().strip()
            for cand in candidates:
                cand_low = cand.lower().strip()
                if cand_low == low or cand_low in low or low in cand_low:
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
        return "₹ 0.00"

def ensure_datetime(df, col="Date"):
    if col in df.columns:
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
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# -------------------------
# Overview Tab
# -------------------------
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <b>Purpose:</b><br><br>
    Centralize multi-channel campaign data, produce automated insights, and provide ML-driven predictions and forecasts to make smarter budget decisions.
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Full-funnel analytics across channels<br>
        • Automated insights and rules-based recommendations<br>
        • Predictive ML models for clicks & conversions<br>
        • Lightweight time-series forecasting (Prophet fallback to linear)<br>
        • Exportable predictions and actionable playbooks
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Reduce wasted ad spend and reallocate to high ROI channels<br>
        • Improve CPL and conversion efficiency<br>
        • Improve forecasting for planning & budgeting<br>
        • Identify underperforming campaigns and creative issues
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Total Impressions</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Total Clicks</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Total Leads</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Total Spend</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Who Should Use This</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    Marketing leaders, performance teams, growth managers, data analysts and decision-makers who need a single source of truth for campaign performance and forecasts.
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# Important Attributes Tab
# -------------------------
with tab2:
    # ---------------------------------------------------------
    # REQUIRED COLUMNS TABLE
    # ---------------------------------------------------------
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    required_dict = {
        "Campaign": "Name of the marketing campaign.",
        "Channel": "Traffic source (FB, Google, Email, etc.).",
        "Date": "Date of activity.",
        "Impressions": "How many times your ad was shown.",
        "Clicks": "Number of people who clicked.",
        "Leads": "Users who showed interest.",
        "Conversions": "Users completing final action.",
        "Spend": "Amount spent on marketing."
    }

    req_df = pd.DataFrame([{"Attribute": k, "Description": v} for k, v in required_dict.items()])

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.dataframe(req_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # INDEPENDENT VARIABLES
    # ---------------------------------------------------------
    st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)

    indep_vars = [
        "Campaign",
        "Channel",
        "Date",
        "Impressions",
        "Clicks",
        "Spend"
    ]

    c1, c2, c3 = st.columns(3)
    for i, v in enumerate(indep_vars):
        (c1 if i % 3 == 0 else c2 if i % 3 == 1 else c3).markdown(
            f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True
        )

    # ---------------------------------------------------------
    # DEPENDENT VARIABLES
    # ---------------------------------------------------------
    st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)

    dep_vars = [
        "Leads",
        "Conversions"
    ]

    c1, c2 = st.columns(2)
    for i, v in enumerate(dep_vars):
        (c1 if i % 2 == 0 else c2).markdown(
            f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True
        )

# -------------------------
# Application Tab
# -------------------------
with tab3:
    st.markdown('<div class="section-title">Step 1: Load dataset</div>', unsafe_allow_html=True)

    df = None
    mode = st.radio("Select Dataset Option:", ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)

    if mode == "Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/marketing.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("Default dataset loaded")
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    elif mode == "Upload CSV":
        st.markdown("#### Download sample CSV (for format reference)")
        SAMPLE_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/marketing.csv"
        try:
            sample_df = pd.read_csv(SAMPLE_URL).head(5)
            st.download_button("Download Sample CSV", sample_df.to_csv(index=False), "sample_dataset.csv", "text/csv")
        except Exception:
            pass

        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("File uploaded")
            st.dataframe(df.head())

    else:  # Upload + mapping
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head())
            st.markdown("Map your columns to the required fields (only required columns shown):")
            mapping = {}
            for req in REQUIRED_MARKETING_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + list(raw.columns))
            if st.button("Apply mapping"):
                missing = [k for k,v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    inv = {v:k for k,v in mapping.items()}
                    df = raw.rename(columns=inv)
                    st.success("Mapping applied")
                    st.dataframe(df.head())

    if df is None:
        st.stop()

    # -------------------------
    # Validate required columns
    # -------------------------
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED_MARKETING_COLS if c not in df.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.info("Use 'Upload CSV + Column mapping' to map columns.")
        st.stop()

    # -------------------------
    # Coerce types, derived metrics
    # -------------------------
    df = df.copy()
    df = ensure_datetime(df, "Date")
    for col in ["Impressions","Clicks","Leads","Conversions","Spend"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["CTR"] = np.where(df["Impressions"]>0, df["Clicks"] / df["Impressions"], 0)
    df["CPC"] = np.where(df["Clicks"]>0, df["Spend"] / df["Clicks"], 0)
    df["Cost_per_Lead"] = np.where(df["Leads"]>0, df["Spend"] / df["Leads"], np.nan)
    df["Conversion_Rate"] = np.where(df["Clicks"]>0, df["Conversions"] / df["Clicks"], 0)

    # -------------------------
    # Filters & Preview
    # -------------------------
    st.markdown('<div class="section-title">Filters & Preview</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2,2,1])
    campaigns = sorted(df["Campaign"].dropna().unique().tolist())
    channels = sorted(df["Channel"].dropna().unique().tolist())
    with c1:
        sel_campaigns = st.multiselect("Campaign", options=campaigns, default=campaigns[:5])
    with c2:
        sel_channels = st.multiselect("Channel", options=channels, default=channels[:3])
    with c3:
        try:
            min_d = df["Date"].min().date()
            max_d = df["Date"].max().date()
            date_range = st.date_input("Date range", value=(min_d, max_d))
        except Exception:
            date_range = st.date_input("Date range")

    filt = df.copy()
    if sel_campaigns:
        filt = filt[filt["Campaign"].isin(sel_campaigns)]
    if sel_channels:
        filt = filt[filt["Channel"].isin(sel_channels)]
    try:
        if date_range and len(date_range) == 2:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            filt = filt[(filt["Date"] >= start) & (filt["Date"] <= end)]
    except Exception:
        pass

    st.markdown('<div class="section-title">Preview (first 10 rows)</div>', unsafe_allow_html=True)
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(500), "filtered_preview.csv")

    # -------------------------
    # KPIs (blue KPI cards)
    # -------------------------
    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    def safe_sum(s):
        try:
            return int(pd.to_numeric(s, errors="coerce").sum())
        except:
            return 0

    k1.markdown(f"<div class='kpi'>Total Impressions<br><span style='font-size:12px;color:#222;font-weight:500;'>(dataset)</span></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'>Total Clicks<br><span style='font-size:12px;color:#222;font-weight:500;'>(dataset)</span></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'>Total Leads<br><span style='font-size:12px;color:#222;font-weight:500;'>(dataset)</span></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'>Total Spend<br><span style='font-size:12px;color:#222;font-weight:500;'>(dataset)</span></div>", unsafe_allow_html=True)

    # also show numeric metrics real below (safe)
    k1_val, k2_val, k3_val, k4_val = st.columns(4)
    k1_val.metric("Impressions", f"{safe_sum(filt['Impressions']):,}")
    k2_val.metric("Clicks", f"{safe_sum(filt['Clicks']):,}")
    k3_val.metric("Leads", f"{safe_sum(filt['Leads']):,}")
    k4_val.metric("Spend", to_currency(filt["Spend"].sum()))

    # -------------------------
    # Charts & EDA
    # -------------------------
    st.markdown('<div class="section-title">Charts & EDA</div>', unsafe_allow_html=True)

    # Campaign aggregated bar (Clicks & Conversions)
    agg = filt.groupby("Campaign").agg({"Impressions":"sum","Clicks":"sum","Leads":"sum","Conversions":"sum","Spend":"sum"}).reset_index().sort_values("Clicks", ascending=False)
    if not agg.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Clicks"], name="Clicks", text=agg["Clicks"], textposition="outside"))
        fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Conversions"], name="Conversions", text=agg["Conversions"], textposition="outside"))
        fig.update_layout(barmode='group', xaxis_title="<b>Campaign</b>", yaxis_title="<b>Count</b>", template="plotly_white", legend_title="Metric")
        fig.update_xaxes(tickangle=-45, showline=True, linewidth=1, linecolor="black")
        fig.update_yaxes(showline=True, linewidth=1, linecolor="black")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No campaign aggregated data to plot.")

    # Channel ROI (Leads per ₹)
    roi_df = filt.groupby("Channel").agg({"Leads":"sum","Spend":"sum"}).reset_index()
    if not roi_df.empty:
        roi_df["Leads_per_Rs"] = np.where(roi_df["Spend"]>0, roi_df["Leads"]/roi_df["Spend"], 0)
        fig2 = px.bar(roi_df.sort_values("Leads_per_Rs", ascending=False), x="Channel", y="Leads_per_Rs", text=roi_df["Leads_per_Rs"].round(4), template="plotly_white")
        fig2.update_traces(textposition="outside")
        fig2.update_layout(yaxis_title="Leads per ₹", xaxis_title="<b>Channel</b>")
        st.plotly_chart(fig2, use_container_width=True)

    # Time-series Clicks with MA
    ts = filt.dropna(subset=["Date"]).groupby("Date")["Clicks"].sum().reset_index().sort_values("Date")
    if not ts.empty:
        ts["MA_7"] = ts["Clicks"].rolling(7, min_periods=1).mean()
        fig3 = px.line(ts, x="Date", y=["Clicks","MA_7"], labels={"value":"Clicks","variable":"Series"}, template="plotly_white")
        fig3.update_layout(xaxis_title="<b>Date</b>", yaxis_title="<b>Clicks</b>")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Not enough date data to plot time-series.")

    # -------------------------
    # ML: Clicks regression (RandomForest) + quick predict widget
    # -------------------------
    st.markdown('<div class="section-title">ML — Clicks Regression (RandomForest)</div>', unsafe_allow_html=True)
    ml_exp = st.expander("Train & try clicks regression (requires >=40 rows)", expanded=False)
    with ml_exp:
        ml_df = filt.copy().dropna(subset=["Clicks","Impressions","Spend"])
        if "Date" in ml_df.columns:
            ml_df["dayofweek"] = ml_df["Date"].dt.dayofweek
            ml_df["month"] = ml_df["Date"].dt.month
        feat_cols = ["Channel","Campaign","Impressions","Spend","dayofweek","month"]
        feat_cols = [c for c in feat_cols if c in ml_df.columns]
        if len(ml_df) < 40 or len(feat_cols) < 2:
            st.info("Not enough data or features to train a robust model (need >=40 rows).")
        else:
            X = ml_df[feat_cols].copy()
            y = ml_df["Clicks"].astype(float)
            cat_cols = [c for c in X.columns if X[c].dtype == "object"]
            num_cols = [c for c in X.columns if c not in cat_cols]
            preprocessor = ColumnTransformer(transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols) if cat_cols else ("noop","passthrough", []),
                ("num", StandardScaler(), num_cols) if num_cols else ("noop2","passthrough", [])
            ], remainder="drop")
            try:
                X_t = preprocessor.fit_transform(X)
            except Exception as e:
                st.error("Preprocessing failed: " + str(e))
                X_t = None

            if X_t is not None:
                X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)
                rf = RandomForestRegressor(n_estimators=150, random_state=42)
                with st.spinner("Training RandomForest..."):
                    rf.fit(X_train, y_train)
                preds = rf.predict(X_test)
                rmse = math.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)
                st.write(f"Clicks regression — RMSE: {rmse:.2f}, R²: {r2:.3f}")
                # feature importances
                try:
                    ohe = preprocessor.named_transformers_.get("cat")
                    cat_names = list(ohe.get_feature_names_out(cat_cols)) if (cat_cols and hasattr(ohe, 'get_feature_names_out')) else []
                except Exception:
                    cat_names = []
                feature_names = cat_names + num_cols
                try:
                    fi = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False).head(30)
                    st.markdown("Top feature importances")
                    st.dataframe(fi.reset_index().rename(columns={"index":"feature",0:"importance"}).head(20))
                except Exception:
                    st.info("Feature importance unavailable (unexpected input shapes).")

                # Quick single-row prediction UI
                st.markdown("Quick predict (single row):")
                sample_campaign = ml_df["Campaign"].iloc[0] if "Campaign" in ml_df.columns else ""
                sample_channel = ml_df["Channel"].iloc[0] if "Channel" in ml_df.columns else ""
                c1,c2,c3 = st.columns(3)
                with c1:
                    sel_campaign = st.selectbox("Campaign", options=ml_df["Campaign"].unique(), index=0)
                with c2:
                    sel_channel = st.selectbox("Channel", options=ml_df["Channel"].unique(), index=0)
                with c3:
                    inp_impr = st.number_input("Impressions", min_value=0, value=int(ml_df["Impressions"].median()))
                c4,c5 = st.columns(2)
                with c4:
                    inp_spend = st.number_input("Spend (INR)", min_value=0.0, value=float(ml_df["Spend"].median()))
                with c5:
                    sel_day = st.selectbox("Day of week (0=Mon)", options=list(range(7)), index=int(ml_df["dayofweek"].median() if "dayofweek" in ml_df else 0))
                if st.button("Predict Clicks"):
                    row = pd.DataFrame([{"Campaign": sel_campaign, "Channel": sel_channel, "Impressions": inp_impr, "Spend": inp_spend, "dayofweek": sel_day, "month": int(pd.to_datetime(ml_df["Date"].iloc[-1]).month) if "Date" in ml_df.columns else 1}])
                    try:
                        row_t = preprocessor.transform(row)
                        pred_clicks = rf.predict(row_t)[0]
                        st.success(f"Predicted Clicks: {int(round(pred_clicks))}")
                    except Exception as e:
                        st.error("Prediction failed: " + str(e))

    # -------------------------
    # Forecasting: tries Prophet, else linear fallback
    # -------------------------
    st.markdown('<div class="section-title">Forecasting</div>', unsafe_allow_html=True)
    with st.expander("Forecast settings", expanded=False):
        st.write("Choose metric and granularity. Prophet is used if available; otherwise a linear fallback is used.")
    metric = st.selectbox("Forecast metric", ["Clicks","Spend","Conversions"], index=0)
    gran = st.selectbox("Granularity", ["Overall","By Campaign","By Channel"], index=0)
    months = st.number_input("Months to forecast", min_value=1, max_value=24, value=3)

    # pick group if needed
    group_field = None
    pick = None
    if gran == "By Campaign":
        group_field = "Campaign"
        pick = st.selectbox("Choose campaign", options=sorted(filt["Campaign"].unique()))
    elif gran == "By Channel":
        group_field = "Channel"
        pick = st.selectbox("Choose channel", options=sorted(filt["Channel"].unique()))

    ts_df = filt.dropna(subset=["Date"])
    if group_field and pick:
        ts_df = ts_df[ts_df[group_field] == pick]
    ts_agg = ts_df.groupby(pd.Grouper(key="Date", freq="D")).agg({metric:"sum"}).reset_index().sort_values("Date")

    if ts_agg[metric].sum() == 0 or len(ts_agg) < 6:
        st.warning("Not enough history for reliable forecast (need ~6 daily points with non-zero values).")
    else:
        try:
            from prophet import Prophet
            prophet_ok = True
        except Exception:
            prophet_ok = False

        if prophet_ok:
            model_df = ts_agg.rename(columns={"Date":"ds", metric:"y"})[["ds","y"]]
            model_df = model_df.set_index("ds").asfreq("D").fillna(0).reset_index()
            m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            with st.spinner("Training Prophet model..."):
                m.fit(model_df)
            future = m.make_future_dataframe(periods=months*30)
            forecast = m.predict(future)
            figf = px.line(forecast, x="ds", y=["yhat","yhat_lower","yhat_upper"], labels={"ds":"Date"}, template="plotly_white", title=f"Forecast for {metric}")
            st.plotly_chart(figf, use_container_width=True)
            st.markdown("Forecast sample (last rows)")
            st.dataframe(forecast[["ds","yhat"]].tail(10).rename(columns={"ds":"Date","yhat":f"Pred_{metric}"}))
        else:
            st.info("Prophet not available. Using linear fallback.")
            ts_lr = ts_agg.set_index("Date").resample("D").sum().fillna(0).reset_index()
            ts_lr["t"] = np.arange(len(ts_lr))
            X = ts_lr[["t"]]
            y = ts_lr[metric]
            lr = LinearRegression()
            lr.fit(X, y)
            future_t = np.arange(len(ts_lr), len(ts_lr)+months*30).reshape(-1,1)
            preds = lr.predict(future_t)
            hist = go.Scatter(x=ts_lr["Date"], y=ts_lr[metric], name="Actual")
            fut_dates = pd.date_range(ts_lr["Date"].max() + pd.Timedelta(days=1), periods=months*30, freq="D")
            fut = go.Scatter(x=fut_dates, y=preds, name="Forecast")
            fig = go.Figure([hist, fut])
            fig.update_layout(title=f"Forecast (linear fallback) for {metric}", xaxis_title="Date", yaxis_title=metric, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pd.DataFrame({"Date":fut_dates, f"Pred_{metric}":preds}).head(10))

    # -------------------------
    # Automated insights
    # -------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)
    with st.expander("Generate automated insights", expanded=True):
        insights = []
        # Channel performance by leads per spend
        channel_perf = filt.groupby("Channel").agg({"Leads":"sum","Spend":"sum"}).reset_index()
        channel_perf["Leads_per_Rs"] = np.where(channel_perf["Spend"]>0, channel_perf["Leads"]/channel_perf["Spend"], 0)
        if not channel_perf.empty:
            best = channel_perf.sort_values("Leads_per_Rs", ascending=False).iloc[0]
            worst = channel_perf.sort_values("Leads_per_Rs", ascending=True).iloc[0]
            insights.append(f"Best channel (leads/₹): {best['Channel']} (~{best['Leads_per_Rs']:.4f} leads/₹). Consider scaling.")
            insights.append(f"Worst channel (leads/₹): {worst['Channel']} (~{worst['Leads_per_Rs']:.4f} leads/₹). Review or pause.")
        # Campaign CPL check
        camp = filt.groupby("Campaign").agg({"Leads":"sum","Spend":"sum"}).reset_index()
        camp["CPL"] = np.where(camp["Leads"]>0, camp["Spend"]/camp["Leads"], np.nan)
        if not camp.empty:
            best_cpl = camp.sort_values("CPL").head(1)
            worst_cpl = camp.sort_values("CPL", ascending=False).head(1)
            if not best_cpl.empty:
                insights.append(f"Lowest CPL campaign: {best_cpl.iloc[0]['Campaign']} (CPL ≈ {to_currency(best_cpl.iloc[0]['CPL'])}).")
            if not worst_cpl.empty and not pd.isna(worst_cpl.iloc[0]["CPL"]):
                insights.append(f"High CPL campaign: {worst_cpl.iloc[0]['Campaign']} (CPL ≈ {to_currency(worst_cpl.iloc[0]['CPL'])}). Investigate landing page/targeting.")

        # Trend based insight
        if "Date" in filt.columns and not filt["Date"].isna().all():
            monthly_leads = filt.groupby(filt["Date"].dt.month).agg({"Leads":"sum"}).reset_index()
            if not monthly_leads.empty:
                top_month = int(monthly_leads.sort_values("Leads", ascending=False).iloc[0]["Date"])
                insights.append(f"Peak lead month historically: {top_month}. Consider preparing campaigns early for this month.")

        # Conversions momentum
        conv_trend = filt.groupby("Date")["Conversions"].sum().reset_index().sort_values("Date")
        if len(conv_trend) > 3:
            last3 = conv_trend["Conversions"].tail(3).values
            if last3[-1] > last3[0]:
                insights.append("Conversions trending up recently — test scale.")
            else:
                insights.append("Conversions dipping recently — check creatives, landing pages, and targeting.")

        if not insights:
            st.info("No insights generated for the selected filters.")
        else:
            for i,ins in enumerate(insights, 1):
                st.markdown(f"**Insight {i}:** {ins}")

    # -------------------------
    # Export & Predictions
    # -------------------------
    st.markdown('<div class="section-title">Predict & Export</div>', unsafe_allow_html=True)
    if st.button("Predict conversions (naive) for filtered dataset"):
        if "Conversion_Rate" in filt.columns:
            median_conv = filt["Conversion_Rate"].median()
            preds = (filt["Clicks"] * median_conv).round().astype(int)
            out = filt.copy()
            out["Predicted_Conversions"] = preds
            st.dataframe(out.head(10))
            download_df(out[["Campaign","Channel","Date","Clicks","Predicted_Conversions"]], "predicted_conversions.csv")
        else:
            st.info("Conversion_Rate not available. Cannot run naive prediction.")

    st.markdown("### Done — export filtered dataset if you want")
    st.download_button("Download filtered dataset", filt.to_csv(index=False), "marketing_filtered.csv", "text/csv")

# End of file
