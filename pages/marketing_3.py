# app_marketing_ml.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import math
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

st.set_page_config(page_title="Marketing ML — Single Page", layout="wide")

# ---------------------------
# CSS (cards, metric glow, tooltip)
# ---------------------------
st.markdown(
    """
<style>
/* Card style */
.card {
    background:#fff;border-radius:12px;padding:18px;margin-bottom:14px;
    box-shadow:0 6px 24px rgba(16,24,40,0.08);
    border: 1px solid rgba(16,24,40,0.04);
}
.metric-card {
    background:#f6fbff;padding:14px;border-radius:10px;text-align:center;
    box-shadow:0 6px 18px rgba(16,24,40,0.06);
    transition: all .18s ease;
    font-weight:600;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow:0 12px 30px rgba(16,24,40,0.10);
    cursor: pointer;
}
/* tooltip */
.metric-card[title] { position: relative; }
.metric-card[title]:hover:after{
    content: attr(title);
    position: absolute;
    bottom: -36px;
    left: 50%;
    transform: translateX(-50%);
    background: #111;color:#fff;padding:6px 10px;border-radius:6px;font-size:12px;
    white-space:nowrap;box-shadow:0 6px 18px rgba(0,0,0,0.25);
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Helpers
# ---------------------------
REQUIRED = ["Campaign", "Channel", "Date", "Impressions", "Clicks", "Leads", "Conversions", "Spend"]

AUTO_MAPS = {
    "Campaign": ["campaign name", "campaign_name", "campaign", "campaign_name"],
    "Channel": ["page name", "page", "channel", "source", "platform", "medium"],
    "Date": ["date", "day", "reporting starts", "reporting ends", "start date", "end date"],
    "Impressions": ["impressions", "impression"],
    "Clicks": ["link clicks", "clicks", "clicks (all)"],
    "Leads": ["results", "leads", "lead"],
    "Conversions": ["conversions", "website conversions", "purchases", "conversion"],
    "Spend": ["amount spent (inr)", "amount spent", "spend", "cost", "amount_spent"]
}


def auto_map_columns(df):
    rename = {}
    cols = [c.strip() for c in df.columns]
    for req, candidates in AUTO_MAPS.items():
        for c in cols:
            low = c.lower().strip()
            matched = False
            for cand in candidates:
                cand_low = cand.lower()
                if cand_low == low or cand_low in low or low in cand_low:
                    rename[c] = req
                    matched = True
                    break
            if matched:
                break
    if rename:
        return df.rename(columns=rename)
    return df


def to_currency(x):
    try:
        return "₹ " + f"{float(x):,.2f}"
    except:
        return x


def ensure_date(df, col="Date"):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def download_df(df, filename):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button("Download CSV", b, file_name=filename, mime="text/csv")


# ---------------------------
# Header / Tabs like RealEstate apps
# ---------------------------
st.markdown("<h1 style='font-size:34px;margin-bottom:6px'>Marketing Intelligence — ML Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<div class='card'>Enterprise marketing analytics: EDA, ML predictions and simple forecasting for campaigns & channels.</div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Overview", "Application"])

with tab1:
    st.markdown("### Overview")
    st.markdown("<div class='card'>A single-page marketing ML workspace to load campaign data, run EDA and train classical ML models (regression/classification) for prediction and action.</div>", unsafe_allow_html=True)

    st.markdown("### Purpose")
    st.markdown("<div class='card'>• Evaluate campaign/channel performance<br>• Predict clicks/leads/conversions using classical ML<br>• Generate simple time-series forecasts<br>• Provide actionable KPIs for budget allocation</div>", unsafe_allow_html=True)

    st.markdown("### KPIs")
    K1, K2, K3, K4 = st.columns(4)
    K1.markdown("<div class='metric-card' title='Total clicks across selected data'>Total Clicks</div>", unsafe_allow_html=True)
    K2.markdown("<div class='metric-card' title='Total leads captured'>Total Leads</div>", unsafe_allow_html=True)
    K3.markdown("<div class='metric-card' title='Conversion Rate (Conversions / Clicks)'>Conversion Rate</div>", unsafe_allow_html=True)
    K4.markdown("<div class='metric-card' title='Return on Ad Spend (Revenue / Spend) — if Revenue exists'>ROAS</div>", unsafe_allow_html=True)

with tab2:
    st.header("Application")

    # ---------------------------
    # Data source selector (3 modes)
    # ---------------------------
    st.markdown("### Step 1 — Load dataset (choose one)")
    mode = st.radio("Dataset option:", ["Default Dataset", "Upload CSV", "Upload CSV + Column Mapping"], horizontal=True)

    df = None
    DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/marketing.csv"

    if mode == "Default Dataset":
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("Default dataset loaded (auto-mapping attempted).")
            st.dataframe(df.head(5))
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))

    elif mode == "Upload CSV":
        st.markdown("#### (Optional) Download small sample from repo for reference")
        try:
            sample = pd.read_csv(DEFAULT_URL).head(5).to_csv(index=False)
            st.download_button("Download sample CSV (5 rows)", sample, "sample_marketing_5rows.csv", "text/csv")
        except:
            pass

        uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("Uploaded and auto-mapped (if possible).")
            st.dataframe(df.head(5))
            st.download_button("Download uploaded sample (5 rows)", df.head(5).to_csv(index=False), "uploaded_sample_5rows.csv", "text/csv")

    else:  # Upload + Column Mapping
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.write("Preview (first 5 rows)")
            st.dataframe(raw.head(5))
            st.markdown("Map your columns to required fields:")
            mapping = {}
            for req in REQUIRED:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + list(raw.columns))
            if st.button("Apply Mapping"):
                missing_map = [k for k, v in mapping.items() if v == "-- Select --"]
                if missing_map:
                    st.error("Map all required columns: " + ", ".join(missing_map))
                else:
                    df = raw.rename(columns={v: k for k, v in mapping.items()})
                    st.success("Mapping applied.")
                    st.dataframe(df.head(5))
                    st.download_button("Download mapped sample (5 rows)", df.head(5).to_csv(index=False), "mapped_sample_5rows.csv", "text/csv")

    # stop if no df
    if df is None:
        st.info("Upload or select a dataset to proceed.")
        st.stop()

    # ---------------------------
    # Validate & prepare
    # ---------------------------
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        st.error("The following required columns are missing: " + ", ".join(missing))
        st.info("Use 'Upload CSV + Column Mapping' to map your columns or adjust your CSV.")
        st.stop()

    # ensure date and numeric types
    df = ensure_date(df, "Date")
    for col in ["Impressions", "Clicks", "Leads", "Conversions", "Spend"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Derived metrics
    df["CTR"] = np.where(df["Impressions"] > 0, df["Clicks"] / df["Impressions"], 0)
    df["CPC"] = np.where(df["Clicks"] > 0, df["Spend"] / df["Clicks"], 0)
    df["CPL"] = np.where(df["Leads"] > 0, df["Spend"] / df["Leads"], np.nan)
    df["Conversion_Rate"] = np.where(df["Clicks"] > 0, df["Conversions"] / df["Clicks"], 0)

    # ---------------------------
    # Filters
    # ---------------------------
    st.markdown("### Step 2 — Filters & preview")
    c1, c2, c3 = st.columns([2, 2, 1])
    campaigns = sorted(df["Campaign"].dropna().unique().tolist())
    channels = sorted(df["Channel"].dropna().unique().tolist())

    with c1:
        sel_campaigns = st.multiselect("Campaign", options=campaigns, default=campaigns[:6])
    with c2:
        sel_channels = st.multiselect("Channel", options=channels, default=channels[:3])
    with c3:
        # safe default dates
        dmin = df["Date"].min() if not df["Date"].isna().all() else None
        dmax = df["Date"].max() if not df["Date"].isna().all() else None
        date_range = st.date_input("Date range", value=(dmin.date() if dmin is not None else None, dmax.date() if dmax is not None else None))

    filt = df.copy()
    if sel_campaigns:
        filt = filt[filt["Campaign"].isin(sel_campaigns)]
    if sel_channels:
        filt = filt[filt["Channel"].isin(sel_channels)]
    if date_range and date_range[0] is not None:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filt = filt[(filt["Date"] >= start) & (filt["Date"] <= end)]

    st.markdown("#### Data preview (filtered)")
    st.dataframe(filt.head(5), use_container_width=True)
    download_df(filt.head(200), "filtered_preview.csv")

    # ---------------------------
    # KPIs
    # ---------------------------
    st.markdown("### Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Impressions", f"{int(filt['Impressions'].sum()):,}")
    k2.metric("Total Clicks", f"{int(filt['Clicks'].sum()):,}")
    k3.metric("Total Leads", f"{int(filt['Leads'].sum()):,}")
    k4.metric("Total Spend", to_currency(filt['Spend'].sum()))

    # ---------------------------
    # Charts
    # ---------------------------
    st.markdown("### Campaign-wise Clicks & Conversions")
    agg = filt.groupby("Campaign").agg({"Impressions": "sum", "Clicks": "sum", "Conversions": "sum", "Spend": "sum"}).reset_index().sort_values("Clicks", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Clicks"], name="Clicks", text=agg["Clicks"], textposition="outside"))
    fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Conversions"], name="Conversions", text=agg["Conversions"], textposition="outside"))
    fig.update_layout(barmode="group", xaxis_tickangle=-45, template="plotly_white", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Channel ROI (Leads per ₹)")
    roi_df = filt.groupby("Channel").agg({"Leads": "sum", "Spend": "sum"}).reset_index()
    roi_df["Leads_per_Rs"] = np.where(roi_df["Spend"] > 0, roi_df["Leads"] / roi_df["Spend"], 0)
    fig_roi = px.bar(roi_df.sort_values("Leads_per_Rs", ascending=False), x="Channel", y="Leads_per_Rs", text=roi_df["Leads_per_Rs"].round(4))
    fig_roi.update_traces(textposition="outside")
    st.plotly_chart(fig_roi, use_container_width=True)

    st.markdown("### Time-series: Clicks (daily, with 7-day MA)")
    ts = filt.dropna(subset=["Date"]).groupby(pd.Grouper(key="Date", freq="D")).agg({"Clicks": "sum"}).reset_index().sort_values("Date")
    if not ts.empty:
        ts["MA_7"] = ts["Clicks"].rolling(7, min_periods=1).mean()
        fig_ts = px.line(ts, x="Date", y=["Clicks", "MA_7"], labels={"value": "Clicks", "variable": "Series"})
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("Not enough date data to plot time-series.")

    # ---------------------------
    # ML section (Clicks regression as example)
    # ---------------------------
    st.markdown("### ML: Predictive models (Clicks regression example)")
    mlcol1, mlcol2 = st.columns([2, 1])
    with mlcol1:
        st.markdown("Feature engineering used: Channel, Campaign, Impressions, Spend, dayofweek, month")
    with mlcol2:
        st.markdown("Requirements: >= 40 rows recommended")

    ml_df = filt.copy().dropna(subset=["Clicks", "Impressions", "Spend"])
    if "Date" in ml_df.columns:
        ml_df["dayofweek"] = ml_df["Date"].dt.dayofweek
        ml_df["month"] = ml_df["Date"].dt.month

    feat_cols = [c for c in ["Channel", "Campaign", "Impressions", "Spend", "dayofweek", "month"] if c in ml_df.columns]
    if len(ml_df) < 40 or len(feat_cols) < 2:
        st.info("Not enough data or features to train a robust model. Provide more rows or ensure key columns exist.")
    else:
        X = ml_df[feat_cols].copy()
        y = ml_df["Clicks"].astype(float)

        # identify categorical & numerical
        cat_cols = [c for c in X.columns if X[c].dtype == "object" or X[c].dtype.name == "category"]
        num_cols = [c for c in X.columns if c not in cat_cols]

        # OneHotEncoder compatibility
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        transformers = []
        if cat_cols:
            transformers.append(("cat", ohe, cat_cols))
        if num_cols:
            transformers.append(("num", StandardScaler(), num_cols))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

        model = RandomForestRegressor(n_estimators=150, random_state=42)
        pipeline = Pipeline([("prep", preprocessor), ("model", model)])

        test_size = st.slider("Test size (train/test split)", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        if st.button("Train Clicks Model"):
            X_t = X.copy()
            X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=test_size, random_state=42)
            with st.spinner("Training RandomForest..."):
                pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            rmse = math.sqrt(((y_test - preds) ** 2).mean())
            r2 = pipeline.score(X_test, y_test)
            st.success(f"Trained — RMSE: {rmse:.2f}   R²: {r2:.3f}")

            # feature importance (attempt to map feature names)
            try:
                # get categorical feature names if present
                feature_names = []
                if cat_cols:
                    cat_names = pipeline.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
                    feature_names += cat_names
                if num_cols:
                    feature_names += num_cols
                importances = pipeline.named_steps["model"].feature_importances_
                fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(40)
                st.markdown("Top feature importances")
                st.dataframe(fi.reset_index().rename(columns={"index": "feature", 0: "importance"}))
            except Exception:
                st.info("Feature importance not available / failed to map feature names.")

           # ------------------------------
           # ----------------------------------------------------
            # QUICK PREDICT — ALWAYS SHOW IF PIPELINE EXISTS
            # ----------------------------------------------------
            
            if "pipeline" in st.session_state:
                st.subheader("Quick Predict (Single Row)")
            
                feat_cols = st.session_state.X_columns
                num_cols = st.session_state.num_cols
                cat_cols = st.session_state.cat_cols
            
                cols = st.columns(3)
                values = {}
            
                for i, col in enumerate(feat_cols):
                    with cols[i % 3]:
                        if col in num_cols:
                            median_val = float(st.session_state.train_df[col].median())
                            values[col] = st.number_input(col, value=median_val)
                        else:
                            opts = sorted(st.session_state.train_df[col].dropna().unique().tolist())
                            opts = [str(x) for x in opts] if opts else ["Unknown"]
                            values[col] = st.selectbox(col, opts)
            
                if st.button("Predict (Single Row)"):
                    try:
                        row = pd.DataFrame([values])
            
                        # type fix
                        for c in num_cols:
                            row[c] = pd.to_numeric(row[c], errors="coerce").fillna(0)
                        for c in cat_cols:
                            row[c] = row[c].astype(str)
            
                        # keep exact column order
                        row = row[feat_cols]
            
                        pred = st.session_state.pipeline.predict(row)[0]
                        st.success(f"Prediction: {round(float(pred), 2)}")
            
                    except Exception as e:
                        st.error("Prediction failed: " + str(e))
            
            else:
                st.info("Train a model first to enable quick prediction.")


    st.markdown("---")
