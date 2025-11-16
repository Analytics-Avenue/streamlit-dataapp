import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import math
import warnings
warnings.filterwarnings("ignore")

from openai import OpenAI

# Load API Key from Streamlit Secrets
client = OpenAI(api_key="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")

st.title("Gen-AI Marketing Assistant")

st.markdown("Ask anything about your marketing dataset. The AI will analyze and respond.")

user_input = st.text_area("Write your question", "")

if st.button("Generate Insights"):
    if not user_input.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing…"):
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a data analytics expert specialized in marketing analytics."},
                    {"role": "user", "content": user_input}
                ]
            )

            ai_output = response.choices[0].message["content"]
            st.success("Response:")
            st.write(ai_output)


st.set_page_config(page_title="Marketing Intelligence & Forecasting Lab", layout="wide")

# -------------------------
# Helper utils & mapping
# -------------------------
REQUIRED_MARKETING_COLS = [
    "Campaign", "Channel", "Date", "Impressions", "Clicks", "Leads", "Conversions", "Spend"
]

# Candidate auto-mapping keywords to friendly required names
AUTO_MAPS = {
    "Campaign": ["campaign name", "campaign_name", "campaign", "Campaign name", "Campaign Name"],
    "Date": ["date", "day", "reporting starts", "reporting ends", "day"],
    "Impressions": ["impressions", "Impression", "Impressions"],
    "Channel": [
        "page name", "page", "channel", "source", "platform",
        "adset", "adset name", "placement", "medium"
    ],
    "Date": [
        "date", "day", "reporting starts", "reporting ends",
        "start date", "end date"
    ],
    "Clicks": [
        "link clicks", "clicks", "all clicks", "total clicks"
    ],
    "Leads": [
        "results", "leads", "lead", "cpl results"
    ],
    "Conversions": [
        "conversions", "website conversions", "purchase",
        "add to cart", "complete registration"
    ],
    "Spend": [
        "amount spent (inr)", "amount spent", "spend",
        "cost", "ad spend", "budget used"
    ]
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
# Page header + overview
# -------------------------
st.markdown("<h1 style='margin-bottom:0.2rem'>Marketing Intelligence & Forecasting Lab</h1>", unsafe_allow_html=True)
st.markdown("Enterprise-capable marketing analytics: ML predictions, time-series forecasts and automated insights from campaign data. No fluff — only actionable outputs.")

st.markdown("""
<style>

 /* Stronger selector: forces Streamlit to obey */
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

/* Glow + scale hover */
div[data-testid="stMarkdownContainer"] .metric-card:hover {
    background: rgba(255,255,255,0.20);
    border: 1px solid rgba(255,255,255,0.55);
    box-shadow: 0 0 18px rgba(255,255,255,0.4);
    transform: scale(1.04);
    cursor: pointer;
}

/* Tooltip (unchanged) */
.metric-card[title] {
    position: relative;
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

tabs = st.tabs(["Overview", "Application"])

with tabs[0]:
    st.markdown("### Overview")

    st.markdown("""
    <div class='card'>
        This app analyzes multi-channel marketing performance,
        predicts outcomes using ML, and gives forecasting insights for
        smarter budget allocation.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Purpose")
    st.markdown("""
    <div class='card'>
        • Track campaign performance across channels<br>
        • Identify high-ROI campaigns and adsets<br>
        • Forecast future Clicks, Leads & Conversions<br>
        • Optimize budget allocation using ML-driven insights<br>
        • Understand channel trends and seasonality<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### KPIs")

    k1, k2, k3, k4 = st.columns(4)

    k1.markdown("""
        <div class='metric-card' title='Total clicks generated across all campaigns'>
            Total Clicks
        </div>
    """, unsafe_allow_html=True)

    k2.markdown("""
        <div class='metric-card' title='Total leads captured from marketing campaigns'>
            Total Leads
        </div>
    """, unsafe_allow_html=True)

    k3.markdown("""
        <div class='metric-card' title='Overall conversion efficiency across campaigns'>
            Conversion Rate
        </div>
    """, unsafe_allow_html=True)

    k4.markdown("""
        <div class='metric-card' title='Return on Ad Spend (Revenue generated per ₹1 spent)'>
            ROAS
        </div>
    """, unsafe_allow_html=True)

   


with tabs[1]:
    st.header("Application")

    # -------------------------
    # Dataset input: default, upload, mapping
    # -------------------------
    st.markdown("### Step 1 — Load dataset")
    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)
    df = None

    if mode == "Default dataset":
        # Default dataset URL (user-provided repo). Change if different.
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
        st.markdown("#### Download Sample CSV for Reference")
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/Marketing_Analytics.csv"
        try:
            # Load default dataset
            sample_df = pd.read_csv(URL).head(5)  # Take first 5 rows
            sample_csv = sample_df.to_csv(index=False)
            st.download_button("Download Sample CSV", sample_csv, "sample_dataset.csv", "text/csv")
        except Exception as e:
            st.info(f"Sample CSV unavailable: {e}")
    
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            st.success("File uploaded.")
            # try auto-map
            df = auto_map_columns(df)
            st.dataframe(df.head())
            # provide small sample download for mapping reference
            sample_small = df.head(5).to_csv(index=False)
            st.download_button("Download sample (first 5 rows)", sample_small, "sample_uploaded_5rows.csv", "text/csv")

    else:  # Upload + mapping
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head())
            st.markdown("Map your columns to required fields (only required fields shown).")
            mapping = {}
            for req in REQUIRED_MARKETING_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + list(raw.columns))
            if st.button("Apply mapping"):
                missing = [k for k,v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v:k for k,v in mapping.items()})
                    st.success("Mapping applied.")
                    st.dataframe(df.head())
                    sample_small = df.head(5).to_csv(index=False)
                    st.download_button("Download mapped sample (5 rows)", sample_small, "mapped_sample_5rows.csv", "text/csv")

    # -------------------------
    # Validate dataset presence
    # -------------------------
    if df is None:
        st.stop()

    # strip spaces from columns
    df.columns = df.columns.str.strip()

    # Try to ensure required columns exist. If not, surface which ones so user can map/upload properly.
    missing = [c for c in REQUIRED_MARKETING_COLS if c not in df.columns]
    if missing:
        st.error("The following required columns are missing: " + ", ".join(missing))
        st.info("Tip: use 'Upload CSV + Column mapping' and map your dataset's columns to the required names.")
        st.stop()

    # -------------------------
    # Type conversions & derived metrics
    # -------------------------
    df = df.copy()
    df = ensure_datetime(df, "Date")
    # numeric conversion for key columns
    for col in ["Impressions", "Clicks", "Leads", "Conversions", "Spend"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Add simple derived metrics
    df["CTR"] = np.where(df["Impressions"]>0, df["Clicks"] / df["Impressions"], 0)
    df["CPC"] = np.where(df["Clicks"]>0, df["Spend"] / df["Clicks"], 0)
    df["Cost_per_Lead"] = np.where(df["Leads"]>0, df["Spend"] / df["Leads"], np.nan)
    df["Conversion_Rate"] = np.where(df["Clicks"]>0, df["Conversions"] / df["Clicks"], 0)

    # -------------------------
    # Filters
    # -------------------------
    st.markdown("### Step 2 — Filters & preview")
    c1, c2, c3 = st.columns([2,2,1])
    campaigns = sorted(df["Campaign"].dropna().unique().tolist())
    channels = sorted(df["Channel"].dropna().unique().tolist())

    with c1:
        sel_campaigns = st.multiselect("Campaign", options=campaigns, default=campaigns[:5])
    with c2:
        sel_channels = st.multiselect("Channel", options=channels, default=channels[:3])
    with c3:
        date_range = st.date_input("Date range", value=(df["Date"].min().date() if not df["Date"].isna().all() else None,
                                                      df["Date"].max().date() if not df["Date"].isna().all() else None))

    filt = df.copy()
    if sel_campaigns:
        filt = filt[filt["Campaign"].isin(sel_campaigns)]
    if sel_channels:
        filt = filt[filt["Channel"].isin(sel_channels)]
    if date_range and date_range[0] is not None:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filt = filt[(filt["Date"] >= start) & (filt["Date"] <= end)]

    st.markdown("Filtered preview")
    st.dataframe(filt.head(5), use_container_width=True)
    download_df(filt.head(5), "filtered_preview.csv")
    
    # --------------AI

    if st.button("Generate AI Insights"):
        prompt = f"""
        Give a full marketing insight summary based on this dataset:
        {filt.head(50).to_dict()}
        """
    
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )
    
        st.write("### AI Insights")
        st.write(response.choices[0].message["content"])


    # -------------------------
    # KPIs
    # -------------------------
    st.markdown("### Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Impressions", f"{int(filt['Impressions'].sum()):,}")
    k2.metric("Total Clicks", f"{int(filt['Clicks'].sum()):,}")
    k3.metric("Total Leads", f"{int(filt['Leads'].sum()):,}")
    k4.metric("Total Spend", to_currency(filt['Spend'].sum()))

    # -------------------------
    # Charts (colorful + data labels)
    # -------------------------
    st.markdown("### Campaign performance chart: Clicks and Conversions")
    # aggregated by Campaign
    agg = filt.groupby("Campaign").agg({
        "Impressions":"sum","Clicks":"sum","Leads":"sum","Conversions":"sum","Spend":"sum"
    }).reset_index().sort_values("Clicks", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Clicks"], name="Clicks", text=agg["Clicks"], textposition="outside"))
    fig.add_trace(go.Bar(x=agg["Campaign"], y=agg["Conversions"], name="Conversions", text=agg["Conversions"], textposition="outside"))
    fig.update_layout(barmode='group', xaxis_title="<b>Campaign</b>", yaxis_title="<b>Count</b>", legend_title_text="Metric", template="plotly_white")
    fig.update_xaxes(tickangle= -45, showline=True, linewidth=2, linecolor="black")
    fig.update_yaxes(showline=True, linewidth=2, linecolor="black")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Channel ROI (Expected leads per ₹)")
    roi_df = filt.groupby("Channel").agg({"Leads":"sum","Spend":"sum"}).reset_index()
    roi_df["Leads_per_Rs"] = np.where(roi_df["Spend"]>0, roi_df["Leads"]/roi_df["Spend"], 0)
    fig2 = px.bar(roi_df.sort_values("Leads_per_Rs", ascending=False), x="Channel", y="Leads_per_Rs", text=roi_df["Leads_per_Rs"].round(4))
    fig2.update_traces(textposition="outside")
    fig2.update_layout(yaxis_title="Leads per ₹", xaxis_title="<b>Channel</b>", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    # Time-series aggregated
    st.markdown("### Time-series: Clicks over time (trend + moving average)")
    ts = filt.dropna(subset=["Date"]).groupby("Date")["Clicks"].sum().reset_index().sort_values("Date")
    if not ts.empty:
        ts["MA_7"] = ts["Clicks"].rolling(7, min_periods=1).mean()
        fig3 = px.line(ts, x="Date", y=["Clicks","MA_7"], labels={"value":"Clicks","variable":"Series"}, template="plotly_white")
        fig3.update_layout(xaxis_title="<b>Date</b>", yaxis_title="<b>Clicks</b>")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Not enough date data to plot time-series.")

    # -------------------------
    # Machine learning: predict Clicks (regression) and Conversion probability (classification)
    # -------------------------
    st.markdown("### ML: Predictive models (Clicks regression + Conversion classification)")
    ml_expander = st.expander("ML settings & results", expanded=True)
    with ml_expander:
        st.markdown("**Feature engineering**: channel, campaign, spend, impressions, date features.")
        # Prepare dataset for ML
        ml_df = filt.copy().dropna(subset=["Clicks","Impressions","Spend"])
        # Basic date features
        if "Date" in ml_df.columns:
            ml_df["dayofweek"] = ml_df["Date"].dt.dayofweek
            ml_df["month"] = ml_df["Date"].dt.month
        # Features and target for Clicks regression
        feat_cols = ["Channel","Campaign","Impressions","Spend","dayofweek","month"]
        feat_cols = [c for c in feat_cols if c in ml_df.columns]
        if len(ml_df) < 40 or len(feat_cols) < 2:
            st.info("Not enough data or features to train robust ML models (need >=40 rows).")
        else:
            X = ml_df[feat_cols].copy()
            y_reg = ml_df["Clicks"].astype(float)
            # Preprocessing for categorical
            cat_cols = [c for c in X.columns if X[c].dtype == "object"]
            num_cols = [c for c in X.columns if c not in cat_cols]
            preprocessor = ColumnTransformer(transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                ("num", StandardScaler(), num_cols)
            ], remainder="drop")
            X_t = preprocessor.fit_transform(X)
            # Split
            X_train, X_test, y_train, y_test = train_test_split(X_t, y_reg, test_size=0.2, random_state=42)
            # Model: RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=200, random_state=42)
            with st.spinner("Training RandomForest (clicks regression)..."):
                rf.fit(X_train, y_train)
            preds = rf.predict(X_test)
            rmse = math.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            st.write(f"Clicks regression — RMSE: {rmse:.2f}, R²: {r2:.3f}")
            # feature importance (map names)
            try:
                ohe = preprocessor.named_transformers_["cat"]
                cat_names = list(ohe.get_feature_names_out(cat_cols)) if cat_cols else []
            except Exception:
                # fallback manual
                cat_names = []
            feature_names = cat_names + num_cols
            fi = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False).head(20)
            st.markdown("Top feature importances (clicks model)")
            st.dataframe(fi.reset_index().rename(columns={"index":"feature",0:"importance"}).head(20))

            # Provide a simple predict row UI
            st.markdown("Try a quick prediction (single row):")
            inp_cols = {}
            # choose a sample row as defaults if possible
            sample_row = ml_df.iloc[0]
            c1,c2,c3 = st.columns(3)
            with c1:
                sel_campaign = st.selectbox("Campaign (for prediction)", options=ml_df["Campaign"].unique(), index=0)
            with c2:
                sel_channel = st.selectbox("Channel (for prediction)", options=ml_df["Channel"].unique(), index=0)
            with c3:
                inp_impr = st.number_input("Impressions", value=int(ml_df["Impressions"].median()))
            c4,c5 = st.columns(2)
            with c4:
                inp_spend = st.number_input("Spend (INR)", value=float(ml_df["Spend"].median()))
            with c5:
                sel_day = st.selectbox("Day of week (0=Mon)", options=list(range(0,7)), index=int(ml_df["dayofweek"].median()))
            if st.button("Predict Clicks"):
                row = pd.DataFrame([{
                    "Campaign": sel_campaign,
                    "Channel": sel_channel,
                    "Impressions": inp_impr,
                    "Spend": inp_spend,
                    "dayofweek": sel_day,
                    "month": int(pd.to_datetime(ml_df["Date"].iloc[-1]).month) if "Date" in ml_df.columns else 1
                }])
                row_t = preprocessor.transform(row)
                pred_clicks = rf.predict(row_t)[0]
                st.success(f"Predicted Clicks: {int(round(pred_clicks))}")

    # -------------------------
    # Forecasting: spend / clicks / conversions with fallback
    # -------------------------
    st.markdown("### Time-series forecasting (campaign or channel level)")
    with st.expander("Forecast settings", expanded=False):
        st.write("Choose metric and granularity. The app will try Prophet; if not installed it uses a simple linear model or moving-average fallback.")
    metric = st.selectbox("Forecast metric:", ["Clicks","Spend","Conversions"])
    granularity = st.selectbox("Granularity:", ["Overall", "By Campaign", "By Channel"])
    periods = st.number_input("Months to forecast", min_value=1, max_value=24, value=6)

    # pick group
    group_field = None
    if granularity == "By Campaign":
        group_field = "Campaign"
        pick = st.selectbox("Choose campaign:", options=sorted(filt["Campaign"].unique()))
    elif granularity == "By Channel":
        group_field = "Channel"
        pick = st.selectbox("Choose channel:", options=sorted(filt["Channel"].unique()))
    else:
        pick = None

    # build timeseries df
    ts_df = filt.dropna(subset=["Date"])
    if group_field and pick:
        ts_df = ts_df[ts_df[group_field]==pick]
    ts_agg = ts_df.groupby(pd.Grouper(key="Date", freq="D")).agg({metric:"sum"}).reset_index().sort_values("Date")

    if ts_agg[metric].sum() == 0 or len(ts_agg) < 6:
        st.warning("Not enough historical points for a reliable forecast. Need at least ~6 daily points with non-zero values.")
    else:
        # try Prophet
        try:
            from prophet import Prophet
            prophet_ok = True
        except Exception:
            prophet_ok = False

        if prophet_ok:
            model_df = ts_agg.rename(columns={"Date":"ds", metric:"y"})[["ds","y"]]
            # fill missing dates
            model_df = model_df.set_index("ds").asfreq("D").fillna(0).reset_index()
            m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            with st.spinner("Training Prophet model..."):
                m.fit(model_df)
            future = m.make_future_dataframe(periods=periods*30)  # approximate months*30 days
            forecast = m.predict(future)
            figf = px.line(forecast, x="ds", y=["yhat","yhat_lower","yhat_upper"], labels={"ds":"Date"}, title=f"Forecast for {metric}")
            st.plotly_chart(figf, use_container_width=True)
            # show last predicted periods
            ff = forecast[["ds","yhat"]].tail(periods*30)
            st.markdown("Forecast sample (last rows)")
            st.dataframe(ff.tail(10).rename(columns={"ds":"Date","yhat":f"Pred_{metric}"}).reset_index(drop=True))
            # basic summary
            last_actual = model_df["y"].iloc[-1]
            next_pred = forecast["yhat"].iloc[-1]
            st.write(f"Last actual ({metric}): {last_actual:.1f}  | Next predicted ({metric}): {next_pred:.1f}")
        else:
            # fallback: simple seasonal + trend using linear regression on time index
            ts_lr = ts_agg.copy()
            ts_lr = ts_lr.set_index("Date").resample("D").sum().fillna(0).reset_index()
            ts_lr["t"] = np.arange(len(ts_lr))
            X = ts_lr[["t"]]
            y = ts_lr[metric]
            lr = LinearRegression()
            lr.fit(X, y)
            future_t = np.arange(len(ts_lr), len(ts_lr)+periods*30).reshape(-1,1)
            preds = lr.predict(future_t)
            # Plot historical + preds
            hist = go.Scatter(x=ts_lr["Date"], y=ts_lr[metric], name="Actual")
            fut_dates = pd.date_range(ts_lr["Date"].max() + pd.Timedelta(days=1), periods=periods*30, freq="D")
            fut = go.Scatter(x=fut_dates, y=preds, name="Forecast")
            fig = go.Figure([hist, fut])
            fig.update_layout(title=f"Forecast (linear fallback) for {metric}", xaxis_title="Date", yaxis_title=metric)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pd.DataFrame({"Date":fut_dates, f"Pred_{metric}":preds}).head(10))

    # -------------------------
    # Automated insights (rule-based "Gen-AI-like")
    # -------------------------
    st.markdown("### Automated Insights (rule-based)")
    with st.expander("Generate insights", expanded=True):
        # Generate simple rules
        insights = []
        # take top & bottom channels by leads per spend
        channel_perf = filt.groupby("Channel").agg({"Leads":"sum","Spend":"sum"}).reset_index()
        channel_perf["Leads_per_Rs"] = np.where(channel_perf["Spend"]>0, channel_perf["Leads"]/channel_perf["Spend"], 0)
        if not channel_perf.empty:
            best_channel = channel_perf.sort_values("Leads_per_Rs", ascending=False).iloc[0]
            worst_channel = channel_perf.sort_values("Leads_per_Rs", ascending=True).iloc[0]
            insights.append(f"Best channel by leads-per-rupee: {best_channel['Channel']} (≈ {best_channel['Leads_per_Rs']:.4f} leads/₹). Consider scaling budget there.")
            insights.append(f"Worst channel by leads-per-rupee: {worst_channel['Channel']} (≈ {worst_channel['Leads_per_Rs']:.4f} leads/₹). Review creatives and targeting or pause for now.")
        # Campaign-level swing
        camp_perf = filt.groupby("Campaign").agg({"Leads":"sum","Spend":"sum","Clicks":"sum"}).reset_index()
        camp_perf["CPL"] = np.where(camp_perf["Leads"]>0, camp_perf["Spend"]/camp_perf["Leads"], np.nan)
        cheap = camp_perf.sort_values("CPL").head(1)
        expensive = camp_perf.sort_values("CPL", ascending=False).head(1)
        if not cheap.empty:
            insights.append(f"Lowest cost-per-lead campaign: {cheap.iloc[0]['Campaign']} (CPL ≈ {to_currency(cheap.iloc[0]['CPL'])}).")
        if not expensive.empty and not math.isfinite(expensive.iloc[0]['CPL'])==False:
            insights.append(f"High CPL campaign: {expensive.iloc[0]['Campaign']} (CPL ≈ {to_currency(expensive.iloc[0]['CPL'])}). Investigate landing pages or audiences.")
        # seasonal lookout
        if "Date" in filt.columns and not filt["Date"].isna().all():
            monthly = filt.groupby(filt["Date"].dt.month).agg({"Leads":"sum"}).reset_index()
            top_month = monthly.sort_values("Leads", ascending=False).iloc[0]["Date"]
            insights.append(f"Peak lead month historically: {int(top_month)}. Consider increasing ad frequency before this window.")
        # conversions trend
        conv_trend = filt.groupby("Date")["Conversions"].sum().reset_index().sort_values("Date")
        if len(conv_trend) > 3:
            last3 = conv_trend["Conversions"].tail(3).values
            if last3[-1] > last3[0]:
                insights.append("Conversions are trending up in recent days — allocate a bit more budget to test scale.")
            else:
                insights.append("Conversions have dipped recently — check creatives, targeting and landing page health.")

        for i,ins in enumerate(insights):
            st.markdown(f"**Insight {i+1}:** {ins}")

    # -------------------------
    # Predictions / Export
    # -------------------------
    st.markdown("### Export predictions & results")
    if st.button("Predict conversions for filtered dataset (ML simple)"):
        # quick naive estimate: conversion rate median * clicks
        if "Conversion_Rate" in filt.columns:
            median_conv_rate = filt["Conversion_Rate"].median()
            preds_conv = (filt["Clicks"] * median_conv_rate).round().astype(int)
            out = filt.copy()
            out["Predicted_Conversions"] = preds_conv
            st.dataframe(out.head())
            download_df(out[["Campaign","Channel","Date","Clicks","Predicted_Conversions"]], "predicted_conversions.csv")
        else:
            st.info("Conversion_Rate not available — cannot predict.")

    st.success("App ready. If you want a tighter Gen-AI integration (call to external LLM), paste an API key and I will add a button that sends a structured prompt to OpenAI/Anthropic — but that requires outbound network and an API key from you.")

# End of app

