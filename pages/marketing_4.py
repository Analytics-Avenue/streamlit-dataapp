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

# -------------------------
# Global CSS (master design spec applied)
# - All text: PURE BLACK (#000)
# - KPI card values: blue
# - Independent variable labels and values: blue
# - Card layout, KPI design, table styling, variable-box, fade-in animation
# - 3-tab layout
# -------------------------
st.markdown("""
<style>
/* Global text color: pure black everywhere */
html, body, [data-testid="stAppViewContainer"] *, .css-1d391kg * { color: #000 !important; }

/* Background and container tweaks to preserve aesthetic */
.css-1d391kg { background-color: #ffffff00; }

/* Card styles (standardized) */
.master-card {
  background: #ffffff; /* white card */
  padding: 18px 20px;
  border-radius: 14px;
  margin-bottom: 18px;
  border: 1px solid rgba(0,0,0,0.08);
  box-shadow: 0 6px 18px rgba(0,0,0,0.06);
  transition: transform 0.35s ease, box-shadow 0.35s ease, opacity 0.6s ease;
  opacity: 0; /* start hidden for fade-in */
  animation: fadeIn 0.9s ease forwards;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}

/* KPI card layout */
.kpi-row { display:flex; gap:14px; }
.kpi-card {
  flex:1; padding:18px; border-radius:12px; text-align:center; background:#f8fbff;
  border:1px solid rgba(0,0,0,0.06); box-shadow: 0 4px 12px rgba(0,0,0,0.04);
}
.kpi-title { font-size:13px; font-weight:700; color:#000; margin-bottom:6px; }
.kpi-value { font-size:20px; font-weight:800; color:#064b86; }
.kpi-sublabel { font-size:12px; color:#333; }

/* Variable box (Independent variables) */
.var-box { display:flex; gap:12px; flex-wrap:wrap; }
.var-item { padding:10px 12px; border-radius:10px; border:1px dashed rgba(0,0,0,0.06); background:#fff; }
.var-label { font-size:12px; font-weight:700; color:#064b86; }
.var-value { font-size:13px; color:#064b86; }

/* Table styling - index-safe HTML renderer */
.styled-table { border-collapse: collapse; font-size:13px; font-family: inherit; width:100%; }
.styled-table thead tr { background: #f1f5f9; }
.styled-table th, .styled-table td { padding:8px 10px; border:1px solid rgba(0,0,0,0.06); text-align:left; }
.styled-table tbody tr:hover { background: #fafafa; }
.styled-table caption { caption-side: top; text-align:left; padding-bottom:8px; font-weight:700; }

/* Make sure Streamlit native widgets keep black text */
.css-1q8dd3e, .css-1q8dd3e * { color: #000 !important; }

/* Utility */
.small-muted { font-size:12px; color:#333; }

</style>
""", unsafe_allow_html=True)

# -------------------------
# Header / Logo
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div class="master-card" style="display:flex; align-items:center; gap:14px;">
  <img src="{logo_url}" width="56" style="border-radius:6px;"/>
  <div>
    <div style="font-size:20px; font-weight:800;">Analytics Avenue &amp; Advanced Analytics</div>
    <div class="small-muted">Marketing Intelligence & Forecasting Lab — standardized UI</div>
  </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Required columns & mapping
# -------------------------
REQUIRED_MARKETING_COLS = [
    "Campaign", "Channel", "Date", "Impressions", "Clicks", "Leads",
    "Conversions", "Spend", "Revenue", "ROAS", "Device", "AgeGroup",
    "Gender", "AdSet", "Creative"
]

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


def render_index_safe_html_table(df: pd.DataFrame, caption: str = None):
    """Render an index-safe HTML table using Streamlit's markdown.
    - We explicitly escape values to avoid accidental HTML injection.
    - Keep index as a separate column so layout and rendering are stable.
    """
    tmp = df.copy()
    # Ensure index is a column for index-safe rendering
    tmp.reset_index(inplace=True)
    # Escape text values safely
    html = tmp.to_html(index=False, classes="styled-table", escape=True)
    if caption:
        # Inject caption in a safe place
        html = html.replace('<table class="styled-table">', f'<table class="styled-table"><caption>{caption}</caption>')
    st.markdown(html, unsafe_allow_html=True)

# -------------------------
# Tabs: Overview | Application | Modeling
# -------------------------
tabs = st.tabs(["Overview", "Application", "Modeling & Forecast"])

with tabs[0]:
    st.markdown("""
    <div class="master-card">
      <div style="font-weight:800; font-size:18px;">Marketing Performance Analysis — Overview</div>
      <div class="small-muted">An enterprise-grade layout following the master spec: standardized cards, KPI design, variable boxes and index-safe HTML tables.</div>
      <hr/>
      <div style="margin-top:10px;">
        <strong>Capabilities</strong>
        <ul>
          <li>Multi-channel tracking, demographic breakdowns, creative-level analysis</li>
          <li>Predictive models for revenue and conversions (RandomForest + linear fallback)</li>
          <li>Automated insights and downloadable reports</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)

with tabs[1]:
    st.markdown("""
    <div class="master-card">
      <div style="font-weight:800; font-size:16px;">Step 1 — Load dataset</div>
      <div class="small-muted">Upload a CSV or use the default sample. App will auto-map common column names.</div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)
    df = None

    if mode == "Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/marketing.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("Default dataset loaded")
            render_index_safe_html_table(df.head(5), caption="Default dataset preview (first 5 rows)")
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    elif mode == "Upload CSV":
        st.markdown("#### Download Sample CSV for Reference")
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/marketing.csv"
        try:
            sample_df = pd.read_csv(URL).head(5)
            sample_csv = sample_df.to_csv(index=False)
            st.download_button("Download Sample CSV", sample_csv, "sample_dataset.csv", "text/csv")
        except Exception as e:
            st.info(f"Sample CSV unavailable: {e}")

        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            st.success("Uploaded dataset preview")
            render_index_safe_html_table(df.head(5), caption="Uploaded dataset preview (first 5 rows)")

    else:  # Upload + mapping
        uploaded = st.file_uploader("Upload CSV to map", type=["csv")
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.write("Preview (first 5 rows):")
            render_index_safe_html_table(raw.head(5), caption="Raw uploaded preview")
            st.markdown("Map your columns to required fields.")
            mapping = {}
            cols = list(raw.columns)
            for req in REQUIRED_MARKETING_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + cols)
            if st.button("Apply mapping"):
                missing = [k for k,v in mapping.items() if v=="-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v:k for k,v in mapping.items()})
                    st.success("Mapping applied.")
                    render_index_safe_html_table(df.head(5), caption="Mapped dataset preview")

    if df is None:
        st.stop()

    # Keep only required columns that exist
    df = df[[c for c in REQUIRED_MARKETING_COLS if c in df.columns]].copy()
    df = ensure_datetime(df, "Date")
    for col in ["Impressions","Clicks","Leads","Conversions","Spend","Revenue","ROAS"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    st.markdown("""
    <div class="master-card">
      <div style="font-weight:800; font-size:16px;">Step 2 — Filters & Preview</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2,2,1])
    campaigns = sorted(df["Campaign"].dropna().unique()) if "Campaign" in df.columns else []
    channels = sorted(df["Channel"].dropna().unique()) if "Channel" in df.columns else []
    devices = sorted(df["Device"].dropna().unique()) if "Device" in df.columns else []
    agegroups = sorted(df["AgeGroup"].dropna().unique()) if "AgeGroup" in df.columns else []
    genders = sorted(df["Gender"].dropna().unique()) if "Gender" in df.columns else []

    with c1:
        sel_campaigns = st.multiselect("Campaign", options=campaigns, default=campaigns[:5])
    with c2:
        sel_channels = st.multiselect("Channel", options=channels, default=channels[:3])
    with c3:
        try:
            date_range = st.date_input("Date range", value=(df["Date"].min().date(), df["Date"].max().date()))
        except Exception:
            date_range = None

    filt = df.copy()
    if sel_campaigns: filt = filt[filt["Campaign"].isin(sel_campaigns)]
    if sel_channels: filt = filt[filt["Channel"].isin(sel_channels)]
    if date_range:
        filt = filt[(filt["Date"]>=pd.to_datetime(date_range[0])) & (filt["Date"]<=pd.to_datetime(date_range[1]))]

    render_index_safe_html_table(filt.head(10), caption="Filtered preview (first 10 rows)")
    download_df(filt.head(10), "filtered_preview.csv")

    # -------------------------
    # Key metrics cards (KPI values must be blue)
    # -------------------------
    st.markdown("<div class='master-card'><div style='font-weight:800;'>Key Metrics</div></div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    def kpi_html(title, value, sub=None):
        sub_html = f"<div class='kpi-sublabel'>{sub}</div>" if sub else ""
        return f"<div class='kpi-card'><div class='kpi-title'>{title}</div><div class='kpi-value'>{value}</div>{sub_html}</div>"

    k1.markdown(kpi_html("Total Revenue", to_currency(filt["Revenue"].sum()), "Sum of Revenue"), unsafe_allow_html=True)
    k2.markdown(kpi_html("ROAS", f"{filt['ROAS'].mean():.2f}", "Avg ROAS"), unsafe_allow_html=True)
    k3.markdown(kpi_html("Total Leads", int(filt["Leads"].sum()), "Sum of Leads"), unsafe_allow_html=True)
    conv_rate = (filt['Conversions'].sum()/max(filt['Clicks'].sum(),1)) if ('Conversions' in filt.columns and 'Clicks' in filt.columns) else 0
    k4.markdown(kpi_html("Conversion Rate", f"{conv_rate:.2%}", "Conversions/Clicks"), unsafe_allow_html=True)

    # Independent variables (blue text requirement)
    st.markdown("<div class='master-card'><div style='font-weight:800;'>Independent Variables</div></div>", unsafe_allow_html=True)
    iv_cols = [c for c in ['Channel','Device','AgeGroup','Gender','AdSet','Creative'] if c in df.columns]
    iv_html = "<div class='var-box'>"
    for c in iv_cols:
        vals = filt[c].dropna().unique()[:6]
        vals_display = ", ".join([str(v) for v in vals])
        iv_html += f"<div class='var-item'><div class='var-label'>{c}</div><div class='var-value'>{vals_display}</div></div>"
    iv_html += "</div>"
    st.markdown(iv_html, unsafe_allow_html=True)

    # -------------------------
    # Charts
    # -------------------------
    st.markdown("<div class='master-card'><div style='font-weight:800;'>Campaign Revenue & Conversions</div></div>", unsafe_allow_html=True)
    if 'Campaign' in filt.columns and 'Revenue' in filt.columns:
        agg = filt.groupby('Campaign')[['Revenue','Conversions']].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=agg['Campaign'], y=agg['Revenue'], name='Revenue'))
        if 'Conversions' in agg.columns:
            fig.add_trace(go.Bar(x=agg['Campaign'], y=agg['Conversions'], name='Conversions'))
        fig.update_layout(barmode='group', xaxis_title='Campaign', yaxis_title='Value', plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Campaign/Revenue columns required to render chart')

    st.markdown("<div class='master-card'><div style='font-weight:800;'>Device / Gender / AgeGroup performance</div></div>", unsafe_allow_html=True)
    group_cols = ['Device','Gender','AgeGroup']
    for g in group_cols:
        if g in filt.columns:
            grp = filt.groupby(g)[['Revenue','Conversions']].sum().reset_index()
            fig = px.bar(grp, x=g, y='Revenue', text='Revenue', title=f"{g} Revenue")
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.markdown("""
    <div class="master-card">
      <div style="font-weight:800; font-size:16px;">Modeling & Forecast</div>
      <div class="small-muted">Train a RandomForest regressor for Revenue prediction. Linear regression fallback available.</div>
    </div>
    """, unsafe_allow_html=True)

    ml_df = filt.copy().dropna(subset=['Revenue']) if 'Revenue' in filt.columns else pd.DataFrame()
    feat_cols = ['Channel','Campaign','Device','AgeGroup','Gender','AdSet','Impressions','Clicks','Spend']
    feat_cols = [c for c in feat_cols if c in ml_df.columns]

    if ml_df.empty or len(ml_df) < 30 or len(feat_cols) < 2:
        st.info('Not enough data to train ML model (>=30 rows and at least 2 features required)')
    else:
        X = ml_df[feat_cols]
        y = ml_df['Revenue']
        cat_cols = [c for c in X.columns if X[c].dtype == 'object']
        num_cols = [c for c in X.columns if c not in cat_cols]
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('num', StandardScaler(), num_cols)
        ], remainder='drop')

        X_t = preprocessor.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)

        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        with st.spinner('Training RandomForest...'):
            rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        rmse = math.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        st.markdown(f"<div class='master-card'><div style='font-weight:800;'>Model Performance</div><div class='small-muted'>RMSE: {rmse:.2f} | R²: {r2:.3f}</div></div>", unsafe_allow_html=True)

        # Present test results in index-safe HTML table
        X_test_df = pd.DataFrame(X_test, columns=[f"Feature_{i}" for i in range(X_test.shape[1])])
        X_test_df['Actual_Revenue'] = y_test.reset_index(drop=True)
        X_test_df['Predicted_Revenue'] = preds
        render_index_safe_html_table(X_test_df.head(10), caption='ML Predictions (test set - first 10 rows)')
        download_df(X_test_df, 'ml_revenue_predictions.csv')

        # Quick feature importance if we can map back columns (best-effort)
        try:
            # If OneHotEncoder used, feature names are many; we provide generic ranking
            importances = rf.feature_importances_
            fi = pd.DataFrame({'feature': [f'F{i}' for i in range(len(importances))], 'importance': importances})
            fi = fi.sort_values('importance', ascending=False).head(10)
            st.markdown('<div class="master-card"><div style="font-weight:800;">Top Feature Importances (approx)</div></div>', unsafe_allow_html=True)
            render_index_safe_html_table(fi, caption='Top feature importances')
        except Exception:
            pass

    # Automated insights table
    st.markdown('<div class="master-card"><div style="font-weight:800;">Automated Insights</div></div>', unsafe_allow_html=True)
    insights_list = []
    if 'Channel' in filt.columns and 'Revenue' in filt.columns and 'Spend' in filt.columns:
        ch_perf = filt.groupby('Channel')[['Revenue','Spend']].sum().reset_index()
        ch_perf['Revenue_per_Rs'] = np.where(ch_perf['Spend']>0, ch_perf['Revenue']/ch_perf['Spend'],0)
        best = ch_perf.sort_values('Revenue_per_Rs', ascending=False).iloc[0]
        worst = ch_perf.sort_values('Revenue_per_Rs', ascending=True).iloc[0]
        insights_list.append({'Insight':'Best Channel ROI','Channel':best['Channel'], 'Revenue_per_Rs':best['Revenue_per_Rs']})
        insights_list.append({'Insight':'Lowest Channel ROI','Channel':worst['Channel'], 'Revenue_per_Rs':worst['Revenue_per_Rs']})

    insights_df = pd.DataFrame(insights_list)
    if not insights_df.empty:
        render_index_safe_html_table(insights_df, caption='Automated insights')
        download_df(insights_df, 'automated_insights.csv')
    else:
        st.markdown('<div class="master-card">No automated insights available for the selected filters.</div>', unsafe_allow_html=True)

# Final note card (hidden in master-card style)
st.markdown("""
<div class="master-card"><div class="small-muted">UI spec: Text is PURE BLACK except KPI card values and Independent Variables which are blue. Table rendered via index-safe HTML. Card layout, variable boxes and fade-in animation applied.</div></div>
""", unsafe_allow_html=True)
