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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Solar Power Generation Forecasting Lab", layout="wide")

# Hide default sidebar nav
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display:none;}
section[data-testid="stSidebar"] {display:none;}
</style>
""", unsafe_allow_html=True)

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
# Global CSS – Marketing Lab standard
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

st.markdown("<div class='big-header'>Solar Power Generation Forecasting Lab</div>", unsafe_allow_html=True)

# -------------------------
# Helpers & constants
# -------------------------
DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/Solar/solar_forecasting.csv"

REQUIRED_SOLAR_COLS = [
    "timestamp",
    "solar_radiation_wm2",
    "diffuse_radiation_wm2",
    "direct_normal_irradiance_wm2",
    "temperature_c",
    "humidity_pct",
    "wind_speed_mps",
    "wind_direction_deg",
    "cloud_cover_pct",
    "pressure_hpa",
    "dew_point_c",
    "forecast_generation_kwh",
    "actual_generation_kwh",
    "generation_error_kwh",
    "plant_id",
    "inverter_cluster_id",
    "weather_condition",
    "sunrise_time",
    "sunset_time",
    "day_of_year",
    "clearness_index",
    "panel_tilt_deg",
    "panel_azimuth_deg",
    "is_holiday_flag"
]

def to_float_series(s):
    return pd.to_numeric(s, errors="coerce")

def ensure_datetime(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def download_df(df, filename, label):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# ==========================================================
# TAB 1 - OVERVIEW
# ==========================================================
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <b>Purpose:</b><br><br>
    This lab brings together plant SCADA data and granular weather signals to forecast solar power generation at
    hourly / sub-hourly resolution. It helps operations teams understand when the plant will under- or over-perform
    vs plan, and how cloud cover, temperature, and irradiation patterns drive that variance.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">What This Lab Does</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Tracks <b>forecast vs actual_generation_kwh</b> at timestamp level<br>
        • Quantifies impact of <b>cloud_cover_pct, solar_radiation_wm2, temperature_c</b><br>
        • Surfaces time-of-day & weather regimes with high <b>generation_error_kwh</b><br>
        • Builds ML models to predict <b>actual_generation_kwh</b> from weather & plant signals<br>
        • Provides plant / inverter-cluster level ranking by forecast accuracy
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Better <b>day-ahead & intra-day scheduling</b> for grid commitments<br>
        • Lower penalties & curtailment due to <b>unplanned swings</b><br>
        • Higher reliability of <b>solar + storage dispatch</b><br>
        • Quantified view of <b>weather risk</b> on generation<br>
        • Stronger analytics story for <b>lenders, regulators & offtakers</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Key KPIs Tracked</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Total Actual Generation (kWh)</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>RMSE / MAE vs Forecast</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>MAPE by Plant / Cluster</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Cloud- & Weather-Driven Loss</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Who Should Use This</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    Solar plant O&M teams, control-room operators, grid planning teams, energy traders, asset managers and data
    engineers who want a single workspace for <b>solar forecasting, error diagnostics, and plant benchmarking</b>.
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# TAB 2 - IMPORTANT ATTRIBUTES
# ==========================================================
with tab2:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    required_dict = {
        "timestamp": "Date-time stamp for each measurement interval.",
        "solar_radiation_wm2": "Global horizontal irradiance (GHI) in W/m².",
        "diffuse_radiation_wm2": "Diffuse component of solar irradiance in W/m².",
        "direct_normal_irradiance_wm2": "DNI (beam radiation) in W/m².",
        "temperature_c": "Ambient temperature at site (°C).",
        "humidity_pct": "Relative humidity (%).",
        "wind_speed_mps": "Wind speed at site (m/s).",
        "wind_direction_deg": "Wind direction in degrees (0–360).",
        "cloud_cover_pct": "Cloud cover percentage (%).",
        "pressure_hpa": "Atmospheric pressure (hPa).",
        "dew_point_c": "Dew point temperature (°C).",
        "forecast_generation_kwh": "Forecasted energy generation for the interval (kWh).",
        "actual_generation_kwh": "Actual energy generated (kWh).",
        "generation_error_kwh": "Error = actual_generation_kwh − forecast_generation_kwh.",
        "plant_id": "Identifier for the solar plant.",
        "inverter_cluster_id": "Cluster / block or inverter group ID.",
        "weather_condition": "Categorical weather label (Clear / Cloudy / Overcast / etc.).",
        "sunrise_time": "Sunrise time for the plant location on that date.",
        "sunset_time": "Sunset time for the plant location on that date.",
        "day_of_year": "Day index in the year (1–365/366).",
        "clearness_index": "Ratio of surface radiation to extraterrestrial radiation.",
        "panel_tilt_deg": "Tilt angle of the PV array (degrees).",
        "panel_azimuth_deg": "Azimuth of the PV array (degrees).",
        "is_holiday_flag": "Indicator if the day is a holiday (0/1 or True/False)."
    }

    req_df = pd.DataFrame(
        [{"Attribute": k, "Description": v} for k, v in required_dict.items()]
    )

    # PURE black table styling override (index-safe renderer)
    st.markdown("""
        <style>
            .required-table th {
                background:#ffffff !important;
                color:#000 !important;
                font-size:18px !important;
                border-bottom:2px solid #000 !important;
            }
            .required-table td {
                color:#000 !important;
                font-size:16px !important;
                padding:8px !important;
                border-bottom:1px solid #dcdcdc !important;
            }
            .required-table tr:hover td {
                background:#f8f8f8 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.dataframe(
        req_df.style.set_table_attributes('class="required-table"'),
        use_container_width=True
    )

    # Independent vs Dependent variables
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        independents = [
            "solar_radiation_wm2",
            "diffuse_radiation_wm2",
            "direct_normal_irradiance_wm2",
            "temperature_c",
            "humidity_pct",
            "wind_speed_mps",
            "wind_direction_deg",
            "cloud_cover_pct",
            "pressure_hpa",
            "dew_point_c",
            "plant_id",
            "inverter_cluster_id",
            "weather_condition",
            "day_of_year",
            "clearness_index",
            "panel_tilt_deg",
            "panel_azimuth_deg",
            "is_holiday_flag"
        ]
        for v in independents:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        dependents = [
            "forecast_generation_kwh",
            "actual_generation_kwh",
            "generation_error_kwh"
        ]
        for v in dependents:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# ==========================================================
# TAB 3 - APPLICATION
# ==========================================================
with tab3:
    st.markdown('<div class="section-title">Step 1: Load dataset</div>', unsafe_allow_html=True)

    df = None
    mode = st.radio(
        "Select Dataset Option:",
        ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"],
        horizontal=True
    )

    # -------------------------
    # Default dataset
    # -------------------------
    if mode == "Default dataset":
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            st.success("Default dataset loaded from GitHub.")
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    # -------------------------
    # Upload CSV (with sample preview)
    # -------------------------
    elif mode == "Upload CSV":
        st.markdown("#### Sample structure (from standard Solar Forecasting dataset)", unsafe_allow_html=True)
        try:
            sample_df = pd.read_csv(DEFAULT_URL).head(10)
            st.dataframe(sample_df, use_container_width=True)
            sample_csv = sample_df.to_csv(index=False)
            st.download_button(
                "Download Sample CSV",
                sample_csv,
                "sample_solar_forecasting.csv",
                "text/csv"
            )
        except Exception as e:
            st.info(f"Sample CSV unavailable from GitHub: {e}")

        uploaded = st.file_uploader("Upload your Solar Forecasting dataset", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            st.success("File uploaded.")
            st.dataframe(df.head(10), use_container_width=True)

    # -------------------------
    # Upload + column mapping
    # -------------------------
    else:
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.write("Preview (first 10 rows):")
            st.dataframe(raw.head(10), use_container_width=True)
            st.markdown("Map your columns to the required fields:", unsafe_allow_html=True)
            mapping = {}
            for req in REQUIRED_SOLAR_COLS:
                mapping[req] = st.selectbox(
                    f"Map → {req}",
                    options=["-- Select --"] + list(raw.columns)
                )
            if st.button("Apply mapping"):
                missing = [k for k, v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    inv = {v: k for k, v in mapping.items()}
                    df = raw.rename(columns=inv)
                    st.success("Mapping applied.")
                    st.dataframe(df.head(10), use_container_width=True)

    if df is None:
        st.stop()

    # -------------------------
    # Validate required columns
    # -------------------------
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED_SOLAR_COLS if c not in df.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.info("Use 'Upload CSV + Column mapping' if your column names differ.")
        st.stop()

    # -------------------------
    # Type coercion
    # -------------------------
    df = ensure_datetime(df, "timestamp")
    # sunrise_time & sunset_time we treat as strings for UI; no need to convert to datetime here

    num_cols = [
        "solar_radiation_wm2",
        "diffuse_radiation_wm2",
        "direct_normal_irradiance_wm2",
        "temperature_c",
        "humidity_pct",
        "wind_speed_mps",
        "wind_direction_deg",
        "cloud_cover_pct",
        "pressure_hpa",
        "dew_point_c",
        "forecast_generation_kwh",
        "actual_generation_kwh",
        "generation_error_kwh",
        "day_of_year",
        "clearness_index",
        "panel_tilt_deg",
        "panel_azimuth_deg"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = to_float_series(df[c])

    # -------------------------
    # Filters
    # -------------------------
    st.markdown('<div class="section-title">Step 2: Filters & Preview</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    plants = sorted(df["plant_id"].dropna().unique().tolist())
    clusters = sorted(df["inverter_cluster_id"].dropna().unique().tolist()) if "inverter_cluster_id" in df.columns else []
    weathers = sorted(df["weather_condition"].dropna().unique().tolist()) if "weather_condition" in df.columns else []

    with c1:
        sel_plants = st.multiselect("Plant ID", options=plants, default=plants[:5] if plants else [])
    with c2:
        sel_clusters = st.multiselect("Inverter Cluster", options=clusters, default=clusters[:5] if clusters else [])
    with c3:
        sel_weather = st.multiselect("Weather Condition", options=weathers, default=weathers if weathers else [])

    # Date range filter
    if "timestamp" in df.columns and not df["timestamp"].isna().all():
        try:
            min_d = df["timestamp"].min().date()
            max_d = df["timestamp"].max().date()
            date_range = st.date_input("Date range", value=(min_d, max_d))
        except Exception:
            date_range = st.date_input("Date range")
    else:
        date_range = None

    filt = df.copy()
    if sel_plants:
        filt = filt[filt["plant_id"].isin(sel_plants)]
    if sel_clusters and "inverter_cluster_id" in filt.columns:
        filt = filt[filt["inverter_cluster_id"].isin(sel_clusters)]
    if sel_weather and "weather_condition" in filt.columns:
        filt = filt[filt["weather_condition"].isin(sel_weather)]

    if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        try:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            if "timestamp" in filt.columns:
                filt = filt[(filt["timestamp"] >= start) & (filt["timestamp"] <= end + pd.Timedelta(days=1))]
        except Exception:
            pass

    # Filter count display
    st.markdown(f"""
    <div style='padding:8px; background:#eef4ff; border-radius:8px; font-weight:600; margin-bottom:8px;'>
    Filtered Rows: {len(filt)} of {len(df)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Data Preview (first 15 rows)")
    st.dataframe(filt.head(15), use_container_width=True)
    download_df(filt.head(500), "solar_filtered_preview.csv", "Download filtered preview (first 500 rows)")

    if filt.empty:
        st.warning("Filtered dataset is empty. Adjust filters above.")
        st.stop()

    # -------------------------
    # KPIs (numeric)
    # -------------------------
    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    def safe_sum(s):
        try:
            return float(pd.to_numeric(s, errors="coerce").sum())
        except Exception:
            return float("nan")

    def safe_mean(s):
        try:
            return float(pd.to_numeric(s, errors="coerce").mean())
        except Exception:
            return float("nan")

    total_actual = safe_sum(filt["actual_generation_kwh"])
    total_forecast = safe_sum(filt["forecast_generation_kwh"])
    mae = mean_absolute_error(
        pd.to_numeric(filt["actual_generation_kwh"], errors="coerce"),
        pd.to_numeric(filt["forecast_generation_kwh"], errors="coerce")
    ) if len(filt) > 0 else float("nan")
    rmse = math.sqrt(mean_squared_error(
        pd.to_numeric(filt["actual_generation_kwh"], errors="coerce"),
        pd.to_numeric(filt["forecast_generation_kwh"], errors="coerce")
    )) if len(filt) > 0 else float("nan")

    # MAPE
    actual_nonzero = filt["actual_generation_kwh"].replace(0, np.nan)
    if actual_nonzero.notna().any():
        mape = (np.abs(filt["actual_generation_kwh"] - filt["forecast_generation_kwh"]) / actual_nonzero).mean() * 100
    else:
        mape = float("nan")

    k1.metric("Total Actual Generation (kWh)", f"{total_actual:,.2f}")
    k2.metric("Total Forecast Generation (kWh)", f"{total_forecast:,.2f}")
    k3.metric("MAE / RMSE (kWh)", f"{mae:.2f} / {rmse:.2f}" if not (math.isnan(mae) or math.isnan(rmse)) else "N/A")
    k4.metric("MAPE (%)", f"{mape:.2f}%" if not math.isnan(mape) else "N/A")

    # -------------------------
    # Charts & Diagnostics
    # -------------------------
    st.markdown('<div class="section-title">Charts & Diagnostics</div>', unsafe_allow_html=True)

    # 1) Time series: actual vs forecast
    if "timestamp" in filt.columns:
        ts = filt.dropna(subset=["timestamp"]).sort_values("timestamp")
        if not ts.empty:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=ts["timestamp"], y=ts["actual_generation_kwh"],
                name="Actual", mode="lines"
            ))
            fig_ts.add_trace(go.Scatter(
                x=ts["timestamp"], y=ts["forecast_generation_kwh"],
                name="Forecast", mode="lines"
            ))
            fig_ts.update_layout(
                title="Actual vs Forecast Generation (kWh)",
                xaxis_title="Timestamp",
                yaxis_title="Generation (kWh)",
                template="plotly_white"
            )
            st.plotly_chart(fig_ts, use_container_width=True)

    # 2) Scatter: Actual vs Forecast
    scat_df = filt[["actual_generation_kwh", "forecast_generation_kwh"]].dropna()
    if not scat_df.empty:
        fig_sc = px.scatter(
            scat_df,
            x="forecast_generation_kwh",
            y="actual_generation_kwh",
            labels={
                "forecast_generation_kwh": "Forecast (kWh)",
                "actual_generation_kwh": "Actual (kWh)"
            },
            title="Actual vs Forecast Generation Scatter"
        )
        fig_sc.add_shape(
            type="line",
            x0=scat_df["forecast_generation_kwh"].min(),
            y0=scat_df["forecast_generation_kwh"].min(),
            x1=scat_df["forecast_generation_kwh"].max(),
            y1=scat_df["forecast_generation_kwh"].max(),
            line=dict(color="red", dash="dash")
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # 3) Error by weather_condition
    if "weather_condition" in filt.columns:
        err_weather = filt.copy()
        err_weather["abs_error"] = (err_weather["actual_generation_kwh"] - err_weather["forecast_generation_kwh"]).abs()
        grp = err_weather.groupby("weather_condition")["abs_error"].mean().reset_index()
        if not grp.empty:
            fig_w = px.bar(
                grp,
                x="weather_condition",
                y="abs_error",
                text="abs_error",
                title="Average Absolute Error by Weather Condition"
            )
            fig_w.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_w.update_layout(xaxis_title="Weather condition", yaxis_title="Avg abs error (kWh)")
            st.plotly_chart(fig_w, use_container_width=True)

    # 4) Clearness index vs generation
    if "clearness_index" in filt.columns:
        ci_df = filt[["clearness_index", "actual_generation_kwh"]].dropna()
        if not ci_df.empty:
            fig_ci = px.scatter(
                ci_df,
                x="clearness_index",
                y="actual_generation_kwh",
                title="Clearness Index vs Actual Generation"
            )
            fig_ci.update_layout(xaxis_title="Clearness index", yaxis_title="Actual generation (kWh)")
            st.plotly_chart(fig_ci, use_container_width=True)

    # -------------------------
    # ML — Actual Generation Regression (RandomForest)
    # -------------------------
    st.markdown('<div class="section-title">ML — Actual Generation Regression (RandomForest)</div>', unsafe_allow_html=True)
    with st.expander("Train & evaluate model (requires >= 80 rows)", expanded=False):
        ml_df = filt.dropna(subset=["actual_generation_kwh"]).copy()

        feat_cols = [
            "solar_radiation_wm2",
            "diffuse_radiation_wm2",
            "direct_normal_irradiance_wm2",
            "temperature_c",
            "humidity_pct",
            "wind_speed_mps",
            "cloud_cover_pct",
            "pressure_hpa",
            "dew_point_c",
            "day_of_year",
            "clearness_index",
            "panel_tilt_deg",
            "panel_azimuth_deg",
            "weather_condition",
            "plant_id",
            "inverter_cluster_id",
            "is_holiday_flag"
        ]
        feat_cols = [c for c in feat_cols if c in ml_df.columns]

        if len(ml_df) < 80 or len(feat_cols) < 3:
            st.info("Not enough rows or features to train a robust model (need at least ~200 rows and a few features).")
        else:
            X = ml_df[feat_cols].copy()
            y = ml_df["actual_generation_kwh"].astype(float)

            # Handle holiday flag to numeric if it's boolean/string
            if "is_holiday_flag" in X.columns:
                X["is_holiday_flag"] = X["is_holiday_flag"].astype(str)

            cat_cols = [c for c in X.columns if X[c].dtype == "object"]
            num_cols_ml = [c for c in X.columns if c not in cat_cols]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols) if cat_cols else ("noop", "passthrough", []),
                    ("num", StandardScaler(), num_cols_ml) if num_cols_ml else ("noop2", "passthrough", [])
                ],
                remainder="drop"
            )

            try:
                X_t = preprocessor.fit_transform(X)
            except Exception as e:
                st.error("Preprocessing failed: " + str(e))
                X_t = None

            if X_t is not None:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_t, y, test_size=0.2, random_state=42
                )
                rf = RandomForestRegressor(n_estimators=200, random_state=42)
                with st.spinner("Training RandomForest regression model..."):
                    rf.fit(X_train, y_train)
                preds = rf.predict(X_test)

                rmse_m = math.sqrt(mean_squared_error(y_test, preds))
                r2_m = r2_score(y_test, preds)
                st.write(f"Actual generation regression — RMSE: {rmse_m:.2f} kWh, R²: {r2_m:.3f}")

                res_df = pd.DataFrame({
                    "Actual_kWh": y_test.reset_index(drop=True),
                    "Predicted_kWh": preds
                })
                st.dataframe(res_df.head(20), use_container_width=True)
                download_df(res_df, "solar_generation_predictions.csv", "Download prediction sample")

    # -------------------------
    # Automated Insights
    # -------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)

    insights_rows = []

    # 1) Plant with worst MAPE
    if "plant_id" in filt.columns:
        tmp = filt.copy()
        tmp = tmp[tmp["actual_generation_kwh"] > 0]
        if not tmp.empty:
            tmp["abs_pct_error"] = (
                (tmp["actual_generation_kwh"] - tmp["forecast_generation_kwh"]).abs()
                / tmp["actual_generation_kwh"]
            ) * 100
            plant_err = tmp.groupby("plant_id")["abs_pct_error"].mean().reset_index()
            if not plant_err.empty:
                worst_p = plant_err.sort_values("abs_pct_error", ascending=False).iloc[0]
                insights_rows.append({
                    "Insight": "Plant with highest average forecast error",
                    "Entity": worst_p["plant_id"],
                    "Metric": f"{worst_p['abs_pct_error']:.2f}% MAPE",
                    "Action": "Review forecasting inputs (weather, clipping, curtailment) & model for this plant."
                })

    # 2) Weather condition causing biggest average error
    if "weather_condition" in filt.columns:
        tmp2 = filt.copy()
        tmp2["abs_error"] = (tmp2["actual_generation_kwh"] - tmp2["forecast_generation_kwh"]).abs()
        wc = tmp2.groupby("weather_condition")["abs_error"].mean().reset_index()
        if not wc.empty:
            worst_w = wc.sort_values("abs_error", ascending=False).iloc[0]
            insights_rows.append({
                "Insight": "Worst weather band for forecast accuracy",
                "Entity": worst_w["weather_condition"],
                "Metric": f"{worst_w['abs_error']:.2f} kWh avg abs error",
                "Action": "Train separate model / correction factor for this weather type."
            })

    # 3) Time-of-day with biggest under-generation
    if "timestamp" in filt.columns:
        tmp3 = filt.dropna(subset=["timestamp"]).copy()
        if not tmp3.empty:
            tmp3["hour"] = tmp3["timestamp"].dt.hour
            tmp3["error"] = tmp3["actual_generation_kwh"] - tmp3["forecast_generation_kwh"]
            hour_err = tmp3.groupby("hour")["error"].mean().reset_index()
            if not hour_err.empty:
                worst_h = hour_err.sort_values("error").iloc[0]  # most negative = under-forecast or under-gen
                direction = "under-generation" if worst_h["error"] < 0 else "over-generation"
                insights_rows.append({
                    "Insight": "Hour with strongest systematic bias",
                    "Entity": f"{int(worst_h['hour']):02d}:00",
                    "Metric": f"{worst_h['error']:.2f} kWh avg error ({direction})",
                    "Action": "Apply hour-of-day correction factors in scheduling / trading."
                })

    # 4) Cluster with lowest clearness but decent generation
    if "inverter_cluster_id" in filt.columns and "clearness_index" in filt.columns:
        tmp4 = filt.copy()
        cl = tmp4.groupby("inverter_cluster_id").agg(
            avg_ci=("clearness_index", "mean"),
            avg_gen=("actual_generation_kwh", "mean")
        ).reset_index()
        cl = cl.dropna()
        if not cl.empty:
            candidate = cl.sort_values(["avg_ci", "avg_gen"], ascending=[True, False]).iloc[0]
            insights_rows.append({
                "Insight": "Cluster performing well under low clearness conditions",
                "Entity": candidate["inverter_cluster_id"],
                "Metric": f"CI ≈ {candidate['avg_ci']:.2f}, Gen ≈ {candidate['avg_gen']:.2f} kWh",
                "Action": "Use this cluster as best-practice benchmark (cleaning, tilt, maintenance)."
            })

    if not insights_rows:
        st.info("No insights generated for the current filter selection.")
    else:
        insights_df = pd.DataFrame(insights_rows)
        st.dataframe(insights_df, use_container_width=True)
        download_df(insights_df, "solar_forecasting_insights.csv", "Download insights")

