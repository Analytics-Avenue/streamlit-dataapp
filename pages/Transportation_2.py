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
from sklearn.metrics import mean_squared_error, r2_score

import math
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="ETA Prediction & On-Time Performance Lab", layout="wide")

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

st.markdown("<div class='big-header'>ETA Prediction & On-Time Performance Lab</div>", unsafe_allow_html=True)

# -------------------------
# Helpers & constants
# -------------------------
DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/transportation/eta_prediction.csv"

REQUIRED_ETA_COLS = [
    "trip_id",
    "route_id",
    "driver_id",
    "distance_km",
    "avg_speed_kmph",
    "traffic_index",
    "weather",
    "stop_count",
    "stop_duration_min",
    "planned_eta_min",
    "actual_eta_min",
    "delay_min",
    "peak_hour_flag",
    "road_type",
    "turn_count",
    "driver_experience_years",
    "congestion_density",
    "historical_avg_eta_min",
    "event_disruption_flag",
    "road_quality_score"
]

NUM_COLS = [
    "distance_km",
    "avg_speed_kmph",
    "traffic_index",
    "stop_count",
    "stop_duration_min",
    "planned_eta_min",
    "actual_eta_min",
    "delay_min",
    "turn_count",
    "driver_experience_years",
    "congestion_density",
    "historical_avg_eta_min",
    "road_quality_score"
]

CAT_COLS = [
    "route_id",
    "driver_id",
    "weather",
    "peak_hour_flag",
    "road_type",
    "event_disruption_flag"
]

def to_float_series(s):
    return pd.to_numeric(s, errors="coerce")

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
    This lab converts raw trip-level data into a live control tower for <b>ETA accuracy</b> and <b>on-time performance</b>.
    It lets you see how planning assumptions (speed, distance, traffic, road type, peak hour) translate into actual arrival time,
    and where the system systematically under- or over-promises.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">What This Lab Does</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Tracks <b>planned_eta_min vs actual_eta_min</b> at trip, route & driver level<br>
        • Quantifies impact of <b>traffic_index, peak_hour_flag, events & road_type</b> on delay_min<br>
        • Builds a <b>ML ETA model</b> using historical patterns & congestion_density<br>
        • Flags <b>systemic underestimation</b> of travel time on specific routes<br>
        • Surfaces <b>drivers / routes with chronic late arrivals</b>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Improve on-time delivery % and reduce SLA penalties<br>
        • Align <b>promised ETA</b> with reality to avoid customer frustration<br>
        • Feed ETA APIs in customer portals & tracking apps<br>
        • Use <b>driver_experience_years</b> & road_quality_score to plan allocations<br>
        • Prioritize infra & partner discussions on worst-performing routes
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Operational KPIs Tracked</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>On-Time Trip %</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Avg ETA Error (min)</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Peak-Hour Delay Uplift</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>High-Risk Route & Event Impact</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Who Should Use This</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    Control-tower teams, fleet managers, planning & routing teams, last-mile ops leads, and data analysts who need a 
    <b>single ETA truth layer</b> across all trips, routes and drivers – not just intuition or driver excuses.
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# TAB 2 - IMPORTANT ATTRIBUTES
# ==========================================================
with tab2:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    required_dict = {
        "trip_id": "Unique trip identifier.",
        "route_id": "Route identifier or code used in planning.",
        "driver_id": "Driver assigned to execute the trip.",
        "distance_km": "Total planned / nominal route distance in km.",
        "avg_speed_kmph": "Observed average speed across the trip.",
        "traffic_index": "Numeric index of traffic congestion (low → high).",
        "weather": "Weather condition (Clear/Rain/Fog/etc.).",
        "stop_count": "Number of planned + unplanned stops.",
        "stop_duration_min": "Total time spent in stops (minutes).",
        "planned_eta_min": "Planned ETA (travel time in minutes).",
        "actual_eta_min": "Actual trip duration from start to end (minutes).",
        "delay_min": "actual_eta_min − planned_eta_min (positive = delay).",
        "peak_hour_flag": "Whether the trip happened in peak hours (Yes/No or 1/0).",
        "road_type": "Dominant road type (Highway/Urban/Rural/etc.).",
        "turn_count": "Number of turns / intersections on the route.",
        "driver_experience_years": "Driver’s experience in years.",
        "congestion_density": "Composite congestion measure along the route.",
        "historical_avg_eta_min": "Historical average ETA for this route / corridor.",
        "event_disruption_flag": "Whether special events (strikes, festivals, closures) affected the route.",
        "road_quality_score": "Quality score for road surface / infra (higher is better)."
    }

    dict_df = pd.DataFrame(
        [{"Attribute": k, "Description": v} for k, v in required_dict.items()]
    )

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
        dict_df.style.set_table_attributes('class="required-table"'),
        use_container_width=True
    )

    # Independent vs Dependent variables
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        independents = [
            "route_id",
            "driver_id",
            "distance_km",
            "avg_speed_kmph",
            "traffic_index",
            "weather",
            "stop_count",
            "stop_duration_min",
            "peak_hour_flag",
            "road_type",
            "turn_count",
            "driver_experience_years",
            "congestion_density",
            "historical_avg_eta_min",
            "event_disruption_flag",
            "road_quality_score"
        ]
        for v in independents:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        dependents = [
            "planned_eta_min",
            "actual_eta_min",
            "delay_min"
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
            st.success("Default ETA dataset loaded from GitHub.")
            st.dataframe(df.head(5), use_container_width=True)
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    # -------------------------
    # Upload CSV (with sample preview)
    # -------------------------
    elif mode == "Upload CSV":
        st.markdown("#### Sample structure (from standard ETA dataset)", unsafe_allow_html=True)
        try:
            sample_df = pd.read_csv(DEFAULT_URL).head(5)
            st.dataframe(sample_df, use_container_width=True)
            sample_csv = sample_df.to_csv(index=False)
            st.download_button(
                "Download Sample CSV",
                sample_csv,
                "sample_eta_prediction.csv",
                "text/csv"
            )
        except Exception as e:
            st.info(f"Sample CSV unavailable from GitHub: {e}")

        uploaded = st.file_uploader("Upload your ETA dataset", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            st.success("File uploaded.")
            st.dataframe(df.head(5), use_container_width=True)

    # -------------------------
    # Upload + column mapping
    # -------------------------
    else:
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head(5), use_container_width=True)
            st.markdown("Map your columns to the required fields:", unsafe_allow_html=True)
            mapping = {}
            for req in REQUIRED_ETA_COLS:
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
                    st.dataframe(df.head(5), use_container_width=True)

    if df is None:
        st.stop()

    # -------------------------
    # Validate required columns
    # -------------------------
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED_ETA_COLS if c not in df.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.info("Use 'Upload CSV + Column mapping' if your column names differ.")
        st.stop()

    # -------------------------
    # Coerce numeric columns
    # -------------------------
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = to_float_series(df[c])

    # Drop fully NA rows in core metrics
    df = df.dropna(subset=["planned_eta_min", "actual_eta_min", "delay_min"])

    # -------------------------
    # Filters
    # -------------------------
    st.markdown('<div class="section-title">Step 2: Filters & Preview</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        traffic_min = float(df["traffic_index"].min())
        traffic_max = float(df["traffic_index"].max())
        traffic_range = st.slider(
            "Traffic index range",
            float(round(traffic_min, 1)),
            float(round(traffic_max, 1)),
            (float(round(traffic_min, 1)), float(round(traffic_max, 1)))
        )

    with c2:
        weather_opts = sorted(df["weather"].astype(str).dropna().unique().tolist())
        sel_weather = st.multiselect(
            "Weather",
            options=weather_opts,
            default=weather_opts
        )

    with c3:
        peak_opts = sorted(df["peak_hour_flag"].astype(str).dropna().unique().tolist())
        sel_peak = st.multiselect(
            "Peak hour flag",
            options=peak_opts,
            default=peak_opts
        )

    filt = df.copy()
    # apply filters
    filt = filt[
        (filt["traffic_index"] >= traffic_range[0]) &
        (filt["traffic_index"] <= traffic_range[1])
    ]
    if sel_weather:
        filt = filt[filt["weather"].astype(str).isin(sel_weather)]
    if sel_peak:
        filt = filt[filt["peak_hour_flag"].astype(str).isin(sel_peak)]

    if filt.empty:
        st.warning("Filters removed all rows. Showing full dataset instead.")
        filt = df.copy()

    st.markdown(f"""
    <div style='padding:8px; background:#eef4ff; border-radius:8px; font-weight:600; margin-bottom:8px;'>
    Filtered Trips: {len(filt)} of {len(df)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Data Preview (first 10 rows)")
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(500), "eta_filtered_sample.csv", "Download filtered sample")

    # -------------------------
    # KPIs
    # -------------------------
    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    def safe_mean(col):
        try:
            return float(pd.to_numeric(filt[col], errors="coerce").mean())
        except Exception:
            return float("nan")

    total_trips = len(filt)
    avg_planned = safe_mean("planned_eta_min")
    avg_actual = safe_mean("actual_eta_min")
    avg_delay = safe_mean("delay_min")

    # On-time = delay_min <= 0
    if "delay_min" in filt.columns and total_trips > 0:
        ontime_pct = (filt["delay_min"] <= 0).mean() * 100
    else:
        ontime_pct = float("nan")

    # Peak vs non-peak delay uplift
    peak_mask = filt["peak_hour_flag"].astype(str).isin(["Yes", "Y", "1", "True"])
    if peak_mask.any() and (~peak_mask).any():
        peak_delay = filt.loc[peak_mask, "delay_min"].mean()
        nonpeak_delay = filt.loc[~peak_mask, "delay_min"].mean()
        peak_uplift = peak_delay - nonpeak_delay
    else:
        peak_uplift = float("nan")

    k1.metric("Total Trips", f"{total_trips:,}")
    k2.metric("Avg ETA Error (min)", f"{avg_delay:.2f}" if not math.isnan(avg_delay) else "N/A")
    k3.metric("On-Time Trips (%)", f"{ontime_pct:.1f}%" if not math.isnan(ontime_pct) else "N/A")
    k4.metric("Peak-Hour Delay Uplift (min)", f"{peak_uplift:.2f}" if not math.isnan(peak_uplift) else "N/A")

    # -------------------------
    # Charts & Diagnostics
    # -------------------------
    st.markdown('<div class="section-title">Charts & Diagnostics</div>', unsafe_allow_html=True)

    # 1) Planned vs Actual ETA scatter
    if not filt.empty:
        eta_df = filt[["trip_id", "planned_eta_min", "actual_eta_min"]].dropna()
        if not eta_df.empty:
            fig_eta = px.scatter(
                eta_df,
                x="planned_eta_min",
                y="actual_eta_min",
                hover_name="trip_id",
                trendline="ols",
                labels={
                    "planned_eta_min": "Planned ETA (min)",
                    "actual_eta_min": "Actual ETA (min)"
                },
                title="Planned vs Actual ETA"
            )
            fig_eta.update_layout(template="plotly_white")
            st.plotly_chart(fig_eta, use_container_width=True)

    # 2) Avg delay by traffic_index (binned)
    if "traffic_index" in filt.columns and "delay_min" in filt.columns:
        tmp = filt[["traffic_index", "delay_min"]].dropna()
        if not tmp.empty:
            tmp["traffic_band"] = pd.qcut(tmp["traffic_index"], q=4, duplicates="drop")

            # Convert Interval -> String because Plotly cannot serialize Interval objects
            tmp["traffic_band"] = tmp["traffic_band"].astype(str)
            
            band_df = tmp.groupby("traffic_band")["delay_min"].mean().reset_index()
            
            fig_band = px.bar(
                band_df,
                x="traffic_band",
                y="delay_min",
                text="delay_min",
                title="Average delay by traffic band"
            )
            fig_band.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            
            st.plotly_chart(fig_band, use_container_width=True)


    # 3) Avg delay by road_type
    if "road_type" in filt.columns and "delay_min" in filt.columns:
        rd = filt.groupby("road_type")["delay_min"].mean().reset_index()
        rd = rd.dropna(subset=["delay_min"])
        if not rd.empty:
            fig_rt = px.bar(
                rd,
                x="road_type",
                y="delay_min",
                text="delay_min",
                title="Average delay by road type"
            )
            fig_rt.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_rt.update_layout(template="plotly_white")
            st.plotly_chart(fig_rt, use_container_width=True)

    # -------------------------
    # ML — ETA Prediction (RandomForest)
    # -------------------------
    st.markdown('<div class="section-title">ML — ETA Regression (RandomForest)</div>', unsafe_allow_html=True)
    with st.expander("Train & evaluate model (predict actual_eta_min)", expanded=False):
        ml_df = filt.dropna(subset=["actual_eta_min"]).copy()

        feat_cols = [
            "distance_km",
            "avg_speed_kmph",
            "traffic_index",
            "stop_count",
            "stop_duration_min",
            "planned_eta_min",
            "turn_count",
            "driver_experience_years",
            "congestion_density",
            "historical_avg_eta_min",
            "road_quality_score",
            "route_id",
            "driver_id",
            "weather",
            "peak_hour_flag",
            "road_type",
            "event_disruption_flag"
        ]
        feat_cols = [c for c in feat_cols if c in ml_df.columns]

        if len(ml_df) < 40 or len(feat_cols) < 3:
            st.info("Not enough rows or features to train a reliable model (need at least ~40 rows & a few features).")
        else:
            X = ml_df[feat_cols].copy()
            y = ml_df["actual_eta_min"].astype(float)

            cat_cols_ml = [c for c in X.columns if X[c].dtype == "object"]
            num_cols_ml = [c for c in X.columns if c not in cat_cols_ml]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols_ml) if cat_cols_ml else ("noop", "passthrough", []),
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
                with st.spinner("Training RandomForest ETA model..."):
                    rf.fit(X_train, y_train)
                preds = rf.predict(X_test)

                rmse = math.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)
                st.write(f"ETA regression — RMSE: {rmse:.2f} minutes, R²: {r2:.3f}")

                res_df = pd.DataFrame({
                    "Actual_ETA_min": y_test.reset_index(drop=True),
                    "Predicted_ETA_min": preds
                })
                st.dataframe(res_df.head(20), use_container_width=True)
                download_df(res_df, "eta_predictions_sample.csv", "Download ETA prediction sample")

                # Quick single-trip prediction UI
                st.markdown("#### Quick ETA prediction (single trip input)")
                q1, q2, q3 = st.columns(3)
                with q1:
                    q_distance = st.number_input(
                        "Distance (km)",
                        min_value=0.0,
                        value=float(ml_df["distance_km"].median() if "distance_km" in ml_df.columns else 50.0)
                    )
                with q2:
                    q_speed = st.number_input(
                        "Expected avg speed (km/h)",
                        min_value=1.0,
                        value=float(ml_df["avg_speed_kmph"].median() if "avg_speed_kmph" in ml_df.columns else 40.0)
                    )
                with q3:
                    q_traffic = st.number_input(
                        "Traffic index",
                        min_value=float(ml_df["traffic_index"].min() if "traffic_index" in ml_df.columns else 0.0),
                        value=float(ml_df["traffic_index"].median() if "traffic_index" in ml_df.columns else 5.0)
                    )

                r1, r2c, r3c = st.columns(3)
                with r1:
                    q_weather = st.selectbox(
                        "Weather",
                        options=sorted(ml_df["weather"].astype(str).unique()) if "weather" in ml_df.columns else ["Clear"],
                    )
                with r2c:
                    q_peak = st.selectbox(
                        "Peak hour flag",
                        options=sorted(ml_df["peak_hour_flag"].astype(str).unique()) if "peak_hour_flag" in ml_df.columns else ["No"],
                    )
                with r3c:
                    q_road = st.selectbox(
                        "Road type",
                        options=sorted(ml_df["road_type"].astype(str).unique()) if "road_type" in ml_df.columns else ["Highway"],
                    )

                q_stop_count = st.number_input(
                    "Stop count",
                    min_value=0,
                    value=int(ml_df["stop_count"].median() if "stop_count" in ml_df.columns else 2)
                )
                q_stop_dur = st.number_input(
                    "Total stop duration (min)",
                    min_value=0.0,
                    value=float(ml_df["stop_duration_min"].median() if "stop_duration_min" in ml_df.columns else 10.0)
                )

                q_planned_eta = st.number_input(
                    "Planned ETA (min)",
                    min_value=0.0,
                    value=float(ml_df["planned_eta_min"].median() if "planned_eta_min" in ml_df.columns else 90.0)
                )

                if st.button("Predict ETA for this trip"):
                    # build a single row using median/defaults where needed
                    base = ml_df.iloc[0]  # just to steal available categories
                    row_dict = {}

                    for col in feat_cols:
                        if col == "distance_km":
                            row_dict[col] = q_distance
                        elif col == "avg_speed_kmph":
                            row_dict[col] = q_speed
                        elif col == "traffic_index":
                            row_dict[col] = q_traffic
                        elif col == "stop_count":
                            row_dict[col] = q_stop_count
                        elif col == "stop_duration_min":
                            row_dict[col] = q_stop_dur
                        elif col == "planned_eta_min":
                            row_dict[col] = q_planned_eta
                        elif col == "weather":
                            row_dict[col] = q_weather
                        elif col == "peak_hour_flag":
                            row_dict[col] = q_peak
                        elif col == "road_type":
                            row_dict[col] = q_road
                        else:
                            # fallback to median / mode from ml_df
                            if ml_df[col].dtype == "object":
                                row_dict[col] = ml_df[col].mode().iloc[0]
                            else:
                                row_dict[col] = float(ml_df[col].median())

                    row = pd.DataFrame([row_dict])
                    try:
                        row_t = preprocessor.transform(row)
                        pred_eta = rf.predict(row_t)[0]
                        st.success(f"Predicted Actual ETA: {pred_eta:.2f} minutes")
                    except Exception as e:
                        st.error("Prediction failed: " + str(e))

    # -------------------------
    # Automated Insights
    # -------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)

    insights_rows = []

    # 1) Route with highest average delay
    if "route_id" in filt.columns and "delay_min" in filt.columns and not filt.empty:
        rdelay = filt.groupby("route_id")["delay_min"].mean().reset_index()
        rdelay = rdelay.dropna(subset=["delay_min"])
        if not rdelay.empty:
            worst_route = rdelay.sort_values("delay_min", ascending=False).iloc[0]
            insights_rows.append({
                "Insight": "Route with highest average delay",
                "Entity": worst_route["route_id"],
                "Metric": f"{worst_route['delay_min']:.2f} min avg delay",
                "Suggested Action": "Re-calibrate planned_eta_min and review route design / traffic windows."
            })

    # 2) Weather with worst ETA impact
    if "weather" in filt.columns and "delay_min" in filt.columns and not filt.empty:
        wdelay = filt.groupby("weather")["delay_min"].mean().reset_index()
        wdelay = wdelay.dropna(subset=["delay_min"])
        if not wdelay.empty:
            worst_weather = wdelay.sort_values("delay_min", ascending=False).iloc[0]
            insights_rows.append({
                "Insight": "Worst weather for ETA",
                "Entity": worst_weather["weather"],
                "Metric": f"{worst_weather['delay_min']:.2f} min avg delay",
                "Suggested Action": "Apply ETA buffer or avoid SLAs in this weather band."
            })

    # 3) Peak-hour penalty
    if "peak_hour_flag" in filt.columns and "delay_min" in filt.columns and not filt.empty:
        peak_grp = filt.groupby(filt["peak_hour_flag"].astype(str))["delay_min"].mean().reset_index()
        if len(peak_grp) >= 2:
            try:
                peak_val = peak_grp.loc[peak_grp["peak_hour_flag"].astype(str).isin(["Yes", "Y", "1", "True"]), "delay_min"].mean()
                nonpeak_val = peak_grp.loc[~peak_grp["peak_hour_flag"].astype(str).isin(["Yes", "Y", "1", "True"]), "delay_min"].mean()
                uplift = peak_val - nonpeak_val
                insights_rows.append({
                    "Insight": "Peak-hour delay uplift",
                    "Entity": "Peak vs Non-Peak",
                    "Metric": f"{uplift:.2f} min extra delay",
                    "Suggested Action": "Add peak-hour specific ETA multiplier or restrict tight SLAs."
                })
            except Exception:
                pass

    # 4) Road type with best reliability (lowest absolute delay)
    if "road_type" in filt.columns and "delay_min" in filt.columns and not filt.empty:
        rabs = filt.copy()
        rabs["abs_delay"] = rabs["delay_min"].abs()
        rrel = rabs.groupby("road_type")["abs_delay"].mean().reset_index()
        rrel = rrel.dropna(subset=["abs_delay"])
        if not rrel.empty:
            best_rt = rrel.sort_values("abs_delay", ascending=True).iloc[0]
            insights_rows.append({
                "Insight": "Most reliable road type",
                "Entity": best_rt["road_type"],
                "Metric": f"{best_rt['abs_delay']:.2f} min avg absolute error",
                "Suggested Action": "Prioritize this road type in route design when trade-offs are possible."
            })

    # 5) Driver experience effect (simple correlation-style read)
    if "driver_experience_years" in filt.columns and "delay_min" in filt.columns and not filt.empty:
        try:
            exp_bin = pd.qcut(filt["driver_experience_years"], q=3, duplicates="drop")
            tmp = pd.DataFrame({
                "exp_bin": exp_bin,
                "delay_min": filt["delay_min"]
            }).dropna()
            if not tmp.empty:
                exp_eff = tmp.groupby("exp_bin")["delay_min"].mean().reset_index()
                best_bin = exp_eff.sort_values("delay_min").iloc[0]
                insights_rows.append({
                    "Insight": "Experience band with lowest delay",
                    "Entity": str(best_bin["exp_bin"]),
                    "Metric": f"{best_bin['delay_min']:.2f} min avg delay",
                    "Suggested Action": "Allocate complex / high-traffic routes to this experience band."
                })
        except Exception:
            pass

    if not insights_rows:
        st.info("No insights generated for the current filter selection.")
    else:
        insights_df = pd.DataFrame(insights_rows)
        st.dataframe(insights_df, use_container_width=True)
        download_df(insights_df, "eta_insights.csv", "Download insights")
