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
st.set_page_config(page_title="Route Optimization & Fleet Efficiency Lab", layout="wide")

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
# Global CSS (Marketing Lab style)
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

st.markdown("<div class='big-header'>Route Optimization & Fleet Efficiency Lab</div>", unsafe_allow_html=True)

# -------------------------
# Helpers & constants
# -------------------------
ROUTE_REQUIRED_COLS = [
    "trip_id",
    "vehicle_id",
    "driver_id",
    "planned_route",
    "actual_route",
    "start_time",
    "end_time",
    "planned_distance_km",
    "actual_distance_km",
    "traffic_level",
    "weather_condition",
    "fuel_used_liters",
    "delay_minutes",
    "deviation_flags",
    "avg_speed_kmph",
    "max_speed_kmph",
    "stoppage_count",
    "total_stoppage_minutes",
    "route_risk_score",
    "fuel_efficiency_km_per_liter",
    "gps_signal_loss_minutes",
    "payload_weight_kg",
    "route_toll_cost"
]

DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/transportation/route_optimization.csv"


def ensure_datetime(df, col):
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            pass
    return df


def to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def download_df(df, filename, label="Download CSV"):
    buf = BytesIO()
    buf.write(df.to_csv(index=False).encode("utf-8"))
    buf.seek(0)
    st.download_button(label, buf, file_name=filename, mime="text/csv")


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
    <b>What this workspace does</b><br><br>
    This lab centralizes <b>trip-level telematics & route data</b> and turns it into a 
    control tower for route performance, fuel efficiency, and driver operations.<br><br>
    It helps you answer:
    <ul>
      <li>Which routes are consistently late, and what’s driving the delay?</li>
      <li>Which vehicles & drivers are fuel efficient vs wasteful?</li>
      <li>Where exactly do we lose time – traffic, stoppages, or bad routing?</li>
      <li>How risky is each trip in terms of speed, deviations, and GPS blind spots?</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Route vs actual comparison (distance, travel time, delay)<br>
        • Delay diagnostics by <b>route, driver, vehicle, traffic & weather</b><br>
        • Fleet-level fuel efficiency benchmarking (km/l, cost per km)<br>
        • Route risk scoring using stoppages, GPS loss, speed metrics<br>
        • ML-based delay prediction using historical patterns<br>
        • Export-ready tables for daily ops reviews & monthly MIS
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Reduce fuel cost by shifting volume to efficient routes & vehicles<br>
        • Improve SLA adherence & customer OTIF by targeting chronic delay routes<br>
        • Shorten planning cycles by using <b>real performance, not assumptions</b><br>
        • Build strong, data-backed cases for network redesign & capex decisions<br>
        • Strengthen vendor / 3PL governance with objective performance metrics
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">KPIs (when data is loaded)</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>On-time Trip %</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Avg Delay (min)</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Fleet Fuel Efficiency (km/l)</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>High-Risk Trip Count</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Who Should Use</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    • Fleet & transport managers<br>
    • Network design / logistics excellence teams<br>
    • 3PL / carrier partners managing SLAs<br>
    • Data & analytics teams creating a single version of truth for fleet KPIs
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# TAB 2 - IMPORTANT ATTRIBUTES
# ==========================================================
with tab2:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    required_dict = {
        "trip_id": "Unique trip identifier.",
        "vehicle_id": "Unique vehicle identifier for the fleet.",
        "driver_id": "Unique driver identifier.",
        "planned_route": "Planned route code / name.",
        "actual_route": "Actual route taken (code / name).",
        "start_time": "Trip start timestamp.",
        "end_time": "Trip end timestamp.",
        "planned_distance_km": "Planned distance of the trip in km.",
        "actual_distance_km": "Actual distance travelled in km.",
        "traffic_level": "Traffic level (e.g., Low / Medium / High).",
        "weather_condition": "Weather (Clear, Rain, Fog, etc.).",
        "fuel_used_liters": "Total fuel consumed on the trip.",
        "delay_minutes": "Delay vs planned schedule (min, negative = early).",
        "deviation_flags": "Flag or code indicating route deviations.",
        "avg_speed_kmph": "Average speed during trip.",
        "max_speed_kmph": "Maximum speed recorded on trip.",
        "stoppage_count": "Count of stoppages during trip.",
        "total_stoppage_minutes": "Total minutes of stoppage.",
        "route_risk_score": "Composite risk score for trip / route.",
        "fuel_efficiency_km_per_liter": "Actual km per liter for this trip.",
        "gps_signal_loss_minutes": "Minutes with GPS / telematics signal loss.",
        "payload_weight_kg": "Payload weight in kg.",
        "route_toll_cost": "Total toll cost on this route."
    }

    req_df = pd.DataFrame(
        [{"Attribute": k, "Description": v} for k, v in required_dict.items()]
    )

    st.markdown("""
        <style>
            .required-table th {
                background:#ffffff !important;
                color:#000 !important;
                font-size:22px !important;
                border-bottom:2px solid #000 !important;
            }
            .required-table td {
                color:#000 !important;
                font-size:21px !important;
                padding:10px !important;
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
            "planned_route",
            "vehicle_id",
            "driver_id",
            "planned_distance_km",
            "traffic_level",
            "weather_condition",
            "payload_weight_kg",
            "route_toll_cost",
            "stoppage_count",
            "total_stoppage_minutes",
            "avg_speed_kmph",
            "max_speed_kmph",
            "gps_signal_loss_minutes",
            "deviation_flags"
        ]
        for v in independents:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with c2:
        st.markown = st.markdown  # just to avoid typo issues later
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        dependents = [
            "delay_minutes",
            "fuel_used_liters",
            "fuel_efficiency_km_per_liter",
            "route_risk_score"
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

    # 1) DEFAULT DATASET
    if mode == "Default dataset":
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            st.success("Default dataset loaded from GitHub URL")
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    # 2) UPLOAD CSV (with sample preview from same URL)
    elif mode == "Upload CSV":
        st.markdown("#### Download sample CSV (for format reference)")
        try:
            sample_df = pd.read_csv(DEFAULT_URL).head(5)
            st.download_button(
                "Download Sample CSV",
                sample_df.to_csv(index=False),
                "route_optimization_sample.csv",
                "text/csv"
            )
        except Exception:
            st.info("Sample CSV unavailable from GitHub.")

        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            st.success("File uploaded")
            st.dataframe(df.head(), use_container_width=True)

    # 3) UPLOAD CSV + MAPPING
    else:
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head(), use_container_width=True)
            st.markdown("Map your columns to the required fields:")
            mapping = {}
            for req in ROUTE_REQUIRED_COLS:
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
                    st.success("Mapping applied")
                    st.dataframe(df.head(), use_container_width=True)

    if df is None:
        st.stop()

    # -------------------------
    # Validate required columns
    # -------------------------
    df.columns = df.columns.str.strip()
    missing = [c for c in ROUTE_REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.info("Use 'Upload CSV + Column mapping' to align your dataset.")
        st.stop()

    # -------------------------
    # Type coercion & derived metrics
    # -------------------------
    df = df.copy()
    df = ensure_datetime(df, "start_time")
    df = ensure_datetime(df, "end_time")

    num_cols = [
        "planned_distance_km",
        "actual_distance_km",
        "fuel_used_liters",
        "delay_minutes",
        "avg_speed_kmph",
        "max_speed_kmph",
        "stoppage_count",
        "total_stoppage_minutes",
        "route_risk_score",
        "fuel_efficiency_km_per_liter",
        "gps_signal_loss_minutes",
        "payload_weight_kg",
        "route_toll_cost"
    ]
    df = to_numeric(df, num_cols)

    # Derived
    if "start_time" in df.columns and "end_time" in df.columns:
        df["trip_duration_min"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60.0
    else:
        df["trip_duration_min"] = np.nan

    if "delay_minutes" in df.columns:
        df["on_time_flag"] = df["delay_minutes"] <= 0
    else:
        df["on_time_flag"] = np.nan

    if "planned_distance_km" in df.columns and "actual_distance_km" in df.columns:
        df["distance_deviation_km"] = df["actual_distance_km"] - df["planned_distance_km"]

    if "fuel_used_liters" in df.columns and "actual_distance_km" in df.columns:
        df["fuel_per_km"] = np.where(
            df["actual_distance_km"] > 0,
            df["fuel_used_liters"] / df["actual_distance_km"],
            np.nan
        )

    # -------------------------
    # Filters & preview
    # -------------------------
    st.markdown('<div class="section-title">Filters & Preview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])

    vehicles = sorted(df["vehicle_id"].dropna().unique().tolist())
    drivers = sorted(df["driver_id"].dropna().unique().tolist())
    traffic_levels = sorted(df["traffic_level"].dropna().unique().tolist())

    with c1:
        sel_vehicles = st.multiselect("Vehicle", options=vehicles, default=vehicles[:5] if vehicles else [])
    with c2:
        sel_drivers = st.multiselect("Driver", options=drivers, default=drivers[:5] if drivers else [])
    with c3:
        sel_traffic = st.multiselect("Traffic level", options=traffic_levels, default=traffic_levels)
    with c4:
        try:
            if "start_time" in df.columns and not df["start_time"].isna().all():
                min_d = df["start_time"].min().date()
                max_d = df["start_time"].max().date()
                date_range = st.date_input("Start date range", value=(min_d, max_d))
            else:
                date_range = st.date_input("Start date range")
        except Exception:
            date_range = st.date_input("Start date range")

    filt = df.copy()
    if sel_vehicles:
        filt = filt[filt["vehicle_id"].isin(sel_vehicles)]
    if sel_drivers:
        filt = filt[filt["driver_id"].isin(sel_drivers)]
    if sel_traffic:
        filt = filt[filt["traffic_level"].isin(sel_traffic)]

    try:
        if date_range and len(date_range) == 2 and "start_time" in filt.columns:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            filt = filt[(filt["start_time"] >= start) & (filt["start_time"] <= end + pd.Timedelta(days=1))]
    except Exception:
        pass

    st.markdown('<div class="section-title">Preview (first 10 rows)</div>', unsafe_allow_html=True)
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(500), "route_filtered_preview.csv", label="Download filtered preview (top 500)")

    # -------------------------
    # KPIs
    # -------------------------
    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    if len(filt) > 0:
        on_time_pct = float(filt["on_time_flag"].mean() * 100) if "on_time_flag" in filt.columns else np.nan
        avg_delay = float(filt["delay_minutes"][filt["delay_minutes"].notna()].mean()) if "delay_minutes" in filt.columns else np.nan
        fleet_fe = float(filt["fuel_efficiency_km_per_liter"][filt["fuel_efficiency_km_per_liter"].notna()].mean()) if "fuel_efficiency_km_per_liter" in filt.columns else np.nan

        if "route_risk_score" in filt.columns and filt["route_risk_score"].notna().sum() > 0:
            threshold = filt["route_risk_score"].quantile(0.75)
            high_risk_count = int((filt["route_risk_score"] >= threshold).sum())
        else:
            high_risk_count = 0
    else:
        on_time_pct = avg_delay = fleet_fe = np.nan
        high_risk_count = 0

    k1.metric("On-time Trip %", f"{on_time_pct:.1f}%" if not math.isnan(on_time_pct) else "N/A")
    k2.metric("Average Delay (min)", f"{avg_delay:.1f}" if not math.isnan(avg_delay) else "N/A")
    k3.metric("Avg Fleet Fuel Efficiency (km/l)", f"{fleet_fe:.2f}" if not math.isnan(fleet_fe) else "N/A")
    k4.metric("High-Risk Trip Count", high_risk_count)

    # -------------------------
    # Charts & EDA
    # -------------------------
    st.markdown('<div class="section-title">Charts & EDA</div>', unsafe_allow_html=True)

    # Route-level delay
    if "planned_route" in filt.columns and "delay_minutes" in filt.columns:
        route_delay = (
            filt.groupby("planned_route")["delay_minutes"]
            .mean()
            .reset_index()
            .sort_values("delay_minutes", ascending=False)
            .head(20)
        )
        if not route_delay.empty:
            fig = px.bar(
                route_delay,
                x="planned_route",
                y="delay_minutes",
                text="delay_minutes",
                template="plotly_white",
                title="Average delay by planned route"
            )
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig.update_layout(xaxis_title="Planned route", yaxis_title="Avg delay (min)")
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    # Vehicle fuel efficiency
    if "vehicle_id" in filt.columns and "fuel_efficiency_km_per_liter" in filt.columns:
        veh_fe = (
            filt.groupby("vehicle_id")["fuel_efficiency_km_per_liter"]
            .mean()
            .reset_index()
            .sort_values("fuel_efficiency_km_per_liter", ascending=False)
            .head(20)
        )
        if not veh_fe.empty:
            fig2 = px.bar(
                veh_fe,
                x="vehicle_id",
                y="fuel_efficiency_km_per_liter",
                text="fuel_efficiency_km_per_liter",
                template="plotly_white",
                title="Avg fuel efficiency by vehicle"
            )
            fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig2.update_layout(xaxis_title="Vehicle", yaxis_title="Km per liter")
            fig2.update_xaxes(tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)

    # Time-series of average delay
    if "start_time" in filt.columns and "delay_minutes" in filt.columns and not filt["start_time"].isna().all():
        ts = (
            filt.dropna(subset=["start_time"])
            .groupby(filt["start_time"].dt.date)["delay_minutes"]
            .mean()
            .reset_index()
            .rename(columns={"start_time": "date"})
        )
        ts["start_time"] = pd.to_datetime(ts["start_time"], errors="coerce")
        ts["date"] = ts["start_time"].dt.date
        ts = ts.rename(columns={ts.columns[0]: "date"})
        ts = ts.sort_values("date")
        if not ts.empty:
            fig3 = px.line(
                ts,
                x="date",
                y="delay_minutes",
                template="plotly_white",
                title="Average delay trend over time"
            )
            fig3.update_layout(xaxis_title="Date", yaxis_title="Avg delay (min)")
            st.plotly_chart(fig3, use_container_width=True)

    # -------------------------
    # ML: Delay prediction (RandomForest)
    # -------------------------
    st.markdown('<div class="section-title">ML — Delay Prediction (RandomForest)</div>', unsafe_allow_html=True)
    ml_exp = st.expander("Train delay prediction model (requires >= 60 rows)", expanded=False)

    with ml_exp:
        ml_df = filt.copy()
        target_col = "delay_minutes"
        feature_candidates = [
            "planned_distance_km",
            "actual_distance_km",
            "traffic_level",
            "weather_condition",
            "stoppage_count",
            "total_stoppage_minutes",
            "avg_speed_kmph",
            "max_speed_kmph",
            "payload_weight_kg",
            "route_risk_score"
        ]

        ml_df = ml_df.dropna(subset=[target_col])
        feat_cols = [c for c in feature_candidates if c in ml_df.columns]

        if len(ml_df) < 60 or len(feat_cols) < 2:
            st.info("Not enough rows or features to train a robust model (need ≥ 60 rows & at least 2 features).")
        else:
            X = ml_df[feat_cols].copy()
            y = ml_df[target_col].astype(float)

            cat_cols = [c for c in X.columns if X[c].dtype == "object"]
            num_cols = [c for c in X.columns if c not in cat_cols]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols) if cat_cols else ("noop", "passthrough", []),
                    ("num", StandardScaler(), num_cols) if num_cols else ("noop2", "passthrough", [])
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
                rf = RandomForestRegressor(n_estimators=150, random_state=42)
                rf.fit(X_train, y_train)
                preds_test = rf.predict(X_test)

                rmse = math.sqrt(mean_squared_error(y_test, preds_test))
                r2 = r2_score(y_test, preds_test)
                st.write(f"Delay prediction — RMSE: {rmse:.2f} min, R²: {r2:.3f}")

                # Feature importance (if we can reconstruct names)
                try:
                    # get cat feature names if exist
                    cat_names = []
                    if "cat" in preprocessor.named_transformers_:
                        ohe = preprocessor.named_transformers_["cat"]
                        if hasattr(ohe, "get_feature_names_out") and cat_cols:
                            cat_names = list(ohe.get_feature_names_out(cat_cols))
                    feature_names = cat_names + num_cols
                    fi = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
                    st.markdown("Top feature importances")
                    st.dataframe(
                        fi.reset_index().rename(columns={"index": "feature", 0: "importance"}).head(20),
                        use_container_width=True
                    )
                except Exception:
                    st.info("Feature importance could not be computed cleanly.")

                # Predict delay for full filtered dataset
                try:
                    full_X_t = preprocessor.transform(X)
                    full_preds = rf.predict(full_X_t)
                    pred_df = ml_df.copy()
                    pred_df["predicted_delay_minutes"] = full_preds
                    st.markdown("Sample of predicted delays (filtered dataset)")
                    st.dataframe(
                        pred_df[["trip_id", "vehicle_id", "driver_id", "delay_minutes", "predicted_delay_minutes"]].head(20),
                        use_container_width=True
                    )
                    download_df(pred_df, "route_with_predicted_delays.csv", label="Download dataset with predicted delays")
                except Exception as e:
                    st.info("Could not generate full prediction table: " + str(e))

    # -------------------------
    # Automated Insights
    # -------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)
    with st.expander("Generate automated insights from filtered data", expanded=True):
        insights = []

        if len(filt) == 0:
            st.info("No data in current filter selection.")
        else:
            # Worst & best routes by delay
            if "planned_route" in filt.columns and "delay_minutes" in filt.columns:
                route_agg = (
                    filt.groupby("planned_route")["delay_minutes"]
                    .mean()
                    .reset_index()
                    .dropna()
                )
                if not route_agg.empty:
                    worst = route_agg.sort_values("delay_minutes", ascending=False).head(1).iloc[0]
                    best = route_agg.sort_values("delay_minutes", ascending=True).head(1).iloc[0]
                    insights.append(
                        f"Route with highest average delay: **{worst['planned_route']}** "
                        f"(≈ {worst['delay_minutes']:.1f} min late)."
                    )
                    insights.append(
                        f"Route with best timing: **{best['planned_route']}** "
                        f"(≈ {best['delay_minutes']:.1f} min)."
                    )

            # Vehicle fuel efficiency
            if "vehicle_id" in filt.columns and "fuel_efficiency_km_per_liter" in filt.columns:
                veh = (
                    filt.groupby("vehicle_id")["fuel_efficiency_km_per_liter"]
                    .mean()
                    .reset_index()
                    .dropna()
                )
                if not veh.empty:
                    best_veh = veh.sort_values("fuel_efficiency_km_per_liter", ascending=False).head(1).iloc[0]
                    worst_veh = veh.sort_values("fuel_efficiency_km_per_liter", ascending=True).head(1).iloc[0]
                    insights.append(
                        f"Most fuel-efficient vehicle: **{best_veh['vehicle_id']}** "
                        f"(≈ {best_veh['fuel_efficiency_km_per_liter']:.2f} km/l)."
                    )
                    insights.append(
                        f"Least fuel-efficient vehicle: **{worst_veh['vehicle_id']}** "
                        f"(≈ {worst_veh['fuel_efficiency_km_per_liter']:.2f} km/l) — candidate for maintenance or route change."
                    )

            # Traffic impact
            if "traffic_level" in filt.columns and "delay_minutes" in filt.columns:
                traf = (
                    filt.groupby("traffic_level")["delay_minutes"]
                    .mean()
                    .reset_index()
                    .dropna()
                )
                if not traf.empty:
                    worst_traf = traf.sort_values("delay_minutes", ascending=False).head(1).iloc[0]
                    insights.append(
                        f"Worst traffic band for delay: **{worst_traf['traffic_level']}** "
                        f"(avg delay ≈ {worst_traf['delay_minutes']:.1f} min)."
                    )

            # Stoppage hotspots
            if "planned_route" in filt.columns and "total_stoppage_minutes" in filt.columns:
                stop = (
                    filt.groupby("planned_route")["total_stoppage_minutes"]
                    .mean()
                    .reset_index()
                    .dropna()
                )
                if not stop.empty:
                    hot = stop.sort_values("total_stoppage_minutes", ascending=False).head(1).iloc[0]
                    insights.append(
                        f"Route with highest stoppage time: **{hot['planned_route']}** "
                        f"(≈ {hot['total_stoppage_minutes']:.1f} min per trip)."
                    )

            if not insights:
                st.info("No strong insights generated for the selected filters.")
            else:
                for i, ins in enumerate(insights, 1):
                    st.markdown(f"**Insight {i}:** {ins}")

    # -------------------------
    # Export filtered dataset
    # -------------------------
    st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
    st.markdown("Download the current filtered dataset with engineered metrics.")
    download_df(filt, "route_filtered_full.csv", label="Download filtered dataset")
