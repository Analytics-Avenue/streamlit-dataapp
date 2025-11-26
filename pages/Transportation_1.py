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

st.markdown("<div class='big-header'>Route Optimization & Fleet Efficiency Lab</div>", unsafe_allow_html=True)

# -------------------------
# Helpers & constants
# -------------------------
DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/transportation/route_optimization.csv"

REQUIRED_ROUTE_COLS = [
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
    This lab centralizes trip-level telemetry and operational data to understand how efficiently routes are executed vs how they were planned.
    It helps fleet teams reduce delays, cut fuel wastage, and control risk across vehicles, drivers, and operating conditions.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">What This Lab Does</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Compares <b>planned vs actual</b> distance, time & route<br>
        • Quantifies impact of <b>traffic, weather, stoppages</b> on delay<br>
        • Tracks <b>fuel efficiency</b> across vehicles & payloads<br>
        • Surfaces <b>high-risk trips</b> with elevated route_risk_score<br>
        • Identifies chronic <b>deviations, signal loss & toll-heavy routes</b>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Reduce <b>average delay_minutes</b> and improve SLA compliance<br>
        • Lower fuel costs via <b>better fuel_efficiency_km_per_liter</b><br>
        • Standardize routes & flag <b>unnecessary detours</b><br>
        • Improve <b>driver allocation</b> to complex routes<br>
        • Build a defensible data trail for <b>customer disputes</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Fleet KPIs Tracked</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>On-time vs Delayed Trips</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Avg Fuel Efficiency (km/l)</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Avg Route Risk Score</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Deviation & Stoppage Burden</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Who Should Use This</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    Fleet managers, transport heads, logistics planners, control-tower teams, and data/ops analysts who need a single, 
    standardized workspace to audit trip performance, optimize routes, and manage risk at a vehicle + driver + route level.
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# TAB 2 - IMPORTANT ATTRIBUTES
# ==========================================================
with tab2:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    required_dict = {
        "trip_id": "Unique identifier for each trip.",
        "vehicle_id": "Vehicle assigned to the trip.",
        "driver_id": "Driver who executed the trip.",
        "planned_route": "Planned route identifier / hash.",
        "actual_route": "Actual route identifier / hash.",
        "start_time": "Trip start time (HH:MM or timestamp).",
        "end_time": "Trip end time (HH:MM or timestamp).",
        "planned_distance_km": "Route distance as per planning (in km).",
        "actual_distance_km": "Actual distance driven (in km).",
        "traffic_level": "Qualitative traffic indicator (Low/Medium/High).",
        "weather_condition": "Weather condition during trip (Clear/Rain/Fog/etc.).",
        "fuel_used_liters": "Total fuel consumed during the trip (in liters).",
        "delay_minutes": "Delay vs expected schedule (in minutes).",
        "deviation_flags": "Reason/flag for route deviation (None/Traffic Jam/Accident/etc.).",
        "avg_speed_kmph": "Average speed across the trip (km/h).",
        "max_speed_kmph": "Maximum speed recorded (km/h).",
        "stoppage_count": "Number of stoppages across the trip.",
        "total_stoppage_minutes": "Total time spent in stoppages (minutes).",
        "route_risk_score": "Composite risk score for the route (0–1 / 0–1+).",
        "fuel_efficiency_km_per_liter": "Effective km covered per liter of fuel.",
        "gps_signal_loss_minutes": "Total minutes of GPS signal loss.",
        "payload_weight_kg": "Payload weight carried (kg).",
        "route_toll_cost": "Total tolls paid on this route (in local currency)."
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
            "stoppage_count",
            "total_stoppage_minutes",
            "gps_signal_loss_minutes",
            "payload_weight_kg",
            "route_toll_cost",
            "avg_speed_kmph",
            "max_speed_kmph"
        ]
        for v in independents:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        dependents = [
            "delay_minutes",
            "route_risk_score",
            "fuel_used_liters",
            "fuel_efficiency_km_per_liter"
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
            st.dataframe(df.head(5), use_container_width=True)
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    # -------------------------
    # Upload CSV (with sample preview)
    # -------------------------
    elif mode == "Upload CSV":
        st.markdown("#### Sample structure (from standard Route Optimization dataset)", unsafe_allow_html=True)
        try:
            sample_df = pd.read_csv(DEFAULT_URL).head(5)
            st.dataframe(sample_df, use_container_width=True)
            sample_csv = sample_df.to_csv(index=False)
            st.download_button(
                "Download Sample CSV",
                sample_csv,
                "sample_route_optimization.csv",
                "text/csv"
            )
        except Exception as e:
            st.info(f"Sample CSV unavailable from GitHub: {e}")

        uploaded = st.file_uploader("Upload your Route Optimization dataset", type=["csv"])
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
            for req in REQUIRED_ROUTE_COLS:
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
    missing = [c for c in REQUIRED_ROUTE_COLS if c not in df.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.info("Use 'Upload CSV + Column mapping' if your column names differ.")
        st.stop()

    # -------------------------
    # Coerce numeric columns
    # -------------------------
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
    for c in num_cols:
        if c in df.columns:
            df[c] = to_float_series(df[c])

    # --------------------------
    # FIXED FILTER SYSTEM FOR ROUTE OPTIMIZATION
    # --------------------------
    
    st.markdown("### Step 2: Filters")
    
    # Store defaults when df loads
    if "default_values" not in st.session_state:
        st.session_state.default_values = {
            "traffic": df["traffic_level"].unique().tolist(),
            "weather": df["weather_condition"].unique().tolist(),
            "delay_min": int(df["delay_minutes"].min()),
            "delay_max": int(df["delay_minutes"].max())
        }
    
    # Reset Filters Button
    if st.button("Reset Filters"):
        st.session_state.selected_traffic = st.session_state.default_values["traffic"]
        st.session_state.selected_weather = st.session_state.default_values["weather"]
        st.session_state.selected_delay = (
            st.session_state.default_values["delay_min"],
            st.session_state.default_values["delay_max"]
        )
    
    # Build filter controls
    traffic = st.multiselect(
        "Traffic Level",
        df["traffic_level"].unique(),
        default=st.session_state.get("selected_traffic", df["traffic_level"].unique().tolist())
    )
    
    weather = st.multiselect(
        "Weather Condition",
        df["weather_condition"].unique(),
        default=st.session_state.get("selected_weather", df["weather_condition"].unique().tolist())
    )
    
    delay = st.slider(
        "Delay Minutes",
        int(df["delay_minutes"].min()),
        int(df["delay_minutes"].max()),
        st.session_state.get(
            "selected_delay",
            (int(df["delay_minutes"].min()), int(df["delay_minutes"].max()))
        )
    )
    
    # Apply filters
    filt = df.copy()
    if traffic:
        filt = filt[filt["traffic_level"].isin(traffic)]
    if weather:
        filt = filt[filt["weather_condition"].isin(weather)]
    
    filt = filt[
        (filt["delay_minutes"] >= delay[0]) &
        (filt["delay_minutes"] <= delay[1])
    ]
    
    # Auto-restore if filtered empty
    if filt.empty:
        st.warning("Filters removed all rows. Restoring defaults.")
        filt = df.copy()
        traffic = df["traffic_level"].unique().tolist()
        weather = df["weather_condition"].unique().tolist()
        delay = (int(df["delay_minutes"].min()), int(df["delay_minutes"].max()))
    
    # Save current selections
    st.session_state.selected_traffic = traffic
    st.session_state.selected_weather = weather
    st.session_state.selected_delay = delay
    
    # Filter count display
    st.markdown(f"""
    <div style='padding:8px; background:#eef4ff; border-radius:8px; font-weight:600;'>
    Filtered Rows: {len(filt)} of {len(df)}
    </div>
    """, unsafe_allow_html=True)
    
    # Data preview
    st.markdown("### Data Preview")
    st.dataframe(filt.head(), use_container_width=True)


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
    avg_delay = safe_mean("delay_minutes")
    avg_fuel_eff = safe_mean("fuel_efficiency_km_per_liter")
    avg_risk = safe_mean("route_risk_score")
    deviation_rate = (
        (filt["deviation_flags"].astype(str) != "None").mean()
        if "deviation_flags" in filt.columns and len(filt) > 0
        else 0
    )

    k1.metric("Total Trips", f"{total_trips:,}")
    k2.metric("Avg Delay (min)", f"{avg_delay:.2f}" if not math.isnan(avg_delay) else "N/A")
    k3.metric("Avg Fuel Efficiency (km/l)", f"{avg_fuel_eff:.2f}" if not math.isnan(avg_fuel_eff) else "N/A")
    k4.metric("Deviation Trip %", f"{deviation_rate*100:.1f}%" if total_trips > 0 else "N/A")

    # -------------------------
    # Charts & EDA
    # -------------------------
    st.markdown('<div class="section-title">Charts & Diagnostics</div>', unsafe_allow_html=True)

    # 1) Planned vs Actual distance
    if not filt.empty:
        dist_df = filt[["trip_id", "planned_distance_km", "actual_distance_km"]].dropna()
        if not dist_df.empty:
            dist_df["distance_variance_km"] = dist_df["actual_distance_km"] - dist_df["planned_distance_km"]
            fig_dist = px.scatter(
                dist_df,
                x="planned_distance_km",
                y="actual_distance_km",
                hover_name="trip_id",
                trendline="ols",
                labels={
                    "planned_distance_km": "Planned distance (km)",
                    "actual_distance_km": "Actual distance (km)"
                },
                title="Planned vs Actual Distance"
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    # 2) Avg delay by traffic_level
    if "traffic_level" in filt.columns and "delay_minutes" in filt.columns:
        tl = filt.groupby("traffic_level")["delay_minutes"].mean().reset_index()
        if not tl.empty:
            fig_tl = px.bar(
                tl,
                x="traffic_level",
                y="delay_minutes",
                text="delay_minutes",
                title="Average delay by traffic level"
            )
            fig_tl.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            st.plotly_chart(fig_tl, use_container_width=True)

    # 3) Fuel efficiency by vehicle
    if "vehicle_id" in filt.columns and "fuel_efficiency_km_per_liter" in filt.columns:
        fe = filt.groupby("vehicle_id")["fuel_efficiency_km_per_liter"].mean().reset_index()
        if not fe.empty:
            fe = fe.sort_values("fuel_efficiency_km_per_liter", ascending=False).head(20)
            fig_fe = px.bar(
                fe,
                x="vehicle_id",
                y="fuel_efficiency_km_per_liter",
                text="fuel_efficiency_km_per_liter",
                title="Fuel efficiency by vehicle (top 20)",
            )
            fig_fe.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_fe.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_fe, use_container_width=True)

    # -------------------------
    # ML — Delay prediction (RandomForest)
    # -------------------------
    st.markdown('<div class="section-title">ML — Delay Minutes Regression (RandomForest)</div>', unsafe_allow_html=True)
    with st.expander("Train & evaluate model (requires >=80 rows with non-null delay_minutes)", expanded=False):
        ml_df = filt.dropna(subset=["delay_minutes"]).copy()

        feat_cols = [
            "planned_distance_km",
            "actual_distance_km",
            "fuel_used_liters",
            "avg_speed_kmph",
            "max_speed_kmph",
            "stoppage_count",
            "total_stoppage_minutes",
            "gps_signal_loss_minutes",
            "payload_weight_kg",
            "route_toll_cost",
            "traffic_level",
            "weather_condition"
        ]
        feat_cols = [c for c in feat_cols if c in ml_df.columns]

        if len(ml_df) < 10 or len(feat_cols) < 2:
            st.info("Not enough rows or features to train a reliable model (need at least ~80 rows).")
        else:
            X = ml_df[feat_cols].copy()
            y = ml_df["delay_minutes"].astype(float)

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

                rmse = math.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)
                st.write(f"Delay regression — RMSE: {rmse:.2f} minutes, R²: {r2:.3f}")

                # Build result table
                res_df = pd.DataFrame({
                    "Actual_delay_minutes": y_test.reset_index(drop=True),
                    "Predicted_delay_minutes": preds
                })
                st.dataframe(res_df.head(20), use_container_width=True)
                download_df(res_df, "delay_predictions.csv", "Download delay prediction sample")

    # -------------------------
    # Automated Insights
    # -------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)

    insights_rows = []

    # 1) Vehicle with highest average delay
    if "vehicle_id" in filt.columns and "delay_minutes" in filt.columns and not filt.empty:
        vdelay = filt.groupby("vehicle_id")["delay_minutes"].mean().reset_index()
        vdelay = vdelay.dropna(subset=["delay_minutes"])
        if not vdelay.empty:
            worst = vdelay.sort_values("delay_minutes", ascending=False).iloc[0]
            insights_rows.append({
                "Insight": "Vehicle with highest average delay",
                "Entity": worst["vehicle_id"],
                "Metric": f"{worst['delay_minutes']:.2f} min",
                "Action": "Audit route planning & driver behavior for this vehicle."
            })

    # 2) Vehicle with best fuel efficiency
    if "vehicle_id" in filt.columns and "fuel_efficiency_km_per_liter" in filt.columns and not filt.empty:
        vfe = filt.groupby("vehicle_id")["fuel_efficiency_km_per_liter"].mean().reset_index()
        vfe = vfe.dropna(subset=["fuel_efficiency_km_per_liter"])
        if not vfe.empty:
            best_fe = vfe.sort_values("fuel_efficiency_km_per_liter", ascending=False).iloc[0]
            insights_rows.append({
                "Insight": "Most fuel-efficient vehicle",
                "Entity": best_fe["vehicle_id"],
                "Metric": f"{best_fe['fuel_efficiency_km_per_liter']:.2f} km/l",
                "Action": "Use this vehicle profile as a benchmark for similar payloads."
            })

    # 3) Traffic condition causing highest delay
    if "traffic_level" in filt.columns and "delay_minutes" in filt.columns and not filt.empty:
        tld = filt.groupby("traffic_level")["delay_minutes"].mean().reset_index()
        tld = tld.dropna(subset=["delay_minutes"])
        if not tld.empty:
            worst_t = tld.sort_values("delay_minutes", ascending=False).iloc[0]
            insights_rows.append({
                "Insight": "Worst traffic band for delay",
                "Entity": worst_t["traffic_level"],
                "Metric": f"{worst_t['delay_minutes']:.2f} min avg delay",
                "Action": "Re-time or reroute trips in this traffic band."
            })

    # 4) Weather condition with highest risk score
    if "weather_condition" in filt.columns and "route_risk_score" in filt.columns and not filt.empty:
        wr = filt.groupby("weather_condition")["route_risk_score"].mean().reset_index()
        wr = wr.dropna(subset=["route_risk_score"])
        if not wr.empty:
            worst_w = wr.sort_values("route_risk_score", ascending=False).iloc[0]
            insights_rows.append({
                "Insight": "Highest route risk by weather",
                "Entity": worst_w["weather_condition"],
                "Metric": f"Risk score {worst_w['route_risk_score']:.2f}",
                "Action": "Apply stricter speed & stoppage controls in this weather."
            })

    # 5) Deviation burden
    if "deviation_flags" in filt.columns and not filt.empty:
        dev_share = (filt["deviation_flags"].astype(str) != "None").mean()
        insights_rows.append({
            "Insight": "Route deviation share",
            "Entity": "All filtered trips",
            "Metric": f"{dev_share*100:.1f} % trips with deviation",
            "Action": "Deep dive into frequent deviation flags (Traffic Jam, Accident, etc.)."
        })

    if not insights_rows:
        st.info("No insights generated for the current filter selection.")
    else:
        insights_df = pd.DataFrame(insights_rows)
        st.dataframe(insights_df, use_container_width=True)
        download_df(insights_df, "route_optimization_insights.csv", "Download insights")
