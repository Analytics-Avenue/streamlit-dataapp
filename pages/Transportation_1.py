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
# Helpers
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

DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/main/datasets/transportation/route_optimization.csv"


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

# -------------------------
# Overview Tab
# -------------------------
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <b>What this workspace does</b><br><br>
    This lab centralizes your <b>trip-level telematics data</b> and turns it into 
    a practical control tower for route performance, fuel efficiency and driver operations.<br><br>
    It sits on top of your GPS / TMS / telematics exports and answers questions like:
    <ul>
      <li>Which routes are consistently running late and why?</li>
      <li>Which vehicles and drivers burn the most fuel per km?</li>
      <li>Where do we lose time: traffic, stoppages, or bad routing?</li>
      <li>What is the operational risk on each route today?</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Trip-level route vs actual analysis (distance, time, stoppages)<br>
        • Delay diagnostics by <b>traffic, driver, route & vehicle</b><br>
        • Fuel efficiency benchmarking across vehicles & payloads<br>
        • Route risk scoring using operational signals (speeding, stoppages, GPS loss)<br>
        • ML-based <b>delay prediction</b> from historical patterns<br>
        • Export-ready tables for ops reviews & management reports
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Lower fuel cost by shifting volume to efficient vehicles & routes<br>
        • Reduce late deliveries by fixing <b>chronic bottleneck routes</b><br>
        • Improve SLA adherence through better driver deployment<br>
        • Shorter planning cycles using real performance, not assumptions<br>
        • Build a data-backed case for network redesign & capex decisions
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">KPIs (Examples when data is loaded)</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>On-time Trip %</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Avg Delay (min)</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Fleet Fuel Efficiency (km/l)</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>High-Risk Route Count</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Who Should Use This</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    • Fleet & transportation managers who need <b>daily control</b> on delays and cost<br>
    • Network design / planning teams running scenario analysis<br>
    • 3PL / logistics providers proving performance to clients<br>
    • Data & analytics teams standardizing transport performance metrics
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# Important Attributes Tab
# -------------------------
with tab2:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    required_dict = {
        "trip_id": "Unique trip identifier.",
        "vehicle_id": "Unique vehicle identifier for the fleet.",
        "driver_id": "Unique driver identifier.",
        "planned_route": "Planned route / path code or name.",
        "actual_route": "Actual route taken (code, polyline ID, or description).",
        "start_time": "Trip start timestamp.",
        "end_time": "Trip end timestamp.",
        "planned_distance_km": "Planned trip distance in km.",
        "actual_distance_km": "Actual trip distance in km.",
        "traffic_level": "Traffic condition category (e.g., Low/Medium/High).",
        "weather_condition": "Weather condition on trip (e.g., Clear/Rain/Fog).",
        "fuel_used_liters": "Total fuel consumed during trip in liters.",
        "delay_minutes": "Delay vs planned schedule in minutes (negative = early).",
        "deviation_flags": "Flag/indicator for route deviations / violations.",
        "avg_speed_kmph": "Average speed across the trip.",
        "max_speed_kmph": "Maximum speed recorded on the trip.",
        "stoppage_count": "Count of stoppages during the trip.",
        "total_stoppage_minutes": "Total time spent in stoppages.",
        "route_risk_score": "Composite risk score for trip/route (higher = riskier).",
        "fuel_efficiency_km_per_liter": "Actual km per liter for this trip.",
        "gps_signal_loss_minutes": "Minutes with GPS / telematics signal loss.",
        "payload_weight_kg": "Total payload carried in kg.",
        "route_toll_cost": "Total toll cost incurred on the trip."
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
                font-size

