import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
st.set_page_config(page_title="Battery Analytics & Storage Optimization Lab", layout="wide")

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

st.markdown("<div class='big-header'>Battery Analytics & Storage Optimization Lab</div>", unsafe_allow_html=True)

# -------------------------
# Helpers & constants
# -------------------------
DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/Solar/battery_analytics.csv"

REQUIRED_BATTERY_COLS = [
    "battery_id",
    "timestamp",
    "state_of_charge_pct",
    "state_of_health_pct",
    "charging_power_kw",
    "discharging_power_kw",
    "battery_temperature_c",
    "ambient_temp_c",
    "cycle_count",
    "voltage_v",
    "current_a",
    "internal_resistance_milliohm",
    "charge_c_rate",
    "discharge_c_rate",
    "battery_capacity_kwh",
    "remaining_useful_life_days",
    "cooling_system_status",
    "battery_pack_id",
    "room_id",
    "degradation_risk_flag",
    "thermal_runaway_risk_score",
    "charging_mode"
]

NUMERIC_COLS = [
    "state_of_charge_pct",
    "state_of_health_pct",
    "charging_power_kw",
    "discharging_power_kw",
    "battery_temperature_c",
    "ambient_temp_c",
    "cycle_count",
    "voltage_v",
    "current_a",
    "internal_resistance_milliohm",
    "charge_c_rate",
    "discharge_c_rate",
    "battery_capacity_kwh",
    "remaining_useful_life_days",
    "thermal_runaway_risk_score"
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
    This lab is built to monitor, optimize, and extend the life of <b>energy storage systems</b> in solar plants and microgrids.
    It unifies pack-level telemetry (temperature, SoC, SoH, current, voltage) with risk and degradation indicators to drive 
    smarter charge–discharge strategies and asset planning.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">What This Lab Does</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Tracks <b>State of Charge (SoC)</b> & <b>State of Health (SoH)</b> across batteries & rooms<br>
        • Monitors <b>thermal behavior</b> and flags thermal runaway risk patterns<br>
        • Links <b>cycle_count</b>, C-rates, and temperatures to degradation risk<br>
        • Surfaces <b>high-risk packs & rooms</b> for inspection<br>
        • Supports <b>Remaining Useful Life (RUL)</b) estimation with ML
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Reduce unplanned battery failures and <b>outage risk</b><br>
        • Extend <b>battery lifespan</b> via better charge–discharge strategy<br>
        • Lower <b>replacement CAPEX</b> by deferring premature swaps<br>
        • Improve <b>round-trip efficiency</b> for solar + storage systems<br>
        • Build an auditable history for <b>OEM warranty & performance disputes</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Key Storage KPIs Tracked</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Average State of Health (SoH)</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Average Remaining Useful Life (days)</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>High Degradation Risk Packs</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>High Thermal Runaway Risk (≥ threshold)</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Who Should Use This</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    Plant heads, O&amp;M teams, energy storage OEMs, microgrid operators, and analytics teams who need a unified workspace
    to track <b>battery health</b>, <b>thermal safety</b>, and <b>storage ROI</b> across packs, rooms, and sites.
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# TAB 2 - IMPORTANT ATTRIBUTES
# ==========================================================
with tab2:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    required_dict = {
        "battery_id": "Unique identifier for each battery unit.",
        "timestamp": "Timestamp of the telemetry reading.",
        "state_of_charge_pct": "Battery state of charge (% of capacity currently stored).",
        "state_of_health_pct": "Battery health indicator (% of original capacity / performance).",
        "charging_power_kw": "Active charging power at this timestamp (kW).",
        "discharging_power_kw": "Active discharging power at this timestamp (kW).",
        "battery_temperature_c": "Measured battery temperature (°C).",
        "ambient_temp_c": "Ambient room temperature (°C).",
        "cycle_count": "Number of charge/discharge cycles completed.",
        "voltage_v": "Battery voltage (V).",
        "current_a": "Battery current (A).",
        "internal_resistance_milliohm": "Estimated internal resistance (milliohms).",
        "charge_c_rate": "Charge C-rate (relative rate vs nominal capacity).",
        "discharge_c_rate": "Discharge C-rate.",
        "battery_capacity_kwh": "Nominal energy capacity of the battery (kWh).",
        "remaining_useful_life_days": "Estimated remaining useful life in days.",
        "cooling_system_status": "Cooling state (Normal/Warning/Failure).",
        "battery_pack_id": "Group identifier for aggregated pack.",
        "room_id": "Room / container / rack identifier.",
        "degradation_risk_flag": "Binary/flag if the battery is in high degradation risk state.",
        "thermal_runaway_risk_score": "Risk score for thermal runaway (higher = riskier).",
        "charging_mode": "Operating mode (Idle/Charging/Discharging/Balancing/etc.)."
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
            "battery_id",
            "battery_pack_id",
            "room_id",
            "timestamp",
            "state_of_charge_pct",
            "charging_power_kw",
            "discharging_power_kw",
            "battery_temperature_c",
            "ambient_temp_c",
            "cycle_count",
            "voltage_v",
            "current_a",
            "internal_resistance_milliohm",
            "charge_c_rate",
            "discharge_c_rate",
            "battery_capacity_kwh",
            "cooling_system_status",
            "charging_mode"
        ]
        for v in independents:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        dependents = [
            "state_of_health_pct",
            "remaining_useful_life_days",
            "degradation_risk_flag",
            "thermal_runaway_risk_score"
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
        st.markdown("#### Sample structure (from standard Battery Analytics dataset)", unsafe_allow_html=True)
        try:
            sample_df = pd.read_csv(DEFAULT_URL).head(5)
            st.dataframe(sample_df, use_container_width=True)
            sample_csv = sample_df.to_csv(index=False)
            st.download_button(
                "Download Sample CSV",
                sample_csv,
                "sample_battery_analytics.csv",
                "text/csv"
            )
        except Exception as e:
            st.info(f"Sample CSV unavailable from GitHub: {e}")

        uploaded = st.file_uploader("Upload your Battery Analytics dataset", type=["csv"])
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
            for req in REQUIRED_BATTERY_COLS:
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
    missing = [c for c in REQUIRED_BATTERY_COLS if c not in df.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.info("Use 'Upload CSV + Column mapping' if your column names differ.")
        st.stop()

    # -------------------------
    # Coerce numeric & timestamp
    # -------------------------
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = to_float_series(df[c])

    # -------------------------
    # Filters
    # -------------------------
    st.markdown('<div class="section-title">Step 2: Filters & Preview</div>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        packs = sorted(df["battery_pack_id"].dropna().unique().tolist()) if "battery_pack_id" in df.columns else []
        sel_packs = st.multiselect("Battery Pack", packs, default=packs[:5] if packs else [])
    with f2:
        rooms = sorted(df["room_id"].dropna().unique().tolist()) if "room_id" in df.columns else []
        sel_rooms = st.multiselect("Room", rooms, default=rooms[:5] if rooms else [])
    with f3:
        modes = sorted(df["charging_mode"].dropna().unique().tolist()) if "charging_mode" in df.columns else []
        sel_modes = st.multiselect("Charging Mode", modes, default=modes[:4] if modes else [])
    with f4:
        if "state_of_health_pct" in df.columns:
            soh_min = float(df["state_of_health_pct"].min())
            soh_max = float(df["state_of_health_pct"].max())
            soh_range = st.slider(
                "State of Health (%)",
                float(round(soh_min, 1)),
                float(round(soh_max, 1)),
                (float(round(soh_min, 1)), float(round(soh_max, 1)))
            )
        else:
            soh_range = None

    filt = df.copy()

    if sel_packs:
        filt = filt[filt["battery_pack_id"].isin(sel_packs)]
    if sel_rooms:
        filt = filt[filt["room_id"].isin(sel_rooms)]
    if sel_modes:
        filt = filt[filt["charging_mode"].isin(sel_modes)]
    if soh_range and "state_of_health_pct" in filt.columns:
        filt = filt[
            (filt["state_of_health_pct"] >= soh_range[0]) &
            (filt["state_of_health_pct"] <= soh_range[1])
        ]

    if filt.empty:
        st.warning("Filters removed all rows. Showing full dataset.")
        filt = df.copy()

    st.markdown(f"""
    <div style='padding:8px; background:#eef4ff; border-radius:8px; font-weight:600;'>
    Filtered Rows: {len(filt)} of {len(df)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Preview (first 10 rows)")
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(500), "battery_filtered_preview.csv", "Download filtered preview (first 500 rows)")

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

    def safe_pct(condition_series):
        if len(condition_series) == 0:
            return 0.0
        return float(condition_series.mean()) * 100

    avg_soh = safe_mean("state_of_health_pct")
    avg_rul = safe_mean("remaining_useful_life_days")
    high_deg_pct = safe_pct(filt["degradation_risk_flag"].astype(float)) if "degradation_risk_flag" in filt.columns else 0.0

    if "thermal_runaway_risk_score" in filt.columns:
        thr_threshold = float(np.nanpercentile(filt["thermal_runaway_risk_score"], 90))
        high_thr_pct = safe_pct(filt["thermal_runaway_risk_score"] >= thr_threshold)
    else:
        high_thr_pct = 0.0

    k1.metric("Avg SoH (%)", f"{avg_soh:.2f}" if not math.isnan(avg_soh) else "N/A")
    k2.metric("Avg Remaining Life (days)", f"{avg_rul:.1f}" if not math.isnan(avg_rul) else "N/A")
    k3.metric("High Degradation Risk (%)", f"{high_deg_pct:.1f}%")
    k4.metric("High Thermal Runaway Risk (%)", f"{high_thr_pct:.1f}%")

    # -------------------------
    # Charts & Diagnostics
    # -------------------------
    st.markdown('<div class="section-title">Charts & Diagnostics</div>', unsafe_allow_html=True)

    # 1) SoC vs SoH scatter
    if "state_of_charge_pct" in filt.columns and "state_of_health_pct" in filt.columns:
        fig_sc = px.scatter(
            filt.dropna(subset=["state_of_charge_pct", "state_of_health_pct"]),
            x="state_of_charge_pct",
            y="state_of_health_pct",
            color="degradation_risk_flag" if "degradation_risk_flag" in filt.columns else None,
            hover_data=["battery_id", "battery_pack_id", "room_id"] if "battery_pack_id" in filt.columns else ["battery_id"],
            labels={
                "state_of_charge_pct": "State of Charge (%)",
                "state_of_health_pct": "State of Health (%)"
            },
            title="SoC vs SoH by battery"
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # 2) Avg SoH by pack
    if "battery_pack_id" in filt.columns and "state_of_health_pct" in filt.columns:
        soh_pack = filt.groupby("battery_pack_id")["state_of_health_pct"].mean().reset_index()
        if not soh_pack.empty:
            soh_pack = soh_pack.sort_values("state_of_health_pct", ascending=False)
            fig_pack = px.bar(
                soh_pack,
                x="battery_pack_id",
                y="state_of_health_pct",
                text="state_of_health_pct",
                title="Average SoH by battery pack"
            )
            fig_pack.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_pack.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_pack, use_container_width=True)

    # 3) Thermal risk vs battery temperature
    if "battery_temperature_c" in filt.columns and "thermal_runaway_risk_score" in filt.columns:
        fig_thr = px.scatter(
            filt.dropna(subset=["battery_temperature_c", "thermal_runaway_risk_score"]),
            x="battery_temperature_c",
            y="thermal_runaway_risk_score",
            color="room_id" if "room_id" in filt.columns else None,
            hover_data=["battery_id", "battery_pack_id"] if "battery_pack_id" in filt.columns else ["battery_id"],
            labels={
                "battery_temperature_c": "Battery Temperature (°C)",
                "thermal_runaway_risk_score": "Thermal Runaway Risk Score"
            },
            title="Thermal runaway risk vs battery temperature"
        )
        st.plotly_chart(fig_thr, use_container_width=True)

    # 4) Time-series SoC or SoH (daily)
    if "timestamp" in filt.columns and not filt["timestamp"].isna().all():
        ts = filt.dropna(subset=["timestamp"]).copy()
        ts["date"] = ts["timestamp"].dt.date
        if "state_of_charge_pct" in ts.columns:
            ts_agg = ts.groupby("date")["state_of_charge_pct"].mean().reset_index()
            fig_ts = px.line(
                ts_agg,
                x="date",
                y="state_of_charge_pct",
                labels={"date": "Date", "state_of_charge_pct": "Avg SoC (%)"},
                title="Daily average State of Charge"
            )
            st.plotly_chart(fig_ts, use_container_width=True)

    # -------------------------
    # ML — Remaining Useful Life Regression
    # -------------------------
    st.markdown('<div class="section-title">ML — Remaining Useful Life Regression (RandomForest)</div>', unsafe_allow_html=True)
    with st.expander("Train & evaluate model (requires >=80 rows with non-null remaining_useful_life_days)", expanded=False):
        if "remaining_useful_life_days" not in filt.columns:
            st.info("Column 'remaining_useful_life_days' not available in dataset.")
        else:
            ml_df = filt.dropna(subset=["remaining_useful_life_days"]).copy()

            feat_cols = [
                "state_of_charge_pct",
                "state_of_health_pct",
                "battery_temperature_c",
                "ambient_temp_c",
                "cycle_count",
                "voltage_v",
                "current_a",
                "internal_resistance_milliohm",
                "charge_c_rate",
                "discharge_c_rate",
                "battery_capacity_kwh",
                "cooling_system_status",
                "charging_mode",
                "room_id",
                "battery_pack_id"
            ]
            feat_cols = [c for c in feat_cols if c in ml_df.columns]

            if len(ml_df) < 80 or len(feat_cols) < 3:
                st.info("Not enough rows or features to train a robust model (need at least ~80 rows and a few features).")
            else:
                X = ml_df[feat_cols].copy()
                y = ml_df["remaining_useful_life_days"].astype(float)

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
                    st.write(f"RUL regression — RMSE: {rmse:.2f} days, R²: {r2:.3f}")

                    res_df = pd.DataFrame({
                        "Actual_RUL_days": y_test.reset_index(drop=True),
                        "Predicted_RUL_days": preds
                    })
                    st.dataframe(res_df.head(20), use_container_width=True)
                    download_df(res_df, "battery_rul_predictions.csv", "Download RUL prediction sample")

    # -------------------------
    # Automated Insights
    # -------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)

    insights_rows = []

    # 1) Pack with lowest average SoH
    if "battery_pack_id" in filt.columns and "state_of_health_pct" in filt.columns and not filt.empty:
        pack_soh = filt.groupby("battery_pack_id")["state_of_health_pct"].mean().reset_index()
        pack_soh = pack_soh.dropna(subset=["state_of_health_pct"])
        if not pack_soh.empty:
            worst_pack = pack_soh.sort_values("state_of_health_pct", ascending=True).iloc[0]
            insights_rows.append({
                "Insight": "Weakest battery pack (SoH)",
                "Entity": worst_pack["battery_pack_id"],
                "Metric": f"{worst_pack['state_of_health_pct']:.1f}% avg SoH",
                "Action": "Schedule inspection, calibration, or partial replacement for this pack."
            })

    # 2) Room with highest thermal risk
    if "room_id" in filt.columns and "thermal_runaway_risk_score" in filt.columns and not filt.empty:
        room_thr = filt.groupby("room_id")["thermal_runaway_risk_score"].mean().reset_index()
        room_thr = room_thr.dropna(subset=["thermal_runaway_risk_score"])
        if not room_thr.empty:
            riskiest_room = room_thr.sort_values("thermal_runaway_risk_score", ascending=False).iloc[0]
            insights_rows.append({
                "Insight": "Room with highest thermal risk",
                "Entity": riskiest_room["room_id"],
                "Metric": f"Risk score {riskiest_room['thermal_runaway_risk_score']:.2f}",
                "Action": "Audit cooling, airflow and loading in this room."
            })

    # 3) Cooling system status issue rate
    if "cooling_system_status" in filt.columns and not filt.empty:
        issue_pct = safe_pct(filt["cooling_system_status"].astype(str).isin(["Warning", "Failure"]))
        insights_rows.append({
            "Insight": "Cooling system issue rate",
            "Entity": "All filtered batteries",
            "Metric": f"{issue_pct:.1f}% in Warning/Failure",
            "Action": "Prioritize maintenance for rooms with repeated cooling alerts."
        })

    # 4) High degradation risk share
    if "degradation_risk_flag" in filt.columns and not filt.empty:
        deg_pct = safe_pct(filt["degradation_risk_flag"].astype(float))
        insights_rows.append({
            "Insight": "High degradation risk share",
            "Entity": "All filtered batteries",
            "Metric": f"{deg_pct:.1f}% flagged high risk",
            "Action": "Consider derating, rebalancing cycles, or targeted replacements."
        })

    # 5) Pack with best health & life
    if "battery_pack_id" in filt.columns and "state_of_health_pct" in filt.columns and "remaining_useful_life_days" in filt.columns and not filt.empty:
        combo = filt.groupby("battery_pack_id").agg({
            "state_of_health_pct": "mean",
            "remaining_useful_life_days": "mean"
        }).reset_index()
        combo = combo.dropna()
        if not combo.empty:
            combo["score"] = combo["state_of_health_pct"].rank(pct=True) + combo["remaining_useful_life_days"].rank(pct=True)
            best_pack = combo.sort_values("score", ascending=False).iloc[0]
            insights_rows.append({
                "Insight": "Best performing battery pack",
                "Entity": best_pack["battery_pack_id"],
                "Metric": f"SoH {best_pack['state_of_health_pct']:.1f}%, RUL {best_pack['remaining_useful_life_days']:.1f} days",
                "Action": "Use this pack profile as reference for fleet-wide tuning."
            })

    if not insights_rows:
        st.info("No insights generated for the current filter selection.")
    else:
        insights_df = pd.DataFrame(insights_rows)
        st.dataframe(insights_df, use_container_width=True)
        download_df(insights_df, "battery_analytics_insights.csv", "Download insights")

    # -------------------------
    # Export full filtered dataset
    # -------------------------
    st.markdown('<div class="section-title">Export Filtered Dataset</div>', unsafe_allow_html=True)
    download_df(filt, "battery_filtered_full.csv", "Download full filtered dataset")

