import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
import math
import warnings

warnings.filterwarnings("ignore")

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Predictive Maintenance & Breakdown Risk Lab", layout="wide")

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

st.markdown("<div class='big-header'>Predictive Maintenance & Breakdown Risk Lab</div>", unsafe_allow_html=True)

# -------------------------
# Helpers & constants
# -------------------------
DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/transportation/predictive_maintenance.csv"

REQUIRED_MAINT_COLS = [
    "vehicle_id",
    "engine_temp_c",
    "vibration_level",
    "rpm",
    "oil_pressure",
    "coolant_level_pct",
    "last_service_km",
    "current_km",
    "fault_code",
    "breakdown_within_30_days",
    "battery_voltage",
    "engine_load_pct",
    "fuel_rate_lph",
    "tire_pressure_front_left",
    "tire_pressure_front_right",
    "tire_pressure_back_left",
    "tire_pressure_back_right",
    "ambient_temp_c",
    "vehicle_age_years",
    "service_alert_flag",
    "breakdown_history_count"
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
    This lab brings together telemetry, sensor health, and maintenance history to predict which vehicles are likely to
    <b>break down in the next 30 days</b>. It helps maintenance and fleet teams prioritize workshop time, parts, and
    replacement planning instead of reacting to breakdowns on the road.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">What This Lab Does</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Tracks <b>engine, vibration, pressure & temperature</b> patterns for each vehicle<br>
        • Combines <b>last_service_km, current_km</b> & breakdown_history_count into a maintenance profile<br>
        • Flags vehicles with <b>service_alert_flag</b> and rising risk signals<br>
        • Learns which <b>fault_code patterns</b> often precede a breakdown<br>
        • Produces an ML-based <b>breakdown_within_30_days</b> risk estimate
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Reduce <b>unplanned breakdowns</b> and roadside failures<br>
        • Improve <b>vehicle uptime</b> and asset utilization<br>
        • Lower <b>emergency repair</b> and towing costs<br>
        • Better <b>workshop scheduling</b> and parts planning<br>
        • Enable data-backed <b>scrap / refurbish / replace</b> decisions
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Key Maintenance KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Breakdown Rate (30 days)</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Average Engine Temperature</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Average Vibration Level</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>High-Risk Vehicle Share</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Who Should Use This</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    Maintenance heads, fleet managers, control room engineers, OEM after-sales teams, and analytics teams responsible for
    building a proactive maintenance program instead of firefighting breakdowns.
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# TAB 2 - IMPORTANT ATTRIBUTES
# ==========================================================
with tab2:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    required_dict = {
        "vehicle_id": "Unique ID of the vehicle.",
        "engine_temp_c": "Engine temperature in °C.",
        "vibration_level": "Vibration index from sensors (unitless or scaled).",
        "rpm": "Engine revolutions per minute.",
        "oil_pressure": "Oil pressure reading from engine.",
        "coolant_level_pct": "Coolant level as percentage of ideal capacity.",
        "last_service_km": "Odometer reading at the last service.",
        "current_km": "Current odometer reading.",
        "fault_code": "Diagnostic fault code or category.",
        "breakdown_within_30_days": "Target flag (1 = breakdown in next 30 days, 0 = no breakdown).",
        "battery_voltage": "Battery voltage level.",
        "engine_load_pct": "Engine load as a percentage.",
        "fuel_rate_lph": "Fuel consumption rate (liters per hour).",
        "tire_pressure_front_left": "Front-left tire pressure.",
        "tire_pressure_front_right": "Front-right tire pressure.",
        "tire_pressure_back_left": "Rear-left tire pressure.",
        "tire_pressure_back_right": "Rear-right tire pressure.",
        "ambient_temp_c": "Ambient temperature around the vehicle (°C).",
        "vehicle_age_years": "Age of vehicle in years.",
        "service_alert_flag": "Flag if onboard systems have raised a service alert.",
        "breakdown_history_count": "Historical breakdown count for this vehicle."
    }

    req_df = pd.DataFrame(
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
        req_df.style.set_table_attributes('class="required-table"'),
        use_container_width=True
    )

    # Independent vs Dependent variables
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        independents = [
            "engine_temp_c",
            "vibration_level",
            "rpm",
            "oil_pressure",
            "coolant_level_pct",
            "last_service_km",
            "current_km",
            "fault_code",
            "battery_voltage",
            "engine_load_pct",
            "fuel_rate_lph",
            "tire_pressure_front_left",
            "tire_pressure_front_right",
            "tire_pressure_back_left",
            "tire_pressure_back_right",
            "ambient_temp_c",
            "vehicle_age_years",
            "service_alert_flag",
            "breakdown_history_count",
        ]
        for v in independents:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        dependents = [
            "breakdown_within_30_days"
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
            st.success("Default predictive maintenance dataset loaded from GitHub.")
            st.dataframe(df.head(5), use_container_width=True)
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    # -------------------------
    # Upload CSV (with sample preview)
    # -------------------------
    elif mode == "Upload CSV":
        st.markdown("#### Sample structure (from standard Predictive Maintenance dataset)", unsafe_allow_html=True)
        try:
            sample_df = pd.read_csv(DEFAULT_URL).head(5)
            st.dataframe(sample_df, use_container_width=True)
            sample_csv = sample_df.to_csv(index=False)
            st.download_button(
                "Download Sample CSV",
                sample_csv,
                "sample_predictive_maintenance.csv",
                "text/csv"
            )
        except Exception as e:
            st.info(f"Sample CSV unavailable from GitHub: {e}")

        uploaded = st.file_uploader("Upload your Predictive Maintenance dataset", type=["csv"])
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
            for req in REQUIRED_MAINT_COLS:
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
    missing = [c for c in REQUIRED_MAINT_COLS if c not in df.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.info("Use 'Upload CSV + Column mapping' if your column names differ.")
        st.stop()

    # -------------------------
    # Coerce numeric columns & target
    # -------------------------
    num_cols = [
        "engine_temp_c",
        "vibration_level",
        "rpm",
        "oil_pressure",
        "coolant_level_pct",
        "last_service_km",
        "current_km",
        "battery_voltage",
        "engine_load_pct",
        "fuel_rate_lph",
        "tire_pressure_front_left",
        "tire_pressure_front_right",
        "tire_pressure_back_left",
        "tire_pressure_back_right",
        "ambient_temp_c",
        "vehicle_age_years",
        "breakdown_history_count",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = to_float_series(df[c])

    # target as numeric 0/1
    df["breakdown_within_30_days"] = pd.to_numeric(df["breakdown_within_30_days"], errors="coerce").fillna(0).round().astype(int)

    # -------------------------
    # Filters
    # -------------------------
    st.markdown('<div class="section-title">Step 2: Filters & Preview</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        vehicles = sorted(df["vehicle_id"].astype(str).unique())
        sel_veh = st.multiselect("Vehicle ID", vehicles, default=vehicles[:10] if len(vehicles) > 10 else vehicles)
    with f2:
        fault_codes = sorted(df["fault_code"].astype(str).unique())
        sel_fault = st.multiselect("Fault Code", fault_codes, default=fault_codes)
    with f3:
        try:
            min_age = float(df["vehicle_age_years"].min())
            max_age = float(df["vehicle_age_years"].max())
        except Exception:
            min_age, max_age = 0.0, 20.0
        age_range = st.slider("Vehicle Age (years)", float(min_age), float(max_age), (float(min_age), float(max_age)))

    filt = df.copy()
    if sel_veh:
        filt = filt[filt["vehicle_id"].astype(str).isin(sel_veh)]
    if sel_fault:
        filt = filt[filt["fault_code"].astype(str).isin(sel_fault)]
    filt = filt[(filt["vehicle_age_years"] >= age_range[0]) & (filt["vehicle_age_years"] <= age_range[1])]

    if filt.empty:
        st.warning("Filters removed all rows. Showing full dataset instead.")
        filt = df.copy()

    st.markdown(f"""
    <div style='padding:8px; background:#eef4ff; border-radius:8px; font-weight:600; margin-bottom:8px;'>
    Filtered Rows: {len(filt)} of {len(df)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Data Preview")
    st.dataframe(filt.head(), use_container_width=True)
    download_df(filt.head(500), "predictive_maintenance_filtered_sample.csv", "Download filtered sample CSV")

    # -------------------------
    # KPIs
    # -------------------------
    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    def safe_mean(column):
        try:
            return float(pd.to_numeric(filt[column], errors="coerce").mean())
        except Exception:
            return float("nan")

    total_records = len(filt)
    breakdown_rate = filt["breakdown_within_30_days"].mean() if total_records > 0 else 0.0
    avg_engine_temp = safe_mean("engine_temp_c")
    avg_vibration = safe_mean("vibration_level")
    avg_age = safe_mean("vehicle_age_years")

    k1.metric("Total Vehicle Records", f"{total_records:,}")
    k2.metric("Breakdown Rate (30 days)", f"{breakdown_rate*100:.1f}%" if total_records > 0 else "N/A")
    k3.metric("Avg Engine Temp (°C)", f"{avg_engine_temp:.1f}" if not math.isnan(avg_engine_temp) else "N/A")
    k4.metric("Avg Vehicle Age (years)", f"{avg_age:.1f}" if not math.isnan(avg_age) else "N/A")

    # -------------------------
    # Charts & Diagnostics
    # -------------------------
    st.markdown('<div class="section-title">Charts & Diagnostics</div>', unsafe_allow_html=True)

    # 1) Breakdown rate by vehicle_age_years band
    if "vehicle_age_years" in filt.columns:
        age_bins = pd.cut(filt["vehicle_age_years"], bins=[0,2,5,8,12,20,50], include_lowest=True)
        age_break = filt.groupby(age_bins)["breakdown_within_30_days"].mean().reset_index(name="breakdown_rate")
        age_break = age_break.dropna(subset=["breakdown_rate"])
        if not age_break.empty:
            fig_age = px.bar(
                age_break,
                x="vehicle_age_years",
                y="breakdown_rate",
                text="breakdown_rate",
                labels={"vehicle_age_years": "Vehicle age band", "breakdown_rate": "Breakdown rate"},
                title="Breakdown rate by vehicle age band"
            )
            fig_age.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            st.plotly_chart(fig_age, use_container_width=True)


    # 2) Breakdown rate by fault_code
    if "fault_code" in filt.columns:
        fc = (
            filt.groupby("fault_code")["breakdown_within_30_days"]
            .mean()
            .reset_index(name="breakdown_rate")
        )
        fc = fc.sort_values("breakdown_rate", ascending=False).head(15)
        if not fc.empty:
            fig_fc = px.bar(
                fc,
                x="fault_code",
                y="breakdown_rate",
                text="breakdown_rate",
                title="Breakdown rate by fault code (top 15)"
            )
            fig_fc.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_fc.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_fc, use_container_width=True)

    # 3) Engine temperature vs vibration colored by breakdown flag
    if "engine_temp_c" in filt.columns and "vibration_level" in filt.columns:
        fig_scatter = px.scatter(
            filt,
            x="engine_temp_c",
            y="vibration_level",
            color="breakdown_within_30_days",
            labels={"breakdown_within_30_days": "Breakdown (1=yes,0=no)"},
            title="Engine temp vs Vibration (colored by breakdown)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # -------------------------
    # ML — Breakdown risk classification (RandomForest)
    # -------------------------
    st.markdown('<div class="section-title">ML — Breakdown Within 30 Days (RandomForest Classifier)</div>', unsafe_allow_html=True)
    with st.expander("Train & evaluate model (requires >= 80 rows with valid target)", expanded=False):
        ml_df = filt.dropna(subset=["breakdown_within_30_days"]).copy()

        feat_cols = [
            "engine_temp_c",
            "vibration_level",
            "rpm",
            "oil_pressure",
            "coolant_level_pct",
            "last_service_km",
            "current_km",
            "battery_voltage",
            "engine_load_pct",
            "fuel_rate_lph",
            "tire_pressure_front_left",
            "tire_pressure_front_right",
            "tire_pressure_back_left",
            "tire_pressure_back_right",
            "ambient_temp_c",
            "vehicle_age_years",
            "breakdown_history_count",
            "fault_code",
            "service_alert_flag"
        ]
        feat_cols = [c for c in feat_cols if c in ml_df.columns]

        if len(ml_df) < 80 or len(feat_cols) < 3:
            st.info("Not enough rows or features to train a reliable model (need at least ~80 rows and a few features).")
        else:
            X = ml_df[feat_cols].copy()
            y = ml_df["breakdown_within_30_days"].astype(int)

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
                    X_t, y, test_size=0.2, random_state=42, stratify=y
                )
                rf = RandomForestClassifier(n_estimators=200, random_state=42)
                with st.spinner("Training RandomForest classifier..."):
                    rf.fit(X_train, y_train)
                preds = rf.predict(X_test)
                probs = rf.predict_proba(X_test)[:, 1]

                acc = accuracy_score(y_test, preds)
                try:
                    auc = roc_auc_score(y_test, probs)
                except Exception:
                    auc = float("nan")

                st.write(f"Classification accuracy: **{acc:.3f}**")
                if not math.isnan(auc):
                    st.write(f"ROC-AUC: **{auc:.3f}**")

                # Build result table
                res_df = pd.DataFrame({
                    "Actual_breakdown_30d": y_test.reset_index(drop=True),
                    "Predicted_breakdown_flag": preds,
                    "Predicted_breakdown_probability": probs
                })
                st.dataframe(res_df.head(30), use_container_width=True)
                download_df(res_df, "breakdown_risk_predictions.csv", "Download prediction sample")

    # -------------------------
    # Automated Insights
    # -------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)

    insights_rows = []

    # 1) Vehicles with highest historical breakdowns
    if "vehicle_id" in filt.columns and "breakdown_history_count" in filt.columns:
        bh = (
            filt.groupby("vehicle_id")["breakdown_history_count"]
            .mean()
            .reset_index()
            .sort_values("breakdown_history_count", ascending=False)
        )
        if not bh.empty:
            top_bh = bh.iloc[0]
            insights_rows.append({
                "Insight": "Vehicle with highest historical breakdowns",
                "Entity": top_bh["vehicle_id"],
                "Metric": f"{top_bh['breakdown_history_count']:.1f} breakdowns on average",
                "Action": "Evaluate for intensive maintenance schedule or replacement."
            })

    # 2) Fault code most associated with breakdowns
    if "fault_code" in filt.columns:
        fc_rate = (
            filt.groupby("fault_code")["breakdown_within_30_days"]
            .mean()
            .reset_index(name="breakdown_rate")
            .dropna(subset=["breakdown_rate"])
        )
        if not fc_rate.empty:
            worst_fc = fc_rate.sort_values("breakdown_rate", ascending=False).iloc[0]
            insights_rows.append({
                "Insight": "Fault code with highest breakdown association",
                "Entity": str(worst_fc["fault_code"]),
                "Metric": f"{worst_fc['breakdown_rate']*100:.1f}% of records result in breakdown",
                "Action": "Prioritize root-cause analysis and pre-emptive checks for this fault code."
            })

    # 3) Service alert impact
    if "service_alert_flag" in filt.columns:
        sa_rate = (
            filt.groupby("service_alert_flag")["breakdown_within_30_days"]
            .mean()
            .reset_index(name="breakdown_rate")
        )
        if len(sa_rate) >= 2:
            row_yes = sa_rate.iloc[sa_rate["breakdown_rate"].argmax()]
            insights_rows.append({
                "Insight": "Service alert vs breakdown risk",
                "Entity": f"service_alert_flag = {row_yes['service_alert_flag']}",
                "Metric": f"Breakdown rate ≈ {row_yes['breakdown_rate']*100:.1f}%",
                "Action": "Treat service alerts as hard triggers for immediate inspection."
            })

    # 4) Older vehicle risk
    if "vehicle_age_years" in filt.columns:
        older = filt[filt["vehicle_age_years"] >= filt["vehicle_age_years"].median()]
        younger = filt[filt["vehicle_age_years"] < filt["vehicle_age_years"].median()]
        if not older.empty and not younger.empty:
            rate_old = older["breakdown_within_30_days"].mean()
            rate_young = younger["breakdown_within_30_days"].mean()
            insights_rows.append({
                "Insight": "Older vs newer vehicle breakdown risk",
                "Entity": "Age above vs below median",
                "Metric": f"Older: {rate_old*100:.1f}%, Newer: {rate_young*100:.1f}%",
                "Action": "Shift older vehicles to shorter or less critical routes."
            })

    if not insights_rows:
        st.info("No insights generated for the current filter selection.")
    else:
        insights_df = pd.DataFrame(insights_rows)
        st.dataframe(insights_df, use_container_width=True)
        download_df(insights_df, "predictive_maintenance_insights.csv", "Download insights")
