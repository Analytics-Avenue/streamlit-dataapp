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
st.set_page_config(page_title="Inverter Failure Prediction Lab", layout="wide")

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

st.markdown("<div class='big-header'>Inverter Failure Prediction Lab</div>", unsafe_allow_html=True)

# -------------------------
# Helpers & constants
# -------------------------
DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/Solar/inverter_failure.csv"

REQUIRED_INVERTER_COLS = [
    "inverter_id",
    "timestamp",
    "temperature_c",
    "heatsink_temp_c",
    "voltage_v",
    "current_a",
    "vibration_mms",
    "dc_input_kw",
    "ac_output_kw",
    "pf_power_factor",
    "frequency_hz",
    "mppt_voltage_v",
    "mppt_current_a",
    "total_energy_generated_kwh",
    "lifetime_hours",
    "ambient_temperature_c",
    "fan_speed_rpm",
    "fault_code",
    "string_count_active",
    "grid_status",
    "failure_within_7_days"
]

def to_float_series(s):
    return pd.to_numeric(s, errors="coerce")

def ensure_datetime(df, col):
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            pass
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
    This lab turns raw inverter telemetry into a failure early-warning system. It tracks thermal stress, electrical
    anomalies, vibration, and loading patterns to predict which inverters are likely to fail in the next few days.
    The goal is simple: <b>fix before failure</b>, not after generation is already lost.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">What This Lab Does</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Consolidates <b>temperature, voltage, current, vibration</b> & loading data<br>
        • Flags inverters with high <b>failure_within_7_days</b> risk<br>
        • Learns patterns from <b>historical breakdowns</b><br>
        • Highlights risky <b>fault_code</b> and <b>grid_status</b> combinations<br>
        • Surfaces chronic underperformers based on <b>lifetime_hours & energy</b>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Fewer <b>unexpected inverter shutdowns</b><br>
        • Higher daily generation through <b>predictive maintenance</b><br>
        • Reduced emergency O&M costs and truck rolls<br>
        • Longer inverter lifetime and better warranty defense<br>
        • Cleaner communication with IPPs, DISCOMs & investors
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">O&M KPIs Tracked</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Failure Rate (7 Days)</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>High-Risk Inverters</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Avg Temperature & Thermal Stress</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Energy Lost Risk (kWh)</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Who Should Use This</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    Solar plant O&M teams, control-room operators, asset managers, OEM service teams, and data engineers
    building reliability models for large-scale solar portfolios.
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# TAB 2 - IMPORTANT ATTRIBUTES
# ==========================================================
with tab2:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    required_dict = {
        "inverter_id": "Unique identifier of the inverter.",
        "timestamp": "Timestamp of the telemetry reading.",
        "temperature_c": "Internal inverter temperature (°C).",
        "heatsink_temp_c": "Heatsink temperature (°C), proxy for thermal stress.",
        "voltage_v": "AC or DC bus voltage (V).",
        "current_a": "Current flowing through inverter (A).",
        "vibration_mms": "Vibration level (mm/s) indicating mechanical stress.",
        "dc_input_kw": "Total DC input power from PV strings (kW).",
        "ac_output_kw": "AC power output injected to grid (kW).",
        "pf_power_factor": "Power factor of the output.",
        "frequency_hz": "Grid frequency during operation (Hz).",
        "mppt_voltage_v": "MPPT input voltage (V).",
        "mppt_current_a": "MPPT input current (A).",
        "total_energy_generated_kwh": "Lifetime energy generated so far (kWh).",
        "lifetime_hours": "Total operating hours.",
        "ambient_temperature_c": "Ambient temperature around the inverter (°C).",
        "fan_speed_rpm": "Cooling fan speed (RPM).",
        "fault_code": "Current/last inverter fault code.",
        "string_count_active": "Number of active DC strings connected.",
        "grid_status": "Grid connectivity/quality status.",
        "failure_within_7_days": "Label: 1 if inverter failed within next 7 days, else 0."
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

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        independents = [
            "temperature_c",
            "heatsink_temp_c",
            "voltage_v",
            "current_a",
            "vibration_mms",
            "dc_input_kw",
            "ac_output_kw",
            "pf_power_factor",
            "frequency_hz",
            "mppt_voltage_v",
            "mppt_current_a",
            "total_energy_generated_kwh",
            "lifetime_hours",
            "ambient_temperature_c",
            "fan_speed_rpm",
            "fault_code",
            "string_count_active",
            "grid_status",
            "inverter_id"
        ]
        for v in independents:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        dependents = [
            "failure_within_7_days"
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
        st.markdown("#### Sample structure (from standard Inverter Failure dataset)", unsafe_allow_html=True)
        try:
            sample_df = pd.read_csv(DEFAULT_URL).head(5)
            st.dataframe(sample_df, use_container_width=True)
            sample_csv = sample_df.to_csv(index=False)
            st.download_button(
                "Download Sample CSV",
                sample_csv,
                "sample_inverter_failure.csv",
                "text/csv"
            )
        except Exception as e:
            st.info(f"Sample CSV unavailable from GitHub: {e}")

        uploaded = st.file_uploader("Upload your Inverter Failure dataset", type=["csv"])
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
            for req in REQUIRED_INVERTER_COLS:
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
    missing = [c for c in REQUIRED_INVERTER_COLS if c not in df.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.info("Use 'Upload CSV + Column mapping' if your column names differ.")
        st.stop()

    # -------------------------
    # Coerce types
    # -------------------------
    df = ensure_datetime(df, "timestamp")

    num_cols = [
        "temperature_c",
        "heatsink_temp_c",
        "voltage_v",
        "current_a",
        "vibration_mms",
        "dc_input_kw",
        "ac_output_kw",
        "pf_power_factor",
        "frequency_hz",
        "mppt_voltage_v",
        "mppt_current_a",
        "total_energy_generated_kwh",
        "lifetime_hours",
        "ambient_temperature_c",
        "fan_speed_rpm",
        "string_count_active"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = to_float_series(df[c])

    # Make sure label is numeric 0/1
    df["failure_within_7_days"] = pd.to_numeric(df["failure_within_7_days"], errors="coerce").fillna(0).astype(int)

    # -------------------------
    # Filters
    # -------------------------
    st.markdown("### Step 2: Filters", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)

    with f1:
        invs = sorted(df["inverter_id"].astype(str).unique())
        sel_inverters = st.multiselect("Inverter ID", invs, default=invs[:10] if len(invs) > 10 else invs)

    with f2:
        grids = sorted(df["grid_status"].astype(str).unique())
        sel_grid = st.multiselect("Grid Status", grids, default=grids)

    with f3:
        faults = sorted(df["fault_code"].astype(str).unique())
        sel_fault = st.multiselect("Fault Code", faults, default=faults)

    filt = df.copy()
    if sel_inverters:
        filt = filt[filt["inverter_id"].astype(str).isin(sel_inverters)]
    if sel_grid:
        filt = filt[filt["grid_status"].astype(str).isin(sel_grid)]
    if sel_fault:
        filt = filt[filt["fault_code"].astype(str).isin(sel_fault)]

    if filt.empty:
        st.warning("Filters removed all rows. Resetting to full dataset.")
        filt = df.copy()

    st.markdown(f"""
    <div style='padding:8px; background:#eef4ff; border-radius:8px; font-weight:600; margin-bottom:10px;'>
    Filtered Rows: {len(filt)} of {len(df)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Data Preview", unsafe_allow_html=True)
    st.dataframe(filt.head(10), use_container_width=True)

    # -------------------------
    # KPIs
    # -------------------------
    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    total_rows = len(filt)
    failure_rate = filt["failure_within_7_days"].mean() if total_rows > 0 else 0.0
    avg_temp = filt["temperature_c"].mean() if "temperature_c" in filt.columns else float("nan")
    avg_heatsink = filt["heatsink_temp_c"].mean() if "heatsink_temp_c" in filt.columns else float("nan")
    avg_energy = filt["total_energy_generated_kwh"].mean() if "total_energy_generated_kwh" in filt.columns else float("nan")

    k1.metric("Records in scope", f"{total_rows:,}")
    k2.metric("7-day Failure Rate", f"{failure_rate*100:.2f}%" if total_rows > 0 else "N/A")
    k3.metric("Avg Temp / Heatsink (°C)",
              f"{avg_temp:.1f} / {avg_heatsink:.1f}" if not (math.isnan(avg_temp) or math.isnan(avg_heatsink)) else "N/A")
    k4.metric("Avg Lifetime Energy (kWh)", f"{avg_energy:,.1f}" if not math.isnan(avg_energy) else "N/A")

    # -------------------------
    # Charts & Diagnostics
    # -------------------------
    st.markdown('<div class="section-title">Charts & Diagnostics</div>', unsafe_allow_html=True)

    # 1) Failure rate by inverter
    if not filt.empty:
        fr_inv = filt.groupby("inverter_id")["failure_within_7_days"].mean().reset_index()
        fr_inv["failure_rate_pct"] = fr_inv["failure_within_7_days"] * 100
        fr_inv = fr_inv.sort_values("failure_rate_pct", ascending=False).head(20)
        if not fr_inv.empty:
            fig1 = px.bar(
                fr_inv,
                x="inverter_id",
                y="failure_rate_pct",
                text="failure_rate_pct",
                labels={"failure_rate_pct": "Failure rate (%)"},
                title="Inverter-wise 7-day failure rate (top 20)"
            )
            fig1.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig1.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig1, use_container_width=True)

    # 2) Temperature vs Vibration colored by failure
    num_points = len(filt)
    if "temperature_c" in filt.columns and "vibration_mms" in filt.columns and num_points > 0:
        tmp_df = filt[["temperature_c", "vibration_mms", "failure_within_7_days"]].dropna()
        if not tmp_df.empty:
            tmp_df["failure_label"] = tmp_df["failure_within_7_days"].map({0: "No failure", 1: "Failure"})
            fig2 = px.scatter(
                tmp_df,
                x="temperature_c",
                y="vibration_mms",
                color="failure_label",
                labels={"temperature_c": "Temperature (°C)", "vibration_mms": "Vibration (mm/s)"},
                title="Temperature vs Vibration by failure outcome"
            )
            st.plotly_chart(fig2, use_container_width=True)

    # 3) Energy vs Lifetime hours
    if "lifetime_hours" in filt.columns and "total_energy_generated_kwh" in filt.columns:
        el = filt[["lifetime_hours", "total_energy_generated_kwh"]].dropna()
        if not el.empty:
            fig3 = px.scatter(
                el,
                x="lifetime_hours",
                y="total_energy_generated_kwh",
                labels={"lifetime_hours": "Lifetime hours", "total_energy_generated_kwh": "Total energy (kWh)"},
                title="Lifetime hours vs total energy generated"
            )
            st.plotly_chart(fig3, use_container_width=True)

    # -------------------------
    # ML — Inverter failure prediction (RandomForest)
    # -------------------------
    st.markdown('<div class="section-title">ML — Failure Within 7 Days (Classification)</div>', unsafe_allow_html=True)
    with st.expander("Train & evaluate model (requires >=80 rows with label diversity)", expanded=False):
        ml_df = filt.copy()
        # Ensure we have both classes present
        label_counts = ml_df["failure_within_7_days"].value_counts(dropna=True)
        if len(ml_df) < 80 or len(label_counts) < 2:
            st.info("Not enough rows or label diversity to train a robust model (need at least ~80 rows and both classes 0/1).")
        else:
            feat_cols = [
                "temperature_c",
                "heatsink_temp_c",
                "voltage_v",
                "current_a",
                "vibration_mms",
                "dc_input_kw",
                "ac_output_kw",
                "pf_power_factor",
                "frequency_hz",
                "mppt_voltage_v",
                "mppt_current_a",
                "total_energy_generated_kwh",
                "lifetime_hours",
                "ambient_temperature_c",
                "fan_speed_rpm",
                "string_count_active",
                "fault_code",
                "grid_status",
                "inverter_id"
            ]
            feat_cols = [c for c in feat_cols if c in ml_df.columns]

            X = ml_df[feat_cols].copy()
            y = ml_df["failure_within_7_days"].astype(int)

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
                with st.spinner("Training RandomForest classification model..."):
                    rf.fit(X_train, y_train)
                preds = rf.predict(X_test)
                probs = rf.predict_proba(X_test)[:, 1]

                acc = accuracy_score(y_test, preds)
                try:
                    auc = roc_auc_score(y_test, probs)
                except Exception:
                    auc = float("nan")

                st.write(f"Classification — Accuracy: {acc:.3f}, ROC-AUC: {auc:.3f}" if not math.isnan(auc) else f"Classification — Accuracy: {acc:.3f}")

                res_df = pd.DataFrame({
                    "Actual_failure_within_7_days": y_test.reset_index(drop=True),
                    "Predicted_failure_flag": preds,
                    "Predicted_failure_probability": probs
                })
                st.dataframe(res_df.head(20), use_container_width=True)
                download_df(res_df, "inverter_failure_predictions.csv", "Download prediction sample")

    # -------------------------
    # Automated Insights
    # -------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)

    insights_rows = []

    # 1) Fault code with highest failure rate
    if "fault_code" in filt.columns and not filt.empty:
        fc = filt.groupby("fault_code")["failure_within_7_days"].mean().reset_index()
        fc = fc.dropna(subset=["failure_within_7_days"])
        if not fc.empty:
            worst_fault = fc.sort_values("failure_within_7_days", ascending=False).iloc[0]
            insights_rows.append({
                "Insight": "Fault code with highest failure rate",
                "Entity": str(worst_fault["fault_code"]),
                "Metric": f"{worst_fault['failure_within_7_days']*100:.1f} % failures",
                "Action": "Prioritize proactive checks whenever this fault appears."
            })

    # 2) Grid status with highest failure share
    if "grid_status" in filt.columns and not filt.empty:
        gs = filt.groupby("grid_status")["failure_within_7_days"].mean().reset_index()
        gs = gs.dropna(subset=["failure_within_7_days"])
        if not gs.empty:
            worst_grid = gs.sort_values("failure_within_7_days", ascending=False).iloc[0]
            insights_rows.append({
                "Insight": "Grid condition most associated with failures",
                "Entity": str(worst_grid["grid_status"]),
                "Metric": f"{worst_grid['failure_within_7_days']*100:.1f} % failures",
                "Action": "Review protections / derating when inverter sees this grid condition."
            })

    # 3) Hottest inverters on average
    if "inverter_id" in filt.columns and "temperature_c" in filt.columns:
        hi = filt.groupby("inverter_id")["temperature_c"].mean().reset_index()
        hi = hi.dropna(subset=["temperature_c"])
        if not hi.empty:
            hot = hi.sort_values("temperature_c", ascending=False).head(3)
            entities = ", ".join([str(x) for x in hot["inverter_id"].tolist()])
            insights_rows.append({
                "Insight": "Inverters with highest average temperature",
                "Entity": entities,
                "Metric": f"Top avg temp ≈ {hot['temperature_c'].max():.1f} °C",
                "Action": "Check ventilation, filter clogging and loading for these inverters."
            })

    # 4) High-vibration cohort
    if "vibration_mms" in filt.columns:
        vib_mean = filt["vibration_mms"].mean()
        vib_high = filt[filt["vibration_mms"] > vib_mean * 1.3]
        if not vib_high.empty:
            frac = len(vib_high) / len(filt)
            insights_rows.append({
                "Insight": "High vibration sample",
                "Entity": f"{len(vib_high)} records",
                "Metric": f"{frac*100:.1f}% of points above 1.3× average vibration",
                "Action": "Inspect mounting, fans, and surroundings for mechanical stress."
            })

    # 5) Energy-heavy but high-risk inverters
    if "inverter_id" in filt.columns and "total_energy_generated_kwh" in filt.columns:
        tmp = filt.groupby("inverter_id").agg({
            "total_energy_generated_kwh": "mean",
            "failure_within_7_days": "mean"
        }).reset_index()
        tmp = tmp.dropna()
        if not tmp.empty:
            # high energy & high failure
            thr_energy = tmp["total_energy_generated_kwh"].median()
            thr_fail = tmp["failure_within_7_days"].median()
            high_combo = tmp[(tmp["total_energy_generated_kwh"] >= thr_energy) &
                             (tmp["failure_within_7_days"] >= thr_fail)]
            if not high_combo.empty:
                insights_rows.append({
                    "Insight": "High energy but high failure-risk inverters",
                    "Entity": ", ".join([str(x) for x in high_combo["inverter_id"].tolist()[:5]]),
                    "Metric": f"{len(high_combo)} inverters in high-risk/high-value segment",
                    "Action": "Prioritize preventive maintenance on these high-value assets."
                })

    if not insights_rows:
        st.info("No insights generated for the current filter selection.")
    else:
        insights_df = pd.DataFrame(insights_rows)
        st.dataframe(insights_df, use_container_width=True)
        download_df(insights_df, "inverter_failure_insights.csv", "Download insights")

