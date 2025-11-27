import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import math
import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# HIDE SIDEBAR
# ----------------------------------------------------------
hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="Energy & Resource Wastage Analytics Lab",
    layout="wide",
    page_icon="⚡"
)

# ----------------------------------------------------------
# HEADER & LOGO
# ----------------------------------------------------------
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

# ----------------------------------------------------------
# GLOBAL CSS – STANDARDIZED LAB THEME
# ----------------------------------------------------------
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

st.markdown("<div class='big-header'>Energy & Resource Wastage Analytics Lab</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# CONSTANTS & HELPERS
# ----------------------------------------------------------
DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/manufacturing/energy_wastage.csv"

REQUIRED_ENERGY_COLS = [
    "Timestamp",
    "Machine_ID",
    "Machine_Type",
    "Fuel_Type",
    "Shift",
    "Operator_ID",
    "Energy_Consumption_kWh",
    "Energy_Predicted_kWh",
    "Baseload_Drift_kWh",
    "Cooling_Load_kWh",
    "Fuel_Consumption",
    "Compressed_Air_CFM",
    "Voltage_Instability",
    "Power_Factor",
    "Ambient_Temp_C",
    "Output_Units",
    "Energy_Intensity_kWh_per_Unit",
    "Peak_Load_Flag",
    "Idle_Flag",
    "Wastage_Severity",
    "CO2_Emissions_kg"
]

required_dict = {
    "Timestamp": "Date-time stamp for each energy / sensor reading.",
    "Machine_ID": "Unique identifier for each machine / asset.",
    "Machine_Type": "Type or family of machine (e.g., Boiler, Chiller, Press).",
    "Fuel_Type": "Primary fuel / power source (Grid, Diesel, Gas, etc.).",
    "Shift": "Operational shift label (Shift A / B / C / General).",
    "Operator_ID": "Operator or team responsible during the reading.",
    "Energy_Consumption_kWh": "Actual energy consumed during the interval (kWh).",
    "Energy_Predicted_kWh": "Predicted / expected energy usage for the same condition (kWh).",
    "Baseload_Drift_kWh": "Excess baseload consumption beyond expected idle baseline (kWh).",
    "Cooling_Load_kWh": "Energy tied to cooling / HVAC load (kWh).",
    "Fuel_Consumption": "Fuel consumed in the interval (litres, kg, etc., consistent unit).",
    "Compressed_Air_CFM": "Compressed air flow (CFM) if applicable.",
    "Voltage_Instability": "Voltage fluctuation index for the period.",
    "Power_Factor": "Power factor during the reading (0–1).",
    "Ambient_Temp_C": "Ambient temperature at the plant (°C).",
    "Output_Units": "Production units or output count tied to that interval.",
    "Energy_Intensity_kWh_per_Unit": "Energy consumption per output unit (kWh/unit).",
    "Peak_Load_Flag": "Flag = 1 if the point is in peak load / demand charge band.",
    "Idle_Flag": "Flag = 1 if machine is idle but drawing non-trivial energy.",
    "Wastage_Severity": "Scored indicator (e.g., 0–5) of wastage severity for that record.",
    "CO2_Emissions_kg": "Estimated CO₂ emissions for the energy / fuel used (kg)."
}

def read_csv_safe(url_or_file):
    df = pd.read_csv(url_or_file)
    cols = list(df.columns)
    if len(cols) != len(set(cols)):
        new_cols = []
        seen = {}
        for c in cols:
            base = str(c).strip()
            if base not in seen:
                seen[base] = 0
                new_cols.append(base)
            else:
                seen[base] += 1
                new_cols.append(f"{base}__dup{seen[base]}")
        df.columns = new_cols
    df.columns = [str(c).strip() for c in df.columns]
    return df

def ensure_datetime(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def to_float_series(s):
    return pd.to_numeric(s, errors="coerce")

def download_df(df, filename, label):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")

def add_linear_regression_trace(fig, x, y, name="Linear fit", color=None):
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if mask.sum() < 2:
        return fig
    xp = x[mask]
    yp = y[mask]
    coeffs = np.polyfit(xp, yp, deg=1)
    slope, intercept = coeffs[0], coeffs[1]
    x_line = np.linspace(np.min(xp), np.max(xp), 50)
    y_line = slope * x_line + intercept
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines", name=name,
        line=dict(dash="dash", color=color or "black")
    ))
    return fig

# ----------------------------------------------------------
# TABS
# ----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# ==========================================================
# TAB 1 - OVERVIEW
# ==========================================================
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <b>Purpose:</b><br><br>
    Monitor plant-level and machine-level energy usage, detect wastage patterns, and quantify the savings impact of
    fixing baseload drift, idle running, poor power factor, cooling inefficiencies and fuel misuse.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">What This Lab Does</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Tracks <b>Energy_Consumption_kWh</b> vs <b>Energy_Predicted_kWh</b> across machines & shifts<br>
        • Surfaces <b>baseload_drift_kWh</b>, <b>Cooling_Load_kWh</b>, & idle flags as wastage levers<br>
        • Builds ML models to predict <b>energy consumption</b> from operational & ambient factors<br>
        • Runs <b>anomaly detection</b> to catch spikes & abnormal energy behavior<br>
        • Clusters records into <b>operational modes</b> (efficient vs inefficient regimes)
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • 10–30% reduction in energy cost by eliminating wastage<br>
        • Better <b>CO₂ reporting</b> and climate disclosures<br>
        • Targeted CAPEX & maintenance interventions by machine / shift<br>
        • Cleaner story for <b>ESG, lenders & management</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Key KPIs Tracked</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown("<div class='kpi'>Total Energy (kWh)</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Avg Energy / Unit</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Peak Load Events</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Idle Hours</div>", unsafe_allow_html=True)
    k5.markdown("<div class='kpi'>Estimated Savings (₹)</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Who Should Use This</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    Energy managers, plant heads, operations & maintenance teams, sustainability leads and data analysts who want one
    workspace for <b>energy diagnostics, forecasting and savings estimation</b>.
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# TAB 2 - IMPORTANT ATTRIBUTES
# ==========================================================
with tab2:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

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
            "Machine_ID",
            "Machine_Type",
            "Fuel_Type",
            "Shift",
            "Operator_ID",
            "Cooling_Load_kWh",
            "Fuel_Consumption",
            "Compressed_Air_CFM",
            "Voltage_Instability",
            "Power_Factor",
            "Ambient_Temp_C",
            "Output_Units",
            "Baseload_Drift_kWh",
            "Peak_Load_Flag",
            "Idle_Flag",
            "Wastage_Severity"
        ]
        for v in independents:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Dependent / Target Variables</div>', unsafe_allow_html=True)
        dependents = [
            "Energy_Consumption_kWh",
            "Energy_Predicted_kWh",
            "Energy_Intensity_kWh_per_Unit",
            "CO2_Emissions_kg"
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
            df = read_csv_safe(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            st.success("Default dataset loaded from GitHub.")
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    # -------------------------
    # Upload CSV (direct)
    # -------------------------
    elif mode == "Upload CSV":
        uploaded = st.file_uploader("Upload your Energy Wastage dataset", type=["csv"])
        if uploaded is not None:
            df = read_csv_safe(uploaded)
            df.columns = df.columns.str.strip()
            st.success("File uploaded.")
            st.dataframe(df.head(10), use_container_width=True)
        else:
            st.stop()

    # -------------------------
    # Upload + column mapping
    # -------------------------
    else:
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded is not None:
            raw = read_csv_safe(uploaded)
            raw.columns = raw.columns.str.strip()
            st.write("Preview (first 10 rows):")
            st.dataframe(raw.head(10), use_container_width=True)
            st.markdown("Map your columns to the required fields (map as many as available).", unsafe_allow_html=True)
            mapping = {}
            for req in REQUIRED_ENERGY_COLS:
                mapping[req] = st.selectbox(
                    f"Map → {req}",
                    options=["-- Select --"] + list(raw.columns)
                )
            if st.button("Apply mapping"):
                chosen = {k: v for k, v in mapping.items() if v != "-- Select --"}
                if not chosen:
                    st.error("Please map at least a few required columns.")
                else:
                    inv = {v: k for k, v in chosen.items()}
                    df = raw.rename(columns=inv)
                    st.success("Mapping applied.")
                    st.dataframe(df.head(10), use_container_width=True)
        else:
            st.stop()

    if df is None:
        st.stop()

    # ----------------------------------------------------------
    # VALIDATE & TYPE COERCION
    # ----------------------------------------------------------
    df.columns = df.columns.str.strip()

    df = ensure_datetime(df, "Timestamp")

    num_cols = [
        "Energy_Consumption_kWh",
        "Energy_Predicted_kWh",
        "Baseload_Drift_kWh",
        "Cooling_Load_kWh",
        "Fuel_Consumption",
        "Compressed_Air_CFM",
        "Voltage_Instability",
        "Power_Factor",
        "Ambient_Temp_C",
        "Output_Units",
        "Energy_Intensity_kWh_per_Unit",
        "CO2_Emissions_kg"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = to_float_series(df[c])

    for flag in ["Peak_Load_Flag", "Idle_Flag", "Wastage_Severity"]:
        if flag in df.columns:
            df[flag] = pd.to_numeric(df[flag], errors="coerce").fillna(0)

    for c in num_cols:
        if c in df.columns:
            if df[c].notna().any():
                df[c] = df[c].fillna(df[c].median())
            else:
                df[c] = df[c].fillna(0)

    # ----------------------------------------------------------
    # FILTERS & PREVIEW
    # ----------------------------------------------------------
    st.markdown('<div class="section-title">Step 2: Filters & Preview</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    # Date range
    if "Timestamp" in df.columns and not df["Timestamp"].isna().all():
        try:
            min_d = df["Timestamp"].min().date()
            max_d = df["Timestamp"].max().date()
            date_range = c1.date_input("Date range", value=(min_d, max_d))
        except Exception:
            date_range = c1.date_input("Date range")
    else:
        date_range = None

    # Machine, fuel, shift filters
    machines = sorted(df["Machine_ID"].dropna().unique().tolist()) if "Machine_ID" in df.columns else []
    fuels = sorted(df["Fuel_Type"].dropna().unique().tolist()) if "Fuel_Type" in df.columns else []
    shifts = sorted(df["Shift"].dropna().unique().tolist()) if "Shift" in df.columns else []

    with c2:
        sel_machines = st.multiselect("Machine_ID", options=machines, default=machines[:8] if machines else [])
    with c3:
        sel_fuel = st.multiselect("Fuel_Type", options=fuels, default=fuels if fuels else [])
    # shift filter below
    sel_shift = st.multiselect("Shift", options=shifts, default=shifts if shifts else [])

    filt = df.copy()

    if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        try:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            if "Timestamp" in filt.columns:
                filt = filt[(filt["Timestamp"] >= start) & (filt["Timestamp"] <= end + pd.Timedelta(days=1))]
        except Exception:
            pass

    if sel_machines and "Machine_ID" in filt.columns:
        filt = filt[filt["Machine_ID"].isin(sel_machines)]

    if sel_fuel and "Fuel_Type" in filt.columns:
        filt = filt[filt["Fuel_Type"].isin(sel_fuel)]

    if sel_shift and "Shift" in filt.columns:
        filt = filt[filt["Shift"].isin(sel_shift)]

    st.markdown(f"""
    <div style='padding:8px; background:#eef4ff; border-radius:8px; font-weight:600; margin-bottom:8px;'>
    Filtered Rows: {len(filt)} of {len(df)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Data Preview (first 10 rows)")
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(500), "energy_filtered_preview.csv", "Download filtered preview (first 500 rows)")

    if filt.empty:
        st.warning("Filtered dataset is empty. Adjust filters above.")
        st.stop()

    # ----------------------------------------------------------
    # KPIs
    # ----------------------------------------------------------
    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)

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

    total_energy = safe_sum(filt["Energy_Consumption_kWh"]) if "Energy_Consumption_kWh" in filt.columns else float("nan")
    avg_intensity = safe_mean(filt["Energy_Intensity_kWh_per_Unit"]) if "Energy_Intensity_kWh_per_Unit" in filt.columns else float("nan")
    peak_events = int(filt["Peak_Load_Flag"].sum()) if "Peak_Load_Flag" in filt.columns else 0
    idle_hours = int(filt["Idle_Flag"].sum()) if "Idle_Flag" in filt.columns else 0
    est_savings = 0.0
    if "Wastage_Severity" in filt.columns and "Energy_Consumption_kWh" in filt.columns:
        est_savings = filt["Wastage_Severity"].sum() * 0.05 * np.nanmean(filt["Energy_Consumption_kWh"])

    k1.metric("Total Energy (kWh)", f"{total_energy:,.0f}" if not math.isnan(total_energy) else "N/A")
    k2.metric("Avg Energy / Unit", f"{avg_intensity:.2f} kWh/unit" if not math.isnan(avg_intensity) else "N/A")
    k3.metric("Peak Load Events", f"{peak_events}")
    k4.metric("Idle Hours (flag count)", f"{idle_hours}")
    k5.metric("Est. Savings (₹)", f"{est_savings:,.0f}" if est_savings else "N/A")

    # ----------------------------------------------------------
    # CHARTS & DIAGNOSTICS
    # ----------------------------------------------------------
    st.markdown('<div class="section-title">Charts & Diagnostics</div>', unsafe_allow_html=True)

    # 1) Daily energy trend
    if "Timestamp" in filt.columns and "Energy_Consumption_kWh" in filt.columns:
        daily = filt.set_index("Timestamp")["Energy_Consumption_kWh"].resample("D").sum().reset_index()
        fig_daily = px.area(daily, x="Timestamp", y="Energy_Consumption_kWh",
                            title="Daily Energy Consumption (kWh)")
        st.plotly_chart(fig_daily, use_container_width=True)

    # 2) Hourly heatmap by machine
    if "Timestamp" in filt.columns and "Machine_ID" in filt.columns and "Energy_Consumption_kWh" in filt.columns:
        tmp = filt.copy()
        tmp["hour"] = tmp["Timestamp"].dt.hour
        heat = tmp.groupby(["Machine_ID", "hour"])["Energy_Consumption_kWh"].mean().reset_index()
        if not heat.empty:
            pivot = heat.pivot(index="Machine_ID", columns="hour", values="Energy_Consumption_kWh").fillna(0)
            fig_heat = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="Viridis"
            ))
            fig_heat.update_layout(title="Avg Hourly Energy by Machine", xaxis_title="Hour", yaxis_title="Machine_ID")
            st.plotly_chart(fig_heat, use_container_width=True)

    # 3) Cooling load vs energy (with regression)
    if "Cooling_Load_kWh" in filt.columns and "Energy_Consumption_kWh" in filt.columns:
        fig_sc = px.scatter(filt, x="Cooling_Load_kWh", y="Energy_Consumption_kWh",
                            hover_data=["Machine_ID"] if "Machine_ID" in filt.columns else None,
                            title="Cooling Load vs Energy Consumption")
        add_linear_regression_trace(fig_sc,
                                    filt["Cooling_Load_kWh"].values,
                                    filt["Energy_Consumption_kWh"].values,
                                    name="Linear fit",
                                    color="red")
        st.plotly_chart(fig_sc, use_container_width=True)

    # 4) Avg energy by shift
    if "Shift" in filt.columns and "Energy_Consumption_kWh" in filt.columns:
        shift_agg = filt.groupby("Shift")["Energy_Consumption_kWh"].mean().reset_index().sort_values("Energy_Consumption_kWh", ascending=False)
        fig_shift = px.bar(shift_agg, x="Shift", y="Energy_Consumption_kWh", title="Avg Energy by Shift")
        st.plotly_chart(fig_shift, use_container_width=True)

    # 5) Box: Energy by Fuel_Type
    if "Fuel_Type" in filt.columns and "Energy_Consumption_kWh" in filt.columns:
        fig_box = px.box(filt, x="Fuel_Type", y="Energy_Consumption_kWh", title="Energy Distribution by Fuel Type")
        st.plotly_chart(fig_box, use_container_width=True)

    # 6) Pairwise scatter matrix (sample)
    numeric_for_matrix = [c for c in [
        "Energy_Consumption_kWh",
        "Cooling_Load_kWh",
        "Fuel_Consumption",
        "Compressed_Air_CFM",
        "Ambient_Temp_C"
    ] if c in filt.columns]
    if len(numeric_for_matrix) >= 2:
        sample_df = filt[numeric_for_matrix].sample(n=min(400, len(filt)), random_state=42)
        fig_mat = px.scatter_matrix(sample_df, dimensions=numeric_for_matrix,
                                    title="Pairwise Relationships (sample)")
        st.plotly_chart(fig_mat, use_container_width=True)

    # 7) Treemap: Machine_Type -> Fuel_Type
    if "Machine_Type" in filt.columns and "Fuel_Type" in filt.columns and "Energy_Consumption_kWh" in filt.columns:
        agg = filt.groupby(["Machine_Type", "Fuel_Type"])["Energy_Consumption_kWh"].sum().reset_index()
        fig_tree = px.treemap(agg, path=["Machine_Type", "Fuel_Type"], values="Energy_Consumption_kWh",
                              title="Energy by Machine Type & Fuel Type")
        st.plotly_chart(fig_tree, use_container_width=True)

    # ----------------------------------------------------------
    # ML & MODELING
    # ----------------------------------------------------------
    st.markdown('<div class="section-title">ML — Energy Prediction & Anomalies</div>', unsafe_allow_html=True)

    model_features = [c for c in [
        "Cooling_Load_kWh",
        "Compressed_Air_CFM",
        "Fuel_Consumption",
        "Ambient_Temp_C",
        "Output_Units",
        "Energy_Intensity_kWh_per_Unit",
        "Baseload_Drift_kWh"
    ] if c in filt.columns]

    st.markdown(f"<div class='card'>Features used for ML: <b>{', '.join(model_features) if model_features else 'No numeric features available'}</b></div>", unsafe_allow_html=True)

    if len(filt) < 40 or len(model_features) < 1 or "Energy_Consumption_kWh" not in filt.columns:
        st.info("Not enough data / features for ML. Need ≥40 rows, at least 1 numeric feature, and target Energy_Consumption_kWh.")
    else:
        X = filt[model_features].fillna(0)
        y = filt["Energy_Consumption_kWh"].fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # RandomForest
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        with st.spinner("Training RandomForest Regressor..."):
            rf.fit(X_train, y_train)
        preds_rf = rf.predict(X_test)
        rmse_rf = math.sqrt(mean_squared_error(y_test, preds_rf))
        r2_rf = r2_score(y_test, preds_rf)

        # GradientBoosting
        gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, random_state=42)
        with st.spinner("Training GradientBoosting Regressor..."):
            gbr.fit(X_train, y_train)
        preds_gbr = gbr.predict(X_test)
        rmse_gbr = math.sqrt(mean_squared_error(y_test, preds_gbr))
        r2_gbr = r2_score(y_test, preds_gbr)

        st.markdown(f"""
        <div class='card'>
        <b>Model performance (Energy_Consumption_kWh):</b><br><br>
        RandomForest → RMSE: <b>{rmse_rf:.2f}</b>, R²: <b>{r2_rf:.3f}</b><br>
        GradientBoosting → RMSE: <b>{rmse_gbr:.2f}</b>, R²: <b>{r2_gbr:.3f}</b><br>
        </div>
        """, unsafe_allow_html=True)

        # Isolation Forest — anomaly detection
        iso = IsolationForest(contamination=0.02, random_state=42)
        with st.spinner("Fitting IsolationForest for anomaly detection..."):
            iso.fit(X.fillna(0))
        iso_scores = iso.decision_function(X.fillna(0))
        iso_preds = iso.predict(X.fillna(0))  # -1 anomaly, 1 normal
        filt["_anomaly_score"] = iso_scores
        filt["_is_anomaly"] = np.where(iso_preds == -1, 1, 0)
        st.markdown(f"<div class='card'>Anomalies detected (IsolationForest): <b>{int(filt['_is_anomaly'].sum())}</b> rows flagged.</div>", unsafe_allow_html=True)

        # KMeans clustering
        try:
            kmeans_feats = X.fillna(0)
            n_clusters = min(4, max(2, int(len(kmeans_feats) / 50)))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(kmeans_feats)
            filt["_kmeans_cluster"] = clusters

            pca = PCA(n_components=2)
            comp = pca.fit_transform(scaler.fit_transform(kmeans_feats))
            pcadf = pd.DataFrame(comp, columns=["PC1", "PC2"])
            pcadf["_cluster"] = clusters
            fig_pca = px.scatter(
                pcadf, x="PC1", y="PC2", color="_cluster",
                title="Operational Modes (KMeans clusters in PCA space)"
            )
            st.plotly_chart(fig_pca, use_container_width=True)
        except Exception as e:
            st.info("KMeans clustering failed: " + str(e))

        # Predictions output
        out_rf = X_test.copy().reset_index(drop=True)
        out_rf["Actual_Energy_kWh"] = y_test.reset_index(drop=True)
        out_rf["Pred_RF_kWh"] = preds_rf
        out_rf["Pred_GBR_kWh"] = preds_gbr

        st.markdown("#### ML Prediction Sample (RandomForest + GradientBoosting)")
        st.dataframe(out_rf.head(20), use_container_width=True)
        download_df(out_rf, "energy_predictions_with_features.csv", "Download ML predictions (RF + GBR)")

    # ----------------------------------------------------------
    # AUTOMATED INSIGHTS
    # ----------------------------------------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)

    insights = []

    # High energy intensity machines
    if "Machine_ID" in filt.columns and "Energy_Intensity_kWh_per_Unit" in filt.columns:
        tmp = filt.groupby("Machine_ID")["Energy_Intensity_kWh_per_Unit"].mean().reset_index()
        tmp = tmp.dropna().sort_values("Energy_Intensity_kWh_per_Unit", ascending=False).head(10)
        for _, r in tmp.iterrows():
            insights.append({
                "Insight_Type": "High Energy Intensity",
                "Machine_ID": r["Machine_ID"],
                "Metric": "Energy_Intensity_kWh_per_Unit",
                "Value": round(float(r["Energy_Intensity_kWh_per_Unit"]), 3)
            })

    # Machines with most anomalies
    if "_is_anomaly" in filt.columns and "Machine_ID" in filt.columns:
        anom_counts = filt.groupby("Machine_ID")["_is_anomaly"].sum().reset_index()
        anom_counts = anom_counts.sort_values("_is_anomaly", ascending=False).head(10)
        for _, r in anom_counts.iterrows():
            if int(r["_is_anomaly"]) > 0:
                insights.append({
                    "Insight_Type": "High Anomaly Count",
                    "Machine_ID": r["Machine_ID"],
                    "Metric": "Anomaly Count",
                    "Value": int(r["_is_anomaly"])
                })

    # Peak load summary
    if "Peak_Load_Flag" in filt.columns and "Machine_ID" in filt.columns:
        peaks = filt.groupby("Machine_ID")["Peak_Load_Flag"].sum().reset_index()
        peaks = peaks.sort_values("Peak_Load_Flag", ascending=False).head(10)
        for _, r in peaks.iterrows():
            if int(r["Peak_Load_Flag"]) > 0:
                insights.append({
                    "Insight_Type": "Peak Load Events",
                    "Machine_ID": r["Machine_ID"],
                    "Metric": "Peak_Load_Flag",
                    "Value": int(r["Peak_Load_Flag"])
                })

    # CO2 emissions by machine
    if "CO2_Emissions_kg" in filt.columns and "Machine_ID" in filt.columns:
        co2 = filt.groupby("Machine_ID")["CO2_Emissions_kg"].sum().reset_index()
        co2 = co2.sort_values("CO2_Emissions_kg", ascending=False).head(10)
        for _, r in co2.iterrows():
            insights.append({
                "Insight_Type": "High CO₂ Emitter",
                "Machine_ID": r["Machine_ID"],
                "Metric": "Total_CO2_kg",
                "Value": round(float(r["CO2_Emissions_kg"]), 2)
            })

    # Quick narrative bullets
    st.markdown("<div class='card'><b>Quick Observations</b><ul>", unsafe_allow_html=True)
    if "Energy_Consumption_kWh" in filt.columns:
        avg_energy = round(float(filt["Energy_Consumption_kWh"].mean()), 2)
        st.markdown(f"<li>Average energy consumption in filtered view: <b>{avg_energy} kWh</b></li>", unsafe_allow_html=True)
    if "Power_Factor" in filt.columns:
        avg_pf = round(float(filt["Power_Factor"].mean()), 2)
        st.markdown(f"<li>Average power factor: <b>{avg_pf}</b> (PF correction advisable if this is low).</li>", unsafe_allow_html=True)
    if "Idle_Flag" in filt.columns:
        idle_cnt = int(filt["Idle_Flag"].sum())
        st.markdown(f"<li>Idle flagged records: <b>{idle_cnt}</b> — check scheduling & shutdown discipline.</li>", unsafe_allow_html=True)
    st.markdown("</ul></div>", unsafe_allow_html=True)

    if insights:
        ins_df = pd.DataFrame(insights)
        st.markdown("#### Insights Table")
        st.dataframe(ins_df, use_container_width=True)
        download_df(ins_df, "energy_automated_insights.csv", "Download insights")
    else:
        st.info("No automated insights generated for the current filters.")
