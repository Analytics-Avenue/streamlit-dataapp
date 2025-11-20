# app_energy_wastage.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO, StringIO
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score
from sklearn.ensemble import IsolationForest
import math
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Page Config & Title (left-aligned)
# ---------------------------------------------------------
st.set_page_config(page_title="Fuel, Power & Resource Wastage Analytics", layout="wide", page_icon="⚡")
st.markdown(
    """
    <div style="text-align:left;">
        <h1 style="margin-bottom:0.1rem;">Fuel, Power & Resource Wastage Analytics</h1>
        <div style="color:#555; margin-top:0.2rem; margin-bottom:12px;">
            Detect waste, predict energy consumption & fuel usage, reduce cost with targeted actions.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# CSS: hover glow for cards & KPI visuals
# ---------------------------------------------------------
st.markdown("""
<style>
.card {
    padding: 16px;
    border-radius: 12px;
    background: white;
    border: 1px solid #e6e6e6;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    text-align:left;
}
.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 10px 30px rgba(6,75,134,0.18);
    border-color: #064b86;
}
.kpi {
    padding: 22px;
    border-radius: 12px;
    background: white;
    border: 1px solid #e0e0e0;
    text-align:center;
    font-size:18px;
    font-weight:700;
    color:#064b86;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.kpi:hover {
    transform: translateY(-6px);
    box-shadow: 0 8px 22px rgba(6,75,134,0.20);
}
.small { font-size:13px; color:#666; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def read_csv_safe(url_or_file):
    """
    Read CSV from a URL or uploaded file. If duplicate columns exist, make names unique.
    """
    try:
        df = pd.read_csv(url_or_file)
    except Exception as e:
        # try with python engine if weird separators
        df = pd.read_csv(url_or_file, engine="python")
    cols = list(df.columns)
    if len(cols) != len(set(cols)):
        # make unique
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

def download_df_button(df, filename, label="Download CSV"):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")

def safe_mean(val):
    try:
        return float(np.nanmean(val))
    except:
        return None

# ---------------------------------------------------------
# Default dataset URL (replace with your raw CSV)
# If not reachable, the app will fall back to a small synthetic sample.
# ---------------------------------------------------------
DEFAULT_URL = "https://raw.githubusercontent.com/your-org/your-repo/main/datasets/manufacturing/energy_wastage.csv"

# ---------------------------------------------------------
# Provide a small synthetic sample fallback
# ---------------------------------------------------------
def generate_sample_energy_df(n=500, seed=42):
    rng = np.random.default_rng(seed)
    base_time = datetime(2023,1,1)
    timestamps = [base_time + timedelta(hours=int(i)) for i in range(n)]
    machine_ids = rng.integers(1, 8, size=n)  # 1..7
    machine_types = np.where(machine_ids % 2 == 0, "CNC", "Press")
    fuel_types = np.where(machine_ids % 3 == 0, "Diesel", "Electric")
    shifts = np.where((np.array([t.hour for t in timestamps]) < 8), "Night",
                      np.where(np.array([t.hour for t in timestamps])<16, "Day","Evening"))
    operator_ids = rng.integers(100, 110, size=n)
    energy_consumption = np.abs(100 + machine_ids*5 + rng.normal(0, 20, n) + (np.array([t.hour for t in timestamps])>16)*20)
    energy_pred = energy_consumption + rng.normal(0, 6, n)
    baseload = rng.normal(2, 0.5, n)
    cooling = np.abs(energy_consumption * rng.uniform(0.02, 0.08, n))
    fuel_consumption = np.where(fuel_types=="Diesel", energy_consumption*0.25 + rng.normal(0,2,n), rng.normal(0.5,0.2,n))
    compressed_air = np.abs(50 + rng.normal(0,10,n))
    voltage_inst = np.clip(rng.normal(0,0.02,n), -0.1, 0.1)
    power_factor = np.clip(rng.normal(0.95,0.03,n), 0.6, 1.0)
    ambient_temp = 25 + rng.normal(0,4,n)
    output_units = np.maximum(1, np.round(energy_consumption/5 + rng.normal(0,3,n))).astype(int)
    energy_intensity = energy_consumption / np.maximum(1, output_units)
    peak_flag = (energy_consumption > (np.mean(energy_consumption) + 1.5*np.std(energy_consumption))).astype(int)
    idle_flag = (energy_consumption < (np.mean(energy_consumption) - 1.0*np.std(energy_consumption))).astype(int)
    wastage_severity = np.clip((energy_intensity - np.mean(energy_intensity))/np.std(energy_intensity), -3, 6)
    co2 = energy_consumption * 0.45  # approximate

    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Machine_ID": machine_ids,
        "Machine_Type": machine_types,
        "Fuel_Type": fuel_types,
        "Shift": shifts,
        "Operator_ID": operator_ids,
        "Energy_Consumption_kWh": energy_consumption,
        "Energy_Predicted_kWh": energy_pred,
        "Baseload_Drift_kWh": baseload,
        "Cooling_Load_kWh": cooling,
        "Fuel_Consumption": fuel_consumption,
        "Compressed_Air_CFM": compressed_air,
        "Voltage_Instability": voltage_inst,
        "Power_Factor": power_factor,
        "Ambient_Temp_C": ambient_temp,
        "Output_Units": output_units,
        "Energy_Intensity_kWh_per_Unit": energy_intensity,
        "Peak_Load_Flag": peak_flag,
        "Idle_Flag": idle_flag,
        "Wastage_Severity": wastage_severity,
        "CO2_Emissions_kg": co2
    })
    return df

# ---------------------------------------------------------
# App tabs
# ---------------------------------------------------------
tabs = st.tabs(["Overview", "Application"])

# -------------------------
# Overview tab
# -------------------------
with tabs[0]:
    st.markdown("<div class='card'><b>Purpose</b>: Reduce fuel/energy wastage and optimize scheduling by predicting energy needs, identifying peak events, and surfacing wasted runs.</div>", unsafe_allow_html=True)

    left, right = st.columns([1,1])
    with left:
        st.markdown("#### Capabilities")
        st.markdown("""
        <div class='card'>
        • Predict machine-level energy consumption (regression).<br>
        • Classify peak-load events and idle states.<br>
        • Predict fuel consumption per run.<br>
        • Unsupervised anomaly detection for abnormal energy patterns.<br>
        • Exportable prioritized maintenance / action lists.
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("#### Business impact")
        st.markdown("""
        <div class='card'>
        • 10–30% energy & fuel cost savings in large plants.<br>
        • Reduced CO₂ footprint via optimized scheduling.<br>
        • Improved equipment life by avoiding overstress periods.<br>
        • Lower wasted output and fewer emergency shutdowns.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### KPIs")
    # five KPI cards in a row
    kcols = st.columns(5)
    kcols[0].markdown("<div class='kpi'>Total Energy Tracked</div>", unsafe_allow_html=True)
    kcols[1].markdown("<div class='kpi'>Avg Energy / Shift</div>", unsafe_allow_html=True)
    kcols[2].markdown("<div class='kpi'>Peak Events</div>", unsafe_allow_html=True)
    kcols[3].markdown("<div class='kpi'>Fuel Usage</div>", unsafe_allow_html=True)
    kcols[4].markdown("<div class='kpi'>CO₂ Emissions</div>", unsafe_allow_html=True)

    st.markdown("### Who should use & how")
    st.markdown("""
    <div class='card'>
    <b>Who</b>: Energy managers, plant managers, reliability engineers, sustainability teams.<br><br>
    <b>How</b>: 1) Load data (default / upload). 2) Filter by date/machine/shift. 3) Review anomalies & predicted high-consumption runs. 4) Export prioritized action lists for scheduling or operator retraining.
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# Application tab
# -------------------------
with tabs[1]:
    st.header("Application")
    st.markdown("### Step 1 — Load dataset (choose one of three options)")

    load_mode = st.radio("Dataset option:", ["Default dataset (GitHub URL)", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)
    df = None

    REQUIRED_COLS = [
        "Timestamp","Machine_ID","Machine_Type","Fuel_Type","Shift","Operator_ID",
        "Energy_Consumption_kWh","Energy_Predicted_kWh","Baseload_Drift_kWh","Cooling_Load_kWh",
        "Fuel_Consumption","Compressed_Air_CFM","Voltage_Instability","Power_Factor","Ambient_Temp_C",
        "Output_Units","Energy_Intensity_kWh_per_Unit","Peak_Load_Flag","Idle_Flag","Wastage_Severity","CO2_Emissions_kg"
    ]

    if load_mode == "Default dataset (GitHub URL)":
        try:
            df = read_csv_safe(DEFAULT_URL)
            st.success("Loaded default dataset from DEFAULT_URL.")
        except Exception as e:
            st.warning("Failed to load DEFAULT_URL. Falling back to a generated sample dataset. (Edit DEFAULT_URL in script.)")
            df = generate_sample_energy_df(1000)

    elif load_mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="upload1")
        if uploaded_file is not None:
            try:
                df = read_csv_safe(uploaded_file)
                st.success("Uploaded file read.")
            except Exception as e:
                st.error("Failed to parse uploaded CSV: " + str(e))
                st.stop()
        else:
            st.stop()

    else:  # Upload + mapping
        uploaded_file = st.file_uploader("Upload CSV for mapping", type=["csv"], key="upload2")
        if uploaded_file is not None:
            raw = read_csv_safe(uploaded_file)
            st.write("Preview of uploaded file:")
            st.dataframe(raw.head())
            st.markdown("Map your columns (map at least those that exist in your file).")
            mapping = {}
            cols_list = list(raw.columns)
            for req in REQUIRED_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", ["-- Skip --"] + cols_list, key=f"map_{req}")
            if st.button("Apply mapping"):
                # build rename dict using selected items that are not skip
                rename = {mapping[k]: k for k in mapping if mapping[k] != "-- Skip --"}
                df = raw.rename(columns=rename)
                st.success("Mapping applied.")
                st.dataframe(df.head())
            else:
                st.stop()
        else:
            st.stop()

    # If df is still None, stop
    if df is None:
        st.stop()

    # -------------------------
    # Basic cleaning & canonicalization
    # -------------------------
    df.columns = [str(c).strip() for c in df.columns]

    # Prefer first duplicate column if duplicates exist
    def prefer_col(cols, base):
        for c in cols:
            if c == base:
                return c
        for c in cols:
            if c.startswith(base + "__dup"):
                return c
        return None

    # Map duplicates to canonical
    cols = list(df.columns)
    canonical_map = {}
    for base in ["Timestamp","Machine_ID","Machine_Type","Fuel_Type","Shift","Operator_ID",
                 "Energy_Consumption_kWh","Energy_Predicted_kWh","Baseload_Drift_kWh","Cooling_Load_kWh",
                 "Fuel_Consumption","Compressed_Air_CFM","Voltage_Instability","Power_Factor","Ambient_Temp_C",
                 "Output_Units","Energy_Intensity_kWh_per_Unit","Peak_Load_Flag","Idle_Flag","Wastage_Severity","CO2_Emissions_kg"]:
        found = prefer_col(cols, base)
        if found and found != base:
            canonical_map[found] = base
    if canonical_map:
        df = df.rename(columns=canonical_map)

    # Ensure Timestamp
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    else:
        # if no Timestamp, create index-based synthetic timestamps
        df["Timestamp"] = pd.date_range(start=datetime.now()-timedelta(days=30), periods=len(df), freq='H')

    # Ensure numeric columns exist and convert
    numeric_cols = ["Energy_Consumption_kWh","Energy_Predicted_kWh","Baseload_Drift_kWh","Cooling_Load_kWh",
                    "Fuel_Consumption","Compressed_Air_CFM","Voltage_Instability","Power_Factor","Ambient_Temp_C",
                    "Output_Units","Energy_Intensity_kWh_per_Unit","Wastage_Severity","CO2_Emissions_kg"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # If Peak_Load_Flag or Idle_Flag missing, create default zeros
    for flag in ["Peak_Load_Flag","Idle_Flag"]:
        if flag not in df.columns:
            df[flag] = 0

    # Fill missing numeric values with medians where reasonable
    for c in numeric_cols:
        if c in df.columns:
            if df[c].notna().any():
                df[c] = df[c].fillna(df[c].median())
            else:
                df[c] = df[c].fillna(0)

    # -------------------------
    # Date-range slider (slicer)
    # -------------------------
    st.markdown("### Filters")
    min_ts = df["Timestamp"].min()
    max_ts = df["Timestamp"].max()
    if pd.isna(min_ts) or pd.isna(max_ts):
        st.warning("Timestamps not found or all invalid. Showing full dataset.")
        min_ts = datetime.now() - timedelta(days=30)
        max_ts = datetime.now()

    date_range = st.slider("Select date range", value=(min_ts, max_ts), min_value=min_ts, max_value=max_ts, format="YYYY-MM-DD HH:mm")
    start_dt, end_dt = date_range
    filt = df[(df["Timestamp"] >= pd.to_datetime(start_dt)) & (df["Timestamp"] <= pd.to_datetime(end_dt))].copy()

    # Other filters
    machine_ids = sorted(filt["Machine_ID"].unique().tolist()) if "Machine_ID" in filt.columns else []
    machine_types = sorted(filt["Machine_Type"].unique().tolist()) if "Machine_Type" in filt.columns else []
    shifts = sorted(filt["Shift"].unique().tolist()) if "Shift" in filt.columns else []

    col1, col2, col3 = st.columns([2,2,2])
    with col1:
        sel_machines = st.multiselect("Machine ID", machine_ids, default=machine_ids)
    with col2:
        sel_types = st.multiselect("Machine Type", machine_types, default=machine_types)
    with col3:
        sel_shifts = st.multiselect("Shift", shifts, default=shifts)

    if sel_machines:
        filt = filt[filt["Machine_ID"].isin(sel_machines)]
    if sel_types:
        filt = filt[filt["Machine_Type"].isin(sel_types)]
    if sel_shifts:
        filt = filt[filt["Shift"].isin(sel_shifts)]

    st.markdown("#### Data preview (first 10 rows)")
    st.dataframe(filt.head(10), use_container_width=True)
    download_df_button(filt.head(200), "filtered_energy_preview.csv", "Download preview CSV")

    # -------------------------
    # KPIs (dynamic) - show numbers now
    # -------------------------
    st.markdown("### KPIs (dynamic)")
    kcols = st.columns(5)
    total_energy = float(filt["Energy_Consumption_kWh"].sum()) if "Energy_Consumption_kWh" in filt.columns else 0.0
    avg_energy_shift = float(filt.groupby("Shift")["Energy_Consumption_kWh"].mean().mean()) if "Energy_Consumption_kWh" in filt.columns and "Shift" in filt.columns else 0.0
    peak_events = int(filt["Peak_Load_Flag"].sum()) if "Peak_Load_Flag" in filt.columns else 0
    total_fuel = float(filt["Fuel_Consumption"].sum()) if "Fuel_Consumption" in filt.columns else 0.0
    total_co2 = float(filt["CO2_Emissions_kg"].sum()) if "CO2_Emissions_kg" in filt.columns else 0.0

    kcols[0].markdown(f"<div class='kpi'>Total Energy<br><div style='font-size:16px;font-weight:600'>{total_energy:,.0f} kWh</div></div>", unsafe_allow_html=True)
    kcols[1].markdown(f"<div class='kpi'>Avg Energy / Shift<br><div style='font-size:16px;font-weight:600'>{avg_energy_shift:,.2f} kWh</div></div>", unsafe_allow_html=True)
    kcols[2].markdown(f"<div class='kpi'>Peak Events<br><div style='font-size:16px;font-weight:600'>{peak_events}</div></div>", unsafe_allow_html=True)
    kcols[3].markdown(f"<div class='kpi'>Fuel Usage<br><div style='font-size:16px;font-weight:600'>{total_fuel:,.1f}</div></div>", unsafe_allow_html=True)
    kcols[4].markdown(f"<div class='kpi'>CO₂ Emissions<br><div style='font-size:16px;font-weight:600'>{total_co2:,.0f} kg</div></div>", unsafe_allow_html=True)

    # -------------------------
    # Charts (many, different than earlier)
    # -------------------------
    st.markdown("### Visualizations")

    # 1. Time series: total energy (aggregated by day)
    if "Timestamp" in filt.columns and "Energy_Consumption_kWh" in filt.columns:
        daily = filt.set_index("Timestamp").resample("D").agg({"Energy_Consumption_kWh":"sum"}).reset_index()
        fig = px.line(daily, x="Timestamp", y="Energy_Consumption_kWh", title="Daily Total Energy Consumption")
        st.plotly_chart(fig, use_container_width=True)

    # 2. Rolling average energy per machine (line)
    if "Timestamp" in filt.columns and "Energy_Consumption_kWh" in filt.columns and "Machine_ID" in filt.columns:
        roll_df = filt.sort_values("Timestamp").groupby(["Machine_ID"]).apply(lambda d: d.set_index("Timestamp")["Energy_Consumption_kWh"].rolling("7D").mean().rename("rolling_7d")).reset_index()
        fig2 = px.line(roll_df, x="Timestamp", y="rolling_7d", color="Machine_ID", title="7-day Rolling Avg Energy per Machine")
        st.plotly_chart(fig2, use_container_width=True)

    # 3. Scatter: Energy vs Output Units
    if "Energy_Consumption_kWh" in filt.columns and "Output_Units" in filt.columns:
        fig3 = px.scatter(filt, x="Output_Units", y="Energy_Consumption_kWh", color="Machine_Type", hover_data=["Operator_ID"], title="Energy vs Output Units")
        st.plotly_chart(fig3, use_container_width=True)

    # 4. Fuel consumption distribution by fuel type
    if "Fuel_Consumption" in filt.columns and "Fuel_Type" in filt.columns:
        fig4 = px.box(filt, x="Fuel_Type", y="Fuel_Consumption", title="Fuel Consumption by Fuel Type")
        st.plotly_chart(fig4, use_container_width=True)

    # 5. Heatmap: correlation among numeric features
    num_for_corr = [c for c in numeric_cols if c in filt.columns]
    if len(num_for_corr) >= 3:
        corr = filt[num_for_corr].corr()
        fig5 = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="Blues"))
        fig5.update_layout(title="Correlation heatmap (numeric features)", height=500)
        st.plotly_chart(fig5, use_container_width=True)

    # 6. Bar: Peak counts by Machine_Type
    if "Peak_Load_Flag" in filt.columns and "Machine_Type" in filt.columns:
        peak_counts = filt[filt["Peak_Load_Flag"]==1].groupby("Machine_Type").size().reset_index(name="peak_count")
        fig6 = px.bar(peak_counts, x="Machine_Type", y="peak_count", title="Peak Load Events by Machine Type")
        st.plotly_chart(fig6, use_container_width=True)

    # 7. Histogram - Energy intensity
    if "Energy_Intensity_kWh_per_Unit" in filt.columns:
        fig7 = px.histogram(filt, x="Energy_Intensity_kWh_per_Unit", nbins=40, title="Distribution: Energy Intensity (kWh per Unit)")
        st.plotly_chart(fig7, use_container_width=True)

    # 8. Time of day pattern: avg energy by hour
    if "Timestamp" in filt.columns and "Energy_Consumption_kWh" in filt.columns:
        filt["hour"] = filt["Timestamp"].dt.hour
        by_hour = filt.groupby("hour")["Energy_Consumption_kWh"].mean().reset_index()
        fig8 = px.line(by_hour, x="hour", y="Energy_Consumption_kWh", title="Average Energy by Hour of Day")
        st.plotly_chart(fig8, use_container_width=True)

    # 9. Operator performance: energy intensity by operator
    if "Operator_ID" in filt.columns and "Energy_Intensity_kWh_per_Unit" in filt.columns:
        op_stats = filt.groupby("Operator_ID")["Energy_Intensity_kWh_per_Unit"].mean().reset_index().sort_values(by="Energy_Intensity_kWh_per_Unit")
        fig9 = px.bar(op_stats.head(20), x="Operator_ID", y="Energy_Intensity_kWh_per_Unit", title="Operator Energy Intensity (top 20)")
        st.plotly_chart(fig9, use_container_width=True)

    # 10. Cumulative energy by machine (stacked)
    if "Machine_ID" in filt.columns and "Energy_Consumption_kWh" in filt.columns:
        cum = filt.groupby(["Timestamp","Machine_ID"])["Energy_Consumption_kWh"].sum().reset_index()
        cum_pivot = cum.pivot(index="Timestamp", columns="Machine_ID", values="Energy_Consumption_kWh").fillna(0)
        cum_cum = cum_pivot.cumsum()
        if cum_cum.shape[1] > 0:
            fig10 = go.Figure()
            for col in cum_cum.columns:
                fig10.add_trace(go.Scatter(x=cum_cum.index, y=cum_cum[col], stackgroup='one', name=str(col)))
            fig10.update_layout(title="Cumulative Energy by Machine (stacked)")
            st.plotly_chart(fig10, use_container_width=True)

    # 11. Peak vs non-peak energy box
    if "Peak_Load_Flag" in filt.columns and "Energy_Consumption_kWh" in filt.columns:
        fig11 = px.box(filt, x="Peak_Load_Flag", y="Energy_Consumption_kWh", title="Energy: Peak vs Non-Peak")
        st.plotly_chart(fig11, use_container_width=True)

    # 12. Scatter: Cooling load vs Energy consumption
    if "Cooling_Load_kWh" in filt.columns and "Energy_Consumption_kWh" in filt.columns:
        fig12 = px.scatter(filt, x="Cooling_Load_kWh", y="Energy_Consumption_kWh", trendline="ols", title="Cooling Load vs Energy")
        st.plotly_chart(fig12, use_container_width=True)

    # 13. Fuel vs Output scatter
    if "Fuel_Consumption" in filt.columns and "Output_Units" in filt.columns:
        fig13 = px.scatter(filt, x="Output_Units", y="Fuel_Consumption", color="Fuel_Type", title="Fuel Consumption vs Output Units")
        st.plotly_chart(fig13, use_container_width=True)

    # 14. Voltage Instability distribution by shift
    if "Voltage_Instability" in filt.columns and "Shift" in filt.columns:
        fig14 = px.violin(filt, x="Shift", y="Voltage_Instability", box=True, points="all", title="Voltage Instability by Shift")
        st.plotly_chart(fig14, use_container_width=True)

    # 15. Sankey-like simple flow: top machines -> peak events (approx using bar chart pairs)
    if "Machine_ID" in filt.columns and "Peak_Load_Flag" in filt.columns:
        top_machines = filt.groupby("Machine_ID")["Peak_Load_Flag"].sum().sort_values(ascending=False).head(10).reset_index()
        fig15 = px.bar(top_machines, x="Machine_ID", y="Peak_Load_Flag", title="Top machines by peak events (top 10)")
        st.plotly_chart(fig15, use_container_width=True)

    # -------------------------
    # Automated Insights (text + table)
    # -------------------------
    st.markdown("### Automated Insights (text highlights followed by table)")

    # quick highlights
    highlights = []
    # top energy machines
    if "Machine_ID" in filt.columns and "Energy_Consumption_kWh" in filt.columns:
        top_energy = filt.groupby("Machine_ID")["Energy_Consumption_kWh"].sum().sort_values(ascending=False).head(5)
        highlights.append("Top energy-consuming machines: " + ", ".join([str(x) for x in top_energy.index.tolist()]))

    # machines with highest wastage severity
    if "Machine_ID" in filt.columns and "Wastage_Severity" in filt.columns:
        top_waste = filt.groupby("Machine_ID")["Wastage_Severity"].mean().sort_values(ascending=False).head(5)
        highlights.append("Machines with highest average wastage severity: " + ", ".join([str(x) for x in top_waste.index.tolist()]))

    # shift with highest avg energy
    if "Shift" in filt.columns and "Energy_Consumption_kWh" in filt.columns:
        shift_max = filt.groupby("Shift")["Energy_Consumption_kWh"].mean().idxmax()
        highlights.append(f"Shift with highest avg energy: {shift_max}")

    # display text highlights
    for h in highlights:
        st.markdown(f"- {h}")

    # build insights table (structured)
    insights_rows = []
    if "Machine_ID" in filt.columns:
        grp = filt.groupby("Machine_ID").agg(
            total_energy=("Energy_Consumption_kWh","sum") if "Energy_Consumption_kWh" in filt.columns else ("Energy_Consumption_kWh", lambda x: 0),
            avg_waste=("Wastage_Severity","mean") if "Wastage_Severity" in filt.columns else ("Wastage_Severity", lambda x: 0),
            peak_events=("Peak_Load_Flag","sum") if "Peak_Load_Flag" in filt.columns else ("Peak_Load_Flag", lambda x: 0)
        ).reset_index()
        grp = grp.sort_values("total_energy", ascending=False).head(50)
        for _, r in grp.iterrows():
            insights_rows.append({
                "Machine_ID": r["Machine_ID"],
                "Total_Energy": round(float(r["total_energy"]),2),
                "Avg_Wastage_Severity": round(float(r["avg_waste"]),3) if not pd.isna(r["avg_waste"]) else None,
                "Peak_Events": int(r["peak_events"])
            })
    insights_df = pd.DataFrame(insights_rows)
    if insights_df.empty:
        st.info("No insights could be generated for filtered dataset.")
    else:
        st.dataframe(insights_df.head(200), use_container_width=True)
        download_df_button(insights_df, "automated_insights_energy.csv", "Download insights")

    # -------------------------
    # Anomaly Detection (IsolationForest)
    # -------------------------
    st.markdown("### Unsupervised Anomaly Detection (IsolationForest)")
    iso_features = [c for c in ["Energy_Consumption_kWh","Fuel_Consumption","Energy_Intensity_kWh_per_Unit","Cooling_Load_kWh","Voltage_Instability"] if c in filt.columns]
    anomalies_df = pd.DataFrame()
    if len(iso_features) >= 2 and len(filt) >= 30:
        X_iso = filt[iso_features].fillna(0)
        iso = IsolationForest(contamination=0.02, random_state=42)
        filt["_iso_pred"] = iso.fit_predict(X_iso)
        filt["_iso_score"] = iso.decision_function(X_iso)
        anomalies_df = filt[filt["_iso_pred"] == -1].sort_values("_iso_score").copy()
        st.markdown(f"Detected anomalies: {len(anomalies_df)}")
        if not anomalies_df.empty:
            st.dataframe(anomalies_df.head(200), use_container_width=True)
            download_df_button(anomalies_df, "anomalies_energy.csv", "Download anomalies")
    else:
        st.info("Insufficient data/features for anomaly detection (need >=2 iso features and >=30 rows).")

    # -------------------------
    # Supervised ML models (3) - trains when enough data exists
    # -------------------------
    st.markdown("### ML Models & Predictions")

    # Helper to train and return results
    def train_regression(X, y, name="reg"):
        results = {}
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=150, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = math.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            results.update({"model": model, "X_test_idx": X_test.index if hasattr(X_test, 'index') else None,
                            "y_test": y_test, "preds": preds, "rmse": rmse, "r2": r2})
        except Exception as e:
            results.update({"error": str(e)})
        return results

    def train_classifier(X, y, name="clf"):
        results = {}
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)
            model = RandomForestClassifier(n_estimators=150, random_state=42)
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else model.predict(X_test)
            preds_label = model.predict(X_test)
            acc = accuracy_score(y_test, preds_label)
            try:
                auc = roc_auc_score(y_test, probs)
            except:
                auc = None
            results.update({"model": model, "X_test_idx": X_test.index if hasattr(X_test, 'index') else None,
                            "y_test": y_test, "probs": probs, "preds_label": preds_label, "acc": acc, "auc": auc})
        except Exception as e:
            results.update({"error": str(e)})
        return results

    # Model A: Energy consumption prediction (regression)
    st.markdown("**Model A — Energy Consumption Prediction (regression)**")
    features_A = [c for c in ["Baseload_Drift_kWh","Cooling_Load_kWh","Power_Factor","Ambient_Temp_C","Output_Units","Compressed_Air_CFM"] if c in filt.columns]
    if "Energy_Consumption_kWh" in filt.columns and len(filt) >= 60 and len(features_A) >= 2:
        X_A = filt[features_A].fillna(0)
        y_A = filt["Energy_Consumption_kWh"].fillna(0)
        res_A = train_regression(X_A, y_A, name="energy_reg")
        if "error" in res_A:
            st.error("Energy model error: " + res_A["error"])
        else:
            st.write(f"RMSE: {res_A['rmse']:.2f}, R²: {res_A['r2']:.3f}")
            # Build downloadable results: actual vs predicted + features
            Xtest = X_A.loc[res_A['X_test_idx']].reset_index(drop=True)
            outA = Xtest.copy()
            outA["Actual_Energy"] = res_A["y_test"].reset_index(drop=True)
            outA["Predicted_Energy"] = res_A["preds"]
            st.dataframe(outA.head(12))
            download_df_button(outA, "energy_predictions.csv", "Download energy predictions")
    else:
        st.info("Not enough data/features for Energy prediction (need >=60 rows and >=2 features).")

    # Model B: Peak load classification
    st.markdown("**Model B — Peak Load Classification**")
    features_B = [c for c in ["Energy_Consumption_kWh","Cooling_Load_kWh","Voltage_Instability","Power_Factor","Ambient_Temp_C"] if c in filt.columns]
    if "Peak_Load_Flag" in filt.columns and len(filt) >= 80 and len(features_B) >= 2:
        X_B = filt[features_B].fillna(0)
        y_B = filt["Peak_Load_Flag"].astype(int).fillna(0)
        res_B = train_classifier(X_B, y_B, name="peak_clf")
        if "error" in res_B:
            st.error("Peak classification error: " + res_B["error"])
        else:
            st.write(f"Accuracy: {res_B['acc']:.3f}" + (f", ROC AUC: {res_B['auc']:.3f}" if res_B['auc'] is not None else ""))
            XtestB = X_B.loc[res_B['X_test_idx']].reset_index(drop=True)
            outB = XtestB.copy()
            outB["Actual_Peak"] = res_B["y_test"].reset_index(drop=True)
            outB["Predicted_Prob_Peak"] = res_B["probs"]
            outB["Predicted_Label"] = res_B["preds_label"]
            st.dataframe(outB.head(12))
            download_df_button(outB, "peak_predictions.csv", "Download peak predictions")
    else:
        st.info("Not enough data/features for Peak classification (need >=80 rows and >=2 features).")

    # Model C: Fuel consumption prediction (regression)
    st.markdown("**Model C — Fuel Consumption Prediction (regression)**")
    features_C = [c for c in ["Energy_Consumption_kWh","Output_Units","Ambient_Temp_C","Compressed_Air_CFM"] if c in filt.columns]
    if "Fuel_Consumption" in filt.columns and len(filt) >= 60 and len(features_C) >= 2:
        X_C = filt[features_C].fillna(0)
        y_C = filt["Fuel_Consumption"].fillna(0)
        res_C = train_regression(X_C, y_C, name="fuel_reg")
        if "error" in res_C:
            st.error("Fuel model error: " + res_C["error"])
        else:
            st.write(f"RMSE: {res_C['rmse']:.3f}, R²: {res_C['r2']:.3f}")
            XtestC = X_C.loc[res_C['X_test_idx']].reset_index(drop=True)
            outC = XtestC.copy()
            outC["Actual_Fuel"] = res_C["y_test"].reset_index(drop=True)
            outC["Predicted_Fuel"] = res_C["preds"]
            st.dataframe(outC.head(12))
            download_df_button(outC, "fuel_predictions.csv", "Download fuel predictions")
    else:
        st.info("Not enough data/features for Fuel prediction (need >=60 rows and >=2 features).")

    # -------------------------
    # Final notes & export
    # -------------------------
    st.markdown("### Done — export any prediction or insights CSVs above. Act on the top machines or shifts with high wastage severity first.")
    st.markdown("<div class='small'>Tip: Hook this to your scheduler or maintenance ticketing to auto-create tasks for top-risk machines.</div>", unsafe_allow_html=True)
