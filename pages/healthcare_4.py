import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import math
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------------
# PAGE CONFIG & HIDE SIDEBAR
# --------------------------------------------------------
st.set_page_config(page_title="Ambulance Ops & Routing Lab", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebarNav"] {display:none;}
section[data-testid="stSidebar"] {display:none;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# HEADER & LOGO
# --------------------------------------------------------
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

# --------------------------------------------------------
# GLOBAL CSS – MARKETING LAB STYLE
# --------------------------------------------------------
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
    font-size:18px !important;
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
    padding:16px;
    border-radius:14px;
    background:white;
    border:1px solid #e5e5e5;
    box-shadow:0 2px 10px rgba(0,0,0,0.10);
    transition:0.25s ease;
    text-align:center;
    font-size:17px !important;
    font-weight:500 !important;
    color:#064b86 !important;
    margin-bottom:10px;
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

st.markdown("<div class='big-header'>Ambulance Ops & Routing Lab</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# UTILITIES & CONSTANTS
# --------------------------------------------------------
REQUIRED_COLS = [
    "Record_ID","Ambulance_ID","Driver_ID","Incident_Type","Patient_Severity",
    "Dispatch_Time","Arrival_Time_at_Scene","Depart_Scene_Time","Arrival_Time_at_Hospital",
    "Incident_Lat","Incident_Lon","Nearest_Hospital","Nearest_Hospital_Distance_km",
    "Dropoff_Hospital","Dropoff_Hospital_Distance_km","Distance_km","Travel_Time_min",
    "Response_Time_min","Crew_Count","Transport_Mode","Fuel_Consumed_Ltrs","Trip_Cost_Rs",
    "Weather","Road_Condition","Is_Interfacility_Transfer","Outcome",
    "Dropoff_Hospital_Beds","Dropoff_Hospital_Occupancy_pct"
]

def download_df(df, filename, label):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")

def to_currency(x):
    try:
        return "₹ " + f"{float(x):,.2f}"
    except Exception:
        return "N/A"

def safe_to_datetime(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def normalize_name(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")

def auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort mapping to REQUIRED_COLS based on fuzzy names."""
    orig_cols = df.columns.tolist()
    norm_map = {normalize_name(c): c for c in orig_cols}

    desired_map = {
        "Record_ID": ["record_id", "record", "id"],
        "Ambulance_ID": ["ambulance_id", "ambulance", "amb_id"],
        "Driver_ID": ["driver_id", "driver"],
        "Incident_Type": ["incident_type", "incident"],
        "Patient_Severity": ["patient_severity", "severity"],
        "Dispatch_Time": ["dispatch_time", "dispatch"],
        "Arrival_Time_at_Scene": ["arrival_time_at_scene", "arrival_scene"],
        "Depart_Scene_Time": ["depart_scene_time", "depart_scene"],
        "Arrival_Time_at_Hospital": ["arrival_time_at_hospital", "arrival_hospital"],
        "Incident_Lat": ["incident_lat", "lat", "latitude"],
        "Incident_Lon": ["incident_lon", "lon", "lng", "long", "longitude"],
        "Nearest_Hospital": ["nearest_hospital"],
        "Nearest_Hospital_Distance_km": ["nearest_hospital_distance_km", "nearest_distance_km", "dist_km_to_incident"],
        "Dropoff_Hospital": ["dropoff_hospital", "dropoff"],
        "Dropoff_Hospital_Distance_km": ["dropoff_hospital_distance_km", "dropoff_dist_km"],
        "Distance_km": ["distance_km", "distance_total_km", "distance"],
        "Travel_Time_min": ["travel_time_min", "travel_time", "travel_minutes"],
        "Response_Time_min": ["response_time_min", "response_time", "response_minutes"],
        "Crew_Count": ["crew_count", "crew"],
        "Transport_Mode": ["transport_mode", "transport"],
        "Fuel_Consumed_Ltrs": ["fuel_consumed_ltrs", "fuel_consumed", "fuel_ltrs"],
        "Trip_Cost_Rs": ["trip_cost_rs", "trip_cost", "cost"],
        "Weather": ["weather"],
        "Road_Condition": ["road_condition"],
        "Is_Interfacility_Transfer": ["is_interfacility_transfer", "interfacility_transfer", "is_transfer"],
        "Outcome": ["outcome"],
        "Dropoff_Hospital_Beds": ["dropoff_hospital_beds", "beds"],
        "Dropoff_Hospital_Occupancy_pct": ["dropoff_hospital_occupancy_pct", "occupancy_pct", "occupancy"]
    }

    rename = {}
    for target, candidates in desired_map.items():
        for cand in candidates:
            cand_norm = cand.lower().replace(" ", "_")
            # exact normalized match
            if cand_norm in norm_map:
                rename[norm_map[cand_norm]] = target
                break
            # substring fuzzy match
            for orig in orig_cols:
                if cand_norm in normalize_name(orig):
                    rename[orig] = target
                    break
            if target in rename.values():
                break

    if rename:
        df = df.rename(columns=rename)
    return df

# --------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------
if "df_amb" not in st.session_state:
    st.session_state.df_amb = None
if "pipeline_reg_amb" not in st.session_state:
    st.session_state.pipeline_reg_amb = None
if "pipeline_clf_amb" not in st.session_state:
    st.session_state.pipeline_clf_amb = None

# --------------------------------------------------------
# TABS
# --------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# ========================================================
# TAB 1 – OVERVIEW
# ========================================================
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <b>What this lab does:</b><br><br>
    A unified <b>Ambulance & Hospital Ops Lab</b> to track dispatch, response time, routing, hospital load,
    and transfer decisions. It combines trip telemetry and hospital capacity signals to show where time is being
    lost and where operations can be tightened.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Purpose</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Monitor <b>response time</b> and travel time patterns<br>
        • Identify <b>bottleneck hospitals & routes</b><br>
        • Quantify impact of <b>distance, severity, weather</b> on delays<br>
        • Support <b>dispatch allocation & routing decisions</b><br>
        • Build ML models for <b>response time & transfer probability</b>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Reduced <b>response & on-scene times</b><br>
        • Lower <b>trip cost per incident</b><br>
        • Better <b>hospital throughput & ER load management</b><br>
        • More predictable <b>interfacility transfers</b><br>
        • Stronger audit trail for <b>regulators & payors</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Key KPIs Tracked</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Average Response Time (min)</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Average Travel Time (min)</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Total Trip Cost (₹)</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Interfacility Transfer Share (%)</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Who Should Use This</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    Emergency control room teams, ambulance vendors, hospital operations leaders, and data analysts who need a 
    <b>single operational cockpit</b> for ambulance fleet performance, routing efficiency, hospital drop-off patterns,
    and ML-backed decision support.
    </div>
    """, unsafe_allow_html=True)

# ========================================================
# TAB 2 – IMPORTANT ATTRIBUTES
# ========================================================
with tab2:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    data_dict = {
        "Record_ID": "Unique identifier for each trip/record.",
        "Ambulance_ID": "Identifier of the ambulance assigned to the incident.",
        "Driver_ID": "Identifier of the driver handling the trip.",
        "Incident_Type": "Type/category of incident (cardiac, trauma, stroke, etc.).",
        "Patient_Severity": "Triage severity level (e.g., Low/Medium/High/Critical).",
        "Dispatch_Time": "Timestamp when ambulance was dispatched.",
        "Arrival_Time_at_Scene": "Timestamp when ambulance arrived at incident location.",
        "Depart_Scene_Time": "Timestamp when ambulance departed from the scene.",
        "Arrival_Time_at_Hospital": "Timestamp when ambulance arrived at drop-off hospital.",
        "Incident_Lat": "Latitude of incident location.",
        "Incident_Lon": "Longitude of incident location.",
        "Nearest_Hospital": "Name/ID of geographically nearest hospital from incident.",
        "Nearest_Hospital_Distance_km": "Distance to nearest hospital from incident (km).",
        "Dropoff_Hospital": "Actual hospital where patient was dropped.",
        "Dropoff_Hospital_Distance_km": "Distance from incident to drop-off hospital (km).",
        "Distance_km": "Total trip distance traveled (km).",
        "Travel_Time_min": "Travel time between key legs (typically scene to hospital) in minutes.",
        "Response_Time_min": "Dispatch to scene arrival time difference in minutes.",
        "Crew_Count": "Number of crew members present in the ambulance.",
        "Transport_Mode": "Transport configuration (BLS/ALS/ICU/other).",
        "Fuel_Consumed_Ltrs": "Fuel consumed during the trip (liters).",
        "Trip_Cost_Rs": "Operational cost attributed to the trip (₹).",
        "Weather": "Weather conditions during incident / transport.",
        "Road_Condition": "Road condition label (Normal, Congested, Blocked, etc.).",
        "Is_Interfacility_Transfer": "Flag if this trip is a hospital-to-hospital transfer (0/1).",
        "Outcome": "Clinical or routing outcome label (Completed, Cancelled, etc.).",
        "Dropoff_Hospital_Beds": "Total bed capacity at drop-off hospital (approx).",
        "Dropoff_Hospital_Occupancy_pct": "Percent occupancy at drop-off hospital at that time."
    }

    dict_df = pd.DataFrame(
        [{"Attribute": k, "Description": v} for k, v in data_dict.items()]
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
        independent_vars = [
            "Incident_Type",
            "Patient_Severity",
            "Incident_Lat",
            "Incident_Lon",
            "Nearest_Hospital",
            "Nearest_Hospital_Distance_km",
            "Dropoff_Hospital",
            "Dropoff_Hospital_Distance_km",
            "Distance_km",
            "Crew_Count",
            "Transport_Mode",
            "Fuel_Consumed_Ltrs",
            "Weather",
            "Road_Condition",
            "Dropoff_Hospital_Beds",
            "Dropoff_Hospital_Occupancy_pct"
        ]
        for v in independent_vars:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        dependent_vars = [
            "Response_Time_min",
            "Travel_Time_min",
            "Trip_Cost_Rs",
            "Is_Interfacility_Transfer"
        ]
        for v in dependent_vars:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# ========================================================
# TAB 3 – APPLICATION
# ========================================================
with tab3:
    st.markdown('<div class="section-title">Step 1: Load dataset</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Dataset option:",
        ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"],
        horizontal=True
    )

    df = None

    # -------------------------
    # Default dataset
    # -------------------------
    if mode == "Default dataset":
        DEFAULT_PATH = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/healthcare/ambulance_dataset.csv"
        try:
            df = pd.read_csv(DEFAULT_PATH)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("Default dataset loaded from GitHub.")
            st.dataframe(df.head(10), use_container_width=True)
            st.session_state.df_amb = df
        except Exception as e:
            st.error(f"Failed to load default dataset from {DEFAULT_PATH}: {e}")
            st.stop()

    # -------------------------
    # Upload CSV
    # -------------------------
    elif mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("File uploaded.")
            st.dataframe(df.head(10), use_container_width=True)
            st.session_state.df_amb = df
            sample_small = df.head(5).to_csv(index=False)
            st.download_button("Download sample (first 5 rows)", sample_small, "sample_uploaded_5rows.csv", "text/csv")

    # -------------------------
    # Upload + mapping
    # -------------------------
    else:
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head(5), use_container_width=True)
            st.markdown("Map your columns to required fields:", unsafe_allow_html=True)
            mapping = {}
            for req in REQUIRED_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + list(raw.columns))
            if st.button("Apply mapping"):
                missing = [k for k, v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v: k for k, v in mapping.items()})
                    st.success("Mapping applied.")
                    st.dataframe(df.head(10), use_container_width=True)
                    st.session_state.df_amb = df
                    sample_small = df.head(5).to_csv(index=False)
                    st.download_button("Download mapped sample (5 rows)", sample_small, "mapped_sample_5rows.csv", "text/csv")

    # If still nothing, exit
    if st.session_state.df_amb is None:
        st.stop()

    df = st.session_state.df_amb.copy()

    # -------------------------
    # Clean & type-coerce
    # -------------------------
    for col in ["Dispatch_Time", "Arrival_Time_at_Scene", "Depart_Scene_Time", "Arrival_Time_at_Hospital"]:
        df = safe_to_datetime(df, col)

    numeric_cols = [
        "Nearest_Hospital_Distance_km",
        "Dropoff_Hospital_Distance_km",
        "Distance_km",
        "Travel_Time_min",
        "Response_Time_min",
        "Crew_Count",
        "Fuel_Consumed_Ltrs",
        "Trip_Cost_Rs",
        "Dropoff_Hospital_Beds",
        "Dropoff_Hospital_Occupancy_pct"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Is_Interfacility_Transfer" in df.columns:
        df["Is_Interfacility_Transfer"] = df["Is_Interfacility_Transfer"].apply(
            lambda x: 1 if str(x).lower() in ["1", "true", "yes", "y", "t"] else 0
        )

    # -------------------------
    # Filters & preview
    # -------------------------
    st.markdown('<div class="section-title">Step 2: Filters & Preview</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 2, 1])
    incidents = sorted(df["Incident_Type"].dropna().unique().tolist()) if "Incident_Type" in df.columns else []
    hospitals = sorted(df["Dropoff_Hospital"].dropna().unique().tolist()) if "Dropoff_Hospital" in df.columns else []

    with c1:
        sel_incidents = st.multiselect(
            "Incident Type",
            options=incidents,
            default=incidents[:5] if incidents else []
        )
    with c2:
        sel_hosp = st.multiselect(
            "Dropoff Hospital",
            options=hospitals,
            default=hospitals[:5] if hospitals else []
        )
    with c3:
        if "Dispatch_Time" in df.columns and df["Dispatch_Time"].notna().any():
            min_d = df["Dispatch_Time"].min().date()
            max_d = df["Dispatch_Time"].max().date()
            date_range = st.date_input("Dispatch date range", value=(min_d, max_d))
        else:
            date_range = None

    filt = df.copy()
    if sel_incidents:
        filt = filt[filt["Incident_Type"].isin(sel_incidents)]
    if sel_hosp:
        filt = filt[filt["Dropoff_Hospital"].isin(sel_hosp)]
    if date_range is not None and "Dispatch_Time" in filt.columns:
        start = pd.to_datetime(date_range[0])
        end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filt = filt[(filt["Dispatch_Time"] >= start) & (filt["Dispatch_Time"] <= end)]

    st.markdown(f"""
    <div style='padding:8px; background:#eef4ff; border-radius:8px; font-weight:600; margin-bottom:8px;'>
    Filtered Rows: {len(filt)} of {len(df)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Filtered preview (first 10 rows)")
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(100), "ambulance_filtered_preview.csv", "Download filtered preview (first 100 rows)")

    # -------------------------
    # Key Metrics
    # -------------------------
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Filtered Trips", f"{len(filt):,}")

    if "Response_Time_min" in filt.columns and filt["Response_Time_min"].dropna().shape[0] > 0:
        m2.metric("Avg Response (min)", f"{filt['Response_Time_min'].mean():.1f}")
    else:
        m2.metric("Avg Response (min)", "N/A")

    if "Travel_Time_min" in filt.columns and filt["Travel_Time_min"].dropna().shape[0] > 0:
        m3.metric("Avg Travel (min)", f"{filt['Travel_Time_min'].mean():.1f}")
    else:
        m3.metric("Avg Travel (min)", "N/A")

    if "Trip_Cost_Rs" in filt.columns:
        total_cost = filt["Trip_Cost_Rs"].sum(min_count=1)
        m4.metric("Total Trip Cost", to_currency(total_cost) if not pd.isna(total_cost) else "N/A")
    else:
        m4.metric("Total Trip Cost", "N/A")

    # -------------------------
    # Visuals
    # -------------------------
    st.markdown('<div class="section-title">Visuals</div>', unsafe_allow_html=True)

    # Top dropoff hospitals
    if "Dropoff_Hospital" in filt.columns:
        agg = (
            filt.groupby("Dropoff_Hospital")
            .agg({"Record_ID": "count", "Trip_Cost_Rs": "sum"})
            .rename(columns={"Record_ID": "Trips"})
            .reset_index()
        )
        if not agg.empty:
            agg = agg.sort_values("Trips", ascending=False)
            fig = px.bar(
                agg.head(10),
                x="Dropoff_Hospital",
                y="Trips",
                title="Top Dropoff Hospitals by Trip Count",
                text="Trips"
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    # Incident density map
    if {"Incident_Lat", "Incident_Lon"}.issubset(set(filt.columns)):
        st.markdown("#### Incident density map")
        map_df = filt.dropna(subset=["Incident_Lat", "Incident_Lon"])[
            ["Incident_Lat", "Incident_Lon", "Response_Time_min"]
        ].rename(columns={"Incident_Lat": "lat", "Incident_Lon": "lon"})
        if not map_df.empty:
            st.map(map_df)
        else:
            st.info("No incident lat/lon points to map.")

    # Response vs Distance scatter
    if "Response_Time_min" in filt.columns and "Distance_km" in filt.columns:
        valid = filt.dropna(subset=["Distance_km", "Response_Time_min"])
        if not valid.empty:
            st.markdown("#### Response time vs Distance")
            color_col = "Patient_Severity" if "Patient_Severity" in valid.columns else None
            fig = px.scatter(
                valid,
                x="Distance_km",
                y="Response_Time_min",
                color=color_col,
                title="Response time vs Distance (km)"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Response time distribution
    if "Response_Time_min" in df.columns and df["Response_Time_min"].dropna().shape[0] > 0:
        st.markdown("#### Response time distribution")
        fig_hist = px.histogram(
            df.dropna(subset=["Response_Time_min"]),
            x="Response_Time_min",
            nbins=40,
            title="Response Time (min)"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # -------------------------
    # ML: Regression & Classification
    # -------------------------
    st.markdown('<div class="section-title">ML — Predict Response Time & Interfacility Transfer</div>', unsafe_allow_html=True)

    with st.expander("ML settings & train", expanded=False):
        can_train_reg = (
            "Response_Time_min" in filt.columns and
            len(filt.dropna(subset=["Response_Time_min"])) > 30
        )
        can_train_clf = (
            "Is_Interfacility_Transfer" in filt.columns and
            filt["Is_Interfacility_Transfer"].nunique() > 1 and
            len(filt.dropna(subset=["Is_Interfacility_Transfer"])) > 30
        )

        st.write(f"Regression available: {can_train_reg}  |  Classification available: {can_train_clf}")

        possible_features = [
            c for c in filt.columns
            if c not in [
                "Record_ID",
                "Dispatch_Time",
                "Arrival_Time_at_Scene",
                "Depart_Scene_Time",
                "Arrival_Time_at_Hospital",
                "Outcome"
            ]
        ]
        default_features = [
            "Distance_km",
            "Travel_Time_min",
            "Crew_Count",
            "Patient_Severity",
            "Transport_Mode"
        ]
        features = st.multiselect(
            "Features to use (ML)",
            options=possible_features,
            default=[f for f in default_features if f in possible_features][:6] or possible_features[:6]
        )

        if st.button("Train ML models"):
            if len(features) < 1:
                st.error("Choose at least 1 feature.")
            else:
                X = filt[features].copy()

                # numeric vs categorical split
                num_cols_ml = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
                cat_cols_ml = [c for c in X.columns if c not in num_cols_ml]

                # fill numeric NAs
                for nc in num_cols_ml:
                    X[nc] = pd.to_numeric(X[nc], errors="coerce").fillna(X[nc].median())

                # categorical to string
                for cc in cat_cols_ml:
                    X[cc] = X[cc].astype(str).fillna("NA")

                # ----------------- REGRESSION: Response_Time_min -----------------
                if can_train_reg:
                    y_reg = pd.to_numeric(filt["Response_Time_min"], errors="coerce")
                    Xr = X.loc[y_reg.dropna().index].copy()
                    yr = y_reg.dropna()
                    if len(Xr) < 10:
                        st.error("Not enough rows for regression after cleaning.")
                    else:
                        transformers = []
                        if len(num_cols_ml) > 0:
                            transformers.append(("num", StandardScaler(), num_cols_ml))
                        if len(cat_cols_ml) > 0:
                            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols_ml))

                        preproc = ColumnTransformer(transformers, remainder="drop") if transformers else None
                        reg = RandomForestRegressor(n_estimators=150, random_state=42)
                        if preproc is not None:
                            pipe_reg = Pipeline([("prep", preproc), ("model", reg)])
                        else:
                            pipe_reg = Pipeline([("model", reg)])

                        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
                            Xr, yr, test_size=0.2, random_state=42
                        )
                        with st.spinner("Training regression model..."):
                            pipe_reg.fit(X_train_r, y_train_r)
                        preds_r = pipe_reg.predict(X_test_r)
                        rmse = math.sqrt(mean_squared_error(y_test_r, preds_r))
                        r2 = r2_score(y_test_r, preds_r)
                        st.success(f"Regression trained — RMSE: {rmse:.2f}  |  R²: {r2:.3f}")
                        st.session_state.pipeline_reg_amb = pipe_reg

                # ----------------- CLASSIFICATION: Is_Interfacility_Transfer -----------------
                if can_train_clf:
                    y_clf = filt["Is_Interfacility_Transfer"].astype(int)
                    Xc = X.loc[y_clf.dropna().index].copy()
                    yc = y_clf.dropna()
                    if yc.nunique() < 2 or len(Xc) < 10:
                        st.error("Not enough class variety or rows for classification.")
                    else:
                        transformers_c = []
                        if len(num_cols_ml) > 0:
                            transformers_c.append(("num", StandardScaler(), num_cols_ml))
                        if len(cat_cols_ml) > 0:
                            transformers_c.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols_ml))

                        preproc_c = ColumnTransformer(transformers_c, remainder="drop") if transformers_c else None
                        clf = RandomForestClassifier(n_estimators=150, random_state=42)
                        if preproc_c is not None:
                            pipe_clf = Pipeline([("prep", preproc_c), ("model", clf)])
                        else:
                            pipe_clf = Pipeline([("model", clf)])

                        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
                            Xc, yc, test_size=0.2, random_state=42
                        )
                        with st.spinner("Training classifier..."):
                            pipe_clf.fit(X_train_c, y_train_c)
                        pred_c = pipe_clf.predict(X_test_c)
                        acc = accuracy_score(y_test_c, pred_c)
                        st.success(f"Classifier trained — Accuracy: {acc:.3f}")
                        st.session_state.pipeline_clf_amb = pipe_clf

    # -------------------------
    # Quick Predict
    # -------------------------
    st.markdown('<div class="section-title">Quick Predict (Single Trip)</div>', unsafe_allow_html=True)

    if st.session_state.pipeline_reg_amb is None and st.session_state.pipeline_clf_amb is None:
        st.info("Train at least one model in the ML panel above to enable quick prediction.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            inp_distance = st.number_input(
                "Distance_km",
                value=float(
                    filt["Distance_km"].median()
                    if "Distance_km" in filt.columns and filt["Distance_km"].dropna().shape[0] > 0
                    else 5.0
                )
            )
            inp_travel = st.number_input(
                "Travel_Time_min",
                value=float(
                    filt["Travel_Time_min"].median()
                    if "Travel_Time_min" in filt.columns and filt["Travel_Time_min"].dropna().shape[0] > 0
                    else 10.0
                )
            )
        with col2:
            inp_crew = st.number_input(
                "Crew_Count",
                value=int(
                    filt["Crew_Count"].median()
                    if "Crew_Count" in filt.columns and filt["Crew_Count"].dropna().shape[0] > 0
                    else 2
                )
            )
            if "Patient_Severity" in filt.columns:
                sev_options = sorted(filt["Patient_Severity"].dropna().unique().tolist())
                inp_sev = st.selectbox("Patient_Severity", options=sev_options, index=0 if sev_options else None)
            else:
                inp_sev = st.text_input("Patient_Severity", "Medium")
        with col3:
            if "Transport_Mode" in filt.columns:
                mode_options = sorted(filt["Transport_Mode"].dropna().unique().tolist())
                inp_mode = st.selectbox("Transport_Mode", options=mode_options, index=0 if mode_options else None)
            else:
                inp_mode = st.text_input("Transport_Mode", "Basic Life Support")

        if st.button("Predict (single trip)"):
            row = {}
            if "Distance_km" in filt.columns:
                row["Distance_km"] = inp_distance
            if "Travel_Time_min" in filt.columns:
                row["Travel_Time_min"] = inp_travel
            if "Crew_Count" in filt.columns:
                row["Crew_Count"] = inp_crew
            if "Patient_Severity" in filt.columns:
                row["Patient_Severity"] = inp_sev
            if "Transport_Mode" in filt.columns:
                row["Transport_Mode"] = inp_mode

            row_df = pd.DataFrame([row])
            out_msgs = []

            if st.session_state.pipeline_reg_amb is not None:
                try:
                    pred_r = st.session_state.pipeline_reg_amb.predict(row_df)[0]
                    out_msgs.append(f"Predicted Response_Time_min: {pred_r:.1f} minutes")
                except Exception as e:
                    out_msgs.append(f"Regression predict failed: {e}")

            if st.session_state.pipeline_clf_amb is not None:
                try:
                    pred_c = st.session_state.pipeline_clf_amb.predict(row_df)[0]
                    out_msgs.append(
                        f"Predicted Is_Interfacility_Transfer: {'Yes' if int(pred_c) == 1 else 'No'}"
                    )
                except Exception as e:
                    out_msgs.append(f"Classifier predict failed: {e}")

            for msg in out_msgs:
                st.success(msg)

    # -------------------------
    # Export
    # -------------------------
    st.markdown('<div class="section-title">Export Filtered Data</div>', unsafe_allow_html=True)
    download_df(filt.head(1000), "ambulance_filtered.csv", "Download filtered dataset (up to 1000 rows)")
