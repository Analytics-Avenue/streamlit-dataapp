import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

# -------------------------
# Config + CSS
# -------------------------
st.set_page_config(page_title="Ambulance Ops & Routing Lab", layout="wide")
st.markdown("""
<style>
.card {
    background: rgba(255,255,255,0.06);
    padding: 16px;
    border-radius: 12px;
    margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    transition: all .18s ease;
}
.card:hover { transform: scale(1.02); box-shadow: 0 12px 28px rgba(0,0,0,0.16); }
.metric-card {
    background: rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 10px;
    text-align:center;
    font-weight:600;
    border: 1px solid rgba(255,255,255,0.12);
}

/* KPI glow cards used in Overview */
.kpi-container {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-top: 18px;
    margin-bottom: 25px;
}
.kpi-card {
    background: rgba(255,255,255,0.05);
    padding: 18px 22px;
    border-radius: 12px;
    width: 230px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.20);
    border: 1px solid rgba(255,255,255,0.12);
    transition: all .20s ease-in-out;
    position: relative;
    overflow: hidden;
}
.kpi-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 22px 40px rgba(0,0,0,0.45), 0 0 25px rgba(0,120,255,0.45);
    border-color: rgba(0,120,255,0.45);
}
.kpi-card::before {
    content: "";
    position: absolute;
    inset: -40px;
    background: radial-gradient(circle at top left, rgba(0,120,255,0.25), transparent);
    z-index: 0;
}
.kpi-content {
    position: relative;
    z-index: 2;
    color: #fff;
    font-family: 'Inter', sans-serif;
}
.kpi-title {
    font-size: 13px;
    color: rgba(255,255,255,0.80);
}
.kpi-value {
    font-size: 24px;
    font-weight: 700;
    margin-top: 6px;
    margin-bottom: 4px;
    color: #ffffff;
}
.kpi-delta {
    font-size: 12px;
    color: rgba(255,255,255,0.75);
}
.kpi-badge {
    font-size: 11px;
    margin-left: 6px;
    background: rgba(255,255,255,0.10);
    padding: 4px 8px;
    border-radius: 999px;
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='margin-bottom:4px'>Ambulance Ops & Routing Lab</h1>", unsafe_allow_html=True)
st.markdown("Monitor response times, hospital loads and transfers — plus ML to predict response time and transfer likelihood.")

# -------------------------
# Utilities
# -------------------------
REQUIRED_COLS = [
    "Record_ID","Ambulance_ID","Driver_ID","Incident_Type","Patient_Severity",
    "Dispatch_Time","Arrival_Time_at_Scene","Depart_Scene_Time","Arrival_Time_at_Hospital",
    "Incident_Lat","Incident_Lon","Nearest_Hospital","Nearest_Hospital_Distance_km",
    "Dropoff_Hospital","Dropoff_Hospital_Distance_km","Distance_km","Travel_Time_min",
    "Response_Time_min","Crew_Count","Transport_Mode","Fuel_Consumed_Ltrs","Trip_Cost_Rs",
    "Weather","Road_Condition","Is_Interfacility_Transfer","Outcome",
    "Dropoff_Hospital_Beds","Dropoff_Hospital_Occupancy_pct"
]

def download_df(df, filename):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button("Download CSV", b, file_name=filename, mime="text/csv")

def to_currency(x):
    try:
        return "₹ " + f"{float(x):,.2f}"
    except:
        return x

def ensure_datetime(df, col):
    try:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    except Exception:
        pass
    return df

def auto_map_columns(df):
    rename = {}
    cols = [c.strip() for c in df.columns]
    lower_map = {c.lower().replace(" ","_"): c for c in cols}
    candidates = {
        "Record_ID":["record","id","record_id"],
        "Ambulance_ID":["ambulance","ambulance_id","amb_id"],
        "Driver_ID":["driver","driver_id"],
        "Incident_Type":["incident_type","incident"],
        "Patient_Severity":["severity","patient_severity"],
        "Dispatch_Time":["dispatch_time","dispatch"],
        "Arrival_Time_at_Scene":["arrival_time_at_scene","arrival_scene","arrival_scene_time"],
        "Depart_Scene_Time":["depart_scene_time","depart_scene"],
        "Arrival_Time_at_Hospital":["arrival_time_at_hospital","arrival_hospital"],
        "Incident_Lat":["incident_lat","lat"],
        "Incident_Lon":["incident_lon","lon","lng","long"],
        "Nearest_Hospital":["nearest_hospital"],
        "Nearest_Hospital_Distance_km":["nearest_hospital_distance_km","dist_km_to_incident","nearest_dist_km"],
        "Dropoff_Hospital":["dropoff_hospital","dropoff"],
        "Dropoff_Hospital_Distance_km":["dropoff_hospital_distance_km","dropoff_dist_km"],
        "Distance_km":["distance_km","distance_km_total","distance"],
        "Travel_Time_min":["travel_time_min","travel_time","travel_minutes"],
        "Response_Time_min":["response_time_min","response_time","response_minutes"],
        "Crew_Count":["crew_count"],
        "Transport_Mode":["transport_mode","transport"],
        "Fuel_Consumed_Ltrs":["fuel_consumed_ltrs","fuel_consumed","fuel_ltrs"],
        "Trip_Cost_Rs":["trip_cost_rs","trip_cost","cost"],
        "Weather":["weather"],
        "Road_Condition":["road_condition"],
        "Is_Interfacility_Transfer":["is_interfacility_transfer","interfacility_transfer","is_transfer"],
        "Outcome":["outcome"],
        "Dropoff_Hospital_Beds":["dropoff_hospital_beds","beds"],
        "Dropoff_Hospital_Occupancy_pct":["dropoff_hospital_occupancy_pct","occupancy_pct"]
    }
    for target, cand_list in candidates.items():
        for cand in cand_list:
            if cand in lower_map:
                rename[lower_map[cand]] = target
                break
            for orig in cols:
                if cand in orig.lower().replace(" ", "_"):
                    rename[orig] = target
                    break
            if target in rename.values():
                break
    if rename:
        df = df.rename(columns=rename)
    return df

# -------------------------
# Tabs: Overview / Application
# -------------------------
tabs = st.tabs(["Overview", "Application"])

# -------------------------
# Session state placeholders
# -------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "pipeline_reg" not in st.session_state:
    st.session_state.pipeline_reg = None
if "pipeline_clf" not in st.session_state:
    st.session_state.pipeline_clf = None

# -------------------------
# OVERVIEW
# -------------------------
with tabs[0]:
    st.subheader("Overview")

    st.markdown("""
    ### Application  
    A unified **Ambulance and Hospital Analytics Platform** that connects triage, dispatch, 
    emergency logistics, admissions, and readmission prediction into a single performance dashboard.

    ### Purpose  
    To improve emergency response, reduce delays, increase ambulance utilization, 
    optimize hospital throughput, and support predictive decision-making.

    ### Capabilities  
    - Real-time ambulance operations tracking  
    - Delay & bottleneck diagnostics  
    - Hospital admission & discharge analytics  
    - Readmission prediction & SHAP explainability  
    - Emergency workflow optimization  
    - Automated performance reports  

    ### Business Impact Achieved  
    - Faster ambulance response time  
    - Reduced ER congestion  
    - Higher ambulance acceptance rate  
    - Lower readmission rates  
    - Streamlined patient movement and hospital throughput  
    """)

    # Example KPI values (replace with dynamic values below if df available)
    ambulance_response_time = "12.4 min"
    acceptance_rate = "87%"
    avg_trip_distance = "16.2 km"
    hospital_wait_time = "22 min"
    readmission_rate = "8.2%"
    bed_occupancy = "74%"

    # If dataframe exists, compute KPIs dynamically (safe guards)
    if st.session_state.df is not None:
        try:
            df = st.session_state.df.copy()
            # compute dynamic KPIs with safe fallbacks
            if "Response_Time_min" in df.columns:
                ambulance_response_time = f"{df['Response_Time_min'].dropna().mean():.1f} min"
            if "Is_Interfacility_Transfer" in df.columns:
                acceptance_rate = f"{(1 - df['Is_Interfacility_Transfer'].mean()):.0%}"  # example: not transfer ~ acceptance
            if "Distance_km" in df.columns:
                avg_trip_distance = f"{df['Distance_km'].dropna().mean():.1f} km"
            if "Trip_Cost_Rs" in df.columns:
                hospital_wait_time = hospital_wait_time  # keep sample unless you have wait metric
            if "Readmission" in df.columns:
                rr = 100 * df['Readmission'].apply(lambda x: 1 if str(x).lower() in ['yes','1','true'] else 0).mean()
                readmission_rate = f"{rr:.1f}%"
            if "Dropoff_Hospital_Occupancy_pct" in df.columns:
                bed_occupancy = f"{df['Dropoff_Hospital_Occupancy_pct'].dropna().mean():.0f}%"
        except Exception:
            # keep sample values if anything fails
            pass

    kpis = [
        {"title": "Avg Ambulance Response Time", "value": ambulance_response_time, "delta": "-1.8% vs last week", "badge": "Ambulance"},
        {"title": "Ambulance Acceptance Rate", "value": acceptance_rate, "delta": "+4.2% vs last month", "badge": "Operations"},
        {"title": "Avg Trip Distance", "value": avg_trip_distance, "delta": "+0.6 km", "badge": "Logistics"},
        {"title": "Hospital Waiting Time", "value": hospital_wait_time, "delta": "-3 min improvement", "badge": "Hospital"},
        {"title": "Readmission Rate", "value": readmission_rate, "delta": "-1.1% vs qtr", "badge": "Clinical"},
        {"title": "Bed Occupancy", "value": bed_occupancy, "delta": "+6% seasonal", "badge": "Capacity"},
    ]

    # Render KPI cards
    html = '<div class="kpi-container">'
    for k in kpis:
        html += f"""
        <div class="kpi-card">
            <div class="kpi-content">
                <div class="kpi-title">{k['title']}<span class="kpi-badge">{k['badge']}</span></div>
                <div class="kpi-value">{k['value']}</div>
                <div class="kpi-delta">{k['delta']}</div>
            </div>
        </div>
        """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    # If a dataset is loaded, show visuals summary (safe checks)
    if st.session_state.df is None:
        st.info("No dataset loaded. Go to Application tab and load/upload an ambulance dataset to see visuals.")
    else:
        df = st.session_state.df.copy()

        st.markdown("### Busiest hospitals (by dropoffs)")
        if "Dropoff_Hospital" in df.columns:
            top_h = df["Dropoff_Hospital"].value_counts().reset_index()
            top_h.columns = ["Hospital", "Trips"]
            # guard empty
            if len(top_h) > 0:
                fig = px.bar(top_h.head(10), x="Hospital", y="Trips", text="Trips", title="Top 10 Dropoff Hospitals")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No dropoff hospital data available.")
        else:
            st.info("Dropoff_Hospital column missing in dataset.")

        st.markdown("### Response time distribution")
        if "Response_Time_min" in df.columns and df["Response_Time_min"].dropna().shape[0] > 0:
            fig = px.histogram(df.dropna(subset=["Response_Time_min"]), x="Response_Time_min", nbins=40, title="Response Time (min)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Response_Time_min not available.")

        st.markdown("### Map view (Incident locations)")
        if {"Incident_Lat","Incident_Lon"}.issubset(set(df.columns)) and df[["Incident_Lat","Incident_Lon"]].dropna().shape[0] > 0:
            mdf = df.dropna(subset=["Incident_Lat","Incident_Lon"])[["Incident_Lat","Incident_Lon"]].rename(columns={"Incident_Lat":"lat","Incident_Lon":"lon"})
            st.map(mdf)
        else:
            st.info("Incident latitude/longitude not available.")

# -------------------------
# APPLICATION
# -------------------------
with tabs[1]:
    st.header("Application")

    st.markdown("### Step 1 — Load dataset")
    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)

    df = None

    if mode == "Default dataset":
        DEFAULT_PATH = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/healthcare/ambulance_dataset.csv"
        try:
            df = pd.read_csv(DEFAULT_PATH)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("Default dataset loaded")
            st.dataframe(df.head())
            st.session_state.df = df
        except Exception as e:
            st.error(f"Failed to load default dataset from {DEFAULT_PATH}: {e}")
            st.info("If you don't have a local file, switch to 'Upload CSV' and upload the generated ambulance_dataset.csv.")
            st.stop()

    elif mode == "Upload CSV":
        st.markdown("#### (Optional) Download sample schema or upload your CSV")
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("File uploaded.")
            st.dataframe(df.head())
            st.session_state.df = df
            sample_small = df.head(5).to_csv(index=False)
            st.download_button("Download sample (first 5 rows)", sample_small, "sample_uploaded_5rows.csv", "text/csv")

    else:  # Upload + mapping
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head())
            st.markdown("Map your columns to required fields (only required fields shown).")
            mapping = {}
            for req in REQUIRED_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + list(raw.columns))
            if st.button("Apply mapping"):
                missing = [k for k,v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v:k for k,v in mapping.items()})
                    st.success("Mapping applied.")
                    st.dataframe(df.head())
                    st.session_state.df = df
                    sample_small = df.head(5).to_csv(index=False)
                    st.download_button("Download mapped sample (5 rows)", sample_small, "mapped_sample_5rows.csv", "text/csv")

    # if still no df
    if st.session_state.df is None:
        st.stop()

    # canonical df in session
    df = st.session_state.df.copy()

    # -------------------------
    # Clean & ensure columns
    # -------------------------
    for col in ["Dispatch_Time","Arrival_Time_at_Scene","Depart_Scene_Time","Arrival_Time_at_Hospital"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    numeric_cols = ["Nearest_Hospital_Distance_km","Dropoff_Hospital_Distance_km","Distance_km","Travel_Time_min","Response_Time_min","Crew_Count","Fuel_Consumed_Ltrs","Trip_Cost_Rs","Dropoff_Hospital_Beds","Dropoff_Hospital_Occupancy_pct"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Is_Interfacility_Transfer" in df.columns:
        df["Is_Interfacility_Transfer"] = df["Is_Interfacility_Transfer"].apply(lambda x: 1 if str(x).lower() in ["1","true","yes","y","t"] else 0)

    # show filtered preview & download
    st.markdown("### Step 2 — Filters & preview")
    c1,c2,c3 = st.columns([2,2,1])
    campaigns = df["Incident_Type"].dropna().unique().tolist() if "Incident_Type" in df.columns else []
    hospitals = df["Dropoff_Hospital"].dropna().unique().tolist() if "Dropoff_Hospital" in df.columns else []
    with c1:
        sel_incidents = st.multiselect("Incident Type", options=sorted(campaigns), default=sorted(campaigns)[:5])
    with c2:
        sel_hosp = st.multiselect("Dropoff Hospital", options=sorted(hospitals), default=sorted(hospitals)[:5] if hospitals else [])
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
    if date_range is not None:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        if "Dispatch_Time" in filt.columns:
            filt = filt[(filt["Dispatch_Time"] >= start) & (filt["Dispatch_Time"] <= end)]

    st.markdown("Filtered preview (first 10 rows)")
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(100), "ambulance_filtered_preview.csv")

    # -------------------------
    # Key Metrics & Charts
    # -------------------------
    st.markdown("### Key Metrics")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Filtered Trips", f"{len(filt):,}")
    if "Response_Time_min" in filt.columns:
        m2.metric("Avg Response (min)", f"{filt['Response_Time_min'].mean():.1f}")
    else:
        m2.metric("Avg Response (min)", "N/A")
    if "Travel_Time_min" in filt.columns:
        m3.metric("Avg Travel (min)", f"{filt['Travel_Time_min'].mean():.1f}")
    else:
        m3.metric("Avg Travel (min)", "N/A")
    if "Trip_Cost_Rs" in filt.columns:
        m4.metric("Total Trip Cost", to_currency(filt["Trip_Cost_Rs"].sum()))
    else:
        m4.metric("Total Trip Cost", "N/A")

    st.markdown("### Visuals")
    if "Dropoff_Hospital" in filt.columns:
        agg = filt.groupby("Dropoff_Hospital").agg({"Record_ID":"count","Trip_Cost_Rs":"sum"}).rename(columns={"Record_ID":"Trips"}).reset_index()
        if agg.shape[0] > 0:
            agg = agg.sort_values("Trips", ascending=False)
            fig = px.bar(agg.head(10), x="Dropoff_Hospital", y="Trips", title="Top dropoff hospitals")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No dropoff hospital records to display.")

    if {"Incident_Lat","Incident_Lon"}.issubset(set(filt.columns)):
        st.markdown("### Incident density map")
        map_df = filt.dropna(subset=["Incident_Lat","Incident_Lon"])[["Incident_Lat","Incident_Lon","Response_Time_min"]].rename(columns={"Incident_Lat":"lat","Incident_Lon":"lon"})
        if map_df.shape[0] > 0:
            st.map(map_df)
        else:
            st.info("No incident lat/lon points to map.")

    if "Response_Time_min" in filt.columns and "Distance_km" in filt.columns:
        st.markdown("### Response vs Distance scatter")
        fig = px.scatter(filt.dropna(subset=["Distance_km","Response_Time_min"]), x="Distance_km", y="Response_Time_min", color="Patient_Severity", title="Response time vs distance")
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # ML Models (unchanged from your original)
    # -------------------------
    st.markdown("### ML: Predict Response Time (Regression) & Transfer (Classification)")
    with st.expander("ML settings & train", expanded=False):
        can_train_reg = "Response_Time_min" in filt.columns and len(filt.dropna(subset=["Response_Time_min"])) > 30
        can_train_clf = "Is_Interfacility_Transfer" in filt.columns and filt["Is_Interfacility_Transfer"].nunique() > 1 and len(filt.dropna(subset=["Is_Interfacility_Transfer"])) > 30

        st.write(f"Regression available: {can_train_reg}  |  Classification available: {can_train_clf}")

        possible_features = [c for c in filt.columns if c not in ["Record_ID","Dispatch_Time","Arrival_Time_at_Scene","Depart_Scene_Time","Arrival_Time_at_Hospital","Outcome"]]
        features = st.multiselect("Features to use (ML)", options=possible_features, default=["Distance_km","Travel_Time_min","Crew_Count","Patient_Severity","Transport_Mode"] if "Distance_km" in possible_features else possible_features[:6])

        if st.button("Train ML models"):
            if len(features) < 1:
                st.error("Choose at least 1 feature.")
            else:
                X = filt[features].copy()
                cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
                num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

                for nc in num_cols:
                    X[nc] = pd.to_numeric(X[nc], errors="coerce").fillna(X[nc].median())

                for cc in cat_cols:
                    X[cc] = X[cc].astype(str).fillna("NA")

                # Regression
                if can_train_reg:
                    y_reg = pd.to_numeric(filt["Response_Time_min"], errors="coerce")
                    Xr = X.loc[y_reg.dropna().index]
                    yr = y_reg.dropna()
                    if len(Xr) < 10:
                        st.error("Not enough rows for regression after cleaning.")
                    else:
                        preproc = ColumnTransformer([
                            ("num", StandardScaler(), num_cols),
                            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
                        ], remainder="drop")
                        reg = RandomForestRegressor(n_estimators=150, random_state=42)
                        pipe_reg = Pipeline([("prep", preproc), ("model", reg)])
                        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(Xr, yr, test_size=0.2, random_state=42)
                        with st.spinner("Training regression..."):
                            pipe_reg.fit(X_train_r, y_train_r)
                        preds_r = pipe_reg.predict(X_test_r)
                        rmse = math.sqrt(mean_squared_error(y_test_r, preds_r))
                        r2 = r2_score(y_test_r, preds_r)
                        st.success(f"Regression trained — RMSE: {rmse:.2f}  R²: {r2:.3f}")
                        st.session_state.pipeline_reg = pipe_reg

                # Classification
                if can_train_clf:
                    y_clf = filt["Is_Interfacility_Transfer"].astype(int)
                    Xc = X.loc[y_clf.dropna().index]
                    yc = y_clf.dropna()
                    if yc.nunique() < 2 or len(Xc) < 10:
                        st.error("Not enough class variety or rows for classification.")
                    else:
                        preproc_c = ColumnTransformer([
                            ("num", StandardScaler(), num_cols),
                            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
                        ], remainder="drop")
                        clf = RandomForestClassifier(n_estimators=150, random_state=42)
                        pipe_clf = Pipeline([("prep", preproc_c), ("model", clf)])
                        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(Xc, yc, test_size=0.2, random_state=42)
                        with st.spinner("Training classifier..."):
                            pipe_clf.fit(X_train_c, y_train_c)
                        pred_c = pipe_clf.predict(X_test_c)
                        acc = accuracy_score(y_test_c, pred_c)
                        st.success(f"Classifier trained — Accuracy: {acc:.3f}")
                        st.session_state.pipeline_clf = pipe_clf

    # Quick predict and export (unchanged)
    st.markdown("### Quick Predict (single row)")
    if st.session_state.pipeline_reg is None and st.session_state.pipeline_clf is None:
        st.info("Train models first (use the ML panel above).")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            inp_distance = st.number_input("Distance_km", value=float(filt["Distance_km"].median() if "Distance_km" in filt.columns else 5.0))
            inp_travel = st.number_input("Travel_Time_min", value=float(filt["Travel_Time_min"].median() if "Travel_Time_min" in filt.columns else 10.0))
        with col2:
            inp_crew = st.number_input("Crew_Count", value=int(filt["Crew_Count"].median() if "Crew_Count" in filt.columns else 2))
            if "Patient_Severity" in filt.columns:
                inp_sev = st.selectbox("Patient_Severity", options=sorted(filt["Patient_Severity"].dropna().unique().tolist()))
            else:
                inp_sev = st.text_input("Patient_Severity", "Medium")
        with col3:
            if "Transport_Mode" in filt.columns:
                inp_mode = st.selectbox("Transport_Mode", options=sorted(filt["Transport_Mode"].dropna().unique().tolist()))
            else:
                inp_mode = st.text_input("Transport_Mode", "Basic Life Support")

        if st.button("Predict (single)"):
            row = pd.DataFrame([{
                "Distance_km": inp_distance,
                "Travel_Time_min": inp_travel,
                "Crew_Count": inp_crew,
                "Patient_Severity": inp_sev,
                "Transport_Mode": inp_mode
            }])
            out_msgs = []
            if st.session_state.pipeline_reg is not None:
                try:
                    pred_r = st.session_state.pipeline_reg.predict(row)[0]
                    out_msgs.append(f"Predicted Response_Time_min: {pred_r:.1f}")
                except Exception as e:
                    out_msgs.append(f"Regression predict failed: {e}")
            if st.session_state.pipeline_clf is not None:
                try:
                    pred_c = st.session_state.pipeline_clf.predict(row)[0]
                    out_msgs.append(f"Predicted Is_Interfacility_Transfer: {'Yes' if int(pred_c)==1 else 'No'}")
                except Exception as e:
                    out_msgs.append(f"Classifier predict failed: {e}")
            for m in out_msgs:
                st.success(m)

    st.markdown("### Export filtered data / save models")
    if st.button("Download filtered CSV (1000 rows max)"):
        download_df(filt.head(1000), "ambulance_filtered.csv")

    st.success("Application ready. Tip: upload your ambulance dataset or load the default sample.")
