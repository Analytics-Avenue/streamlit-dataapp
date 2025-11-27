# app_route_optimization_standardized.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import math
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Route Optimization & Logistics Efficiency",
    layout="wide",
    page_icon="🚚"
)

# ---------------------------------------------------------
# Hide sidebar
# ---------------------------------------------------------
hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# ---------------------------------------------------------
# Global CSS: cards, KPIs, basic typography
# ---------------------------------------------------------
st.markdown("""
<style>
.card {
    padding: 18px;
    border-radius: 12px;
    background: #ffffff;
    border: 1px solid #e6e6e6;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    text-align: left;
}
.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 12px 36px rgba(6,75,134,0.18);
    border-color: #064b86;
}

.kpi {
    padding: 24px;
    border-radius: 12px;
    background: #ffffff;
    border: 1px solid #e6e6e6;
    text-align: center;
    font-weight: 700;
    color: #064b86;
    font-size: 20px;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
.kpi:hover {
    transform: translateY(-6px);
    box-shadow: 0 10px 30px rgba(6,75,134,0.18);
    border-color: #064b86;
}

.small { color:#666; font-size:13px; }
.left-align { text-align:left; }

.title-main {
    font-size: 30px;
    font-weight: 800;
    color:#064b86;
    margin:0;
    padding:0;
    text-align:left;
}
.subtitle-main {
    font-size:14px;
    color:#555;
    margin-top:4px;
    margin-bottom:0;
    text-align:left;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Company Logo + Name
# ---------------------------------------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
col_logo, col_head = st.columns([0.18, 3])
with col_logo:
    st.image(logo_url, width=60)
with col_head:
    st.markdown(
        """
        <div>
            <p class="title-main">Route Optimization & Logistics Efficiency</p>
            <p class="subtitle-main">Reduce miles, cut fuel, speed deliveries with data-driven routing.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def read_csv_safe(src):
    """Read CSV and make duplicate column names unique."""
    df = pd.read_csv(src)
    cols = list(df.columns)
    if len(cols) != len(set(cols)):
        new = []
        seen = {}
        for c in cols:
            base = str(c).strip()
            if base not in seen:
                seen[base] = 0
                new.append(base)
            else:
                seen[base] += 1
                new.append(f"{base}__dup{seen[base]}")
        df.columns = new
    df.columns = [str(c).strip() for c in df.columns]
    return df

def download_df(df, filename, label="Download CSV"):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tabs = st.tabs(["Overview", "Data Dictionary", "Application", "Action Playbooks"])

# ---------------------------------------------------------
# OVERVIEW TAB
# ---------------------------------------------------------
with tabs[0]:
    st.markdown("### Overview")
    st.markdown("""
    <div class='card left-align'>
      <b>Purpose</b>: Cut route costs and delivery time by optimizing routes, predicting delays & fuel usage, and prioritising high-impact fleet actions.
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Capabilities")
        st.markdown("""
        <div class='card left-align'>
        • Route efficiency scoring and anomaly detection<br>
        • Predictive travel time and fuel consumption models<br>
        • Clustering of routes/vehicles for capacity planning<br>
        • Multi-filter exploration (vehicle / route / traffic / weather)<br>
        • Exportable prioritized actions for operations teams
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("#### Business impact")
        st.markdown("""
        <div class='card left-align'>
        • Lower fuel & operational cost per km<br>
        • Faster deliveries & higher on-time %<br>
        • Reduced CO₂ per shipment<br>
        • Better fleet utilisation & scheduling<br>
        • Data-driven procurement of vehicles & drivers
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### KPIs")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown("<div class='kpi'>Total Routes Tracked</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Avg Efficiency Score</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Avg Delay (hrs)</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Avg Fuel / Route (L)</div>", unsafe_allow_html=True)
    k5.markdown("<div class='kpi'>On-Time %</div>", unsafe_allow_html=True)

    st.markdown("### Who should use & How")
    st.markdown("""
    <div class='card left-align'>
      <b>Who</b>: Fleet managers, logistics planners, operations heads, sustainability teams.<br><br>
      <b>How</b>: 1) Load dataset (default / upload). 2) Filter by vehicle / route / period. 3) Review top-delay & low-efficiency routes. 4) Export cost simulations, ML predictions & playbooks to drive execution.
    </div>
    """, unsafe_allow_html=True)
# ---------------------------------------------------------
# DATA DICTIONARY TAB
# ---------------------------------------------------------
with tabs[1]:
    st.header("Data Dictionary")

    st.markdown("""
    <div class='card left-align'>
    This section defines all required and optional fields for the Route Optimization dataset.  
    Use this as a reference when preparing CSV files or mapping uploaded datasets.
    </div>
    """, unsafe_allow_html=True)

    data_dict = [
        ["Timestamp", "Datetime", "Trip start timestamp, used for time-series trend analysis."],
        ["Vehicle_ID", "Categorical", "Unique identifier of the vehicle."],
        ["Vehicle_Type", "Categorical", "Type/category of the vehicle (e.g., Van, Truck, 2W)."],
        ["Route_ID", "Categorical", "Unique route identifier for grouping trips."],
        ["Start_City", "Categorical", "Origin city/location of the route."],
        ["End_City", "Categorical", "Destination city/location of the route."],
        ["Route_Distance_km", "Numeric", "Total distance of the route in kilometers."],
        ["Traffic_Level", "Categorical", "Traffic condition (Low, Medium, High)."],
        ["Weather_Condition", "Categorical", "Weather during trip (Clear, Rain, Fog, Storm)."],
        
        ["Predicted_Travel_Hours", "Numeric", "ML-generated expected travel time."],
        ["Actual_Travel_Hours", "Numeric", "Actual recorded travel time."],
        
        ["Predicted_Fuel_Liters", "Numeric", "ML-predicted fuel consumption."],
        ["Actual_Fuel_Liters", "Numeric", "Actual consumed fuel in liters."],
        
        ["Load_Weight_kg", "Numeric", "Cargo weight in kilograms."],
        ["Vehicle_Capacity_kg", "Numeric", "Maximum load capacity of the vehicle."],
        
        ["Delay_Hours", "Numeric", "Difference between expected vs actual arrival time."],
        
        ["Efficiency_Score", "Numeric", "Derived metric: travel & fuel efficiency combined."],
        ["Fuel_L_per_km", "Numeric (derived)", "Fuel burned per km. Auto-computed if missing."],
        
        ["_is_anomaly", "Binary (derived)", "IsolationForest flag: 1=anomaly trip, 0=normal."],
    ]

    df_dict = pd.DataFrame(data_dict, columns=["Column", "Type", "Description"])
    st.dataframe(df_dict, use_container_width=True)

    st.markdown("#### Download Data Dictionary")
    csv_dd = df_dict.to_csv(index=False).encode("utf-8")
    st.download_button("Download Data Dictionary CSV", csv_dd, "route_data_dictionary.csv", "text/csv")

# ---------------------------------------------------------
# APPLICATION TAB
# ---------------------------------------------------------
with tabs[2]:
    st.header("Application")
    st.markdown("### Step 1 — Load dataset")

    load_mode = st.radio(
        "Dataset option:",
        ["Default Data", "Upload CSV", "Upload + Map Columns"],
        horizontal=True
    )

    df = None

    if load_mode == "Default Data":
        url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/supply_chain/route_optimization_dataset.csv"
        try:
            df = read_csv_safe(url)
            st.success("Default dataset loaded.")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    elif load_mode == "Upload CSV":
        st.markdown("#### Download Sample CSV for reference")
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/supply_chain/route_optimization_dataset.csv"
        try:
            sample_df = read_csv_safe(URL).head(5)
            sample_csv = sample_df.to_csv(index=False)
            st.download_button("Download Sample CSV", sample_csv, "sample_route_dataset.csv", "text/csv")
        except Exception as e:
            st.info(f"Sample CSV unavailable: {e}")

        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = read_csv_safe(file)
            st.success("Dataset uploaded.")
            st.dataframe(df.head(), use_container_width=True)

    else:  # Upload + Map
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"], key="upload_map")
        if uploaded:
            raw = read_csv_safe(uploaded)
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head(), use_container_width=True)
            st.markdown("Map columns (map at least Timestamp, Vehicle_ID, Route_ID, distance & fuel if available).")

            cols = list(raw.columns)
            expected = [
                "Timestamp","Vehicle_ID","Vehicle_Type","Route_ID","Start_City","End_City",
                "Route_Distance_km","Traffic_Level","Weather_Condition",
                "Predicted_Travel_Hours","Actual_Travel_Hours",
                "Predicted_Fuel_Liters","Actual_Fuel_Liters","Vehicle_Capacity_kg",
                "Load_Weight_kg","Delay_Hours","Efficiency_Score"
            ]
            mapping = {}
            for e in expected:
                mapping[e] = st.selectbox(f"Map → {e}", ["-- Skip --"] + cols, index=0, key=f"map_{e}")

            if st.button("Apply mapping"):
                rename = {v: k for k, v in mapping.items() if v != "-- Skip --"}
                if not rename:
                    st.error("You must map at least one column.")
                    st.stop()
                df = raw.rename(columns=rename)
                st.success("Mapping applied.")
                st.dataframe(df.head(), use_container_width=True)

    if df is None:
        st.stop()

    # -----------------------------------------------------
    # Cleaning & type handling
    # -----------------------------------------------------
    df.columns = [str(c).strip() for c in df.columns]

    # Timestamp handling
    if "Timestamp" not in df.columns:
        possible_ts = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if possible_ts:
            st.info(f"Using `{possible_ts[0]}` as Timestamp column.")
            df = df.rename(columns={possible_ts[0]: "Timestamp"})

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    numeric_candidates = [
        "Route_Distance_km","Predicted_Travel_Hours","Actual_Travel_Hours",
        "Predicted_Fuel_Liters","Actual_Fuel_Liters","Vehicle_Capacity_kg",
        "Load_Weight_kg","Delay_Hours","Efficiency_Score"
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derive Efficiency_Score if missing
    if "Efficiency_Score" not in df.columns and all(x in df.columns for x in ["Predicted_Travel_Hours","Actual_Travel_Hours","Predicted_Fuel_Liters","Actual_Fuel_Liters"]):
        df["Efficiency_Score"] = (
            (df["Predicted_Travel_Hours"] / df["Actual_Travel_Hours"].replace(0, np.nan)) *
            (df["Predicted_Fuel_Liters"] / df["Actual_Fuel_Liters"].replace(0, np.nan))
        )

    if "Efficiency_Score" not in df.columns:
        df["Efficiency_Score"] = np.nan

    for c in numeric_candidates:
        if c not in df.columns:
            df[c] = np.nan

    # -----------------------------------------------------
    # Filters
    # -----------------------------------------------------
    st.markdown("### Step 2 — Filters & Preview")
    col_a, col_b, col_c = st.columns([2, 2, 2])

    # Date filter
    if "Timestamp" in df.columns and df["Timestamp"].notna().any():
        min_ts = df["Timestamp"].min()
        max_ts = df["Timestamp"].max()
        try:
            date_range = st.slider(
                "Select date range",
                min_value=min_ts,
                max_value=max_ts,
                value=(min_ts, max_ts),
                format="YYYY-MM-DD HH:mm"
            )
            start_sel, end_sel = date_range
        except Exception:
            dates = st.date_input(
                "Select date range (date only)",
                value=(min_ts.date(), max_ts.date())
            )
            start_sel = datetime.combine(dates[0], datetime.min.time())
            end_sel = datetime.combine(dates[1], datetime.max.time())
    else:
        start_sel = None
        end_sel = None
        st.info("No usable Timestamp column; skipping date filter.")

    vehicles = df["Vehicle_ID"].dropna().unique().tolist() if "Vehicle_ID" in df.columns else []
    vtypes = df["Vehicle_Type"].dropna().unique().tolist() if "Vehicle_Type" in df.columns else []
    routes = df["Route_ID"].dropna().unique().tolist() if "Route_ID" in df.columns else []

    sel_vehicles = col_a.multiselect("Vehicle_ID", options=vehicles, default=vehicles[:5] if vehicles else [])
    sel_vtypes = col_b.multiselect("Vehicle_Type", options=vtypes, default=vtypes[:3] if vtypes else [])
    sel_routes = col_c.multiselect("Route_ID", options=routes, default=routes[:5] if routes else [])

    # Apply filters
    filt = df.copy()
    if start_sel is not None and end_sel is not None and "Timestamp" in filt.columns:
        filt = filt[(filt["Timestamp"] >= pd.to_datetime(start_sel)) & (filt["Timestamp"] <= pd.to_datetime(end_sel))]
    if sel_vehicles and "Vehicle_ID" in filt.columns:
        filt = filt[filt["Vehicle_ID"].isin(sel_vehicles)]
    if sel_vtypes and "Vehicle_Type" in filt.columns:
        filt = filt[filt["Vehicle_Type"].isin(sel_vtypes)]
    if sel_routes and "Route_ID" in filt.columns:
        filt = filt[filt["Route_ID"].isin(sel_routes)]

    st.markdown("#### Filtered Data Preview (first 10 rows)")
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(500), "filtered_route_preview.csv", label="Download filtered preview (up to 500 rows)")

    # -----------------------------------------------------
    # -----------------------------------------------------
    # Dynamic KPIs (FULL FIXED BLOCK)
    # -----------------------------------------------------
    st.markdown("### KPIs (Dynamic)")
    dk1, dk2, dk3, dk4, dk5 = st.columns(5)
    
    # Compute KPI raw values safely
    total_routes = (
        filt["Route_ID"].nunique()
        if "Route_ID" in filt.columns
        else len(filt)
    )
    
    avg_eff = (
        float(filt["Efficiency_Score"].mean())
        if "Efficiency_Score" in filt.columns and filt["Efficiency_Score"].notna().any()
        else None
    )
    
    avg_delay = (
        float(filt["Delay_Hours"].mean())
        if "Delay_Hours" in filt.columns and filt["Delay_Hours"].notna().any()
        else None
    )
    
    avg_fuel = (
        float(filt["Actual_Fuel_Liters"].mean())
        if "Actual_Fuel_Liters" in filt.columns and filt["Actual_Fuel_Liters"].notna().any()
        else None
    )
    
    if "Delay_Hours" in filt.columns and filt["Delay_Hours"].notna().any():
        ontime_rate = (filt["Delay_Hours"] <= 0.25).mean() * 100
    else:
        ontime_rate = None
    
    # Convert numbers → safe display text  
    routes_val = f"{total_routes:,}"
    avg_eff_val = f"{avg_eff:.3f}" if avg_eff is not None else "N/A"
    avg_delay_val = f"{avg_delay:.2f}" if avg_delay is not None else "N/A"
    avg_fuel_val = f"{avg_fuel:.2f}" if avg_fuel is not None else "N/A"
    ontime_val = f"{ontime_rate:.1f}%" if ontime_rate is not None else "N/A"
    
    # Render KPI cards
    dk1.markdown(
        f"<div class='kpi'>{routes_val}<div class='small'>Routes in selection</div></div>",
        unsafe_allow_html=True
    )
    dk2.markdown(
        f"<div class='kpi'>{avg_eff_val}<div class='small'>Avg Efficiency Score</div></div>",
        unsafe_allow_html=True
    )
    dk3.markdown(
        f"<div class='kpi'>{avg_delay_val}<div class='small'>Avg Delay (hrs)</div></div>",
        unsafe_allow_html=True
    )
    dk4.markdown(
        f"<div class='kpi'>{avg_fuel_val}<div class='small'>Avg Fuel / Route (L)</div></div>",
        unsafe_allow_html=True
    )
    dk5.markdown(
        f"<div class='kpi'>{ontime_val}<div class='small'>On-Time Deliveries</div></div>",
        unsafe_allow_html=True
    )
    
    # -----------------------------------------------------
    # Exploratory Data Analysis
    # -----------------------------------------------------
    st.markdown("## Exploratory Data Analysis")

    # 1) Route distance histogram
    if "Route_Distance_km" in filt.columns and filt["Route_Distance_km"].notna().any():
        fig_dist = px.histogram(filt, x="Route_Distance_km", nbins=30, title="Route distance distribution (km)")
        st.plotly_chart(fig_dist, use_container_width=True)

    # 2) Efficiency score by vehicle type
    if "Vehicle_Type" in filt.columns and "Efficiency_Score" in filt.columns and filt["Efficiency_Score"].notna().any():
        fig_eff = px.box(filt, x="Vehicle_Type", y="Efficiency_Score", title="Efficiency Score by Vehicle Type")
        st.plotly_chart(fig_eff, use_container_width=True)

    # 3) Delay vs Distance scatter
    if "Delay_Hours" in filt.columns and "Route_Distance_km" in filt.columns and filt["Delay_Hours"].notna().any():
        fig_sc = px.scatter(
            filt,
            x="Route_Distance_km",
            y="Delay_Hours",
            color="Traffic_Level" if "Traffic_Level" in filt.columns else None,
            size="Load_Weight_kg" if "Load_Weight_kg" in filt.columns else None,
            title="Delay vs Distance (bubble size = load)"
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # 4) Fuel efficiency (L/km)
    if "Actual_Fuel_Liters" in filt.columns and "Route_Distance_km" in filt.columns:
        filt["Fuel_L_per_km"] = filt["Actual_Fuel_Liters"] / np.where(filt["Route_Distance_km"] == 0, np.nan, filt["Route_Distance_km"])
        if filt["Fuel_L_per_km"].notna().any():
            fig_fuel = px.violin(filt, y="Fuel_L_per_km", box=True, title="Fuel consumption (L per km) distribution")
            st.plotly_chart(fig_fuel, use_container_width=True)

    # 5) Efficiency over time (daily)
    if "Timestamp" in filt.columns and "Efficiency_Score" in filt.columns and filt["Efficiency_Score"].notna().any():
        ts = filt.set_index("Timestamp").resample("D")["Efficiency_Score"].mean().reset_index()
        if not ts.empty:
            fig_ts = px.line(ts, x="Timestamp", y="Efficiency_Score", title="Daily average Efficiency Score")
            st.plotly_chart(fig_ts, use_container_width=True)

    # 6) KMeans clusters (distance vs delay)
    if all(c in filt.columns for c in ["Route_Distance_km", "Delay_Hours"]) and len(filt) >= 10:
        kdf = filt[["Route_Distance_km","Delay_Hours"]].dropna()
        if len(kdf) >= 10:
            kdf = kdf.sample(n=min(800, len(kdf)), random_state=42)
            try:
                kmeans = KMeans(n_clusters=4, random_state=42)
                klabels = kmeans.fit_predict(kdf)
                kdf_plot = kdf.copy()
                kdf_plot["cluster"] = klabels.astype(str)
                fig_k = px.scatter(
                    kdf_plot,
                    x="Route_Distance_km",
                    y="Delay_Hours",
                    color="cluster",
                    title="KMeans clusters by distance & delay"
                )
                st.plotly_chart(fig_k, use_container_width=True)
            except Exception:
                st.info("KMeans clustering failed (insufficient or problematic data).")

    # 7) Correlation matrix
    numcols = [
        "Route_Distance_km","Predicted_Travel_Hours","Actual_Travel_Hours",
        "Predicted_Fuel_Liters","Actual_Fuel_Liters","Load_Weight_kg",
        "Delay_Hours","Efficiency_Score"
    ]
    existing_numcols = [c for c in numcols if c in filt.columns]
    if len(existing_numcols) >= 2:
        corr = filt[existing_numcols].corr()
        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="Blues"
            )
        )
        fig_corr.update_layout(title="Correlation matrix (numeric variables)")
        st.plotly_chart(fig_corr, use_container_width=True)

    # ---------------------------------------------------------
    # Route-Level Cost Simulation
    # ---------------------------------------------------------
    st.markdown("### Route-Level Cost Simulation")

    if all(c in filt.columns for c in ["Actual_Fuel_Liters","Delay_Hours","Route_Distance_km"]):
        SIM_FUEL_COST = 94.5      # INR per liter
        SIM_DRIVER_COST_HR = 220  # INR per hour
        SIM_VEHICLE_COST_KM = 18  # INR per km

        sim = filt.copy()
        sim["Fuel_Cost_INR"] = sim["Actual_Fuel_Liters"] * SIM_FUEL_COST
        sim["Distance_Cost_INR"] = sim["Route_Distance_km"] * SIM_VEHICLE_COST_KM
        sim["Delay_Penalty_INR"] = sim["Delay_Hours"] * SIM_DRIVER_COST_HR
        sim["Total_Route_Cost_INR"] = sim["Fuel_Cost_INR"] + sim["Distance_Cost_INR"] + sim["Delay_Penalty_INR"]

        st.dataframe(
            sim[["Route_ID","Vehicle_ID","Fuel_Cost_INR","Distance_Cost_INR","Delay_Penalty_INR","Total_Route_Cost_INR"]].head(15),
            use_container_width=True
        )
        download_df(sim, "route_cost_simulation.csv", "Download Cost Simulation")
    else:
        st.info("Required columns missing for cost simulation (need Actual_Fuel_Liters, Delay_Hours, Route_Distance_km).")

    # ---------------------------------------------------------
    # Machine Learning — 4 models
    # ---------------------------------------------------------
    st.markdown("## Machine Learning — Models & Predictions")
    st.markdown("Models run only when sufficient rows and required columns exist (recommended: ≥ 80 rows).")

    # Build modeling dataframe with simple one-hot encoding
    base_features = []
    if "Route_Distance_km" in filt.columns:
        base_features.append("Route_Distance_km")
    if "Load_Weight_kg" in filt.columns:
        base_features.append("Load_Weight_kg")

    cat_cols = [c for c in ["Vehicle_Type","Traffic_Level","Weather_Condition","Start_City","End_City"] if c in filt.columns]

    df_model = filt.copy().reset_index(drop=True)
    if cat_cols:
        df_model = pd.get_dummies(df_model, columns=cat_cols, dummy_na=False)

    for c in df_model.columns:
        if df_model[c].dtype.kind in "biufc" and df_model[c].isna().any():
            df_model[c] = df_model[c].fillna(df_model[c].median())

    def train_eval_reg(X, y, model):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = math.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        return model, X_test, y_test, preds, rmse, r2

    # 1) RandomForestRegressor — predict Actual_Travel_Hours
    if "Actual_Travel_Hours" in df_model.columns:
        target = "Actual_Travel_Hours"
        features = [
            c for c in df_model.columns
            if c not in ["Timestamp","Vehicle_ID","Route_ID","Predicted_Travel_Hours",
                         "Actual_Travel_Hours","Predicted_Fuel_Liters","Actual_Fuel_Liters",
                         "Efficiency_Score"]
        ]
        if features:
            X = df_model[features].select_dtypes(include=[np.number]).fillna(0)
            y = df_model[target].fillna(0)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            rf = RandomForestRegressor(n_estimators=150, random_state=42)
            rf, X_test, y_test, preds, rmse, r2 = train_eval_reg(Xs, y, rf)
            st.markdown("### RandomForest — Predict Actual Travel Hours")
            st.write(f"RMSE: {rmse:.3f}  |  R²: {r2:.3f}")
            out = pd.DataFrame({"Actual": y_test, "Predicted": preds})
            st.dataframe(out.head(10), use_container_width=True)
            download_df(out, "rf_actual_travel_hours_predictions.csv", "Download RF predictions")
        else:
            st.info("No numeric features available for RandomForest model.")
    else:
        st.info("RandomForest: need column Actual_Travel_Hours and at least 80 rows.")

    # 2) GradientBoostingRegressor — predict Actual_Fuel_Liters
    if "Actual_Fuel_Liters" in df_model.columns:
        target2 = "Actual_Fuel_Liters"
        features2 = [
            c for c in df_model.columns
            if c not in ["Timestamp","Vehicle_ID","Route_ID","Predicted_Travel_Hours",
                         "Actual_Travel_Hours","Predicted_Fuel_Liters","Actual_Fuel_Liters",
                         "Efficiency_Score"]
        ]
        if features2:
            X2 = df_model[features2].select_dtypes(include=[np.number]).fillna(0)
            y2 = df_model[target2].fillna(0)
            scaler2 = StandardScaler()
            X2s = scaler2.fit_transform(X2)
            gbr = GradientBoostingRegressor(n_estimators=150, random_state=42)
            gbr, X2_test, y2_test, preds2, rmse2, r22 = train_eval_reg(X2s, y2, gbr)
            st.markdown("### GradientBoosting — Predict Actual Fuel Liters")
            st.write(f"RMSE: {rmse2:.3f}  |  R²: {r22:.3f}")
            out2 = pd.DataFrame({"Actual_Fuel": y2_test, "Predicted_Fuel": preds2})
            st.dataframe(out2.head(10), use_container_width=True)
            download_df(out2, "gbr_actual_fuel_predictions.csv", "Download GBR predictions")
        else:
            st.info("No numeric features available for GradientBoosting model.")
    else:
        st.info("GradientBoosting: need column Actual_Fuel_Liters and at least 80 rows.")

    # 3) KNeighborsRegressor — predict Delay_Hours
    if "Delay_Hours" in df_model.columns:
        target3 = "Delay_Hours"
        features3 = [
            c for c in df_model.columns
            if c not in ["Timestamp","Vehicle_ID","Route_ID","Delay_Hours","Efficiency_Score"]
        ]
        if features3:
            X3 = df_model[features3].select_dtypes(include=[np.number]).fillna(0)
            y3 = df_model[target3].fillna(0)
            scaler3 = StandardScaler()
            X3s = scaler3.fit_transform(X3)
            knn = KNeighborsRegressor(n_neighbors=7)
            knn, X3_test, y3_test, preds3, rmse3, r23 = train_eval_reg(X3s, y3, knn)
            st.markdown("### KNN Regressor — Predict Delay Hours")
            st.write(f"RMSE: {rmse3:.3f}  |  R²: {r23:.3f}")
            out3 = pd.DataFrame({"Actual_Delay": y3_test, "Predicted_Delay": preds3})
            st.dataframe(out3.head(10), use_container_width=True)
            download_df(out3, "knn_delay_predictions.csv", "Download KNN predictions")
        else:
            st.info("No numeric features available for KNN model.")
    else:
        st.info("KNN: need column Delay_Hours and at least 80 rows.")

    # 4) IsolationForest — anomaly detection
    st.markdown("### Anomaly Detection — IsolationForest")

    if "Fuel_L_per_km" not in filt.columns and all(c in filt.columns for c in ["Actual_Fuel_Liters","Route_Distance_km"]):
        filt["Fuel_L_per_km"] = filt["Actual_Fuel_Liters"] / np.where(filt["Route_Distance_km"] == 0, np.nan, filt["Route_Distance_km"])

    iso_cols = [c for c in ["Efficiency_Score","Actual_Fuel_Liters","Fuel_L_per_km","Delay_Hours"] if c in filt.columns]
    if len(iso_cols) >= 2 and len(filt) >= 30:
        iso_feats = filt[iso_cols].fillna(0)
        iso = IsolationForest(contamination=0.03, random_state=42)
        try:
            filt["_is_anomaly"] = iso.fit_predict(iso_feats)
            filt["_is_anomaly"] = np.where(filt["_is_anomaly"] == -1, 1, 0)
            num_anoms = int(filt["_is_anomaly"].sum())
            st.markdown(f"Detected anomalies: **{num_anoms}**")
            if num_anoms > 0:
                anom_cols = [
                    c for c in ["Timestamp","Vehicle_ID","Route_ID","Route_Distance_km",
                                "Actual_Fuel_Liters","Fuel_L_per_km","Efficiency_Score",
                                "Delay_Hours"] if c in filt.columns
                ]
                st.dataframe(filt.loc[filt["_is_anomaly"] == 1, anom_cols].head(20), use_container_width=True)
                download_df(
                    filt.loc[filt["_is_anomaly"] == 1, anom_cols],
                    "anomalies_route.csv",
                    "Download anomalies CSV"
                )
        except Exception as e:
            st.info("Anomaly detection failed: " + str(e))
    else:
        st.info("IsolationForest: need ≥ 30 rows and ≥ 2 numeric features (efficiency / fuel / delay).")

    # ---------------------------------------------------------
    # Automated Insights
    # ---------------------------------------------------------
    st.markdown("## Automated Insights")
    insights = []

    # Low-efficiency routes
    if "Efficiency_Score" in filt.columns and filt["Efficiency_Score"].notna().any() and "Route_ID" in filt.columns:
        grp = filt.groupby("Route_ID").agg(
            avg_eff=("Efficiency_Score","mean"),
            cnt=("Route_ID","count")
        ).reset_index()
        top_bad = grp.sort_values("avg_eff").head(10)
        for _, r in top_bad.iterrows():
            insights.append({
                "Insight_Type": "Low Efficiency Route",
                "Route_ID": r["Route_ID"],
                "Avg_Efficiency": round(float(r["avg_eff"]), 4),
                "Sample_Count": int(r["cnt"])
            })

    # High-delay routes
    if "Delay_Hours" in filt.columns and filt["Delay_Hours"].notna().any() and "Route_ID" in filt.columns:
        grp2 = filt.groupby("Route_ID").agg(
            avg_delay=("Delay_Hours","mean"),
            cnt=("Route_ID","count")
        ).reset_index()
        top_delay = grp2.sort_values("avg_delay", ascending=False).head(10)
        for _, r in top_delay.iterrows():
            insights.append({
                "Insight_Type": "High Avg Delay",
                "Route_ID": r["Route_ID"],
                "Avg_Delay_Hours": round(float(r["avg_delay"]), 3),
                "Sample_Count": int(r["cnt"])
            })

    # Fuel hog vehicles
    if "Fuel_L_per_km" in filt.columns and filt["Fuel_L_per_km"].notna().any() and "Vehicle_ID" in filt.columns:
        veh = filt.groupby("Vehicle_ID").agg(
            avg_fpk=("Fuel_L_per_km","mean"),
            cnt=("Vehicle_ID","count")
        ).reset_index().sort_values("avg_fpk", ascending=False).head(10)
        for _, r in veh.iterrows():
            insights.append({
                "Insight_Type": "High Fuel per km (Vehicle)",
                "Vehicle_ID": r["Vehicle_ID"],
                "Avg_L_per_km": round(float(r["avg_fpk"]), 4),
                "Sample_Count": int(r["cnt"])
            })

    # Overall stats
    if "Efficiency_Score" in filt.columns and filt["Efficiency_Score"].notna().any():
        insights.append({
            "Insight_Type": "Overall Avg Efficiency",
            "Value": round(float(filt["Efficiency_Score"].mean()), 4)
        })
    if "Delay_Hours" in filt.columns and filt["Delay_Hours"].notna().any():
        insights.append({
            "Insight_Type": "Overall Avg Delay (hrs)",
            "Value": round(float(filt["Delay_Hours"].mean()), 3)
        })
    if "Actual_Fuel_Liters" in filt.columns and filt["Actual_Fuel_Liters"].notna().any():
        insights.append({
            "Insight_Type": "Avg Fuel per Route (L)",
            "Value": round(float(filt["Actual_Fuel_Liters"].mean()), 3)
        })

    ins_df = pd.DataFrame(insights)
    if ins_df.empty:
        st.info("No insights generated for the current filter.")
    else:
        st.dataframe(ins_df, use_container_width=True)
        download_df(ins_df, "automated_insights_route.csv", "Download insights CSV")

# ---------------------------------------------------------
# ACTION PLAYBOOKS TAB
# ---------------------------------------------------------
with tabs[3]:
    st.header("Action Playbooks")

    if "filt" not in locals() or filt is None or len(filt) == 0:
        st.warning("No filtered data found. Load and filter data in the Application tab first.")
        st.stop()

    # 1) Routes to reassign
    st.subheader("1. Top Routes to Reassign (low efficiency + high fuel)")
    if "Efficiency_Score" in filt.columns and "Actual_Fuel_Liters" in filt.columns and "Route_ID" in filt.columns:
        play_route = filt.groupby("Route_ID").agg(
            avg_eff=("Efficiency_Score","mean"),
            avg_fuel=("Actual_Fuel_Liters","mean"),
            cnt=("Route_ID","count")
        ).reset_index()
        play_route["rank_score"] = (1 - play_route["avg_eff"].fillna(0)) * play_route["avg_fuel"].fillna(0)
        top_routes = play_route.sort_values("rank_score", ascending=False).head(10)
        st.markdown("These routes combine poor efficiency and high fuel burn. Re-evaluate scheduling, consolidation, or reassignment.")
        st.dataframe(top_routes, use_container_width=True)
        download_df(top_routes, "top_routes_reassign.csv", "Download playbook CSV")
    else:
        st.info("Missing columns for route reassignment insights.")

    # 2) Vehicles to audit
    st.subheader("2. Vehicles to Audit (fuel/km & delay)")
    if "Fuel_L_per_km" not in filt.columns and all(c in filt.columns for c in ["Actual_Fuel_Liters","Route_Distance_km"]):
        filt["Fuel_L_per_km"] = filt["Actual_Fuel_Liters"] / np.where(filt["Route_Distance_km"] == 0, np.nan, filt["Route_Distance_km"])

    if all(c in filt.columns for c in ["Fuel_L_per_km","Delay_Hours","Vehicle_ID"]):
        veh = filt.groupby("Vehicle_ID").agg(
            avg_fpk=("Fuel_L_per_km","mean"),
            avg_delay=("Delay_Hours","mean"),
            cnt=("Vehicle_ID","count")
        ).reset_index()
        veh["rank_score"] = veh["avg_fpk"].fillna(0) * 0.6 + veh["avg_delay"].fillna(0) * 0.4
        audit_veh = veh.sort_values("rank_score", ascending=False).head(10)
        st.markdown("Vehicles with abnormal fuel per km and chronic delays. Prioritise them for maintenance or reassignment.")
        st.dataframe(audit_veh, use_container_width=True)
        download_df(audit_veh, "vehicles_to_audit.csv", "Download playbook CSV")
    else:
        st.info("Missing columns for vehicle audit insights.")

    # 3) Drivers / operators to coach
    st.subheader("3. Drivers / Operators to Coach")
    driver_cols = [c for c in filt.columns if "driver" in c.lower() or "operator" in c.lower()]
    if not driver_cols and "Vehicle_ID" in filt.columns:
        driver_cols = ["Vehicle_ID"]

    if driver_cols:
        dcol = driver_cols[0]
        driver_df = filt.groupby(dcol).agg(
            avg_delay=("Delay_Hours","mean") if "Delay_Hours" in filt.columns else None,
            avg_eff=("Efficiency_Score","mean") if "Efficiency_Score" in filt.columns else None,
            cnt=(dcol,"count")
        ).reset_index()
        driver_df["score"] = (
            (1 - driver_df["avg_eff"].fillna(0)) * 0.5 +
            driver_df["avg_delay"].fillna(0) * 0.5
        )
        top_drivers = driver_df.sort_values("score", ascending=False).head(10)
        st.markdown("These drivers / vehicles show recurring delays or low efficiency. Use for coaching and route pairing decisions.")
        st.dataframe(top_drivers, use_container_width=True)
        download_df(top_drivers, "drivers_to_coach.csv", "Download playbook CSV")
    else:
        st.info("No driver/operator or vehicle identifier found for coaching insights.")

    # 4) High-risk traffic × weather combos
    st.subheader("4. High-Risk Traffic × Weather Conditions")
    if all(c in filt.columns for c in ["Traffic_Level","Weather_Condition","Delay_Hours","Route_ID"]):
        risk = filt.groupby(["Traffic_Level","Weather_Condition"]).agg(
            avg_delay=("Delay_Hours","mean"),
            cnt=("Route_ID","count")
        ).reset_index()
        high_risk = risk.sort_values("avg_delay", ascending=False).head(10)
        st.markdown("These traffic–weather combinations produce the worst delays. Use them to drive dynamic routing rules & SLAs.")
        st.dataframe(high_risk, use_container_width=True)
        download_df(high_risk, "high_risk_conditions.csv", "Download playbook CSV")
    else:
        st.info("Missing columns for traffic × weather risk matrix.")

    # 5) Executive summary
    st.subheader("5. Executive Recommendations")
    st.markdown("""
    <div class='card left-align'>
    • Reassign or redesign the 10 least efficient high-fuel routes.<br>
    • Audit vehicles with the worst fuel/km and highest delay scores.<br>
    • Coach or re-pair drivers / operators flagged by the inefficiency scorecard.<br>
    • Avoid high-risk traffic–weather combinations using pre-set routing rules.<br>
    • Use ML-predicted travel hours for planning time-critical routes.<br>
    • Feed these insights into monthly fleet reviews and vendor contracts.
    </div>
    """, unsafe_allow_html=True)
