# app_route_optimization.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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

# ------------------------
# Page config & title
# ------------------------
st.set_page_config(page_title="Route Optimization & Logistics Efficiency", layout="wide")
# Left-aligned big header
st.markdown("""
<div style="text-align: left;">
  <h1 style="margin:0; padding:0;">Route Optimization & Logistics Efficiency</h1>
  <p style="margin-top:4px; color:#555">Reduce miles, cut fuel, speed deliveries — with data.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------
# Card CSS (hover glow)
# ------------------------
st.markdown("""
<style>
.card {
    padding:20px;
    border-radius:12px;
    background:#ffffff;
    border:1px solid #e6e6e6;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 8px 30px rgba(6,75,134,0.18);
    border-color:#064b86;
}
.kpi {
    padding:28px;
    border-radius:12px;
    background:#ffffff;
    border:1px solid #e6e6e6;
    text-align:center;
    font-weight:700;
    color:#064b86;
    font-size:20px;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.kpi:hover {
    transform: translateY(-6px);
    box-shadow: 0 8px 30px rgba(6,75,134,0.18);
    border-color:#064b86;
}
.small { color:#666; font-size:13px; }
.left-align { text-align: left; }
</style>
""", unsafe_allow_html=True)


def read_csv_safe(src):
    # handles duplicate column names by making them unique
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

# ------------------------
# Tabs
# ------------------------
tabs = st.tabs(["Overview", "Application"])

# ------------------------
# Overview Tab
# ------------------------
with tabs[0]:
    # Left column: Capabilities (left-aligned cards)
    st.markdown("### Overview", unsafe_allow_html=True)
    st.markdown("""
    <div class='card left-align'>
      <b>Purpose</b>: Cut route costs and delivery time by optimizing routes, predicting delays and fuel usage, and prioritizing high-impact improvement opportunities.
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Capabilities")
        st.markdown("""
        <div class='card left-align'>
        • Route efficiency scoring and anomaly detection<br>
        • Predictive travel time and fuel consumption models<br>
        • Cluster similar routes and vehicles for capacity planning<br>
        • Multi-dimensional filters (vehicle, route, traffic, weather)<br>
        • Exportable prioritized actions for operations
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("#### Business impact")
        st.markdown("""
        <div class='card left-align'>
        • Lower fuel & operational cost<br>
        • Faster deliveries & higher on-time rate<br>
        • Reduced CO2 emissions per unit shipped<br>
        • Better fleet utilisation and scheduling<br>
        • Data-driven procurement of vehicles & drivers
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### KPIs")
    # 5 KPI cards in one row
    kpicol1, kpicol2, kpicol3, kpicol4, kpicol5 = st.columns(5)
    kpicol1.markdown("<div class='kpi'>Total Routes Tracked</div>", unsafe_allow_html=True)
    kpicol2.markdown("<div class='kpi'>Avg Efficiency Score</div>", unsafe_allow_html=True)
    kpicol3.markdown("<div class='kpi'>Avg Delay (hrs)</div>", unsafe_allow_html=True)
    kpicol4.markdown("<div class='kpi'>Avg Fuel L (per route)</div>", unsafe_allow_html=True)
    kpicol5.markdown("<div class='kpi'>On-Time %</div>", unsafe_allow_html=True)

    st.markdown("### Who should use this app & How")
    st.markdown("""
    <div class='card left-align'>
      <b>Who</b>: Fleet managers, logistics planners, operations leads, sustainability teams.<br><br>
      <b>How</b>: 1) Load dataset (default/generate or upload). 2) Use filters to focus on routes/vehicles. 3) Review top-risk/inefficient routes. 4) Export prioritized fixes and retrain models periodically.
    </div>
    """, unsafe_allow_html=True)

# ------------------------
# Application Tab
# ------------------------
with tabs[1]:
    st.header("Application")
    st.markdown("### Step 1 — Load dataset (choose one)")

    load_mode = st.radio("Dataset option:", ["Default Data", "Upload CSV", "Upload + Map Columns"], horizontal=True)
    df = None

    if load_mode == "Default Data":
      url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/supply_chain/route_optimization_dataset.csv"
      try:
          df = read_csv_safe(url)
          st.success("Loaded dataset from URL.")
          st.dataframe(df.head())
      except Exception as e:
        st.error("Failed to load CSV from URL: " + str(e))
        st.stop()
    else:
        st.info("Press 'Load from URL' after pasting the raw CSV URL.")
        st.stop()

    elif load_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            try:
                df = read_csv_safe(uploaded)
                st.success("File uploaded.")
                st.dataframe(df.head())
            except Exception as e:
                st.error("Upload failed: " + str(e))
                st.stop()
        else:
            st.stop()

    else:  # Upload + Map
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"], key="upload_map")
        if uploaded:
            raw = read_csv_safe(uploaded)
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head())
            st.markdown("Map columns (map at least the key columns you have).")
            # suggest likely names
            cols = list(raw.columns)
            mapping = {}
            expected = [
                "Timestamp","Vehicle_ID","Vehicle_Type","Route_ID","Start_City","End_City",
                "Route_Distance_km","Traffic_Level","Weather_Condition",
                "Predicted_Travel_Hours","Actual_Travel_Hours",
                "Predicted_Fuel_Liters","Actual_Fuel_Liters","Vehicle_Capacity_kg",
                "Load_Weight_kg","Delay_Hours","Efficiency_Score"
            ]
            for e in expected:
                mapping[e] = st.selectbox(f"Map → {e}", ["-- Skip --"] + cols, index=0)
            if st.button("Apply mapping"):
                rename = {}
                for k,v in mapping.items():
                    if v != "-- Skip --":
                        rename[v] = k
                if not rename:
                    st.error("You must map at least one column.")
                    st.stop()
                df = raw.rename(columns=rename)
                st.success("Mapping applied.")
                st.dataframe(df.head())
        else:
            st.stop()

    # ensure df exists
    if df is None:
        st.stop()

    # Basic cleaning: trim column names
    df.columns = [str(c).strip() for c in df.columns]

    # Ensure Timestamp column exists or try to infer common names
    if "Timestamp" not in df.columns:
        possible_ts = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if possible_ts:
            st.info(f"Using `{possible_ts[0]}` as Timestamp column.")
            df = df.rename(columns={possible_ts[0]:"Timestamp"})
    # coerce Timestamp
    if "Timestamp" in df.columns:
        try:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        except:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Fill / coerce numeric columns
    numeric_candidates = ["Route_Distance_km","Predicted_Travel_Hours","Actual_Travel_Hours",
                          "Predicted_Fuel_Liters","Actual_Fuel_Liters","Vehicle_Capacity_kg",
                          "Load_Weight_kg","Delay_Hours","Efficiency_Score"]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # If Efficiency_Score not present, compute approx from predictions if possible
    if "Efficiency_Score" not in df.columns and all(x in df.columns for x in ["Predicted_Travel_Hours","Actual_Travel_Hours","Predicted_Fuel_Liters","Actual_Fuel_Liters"]):
        df["Efficiency_Score"] = (df["Predicted_Travel_Hours"]/df["Actual_Travel_Hours"]) * (df["Predicted_Fuel_Liters"]/df["Actual_Fuel_Liters"])
    if "Efficiency_Score" not in df.columns:
        df["Efficiency_Score"] = np.nan

    # Safe defaults for missing numeric columns
    for c in numeric_candidates:
        if c not in df.columns:
            df[c] = np.nan

    # ------------------------
    # Step 2 — Filters (date range slider + other filters)
    # ------------------------
    st.markdown("### Step 2 — Filters & Preview")
    col_a, col_b, col_c = st.columns([2,2,2])

    # date slider: create min_ts and max_ts datetime objects of same type
    if df["Timestamp"].notna().any():
        min_ts = df["Timestamp"].min()
        max_ts = df["Timestamp"].max()
        # slider requires values not None and same types; provide fallback
        try:
            date_range = st.slider("Select date range", min_value=min_ts, max_value=max_ts, value=(min_ts, max_ts), format="YYYY-MM-DD HH:mm")
            start_sel, end_sel = date_range
        except Exception:
            # fallback to date_input if slider complains (consistent types)
            st.info("Date slider unsupported in this environment — using date_input instead.")
            dates = st.date_input("Select date range (date only)", value=(min_ts.date(), max_ts.date()))
            start_sel = datetime.combine(dates[0], datetime.min.time())
            end_sel = datetime.combine(dates[1], datetime.max.time())
    else:
        start_sel = None
        end_sel = None
        st.info("No Timestamp column or all null timestamps; skipping date filter.")

    vehicles = df["Vehicle_ID"].dropna().unique().tolist() if "Vehicle_ID" in df.columns else []
    vtypes = df["Vehicle_Type"].dropna().unique().tolist() if "Vehicle_Type" in df.columns else []
    routes = df["Route_ID"].dropna().unique().tolist() if "Route_ID" in df.columns else []

    sel_vehicles = col_a.multiselect("Vehicle_ID", options=vehicles, default=vehicles[:5] if vehicles else [])
    sel_vtypes = col_b.multiselect("Vehicle_Type", options=vtypes, default=vtypes[:3] if vtypes else [])
    sel_routes = col_c.multiselect("Route_ID", options=routes, default=routes[:5] if routes else [])

    # filter
    filt = df.copy()
    if start_sel is not None and end_sel is not None and "Timestamp" in filt.columns:
        filt = filt[(filt["Timestamp"] >= pd.to_datetime(start_sel)) & (filt["Timestamp"] <= pd.to_datetime(end_sel))]
    if sel_vehicles:
        if "Vehicle_ID" in filt.columns:
            filt = filt[filt["Vehicle_ID"].isin(sel_vehicles)]
    if sel_vtypes:
        if "Vehicle_Type" in filt.columns:
            filt = filt[filt["Vehicle_Type"].isin(sel_vtypes)]
    if sel_routes:
        if "Route_ID" in filt.columns:
            filt = filt[filt["Route_ID"].isin(sel_routes)]

    st.markdown("Preview (first 10 rows):")
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(500), "filtered_route_preview.csv", label="Download filtered preview (up to 500 rows)")

    # ------------------------
    # Step 3 — EDA Charts (lots, but not duplicates)
    # ------------------------
    st.markdown("## Exploratory Data Analysis")

    # 1) Route distance histogram
    if filt["Route_Distance_km"].notna().any():
        fig_dist = px.histogram(filt, x="Route_Distance_km", nbins=30, title="Route distance distribution (km)")
        st.plotly_chart(fig_dist, use_container_width=True)
    # 2) Efficiency score by vehicle type (box)
    if "Vehicle_Type" in filt.columns and filt["Efficiency_Score"].notna().any():
        fig_eff = px.box(filt, x="Vehicle_Type", y="Efficiency_Score", title="Efficiency Score by Vehicle Type")
        st.plotly_chart(fig_eff, use_container_width=True)
    # 3) Delay vs Distance scatter with color by traffic
    if filt["Delay_Hours"].notna().any() and filt["Route_Distance_km"].notna().any():
        fig_sc = px.scatter(filt, x="Route_Distance_km", y="Delay_Hours", color="Traffic_Level", 
                            size="Load_Weight_kg" if "Load_Weight_kg" in filt.columns else None,
                            title="Delay vs Distance (bubble size = load)")
        st.plotly_chart(fig_sc, use_container_width=True)
    # 4) Fuel efficiency: Actual_Fuel_Liters per km distribution
    if filt["Actual_Fuel_Liters"].notna().any() and filt["Route_Distance_km"].notna().any():
        filt["Fuel_L_per_km"] = filt["Actual_Fuel_Liters"] / np.where(filt["Route_Distance_km"]==0, np.nan, filt["Route_Distance_km"])
        fig_fuel = px.violin(filt, y="Fuel_L_per_km", box=True, title="Fuel consumption (L per km) distribution")
        st.plotly_chart(fig_fuel, use_container_width=True)
    # 5) Time-series of avg efficiency over time
    if "Timestamp" in filt.columns and filt["Efficiency_Score"].notna().any():
        ts = filt.set_index("Timestamp").resample("D").agg({"Efficiency_Score":"mean"}).reset_index()
        fig_ts = px.line(ts, x="Timestamp", y="Efficiency_Score", title="Daily avg Efficiency Score")
        st.plotly_chart(fig_ts, use_container_width=True)
    # 6) Route clusters (kmeans) visualized by 2 components (distance, delay)
    if filt["Route_Distance_km"].notna().any() and filt["Delay_Hours"].notna().any() and len(filt)>=10:
        kdf = filt[["Route_Distance_km","Delay_Hours"]].dropna().sample(n=min(800,len(filt)), random_state=42)
        try:
            kmeans = KMeans(n_clusters=4, random_state=42)
            klabels = kmeans.fit_predict(kdf)
            kdf_plot = kdf.copy()
            kdf_plot["cluster"] = klabels.astype(str)
            fig_k = px.scatter(kdf_plot, x="Route_Distance_km", y="Delay_Hours", color="cluster", title="KMeans clusters by distance and delay")
            st.plotly_chart(fig_k, use_container_width=True)
        except Exception:
            st.info("KMeans cluster failed (not enough data or other issue).")

    # 7) Pairwise correlations heatmap for numeric columns
    numcols = ["Route_Distance_km","Predicted_Travel_Hours","Actual_Travel_Hours","Predicted_Fuel_Liters","Actual_Fuel_Liters","Load_Weight_kg","Delay_Hours","Efficiency_Score"]
    existing_numcols = [c for c in numcols if c in filt.columns]
    if len(existing_numcols) >= 2:
        corr = filt[existing_numcols].corr()
        fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="Blues"))
        fig_corr.update_layout(title="Correlation matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

    # ------------------------
    # Step 4 — ML Concepts (4 chosen)
    #   1) RandomForestRegressor (predict Actual_Travel_Hours)
    #   2) GradientBoostingRegressor (predict Actual_Fuel_Liters)
    #   3) KNeighborsRegressor (predict Delay_Hours)
    #   4) IsolationForest (anomaly detection on fuel/efficiency)
    # ------------------------
    st.markdown("## Machine Learning — models & predictions")
    st.markdown("Note: Models train only if required columns and sufficient rows exist (>= 80 rows).")

    # Prepare features: do encoding for categorical features (simple)
    base_features = []
    if "Route_Distance_km" in filt.columns:
        base_features.append("Route_Distance_km")
    if "Load_Weight_kg" in filt.columns:
        base_features.append("Load_Weight_kg")
    # one-hot for Vehicle_Type, Traffic_Level, Weather_Condition (use pd.get_dummies)
    cat_cols = [c for c in ["Vehicle_Type","Traffic_Level","Weather_Condition","Start_City","End_City"] if c in filt.columns]
    df_model = filt.copy().reset_index(drop=True)
    if cat_cols:
        df_model = pd.get_dummies(df_model, columns=cat_cols, dummy_na=False)
    # fill numeric NAs with median
    for c in df_model.columns:
        if df_model[c].dtype.kind in 'biufc' and df_model[c].isna().any():
            df_model[c] = df_model[c].fillna(df_model[c].median())

    # Helper to train & evaluate regression
    def train_eval_reg(X, y, model):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = math.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        return model, X_test, y_test, preds, rmse, r2

    # 1) RandomForestRegressor -> predict Actual_Travel_Hours
    if "Actual_Travel_Hours" in df_model.columns and len(df_model) >= 80:
        target = "Actual_Travel_Hours"
        features = [c for c in df_model.columns if c not in ["Timestamp","Vehicle_ID","Route_ID","Predicted_Travel_Hours","Actual_Travel_Hours","Predicted_Fuel_Liters","Actual_Fuel_Liters","Efficiency_Score"]]
        if features:
            X = df_model[features].select_dtypes(include=[np.number]).fillna(0)
            y = df_model[target].fillna(0)
            # scale numeric features
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            rf = RandomForestRegressor(n_estimators=150, random_state=42)
            rf, X_test, y_test, preds, rmse, r2 = train_eval_reg(Xs, y, rf)
            st.markdown("### RandomForest — Predict Actual Travel Hours")
            st.write(f"RMSE: {rmse:.3f}  |  R²: {r2:.3f}")
            out = pd.DataFrame({"Actual": y_test, "Predicted": preds})
            st.dataframe(out.head(10))
            download_df(out, "rf_actual_travel_hours_predictions.csv", label="Download RF predictions")
        else:
            st.info("No numeric features available for RandomForest model.")
    else:
        st.info("Not enough data to train RandomForest for Actual_Travel_Hours (need >=80 rows and the column).")

    # 2) GradientBoostingRegressor -> predict Actual_Fuel_Liters
    if "Actual_Fuel_Liters" in df_model.columns and len(df_model) >= 80:
        target2 = "Actual_Fuel_Liters"
        features2 = [c for c in df_model.columns if c not in ["Timestamp","Vehicle_ID","Route_ID","Predicted_Travel_Hours","Actual_Travel_Hours","Predicted_Fuel_Liters","Actual_Fuel_Liters","Efficiency_Score"]]
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
            st.dataframe(out2.head(10))
            download_df(out2, "gbr_actual_fuel_predictions.csv", label="Download GBR predictions")
        else:
            st.info("No numeric features available for GradientBoosting model.")
    else:
        st.info("Not enough data to train GradientBoosting for Actual_Fuel_Liters (need >=80 rows and the column).")

    # 3) KNeighborsRegressor -> predict Delay_Hours
    if "Delay_Hours" in df_model.columns and len(df_model) >= 80:
        target3 = "Delay_Hours"
        features3 = [c for c in df_model.columns if c not in ["Timestamp","Vehicle_ID","Route_ID","Delay_Hours","Efficiency_Score"]]
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
            st.dataframe(out3.head(10))
            download_df(out3, "knn_delay_predictions.csv", label="Download KNN predictions")
        else:
            st.info("No numeric features available for KNN model.")
    else:
        st.info("Not enough data to train KNN for Delay_Hours (need >=80 rows and the column).")

    # 4) IsolationForest anomaly detection on efficiency and fuel
    st.markdown("### Anomaly Detection — IsolationForest (flag unusual fuel/efficiency)")
    iso_cols = [c for c in ["Efficiency_Score","Actual_Fuel_Liters","Fuel_L_per_km","Delay_Hours"] if c in filt.columns or c=="Fuel_L_per_km"]
    if "Fuel_L_per_km" not in filt.columns and "Actual_Fuel_Liters" in filt.columns and "Route_Distance_km" in filt.columns:
        filt["Fuel_L_per_km"] = filt["Actual_Fuel_Liters"] / np.where(filt["Route_Distance_km"]==0, np.nan, filt["Route_Distance_km"])
    iso_cols = [c for c in ["Efficiency_Score","Actual_Fuel_Liters","Fuel_L_per_km","Delay_Hours"] if c in filt.columns]
    if len(iso_cols) >= 2 and len(filt) >= 30:
        iso_feats = filt[iso_cols].fillna(0)
        iso = IsolationForest(contamination=0.03, random_state=42)
        try:
            filt["_is_anomaly"] = iso.fit_predict(iso_feats)
            filt["_is_anomaly"] = np.where(filt["_is_anomaly"]==-1, 1, 0)
            num_anoms = int(filt["_is_anomaly"].sum())
            st.markdown(f"Detected anomalies: **{num_anoms}** (IsolationForest)")
            if num_anoms>0:
                st.dataframe(filt.loc[filt["_is_anomaly"]==1, ["Timestamp","Vehicle_ID","Route_ID","Route_Distance_km","Actual_Fuel_Liters","Fuel_L_per_km","Efficiency_Score","Delay_Hours"]].head(20))
                download_df(filt.loc[filt["_is_anomaly"]==1, ["Timestamp","Vehicle_ID","Route_ID","Route_Distance_km","Actual_Fuel_Liters","Fuel_L_per_km","Efficiency_Score","Delay_Hours"]], "anomalies_route.csv", label="Download anomalies CSV")
        except Exception as e:
            st.info("Anomaly detection failed: " + str(e))
    else:
        st.info("Not enough numeric columns or rows for anomaly detection (need >=30 rows and >=2 numeric features).")

    # ------------------------
    # Automated Insights
    # ------------------------
    st.markdown("## Automated Insights (generated)")
    insights = []

    # Top inefficient routes (lowest efficiency score)
    if "Efficiency_Score" in filt.columns and filt["Efficiency_Score"].notna().any():
        grp = filt.groupby("Route_ID").agg(avg_eff=("Efficiency_Score","mean"), cnt=("Route_ID","count")).reset_index()
        top_bad = grp.sort_values("avg_eff").head(10)
        for _, r in top_bad.iterrows():
            insights.append({
                "Insight_Type":"Low Efficiency Route",
                "Route_ID": r["Route_ID"],
                "Avg_Efficiency": round(r["avg_eff"],4),
                "Sample_Count": int(r["cnt"])
            })
    # Routes with high average delay
    if "Delay_Hours" in filt.columns and filt["Delay_Hours"].notna().any():
        grp2 = filt.groupby("Route_ID").agg(avg_delay=("Delay_Hours","mean"), cnt=("Route_ID","count")).reset_index()
        top_delay = grp2.sort_values("avg_delay", ascending=False).head(10)
        for _, r in top_delay.iterrows():
            insights.append({
                "Insight_Type":"High Avg Delay",
                "Route_ID": r["Route_ID"],
                "Avg_Delay_Hours": round(r["avg_delay"],3),
                "Sample_Count": int(r["cnt"])
            })
    # Vehicles with high fuel consumption per km
    if "Fuel_L_per_km" in filt.columns and filt["Fuel_L_per_km"].notna().any():
        veh = filt.groupby("Vehicle_ID").agg(avg_fpk=("Fuel_L_per_km","mean"), cnt=("Vehicle_ID","count")).reset_index().sort_values("avg_fpk", ascending=False).head(10)
        for _, r in veh.iterrows():
            insights.append({
                "Insight_Type":"High Fuel per km (Vehicle)",
                "Vehicle_ID": r["Vehicle_ID"],
                "Avg_L_per_km": round(r["avg_fpk"],4),
                "Sample_Count": int(r["cnt"])
            })
    # Overall stats
    if "Efficiency_Score" in filt.columns and filt["Efficiency_Score"].notna().any():
        insights.append({"Insight_Type":"Overall Avg Efficiency", "Value": round(float(filt["Efficiency_Score"].mean()),4)})
    if "Delay_Hours" in filt.columns and filt["Delay_Hours"].notna().any():
        insights.append({"Insight_Type":"Overall Avg Delay (hrs)", "Value": round(float(filt["Delay_Hours"].mean()),3)})
    if "Actual_Fuel_Liters" in filt.columns and filt["Actual_Fuel_Liters"].notna().any():
        insights.append({"Insight_Type":"Avg Fuel per Route (L)", "Value": round(float(filt["Actual_Fuel_Liters"].mean()),3)})

    ins_df = pd.DataFrame(insights)
    if ins_df.empty:
        st.info("No insights generated for the current filter.")
    else:
        st.dataframe(ins_df, use_container_width=True)
        download_df(ins_df, "automated_insights_route.csv", label="Download insights CSV")

    st.markdown("### Done — export what you need and start cutting costs. If the app crashes, send coffee to the nearest data engineer.")

# end of file
