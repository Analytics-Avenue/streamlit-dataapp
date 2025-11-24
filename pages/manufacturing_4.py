import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO, StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import math
import warnings
warnings.filterwarnings("ignore")

# -------------------------

hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# ----------------------------
# Page config and CSS
# ----------------------------
st.set_page_config(page_title="Energy & Resource Wastage Analytics", layout="wide", page_icon="⚡")

# Left-aligned title and header styling, hover-glow for cards and KPI visuals
st.markdown("""
<style>
/* left align header */
.header-row { display:flex; align-items:center; gap:12px; justify-content:flex-start; }
.header-title { font-size:28px; font-weight:700; color:#064b86; margin:0; padding:0; text-align:left; }
.header-sub { font-size:14px; color:#444; margin:0; padding:0; text-align:left; }

/* card + hover glow */
.card {
    padding: 16px;
    border-radius: 12px;
    background: #ffffff;
    border: 1px solid #e6e9ee;
    transition: all 0.22s ease;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 8px 30px rgba(6,75,134,0.14);
    border-color: #064b86;
}

/* KPI visual large labels */
.kpi {
    padding: 24px;
    border-radius: 12px;
    background: #fff;
    border: 1px solid rgba(6,75,134,0.10);
    font-size:22px;
    font-weight:700;
    color:#064b86;
    text-align:center;
    transition: all 0.18s ease;
}
.kpi:hover {
    transform: translateY(-4px);
    box-shadow: 0 6px 24px rgba(6,75,134,0.12);
}

/* left-aligned card text */
.left-align { text-align: left !important; }

/* small helper text */
.small { font-size:12px; color:#666; }

/* make streamlit default containers a bit more airy */
.block-container { padding-top: 12px; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Helper functions
# ----------------------------

def read_csv_safe(url_or_file):
    """
    Read CSV from URL or file-like. If duplicate columns exist, make them unique
    by appending suffixes: col, col__dup1, col__dup2...
    Returns pandas DataFrame.
    """
    df = pd.read_csv(url_or_file)
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

def download_df(df: pd.DataFrame, filename: str, button_label: str = "Download CSV", key=None):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(button_label, b, file_name=filename, mime="text/csv", key=key)

def safe_mean(series):
    try:
        return float(np.nanmean(series))
    except:
        return None

def add_linear_regression_trace(fig, x, y, name="Linear fit", color=None):
    """
    Add a linear regression line (numpy polyfit) to a plotly figure.
    Avoids using statsmodels so no extra dependency.
    """
    # remove missing
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if mask.sum() < 2:
        return fig
    xp = x[mask]
    yp = y[mask]
    coeffs = np.polyfit(xp, yp, deg=1)
    slope, intercept = coeffs[0], coeffs[1]
    x_line = np.linspace(np.min(xp), np.max(xp), 50)
    y_line = slope * x_line + intercept
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name=name,
                             line=dict(dash="dash", color=color or "black")))
    return fig

# ----------------------------
# App header left-aligned
# ----------------------------
col_logo, col_title = st.columns([0.18, 3])
with col_logo:
    st.image("https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png", width=64)
with col_title:
    st.markdown("<div class='header-row'><div><p class='header-title'>Energy & Resource Wastage Analytics</p><p class='header-sub'>Detect energy waste, predict consumption, and prioritize efficiency actions.</p></div></div>", unsafe_allow_html=True)

# ----------------------------
# Tabs: Overview & Application
# ----------------------------
tabs = st.tabs(["Overview", "Application"])

# ----------------------------
# Overview Page
# ----------------------------
with tabs[0]:
    st.markdown("### Overview")
    st.markdown("""<div class='card left-align'>
                    <b>Purpose</b>: Monitor fuel and power usage, identify wastage patterns, and recommend corrective actions at machine, shift and plant level.
                   </div>""", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Capabilities")
        st.markdown("""
            <div class='card left-align'>
            • Machine-level energy consumption monitoring<br>
            • Energy prediction models (short-term & per-shift)<br>
            • Anomaly detection for sudden spikes or baseload drift<br>
            • Operational clustering (modes) to identify inefficient states<br>
            • Downloadable predictions & prioritized action lists
            </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("#### Business impact")
        st.markdown("""
            <div class='card left-align'>
            • Reduce energy cost by ~10-30% across large units<br>
            • Lower CO₂ emissions via targeted interventions<br>
            • Avoid over-consumption due to idle/peak inefficiencies<br>
            • Improve planning for fuel & electricity procurement
            </div>
        """, unsafe_allow_html=True)

    # KPI cards (5 in a row as requested)
    st.markdown("### KPIs")
    kcols = st.columns(5)
    klabels = ["Total Energy (kWh)", "Avg Energy/Unit", "Peak Load Events", "Idle Hours", "Estimated Savings"]
    for kc, label in zip(kcols, klabels):
        kc.markdown(f"<div class='kpi'>{label}</div>", unsafe_allow_html=True)

    st.markdown("### Who should use & How")
    st.markdown("""<div class='card left-align'>
                    <b>Who</b>: Energy managers, plant managers, sustainability teams, operations leads.<br><br>
                    <b>How</b>: Load the plant sensor & meter data, filter by date/machine/shift, review anomaly and ML prediction outputs, export prioritized action lists for maintenance & scheduling.
                   </div>""", unsafe_allow_html=True)

# ----------------------------
# Application Page
# ----------------------------
with tabs[1]:
    st.markdown("### Application")
    st.markdown("Select data load option and proceed to EDA, ML and Automated Insights.")

    data_option = st.radio("Dataset option:", ["Default (GitHub URL)", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)
    df = None

    # default dataset URL suggestion (user will provide actual dataset if needed)
    DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/manufacturing/energy_wastage.csv"
    # NOTE: replace DEFAULT_URL with your actual raw GitHub CSV URL

    REQUIRED_COLS = [
        "Timestamp","Machine_ID","Machine_Type","Fuel_Type","Shift","Operator_ID",
        "Energy_Consumption_kWh","Energy_Predicted_kWh","Baseload_Drift_kWh","Cooling_Load_kWh",
        "Fuel_Consumption","Compressed_Air_CFM","Voltage_Instability","Power_Factor","Ambient_Temp_C",
        "Output_Units","Energy_Intensity_kWh_per_Unit","Peak_Load_Flag","Idle_Flag","Wastage_Severity","CO2_Emissions_kg"
    ]

    if data_option == "Default (GitHub URL)":
        st.info("Loading default dataset from DEFAULT_URL. Replace DEFAULT_URL in the code if needed.")
        try:
            df = read_csv_safe(DEFAULT_URL)
            st.success("Default dataset loaded (from DEFAULT_URL).")
            st.dataframe(df.head())
        except Exception as e:
            st.error("Failed to load default dataset. Check DEFAULT_URL or your network. Error: " + str(e))
            st.info("You can upload a CSV instead.")
            df = None

    elif data_option == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"], help="Upload CSV with columns like Timestamp, Machine_ID, Energy_Consumption_kWh ...")
        if uploaded:
            try:
                df = read_csv_safe(uploaded)
                st.success("File uploaded.")
                st.dataframe(df.head())
            except Exception as e:
                st.error("Failed to read uploaded CSV: " + str(e))
                st.stop()
        else:
            st.stop()

    else:  # mapping
        uploaded = st.file_uploader("Upload CSV for mapping", type=["csv"])
        if uploaded:
            raw = read_csv_safe(uploaded)
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head())
            st.markdown("Map your columns to the required fields below (map as many as available).")
            cols = list(raw.columns)
            mapping = {}
            for req in REQUIRED_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- None --"] + cols, index=0, key=f"map_{req}")
            if st.button("Apply mapping"):
                # build rename map (only for chosen)
                rename_map = {}
                for req, sel in mapping.items():
                    if sel != "-- None --":
                        rename_map[sel] = req
                if rename_map:
                    df = raw.rename(columns=rename_map)
                    st.success("Mapping applied.")
                    st.dataframe(df.head())
                else:
                    st.error("You must map at least one column.")
        else:
            st.stop()

    if df is None:
        st.stop()

    # ---------- Standardize & Clean ----------
    # trim column names
    df.columns = [str(c).strip() for c in df.columns]

    # Try to coerce Timestamp to datetime (many data sets use different formats)
    if "Timestamp" in df.columns:
        try:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        except Exception:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Numeric conversions for expected numeric columns
    numeric_cols = ["Energy_Consumption_kWh","Energy_Predicted_kWh","Baseload_Drift_kWh","Cooling_Load_kWh",
                    "Fuel_Consumption","Compressed_Air_CFM","Voltage_Instability","Power_Factor",
                    "Ambient_Temp_C","Output_Units","Energy_Intensity_kWh_per_Unit","CO2_Emissions_kg"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Flags to numeric
    for flag in ["Peak_Load_Flag","Idle_Flag","Wastage_Severity","Shortage_Flag"]:
        if flag in df.columns:
            try:
                df[flag] = pd.to_numeric(df[flag], errors="coerce").fillna(0).astype(int)
            except:
                pass

    # Fill numeric NaNs with sensible defaults (median or 0)
    for c in numeric_cols:
        if c in df.columns:
            if df[c].notna().any():
                df[c] = df[c].fillna(df[c].median())
            else:
                df[c] = df[c].fillna(0)

    # Basic EDA controls
    st.markdown("### Filters & Preview")
    left_col, right_col = st.columns([2,1])

    # Date range filter: use date_input slicer (safer)
    if "Timestamp" in df.columns:
        min_ts = df["Timestamp"].min().date()
        max_ts = df["Timestamp"].max().date()
        date_range = left_col.date_input("Select date range", value=(min_ts, max_ts), min_value=min_ts, max_value=max_ts)
        # Ensure date_range is a tuple
        if isinstance(date_range, list) or isinstance(date_range, tuple):
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        else:
            start_date, end_date = pd.to_datetime(date_range), pd.to_datetime(date_range)
    else:
        start_date, end_date = None, None

    # Machine / Fuel / Shift filters
    machine_list = sorted(df["Machine_ID"].dropna().unique().tolist()) if "Machine_ID" in df.columns else []
    fuel_list = sorted(df["Fuel_Type"].dropna().unique().tolist()) if "Fuel_Type" in df.columns else []
    shift_list = sorted(df["Shift"].dropna().unique().tolist()) if "Shift" in df.columns else []

    sel_machines = left_col.multiselect("Machine_ID", options=machine_list, default=machine_list[:10])
    sel_fuel = left_col.multiselect("Fuel_Type", options=fuel_list, default=fuel_list)
    sel_shift = left_col.multiselect("Shift", options=shift_list, default=shift_list)

    # Preview after filtering
    filt = df.copy()
    if start_date is not None and end_date is not None and "Timestamp" in filt.columns:
        filt = filt[(filt["Timestamp"] >= pd.to_datetime(start_date)) & (filt["Timestamp"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]
    if sel_machines:
        if "Machine_ID" in filt.columns:
            filt = filt[filt["Machine_ID"].isin(sel_machines)]
    if sel_fuel:
        if "Fuel_Type" in filt.columns:
            filt = filt[filt["Fuel_Type"].isin(sel_fuel)]
    if sel_shift:
        if "Shift" in filt.columns:
            filt = filt[filt["Shift"].isin(sel_shift)]

    right_col.markdown("#### Preview (first 10 rows)")
    right_col.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(500), "filtered_energy_preview.csv", "Download sample preview", key="preview_download")

    # ---------- Dynamic KPIs ----------
    st.markdown("### Key Metrics (Dynamic)")
    kcols = st.columns(5)

    total_energy = filt["Energy_Consumption_kWh"].sum() if "Energy_Consumption_kWh" in filt.columns else 0
    energy_per_unit = (filt["Energy_Intensity_kWh_per_Unit"].mean() if "Energy_Intensity_kWh_per_Unit" in filt.columns else None)
    peak_events = int(filt["Peak_Load_Flag"].sum()) if "Peak_Load_Flag" in filt.columns else 0
    idle_hours = int(filt["Idle_Flag"].sum()) if "Idle_Flag" in filt.columns else 0
    # crude estimated savings = sum of wastage severity * some factor (example)
    est_savings = 0
    if "Wastage_Severity" in filt.columns and "Energy_Consumption_kWh" in filt.columns:
        est_savings = (filt["Wastage_Severity"].sum() * 0.05 * filt["Energy_Consumption_kWh"].mean())  # illustrative

    kvals = [f"{total_energy:,.0f} kWh" if total_energy else "N/A",
             f"{energy_per_unit:.2f} kWh/unit" if energy_per_unit is not None else "N/A",
             f"{peak_events}" if peak_events else "0",
             f"{idle_hours}" if idle_hours else "0",
             f"~₹{est_savings:,.0f}" if est_savings else "N/A"]

    for kc, label, val in zip(kcols, ["Total Energy", "Avg Energy/Unit", "Peak Load Events", "Idle Hours", "Estimated Savings"], kvals):
        kc.markdown(f"<div class='kpi'>{label}<div style='font-size:14px; font-weight:600; color:#444; margin-top:8px'>{val}</div></div>", unsafe_allow_html=True)

    # ---------- Charts (a variety, not repeating previous ones) ----------
    st.markdown("### Visualizations")

    # 1. Time series: total energy consumption per day (area)
    if "Timestamp" in filt.columns and "Energy_Consumption_kWh" in filt.columns:
        daily = filt.set_index("Timestamp").resample("D")["Energy_Consumption_kWh"].sum().reset_index()
        fig_daily = px.area(daily, x="Timestamp", y="Energy_Consumption_kWh", title="Daily Energy Consumption (kWh)")
        st.plotly_chart(fig_daily, use_container_width=True)
    else:
        st.info("Timestamp or Energy_Consumption_kWh missing - skipping daily time series.")

    # 2. Heatmap: hourly average energy by machine (if timestamp present)
    if "Timestamp" in filt.columns and "Machine_ID" in filt.columns:
        temp = filt.copy()
        temp["hour"] = temp["Timestamp"].dt.hour
        heat = temp.groupby(["Machine_ID", "hour"])["Energy_Consumption_kWh"].mean().reset_index()
        if not heat.empty:
            heat_pivot = heat.pivot(index="Machine_ID", columns="hour", values="Energy_Consumption_kWh").fillna(0)
            fig_heat = go.Figure(data=go.Heatmap(z=heat_pivot.values, x=heat_pivot.columns, y=heat_pivot.index,
                                                 colorscale="Viridis"))
            fig_heat.update_layout(title="Avg Energy by Machine (hourly)", xaxis_title="Hour of day", yaxis_title="Machine_ID")
            st.plotly_chart(fig_heat, use_container_width=True)
    # 3. Scatter: Cooling load vs energy with regression overlay (numpy)
    if "Cooling_Load_kWh" in filt.columns and "Energy_Consumption_kWh" in filt.columns:
        fig_sc = px.scatter(filt, x="Cooling_Load_kWh", y="Energy_Consumption_kWh", hover_data=["Machine_ID"], title="Cooling Load vs Energy Consumption")
        add_linear_regression_trace(fig_sc, filt["Cooling_Load_kWh"].values, filt["Energy_Consumption_kWh"].values, name="Lin fit", color="red")
        st.plotly_chart(fig_sc, use_container_width=True)

    # 4. Bar: Avg energy by shift
    if "Shift" in filt.columns and "Energy_Consumption_kWh" in filt.columns:
        bar_shift = filt.groupby("Shift")["Energy_Consumption_kWh"].mean().reset_index().sort_values("Energy_Consumption_kWh", ascending=False)
        fig_bar = px.bar(bar_shift, x="Shift", y="Energy_Consumption_kWh", title="Average Energy by Shift")
        st.plotly_chart(fig_bar, use_container_width=True)

    # 5. Box: Energy distribution by Fuel_Type
    if "Fuel_Type" in filt.columns and "Energy_Consumption_kWh" in filt.columns:
        fig_box = px.box(filt, x="Fuel_Type", y="Energy_Consumption_kWh", title="Energy Distribution by Fuel Type")
        st.plotly_chart(fig_box, use_container_width=True)

    # 6. Scatter matrix (pairwise) for selected numeric columns (small sample)
    numeric_for_matrix = [c for c in ["Energy_Consumption_kWh","Cooling_Load_kWh","Fuel_Consumption","Compressed_Air_CFM","Ambient_Temp_C"] if c in filt.columns]
    if len(numeric_for_matrix) >= 2:
        sample_df = filt[numeric_for_matrix].sample(n=min(500, len(filt)), random_state=42)
        fig_mat = px.scatter_matrix(sample_df, dimensions=numeric_for_matrix, title="Pairwise relationships (sample)")
        st.plotly_chart(fig_mat, use_container_width=True)

    # 7. Sankey-ish: Energy distribution across Machine_Type -> Fuel_Type (aggregate)
    if "Machine_Type" in filt.columns and "Fuel_Type" in filt.columns and "Energy_Consumption_kWh" in filt.columns:
        agg = filt.groupby(["Machine_Type","Fuel_Type"])["Energy_Consumption_kWh"].sum().reset_index()
        # construct simple sankey-like using treemap
        fig_tree = px.treemap(agg, path=["Machine_Type","Fuel_Type"], values="Energy_Consumption_kWh", title="Energy by Machine Type and Fuel")
        st.plotly_chart(fig_tree, use_container_width=True)

    # Additional charts can be added similarly (histograms, cumulative area, etc.)

    # ---------- ML Concepts (choose 4) ----------
    st.markdown("### Machine Learning & Modeling")
    st.markdown("We demonstrate 4 techniques: 1) RandomForest Regression (predict energy), 2) GradientBoosting Regression (predict), 3) IsolationForest (anomaly detection), 4) KMeans (operational clustering).")

    # Shared setup: features selection
    model_features = [c for c in ["Cooling_Load_kWh","Compressed_Air_CFM","Fuel_Consumption","Ambient_Temp_C","Output_Units","Energy_Intensity_kWh_per_Unit"] if c in filt.columns]
    # If Output_Units missing, energy intensity may be present; allow flexible features
    st.write(f"Using features: {model_features if model_features else 'No numeric model features found'}")

    # Minimum rows guard
    if len(filt) < 40 or len(model_features) < 1 or "Energy_Consumption_kWh" not in filt.columns:
        st.info("Not enough data or not enough features for ML. Need >=40 rows, at least 1 numeric feature and target Energy_Consumption_kWh.")
    else:
        X = filt[model_features].fillna(0)
        y = filt["Energy_Consumption_kWh"].fillna(0)

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # scale for some models
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # 1) RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=150, random_state=42)
        with st.spinner("Training RandomForest..."):
            rf.fit(X_train, y_train)
        preds_rf = rf.predict(X_test)
        rmse_rf = math.sqrt(mean_squared_error(y_test, preds_rf))
        r2_rf = r2_score(y_test, preds_rf)
        st.markdown(f"**RandomForest** — RMSE: {rmse_rf:.2f} | R²: {r2_rf:.3f}")

        # 2) GradientBoostingRegressor
        gbr = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42)
        with st.spinner("Training GradientBoosting..."):
            gbr.fit(X_train, y_train)
        preds_gbr = gbr.predict(X_test)
        rmse_gbr = math.sqrt(mean_squared_error(y_test, preds_gbr))
        r2_gbr = r2_score(y_test, preds_gbr)
        st.markdown(f"**GradientBoosting** — RMSE: {rmse_gbr:.2f} | R²: {r2_gbr:.3f}")

        # 3) IsolationForest for anomaly scoring (unsupervised)
        iso_features = X.fillna(0)
        iso = IsolationForest(contamination=0.02, random_state=42)
        with st.spinner("Fitting IsolationForest for anomalies..."):
            iso.fit(iso_features)
        iso_scores = iso.decision_function(iso_features)
        iso_preds = iso.predict(iso_features)  # -1 anomaly, 1 normal
        filt["_anomaly_score"] = iso_scores
        filt["_is_anomaly"] = np.where(iso_preds == -1, 1, 0)
        st.markdown(f"**Anomalies detected (isolation forest):** {int(filt['_is_anomaly'].sum())}")

        # 4) KMeans clustering to identify operational modes
        try:
            kmeans_feats = X.fillna(0)
            kmeans = KMeans(n_clusters=min(4, max(2, int(len(kmeans_feats)/50))), random_state=42)
            kmeans_labels = kmeans.fit_predict(kmeans_feats)
            filt["_kmeans_cluster"] = kmeans_labels
            st.markdown("**KMeans clustering** applied — clusters added to dataset as _kmeans_cluster.")
            # Visualize clusters on first two PCA components
            pca = PCA(n_components=2)
            comp = pca.fit_transform(scaler.fit_transform(kmeans_feats))
            pcadf = pd.DataFrame(comp, columns=["PC1","PC2"])
            pcadf["_cluster"] = kmeans_labels
            fig_pca = px.scatter(pcadf, x="PC1", y="PC2", color="_cluster", title="KMeans clusters (PCA projection)")
            st.plotly_chart(fig_pca, use_container_width=True)
        except Exception as e:
            st.info("KMeans step failed: " + str(e))

        # ---------- Prepare downloadable ML predictions (actual vs predicted + features) ----------
        out_rf = X_test.copy().reset_index(drop=True)
        out_rf["Actual_Energy_kWh"] = y_test.reset_index(drop=True)
        out_rf["Pred_RF_kWh"] = preds_rf
        out_rf["Pred_GBR_kWh"] = preds_gbr
        st.markdown("#### ML Predictions (sample)")
        st.dataframe(out_rf.head(20))
        download_df(out_rf, "energy_predictions_with_features.csv", "Download predictions (RF+GBR)")

    # ---------- Automated Insights ----------
    st.markdown("### Automated Insights")
    insights = []

    # Top machines by avg energy intensity
    if "Machine_ID" in filt.columns and "Energy_Intensity_kWh_per_Unit" in filt.columns:
        tmp = filt.groupby("Machine_ID")["Energy_Intensity_kWh_per_Unit"].mean().reset_index().sort_values(by="Energy_Intensity_kWh_per_Unit", ascending=False).head(10)
        for _, r in tmp.iterrows():
            insights.append({
                "Insight_Type": "High Energy Intensity",
                "Machine_ID": r["Machine_ID"],
                "Energy_Intensity_kWh_per_Unit": round(float(r["Energy_Intensity_kWh_per_Unit"]), 3)
            })

    # Machines with most anomalies
    if "_is_anomaly" in filt.columns and "Machine_ID" in filt.columns:
        anom_counts = filt.groupby("Machine_ID")["_is_anomaly"].sum().reset_index().rename(columns={"_is_anomaly":"anomaly_count"})
        anom_counts = anom_counts.sort_values("anomaly_count", ascending=False).head(10)
        for _, r in anom_counts.iterrows():
            if int(r["anomaly_count"]) > 0:
                insights.append({
                    "Insight_Type": "Anomaly Count",
                    "Machine_ID": r["Machine_ID"],
                    "Anomaly_Count": int(r["anomaly_count"])
                })

    # Peak load summary (per machine)
    if "Peak_Load_Flag" in filt.columns and "Machine_ID" in filt.columns:
        peaks = filt.groupby("Machine_ID")["Peak_Load_Flag"].sum().reset_index().sort_values(by="Peak_Load_Flag", ascending=False).head(10)
        for _, r in peaks.iterrows():
            if int(r["Peak_Load_Flag"]) > 0:
                insights.append({
                    "Insight_Type":"Peak Load Events",
                    "Machine_ID": r["Machine_ID"],
                    "Peak_Events": int(r["Peak_Load_Flag"])
                })

    # CO2 summary: top emitters
    if "CO2_Emissions_kg" in filt.columns and "Machine_ID" in filt.columns:
        co2 = filt.groupby("Machine_ID")["CO2_Emissions_kg"].sum().reset_index().sort_values(by="CO2_Emissions_kg", ascending=False).head(10)
        for _, r in co2.iterrows():
            insights.append({
                "Insight_Type":"CO2 Emissions",
                "Machine_ID": r["Machine_ID"],
                "Total_CO2_kg": round(float(r["CO2_Emissions_kg"]),2)
            })

    # Add simple text insights (left aligned card)
    st.markdown("<div class='card left-align'><b>Quick Observations</b><ul class='small'>", unsafe_allow_html=True)
    # compute some quick bullets
    avg_energy = round(float(filt["Energy_Consumption_kWh"].mean()),2) if "Energy_Consumption_kWh" in filt.columns else None
    avg_pf = round(float(filt["Power_Factor"].mean()),2) if "Power_Factor" in filt.columns else None
    if avg_energy:
        st.markdown(f"<li class='small'>Average energy consumption in filtered range: <b>{avg_energy} kWh</b></li>", unsafe_allow_html=True)
    if avg_pf:
        st.markdown(f"<li class='small'>Average power factor: <b>{avg_pf}</b> (consider PF correction where low)</li>", unsafe_allow_html=True)
    st.markdown("</ul></div>", unsafe_allow_html=True)

    # Prepare automated insights DataFrame and download
    if insights:
        ins_df = pd.DataFrame(insights)
        st.markdown("#### Insights Table")
        st.dataframe(ins_df, use_container_width=True)
        download_df(ins_df, "automated_insights_energy.csv", "Download insights")
    else:
        st.info("No automated insights generated for the current filters.")

    # Final note
    st.markdown("<div class='card left-align'>Done — download predictions, insights, or filtered data. Use clusters and anomaly lists to prioritize maintenance and scheduling changes.</div>", unsafe_allow_html=True)
