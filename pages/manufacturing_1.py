# app_predictive_maintenance.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score
import math
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# App config & header
# -------------------------
st.set_page_config(page_title="Predictive Maintenance App", layout="wide", page_icon="⚙️")

# Logo + company
LOGO_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:12px;">
        <img src="{LOGO_URL}" width="60" style="display:inline-block;">
        <div style="line-height:1;">
            <div style="color:#064b86; font-size:28px; font-weight:700; margin:0;">Analytics Avenue</div>
            <div style="color:#064b86; font-size:14px; font-weight:600; margin:0;">Advanced Analytics</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='margin-top:8px; margin-bottom:6px;'>Production Downtime & Machine Failures</h1>", unsafe_allow_html=True)
st.markdown("Predict failures, detect anomalies, and schedule maintenance using sensor & historical data.")

# -------------------------
# Utilities
# -------------------------
REQUIRED_COLS = [
    "Timestamp", "Machine_ID", "Machine_Type", "Temperature", "Vibration",
    "RPM", "Load", "Run_Hours", "Failure_Flag"
]

AUTO_MAPS = {
    "Timestamp": ["timestamp", "time", "ts", "date"],
    "Machine_ID": ["machine_id", "id", "machine"],
    "Machine_Type": ["machine_type", "type"],
    "Temperature": ["temperature", "temp", "t"],
    "Vibration": ["vibration", "vib", "vibration_level"],
    "RPM": ["rpm", "speed"],
    "Load": ["load", "utilization", "usage"],
    "Run_Hours": ["run_hours", "hours", "runtime"],
    "Failure_Flag": ["failure_flag", "failure", "failed", "breakdown", "failure"]
}

def auto_map_columns(df):
    rename = {}
    cols = [c.strip() for c in df.columns]
    for req, candidates in AUTO_MAPS.items():
        for c in cols:
            low = c.lower().strip()
            for cand in candidates:
                if cand.lower() == low or cand.lower() in low or low in cand.lower():
                    rename[c] = req
                    break
            if c in rename:
                break
    if rename:
        df = df.rename(columns=rename)
    return df

def ensure_datetime(df, col="Timestamp"):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def download_df(df, filename, key=None, label="Download CSV"):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv", key=key)

def safe_mean(x):
    try:
        return float(np.nanmean(x))
    except:
        return None

# -------------------------
# CSS
# -------------------------
st.markdown("""
<style>
.card {background: rgba(255,255,255,0.06); padding:14px; border-radius:10px; margin-bottom:12px; border:1px solid rgba(0,0,0,0.06);}
.metric-card {background: rgba(255,255,255,0.08); padding:14px; border-radius:10px; text-align:center; font-weight:700;}
.small {font-size:13px; color:#666}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Default dataset URL (replace if needed)
# -------------------------
DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/blob/main/datasets/manufacturing/machine_failure_data.csv"

@st.cache_data
def load_default_data(url=DEFAULT_URL):
    df = pd.read_csv(url)
    return df

# -------------------------
# Page navigation (tabs)
# -------------------------
tabs = st.tabs(["Overview", "Application"])

# -------------------------
# Overview tab
# -------------------------
with tabs[0]:
    st.markdown("### Overview")
    st.markdown("""
    <div class='card'>
        <b>Purpose</b>: Predict machine failures and reduce unplanned downtime by combining sensor signals (vibration, temperature, RPM, load) with historical usage and failure records.
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Capabilities")
        st.markdown("""
        <div class='card'>
        • Real-time anomaly detection using sensor baselines.<br>
        • Failure probability prediction using supervised models.<br>
        • Health scoring and per-machine trend visuals.<br>
        • Downloadable ML outputs and feature snapshots.<br>
        • Automated insights and export for maintenance scheduling.
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("#### Business impact")
        st.markdown("""
        <div class='card'>
        • Lower unplanned downtime & production losses.<br>
        • Optimised maintenance scheduling (condition-based).<br>
        • Improved asset lifetime and reduced rush spare orders.<br>
        • Data-driven CAPEX decision support for replacements.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### KPIs")
    k1,k2,k3,k4,k5 = st.columns([1,1,1,1,1])
    k1.markdown("<div class='metric-card'>Avg Downtime / Month</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Failure Rate</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Mean Time To Repair (MTTR)</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Mean Time Between Failures (MTBF)</div>", unsafe_allow_html=True)
    k5.markdown("<div class='metric-card'>% Critical Machines</div>", unsafe_allow_html=True)

    st.markdown("#### Who should use this app & How")
    st.markdown("""
    <div class='card'>
    <b>Who</b>: Maintenance leads, plant managers, reliability engineers, data teams.<br><br>
    <b>How</b>: 1) Upload machine sensor file (or use default). 2) Filter by machine / time / type. 3) Review top-risk machines and download predictions. 4) Schedule maintenance based on priority list exported from the app.
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# Application tab
# -------------------------
with tabs[1]:
    st.header("Application")

    st.markdown("### Step 1 — Load dataset")
    st.markdown("Choose one: default dataset (GitHub), upload file, or upload + map columns.")
    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV (with sample)", "Upload CSV + Column mapping"], horizontal=True)
    df = None

    if mode == "Default dataset":
        try:
            df = load_default_data()
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("Default dataset loaded.")
            st.dataframe(df.head())
        except Exception as e:
            st.error("Failed to load default dataset. Check DEFAULT_URL. Error: " + str(e))
            st.stop()

    elif mode == "Upload CSV (with sample)":
        # offer sample from default url
        try:
            sample = load_default_data().head(100)
            download_df(sample, "sample_machine_data.csv", key="sample_download", label="Download Sample CSV")
        except:
            st.info("No sample available from the default source.")
        uploaded = st.file_uploader("Upload CSV file", type=["csv"], key="upload1")
        if uploaded:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("File uploaded and auto-mapped (best effort).")
            st.dataframe(df.head())
        else:
            st.stop()

    else:  # Upload + mapping
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"], key="upload2")
        if uploaded:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head())
            st.markdown("Map your columns to required fields (map at least those you have):")
            mapping = {}
            for req in REQUIRED_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + list(raw.columns), key=f"map_{req}")
            if st.button("Apply mapping", key="apply_map"):
                missing = [k for k, v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Please map required columns (or choose appropriate columns). Missing: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v: k for k, v in mapping.items()})
                    st.success("Mapping applied.")
                    st.dataframe(df.head())
        else:
            st.stop()

    if df is None:
        st.stop()

    # Standardize column names & ensure timestamp
    df = auto_map_columns(df)
    df = df[[c for c in REQUIRED_COLS if c in df.columns] + [c for c in df.columns if c not in REQUIRED_COLS]]
    df = ensure_datetime(df, "Timestamp")

    # Convert numeric columns safely
    numeric_cols = ["Temperature", "Vibration", "RPM", "Load", "Run_Hours"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0)

    if "Failure_Flag" in df.columns:
        df["Failure_Flag"] = pd.to_numeric(df["Failure_Flag"], errors="coerce").fillna(0).astype(int)
    else:
        df["Failure_Flag"] = 0

    # -------------------------
    # Step 2 — Filters & preview
    # -------------------------
    st.markdown("### Step 2 — Filters & preview")
    c1, c2, c3 = st.columns([2, 2, 1])
    machines = sorted(df["Machine_ID"].dropna().unique()) if "Machine_ID" in df.columns else []
    mtypes = sorted(df["Machine_Type"].dropna().unique()) if "Machine_Type" in df.columns else []
    times = (df["Timestamp"].min(), df["Timestamp"].max()) if "Timestamp" in df.columns else (None, None)

    with c1:
        sel_machines = st.multiselect("Machine_ID", options=machines, default=machines[:5] if machines else [])
    with c2:
        sel_types = st.multiselect("Machine_Type", options=mtypes, default=mtypes[:3] if mtypes else [])
    with c3:
        if times[0] is not None and not pd.isna(times[0]):
            ds = st.date_input("Date range", value=(times[0].date(), times[1].date()))
        else:
            ds = None

    filt = df.copy()
    if sel_machines:
        filt = filt[filt["Machine_ID"].isin(sel_machines)]
    if sel_types:
        filt = filt[filt["Machine_Type"].isin(sel_types)]
    if ds and "Timestamp" in filt.columns:
        start, end = pd.to_datetime(ds[0]), pd.to_datetime(ds[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filt = filt[(filt["Timestamp"] >= start) & (filt["Timestamp"] <= end)]

    st.markdown("Preview (first 10 rows)")
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(100), "filtered_machine_preview.csv", key="preview")

    # -------------------------
    # Health metrics & summary charts
    # -------------------------
    st.markdown("### Machine Health Summary")
    col1, col2, col3, col4 = st.columns(4)
    avg_temp = safe_mean(filt["Temperature"]) if "Temperature" in filt.columns else None
    avg_vib = safe_mean(filt["Vibration"]) if "Vibration" in filt.columns else None
    failure_rate = filt["Failure_Flag"].mean() if "Failure_Flag" in filt.columns else 0
    col1.metric("Avg Temperature", f"{avg_temp:.2f}" if avg_temp is not None else "N/A")
    col2.metric("Avg Vibration", f"{avg_vib:.2f}" if avg_vib is not None else "N/A")
    col3.metric("Failure Rate", f"{failure_rate:.2%}")
    col4.metric("Total Machines Shown", len(filt["Machine_ID"].unique()) if "Machine_ID" in filt.columns else 0)

    # Time-series: temperature & vibration trends
    st.markdown("#### Time series — Temperature & Vibration (aggregated)")
    if "Timestamp" in filt.columns:
        ts = filt.set_index("Timestamp").resample("D").agg({"Temperature": "mean", "Vibration": "mean", "Failure_Flag": "sum"}).reset_index()
        fig_ts = go.Figure()
        if "Temperature" in ts.columns:
            fig_ts.add_trace(go.Scatter(x=ts["Timestamp"], y=ts["Temperature"], name="Avg Temp"))
        if "Vibration" in ts.columns:
            fig_ts.add_trace(go.Scatter(x=ts["Timestamp"], y=ts["Vibration"], name="Avg Vibration", yaxis="y2"))
        fig_ts.update_layout(
            yaxis=dict(title="Temperature"),
            yaxis2=dict(title="Vibration", overlaying="y", side="right"),
            xaxis_title="Date",
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    # Distribution charts
    st.markdown("#### Distributions")
    d1, d2 = st.columns(2)
    if "Temperature" in filt.columns:
        fig1 = px.histogram(filt, x="Temperature", nbins=40, title="Temperature distribution")
        d1.plotly_chart(fig1, use_container_width=True)
    if "Vibration" in filt.columns:
        fig2 = px.histogram(filt, x="Vibration", nbins=40, title="Vibration distribution")
        d2.plotly_chart(fig2, use_container_width=True)

    # -------------------------
    # Anomaly detection (IsolationForest)
    # -------------------------
    st.markdown("### Anomaly Detection (unsupervised)")
    if set(["Temperature", "Vibration", "RPM", "Load"]).issubset(set(filt.columns)):
        iso_feats = filt[["Temperature", "Vibration", "RPM", "Load"]].fillna(0)
        iso = IsolationForest(contamination=0.02, random_state=42)
        iso_preds = iso.fit_predict(iso_feats)
        filt["_anomaly_score"] = iso.decision_function(iso_feats)
        filt["_is_anomaly"] = np.where(iso_preds == -1, 1, 0)
        st.markdown(f"Detected anomalies: {int(filt['_is_anomaly'].sum())}")
        st.dataframe(filt.loc[filt["_is_anomaly"] == 1, ["Timestamp", "Machine_ID", "Machine_Type", "Temperature", "Vibration", "RPM", "Load", "_anomaly_score"]].head(200))
        download_df(filt.loc[filt["_is_anomaly"] == 1, ["Timestamp", "Machine_ID", "Machine_Type", "Temperature", "Vibration", "RPM", "Load", "_anomaly_score"]], "anomalies.csv", key="anoms")
    else:
        st.info("Need Temperature, Vibration, RPM and Load columns for anomaly detection.")

    # -------------------------
    # ML: Failure prediction (supervised) + download
    # -------------------------
    st.markdown("### ML: Failure Probability (RandomForest Classifier)")
    model_feats = ["Temperature", "Vibration", "RPM", "Load", "Run_Hours", "Machine_Type"]
    model_feats = [c for c in model_feats if c in filt.columns]

    if ("Failure_Flag" in filt.columns) and len(filt) >= 50 and len(model_feats) >= 2:
        X = filt[model_feats].copy()
        y = filt["Failure_Flag"].astype(int).reset_index(drop=True)

        # preprocessing
        cat_cols = [c for c in X.columns if X[c].dtype == "object"]
        num_cols = [c for c in X.columns if c not in cat_cols]
        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
            ("num", StandardScaler(), num_cols)
        ], remainder="drop")

        X_t = preprocessor.fit_transform(X)
        # split with indices so we can map back to original rows
        indices = np.arange(X_t.shape[0])
        strat = y if y.nunique() > 1 else None
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=strat)
        X_train = X_t[train_idx]
        X_test = X_t[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        with st.spinner("Training RandomForest classifier..."):
            clf.fit(X_train, y_train)

        preds_prob = clf.predict_proba(X_test)[:, 1]
        preds_label = (preds_prob >= 0.5).astype(int)

        # Metrics
        try:
            auc = roc_auc_score(y_test, preds_prob)
        except:
            auc = None
        acc = accuracy_score(y_test, preds_label)
        st.write(f"Classifier — Accuracy: {acc:.3f}" + (f", ROC AUC: {auc:.3f}" if auc is not None else ""))

        # Build downloadable df aligned with test_idx rows
        X_test_original = filt.reset_index(drop=True).loc[test_idx, model_feats].reset_index(drop=True)
        out_df = X_test_original.copy()
        out_df["Actual_Failure"] = y_test.reset_index(drop=True)
        out_df["Predicted_Prob"] = preds_prob
        out_df["Predicted_Label"] = preds_label
        st.dataframe(out_df.head(20))
        download_df(out_df, "failure_predictions_with_features.csv", key="fail_pred")
    else:
        st.info("Not enough labeled data or features for supervised failure prediction. Need Failure_Flag column, >=50 rows and at least 2 features.")

    # -------------------------
    # ML: Regression (severity/time-to-failure proxy)
    # -------------------------
    st.markdown("### ML: Regression (proxy for severity / time-to-failure)")
    if "Run_Hours" in filt.columns and "Failure_Flag" in filt.columns and len(filt) >= 50:
        reg_df = filt.copy().reset_index(drop=True)
        reg_df["target_severity"] = reg_df["Run_Hours"] * reg_df["Failure_Flag"]
        if reg_df["target_severity"].sum() > 0:
            feat_cols_reg = [c for c in ["Temperature", "Vibration", "RPM", "Load", "Machine_Type"] if c in reg_df.columns]
            Xr = reg_df[feat_cols_reg].copy()
            yr = reg_df["target_severity"].astype(float)
            cat_cols = [c for c in Xr.columns if Xr[c].dtype == "object"]
            num_cols = [c for c in Xr.columns if c not in cat_cols]
            preprocessor_r = ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
                ("num", StandardScaler(), num_cols)
            ], remainder="drop")
            try:
                Xr_t = preprocessor_r.fit_transform(Xr)
                indices_r = np.arange(Xr_t.shape[0])
                train_idx_r, test_idx_r = train_test_split(indices_r, test_size=0.2, random_state=42)
                X_train_r = Xr_t[train_idx_r]
                X_test_r = Xr_t[test_idx_r]
                y_train_r = yr.iloc[train_idx_r]
                y_test_r = yr.iloc[test_idx_r]

                rfr = RandomForestRegressor(n_estimators=150, random_state=42)
                with st.spinner("Training regression model..."):
                    rfr.fit(X_train_r, y_train_r)
                preds_r = rfr.predict(X_test_r)
                st.write(f"Regression — RMSE: {math.sqrt(mean_squared_error(y_test_r, preds_r)):.2f}, R²: {r2_score(y_test_r, preds_r):.3f}")

                out_r = reg_df.loc[test_idx_r, feat_cols_reg].reset_index(drop=True)
                out_r["Actual_Severity"] = y_test_r.reset_index(drop=True)
                out_r["Pred_Severity"] = preds_r
                st.dataframe(out_r.head(20))
                download_df(out_r, "severity_predictions_with_features.csv", key="reg_pred")
            except Exception as e:
                st.info("Regression model training failed: " + str(e))
        else:
            st.info("No failure-severity signal in data to train regression.")
    else:
        st.info("Not enough columns (Run_Hours/Failure_Flag) or rows to train regression model.")

    # -------------------------
    # Automated Insights (table + download)
    # -------------------------
    st.markdown("### Automated Insights (table + download)")
    insights_rows = []

    if "Machine_ID" in filt.columns and "Failure_Flag" in filt.columns:
        grp = filt.groupby("Machine_ID").agg(
            total_records=("Machine_ID", "count"),
            failures=("Failure_Flag", "sum")
        ).reset_index()
        if "Temperature" in filt.columns:
            grp["avg_temp"] = filt.groupby("Machine_ID")["Temperature"].mean().values
        if "Vibration" in filt.columns:
            grp["avg_vib"] = filt.groupby("Machine_ID")["Vibration"].mean().values
        grp["failure_rate"] = np.where(grp["total_records"] > 0, grp["failures"] / grp["total_records"], 0)
        top_risk = grp.sort_values("failure_rate", ascending=False).head(20)
        for _, r in top_risk.iterrows():
            insights_rows.append({
                "Insight_Type": "Top-risk Machine",
                "Machine_ID": r["Machine_ID"],
                "Failure_Rate": round(r["failure_rate"], 4),
                "Failures": int(r["failures"]),
                "Total_Records": int(r["total_records"]),
                "Avg_Temp": round(r.get("avg_temp", np.nan), 2) if not pd.isna(r.get("avg_temp", np.nan)) else None,
                "Avg_Vib": round(r.get("avg_vib", np.nan), 2) if not pd.isna(r.get("avg_vib", np.nan)) else None
            })

    if "_is_anomaly" in filt.columns:
        anom_counts = filt.groupby("Machine_ID")["_is_anomaly"].sum().reset_index().rename(columns={"_is_anomaly": "anomaly_count"})
        anom_counts = anom_counts.sort_values("anomaly_count", ascending=False).head(20)
        for _, r in anom_counts.iterrows():
            insights_rows.append({
                "Insight_Type": "Anomaly Count",
                "Machine_ID": r["Machine_ID"],
                "Anomaly_Count": int(r["anomaly_count"])
            })

    if "Timestamp" in filt.columns and "Failure_Flag" in filt.columns:
        monthly = filt.set_index("Timestamp").resample("M").agg({"Failure_Flag": "sum", "Machine_ID": "nunique"}).reset_index()
        monthly["Failures_per_Month"] = monthly["Failure_Flag"]
        for _, r in monthly.iterrows():
            insights_rows.append({
                "Insight_Type": "Monthly Failures",
                "Month": r["Timestamp"].strftime("%Y-%m"),
                "Failures": int(r["Failures_per_Month"]),
                "Active_Machines": int(r["Machine_ID"])
            })

    insights_df = pd.DataFrame(insights_rows).fillna("")
    if insights_df.empty:
        st.info("No automated insights could be generated on this filtered dataset.")
    else:
        st.dataframe(insights_df.head(200), use_container_width=True)
        download_df(insights_df, "automated_insights_machine.csv", key="insights")

    st.markdown("### Done — download predictions or insights and schedule maintenance before someone yells at you.")
