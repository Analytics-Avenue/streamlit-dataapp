import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import math
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(page_title="HospitalOps — Dashboard", layout="wide")

# Logo + title
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(
    f"""
<div style="display:flex;align-items:center;">
  <img src="{logo_url}" width="60" style="margin-right:10px;">
  <div style="line-height:1;">
    <div style="color:#064b86;font-size:28px;font-weight:bold;margin:0;padding:0;">Analytics Avenue &</div>
    <div style="color:#064b86;font-size:28px;font-weight:bold;margin:0;padding:0;">Advanced Analytics</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.title("HospitalOps Dashboard")

# -------------------------------------------------------------------
# CSS
# -------------------------------------------------------------------
st.markdown(
    """
<style>
*{color:#000 !important;}
.glow-card{
    background:#fff;padding:16px;border-radius:12px;
    box-shadow:0 10px 30px rgba(0,120,255,0.12),0 0 12px rgba(0,120,255,0.06) inset;
    border:1px solid rgba(0,120,255,0.18);margin-bottom:12px;transition:0.3s;
}
.glow-card:hover{box-shadow:0 0 25px #0078ff;transform:translateY(-3px);}
.metric-card {
    background: #ffffff;
    padding: 14px;
    border-radius: 10px;
    text-align: center;
    font-weight: 600;
    font-size: 16px;
    border: 1px solid rgba(0,0,0,0.08);
    box-shadow: 0px 4px 10px rgba(0,0,0,0.04);
    transition: 0.22s ease-in-out;
}
.metric-card:hover {
    box-shadow: 0 0 18px rgba(0, 120, 255, 0.32);
    transform: translateY(-3px) scale(1.02);
    border-color: rgba(0,120,255,0.25);
}
.kpi-value {
    display:block;
    font-size:20px;
    font-weight:700;
    margin-top:8px;
}
[data-testid="stSidebarNav"]{display:none;}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/healthcare/healthcare_syn.csv"
)


def remove_dup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = df.columns.str.strip()
    return df


def download_df(df: pd.DataFrame, filename: str):
    bio = BytesIO()
    bio.write(df.to_csv(index=False).encode("utf-8"))
    bio.seek(0)
    st.download_button("Download CSV", bio, file_name=filename, mime="text/csv")


def safe_onehot_kwargs():
    return {"handle_unknown": "ignore", "sparse_output": False}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\n", " ")
        .str.replace(" ", "_")
        .str.lower()
    )
    return df


def get_safe_col(df: pd.DataFrame, colnames):
    """
    Try multiple variants to find a matching column. Return first match or None.
    colnames: list of candidate names (snake_case)
    """
    cols = set(df.columns)
    for c in colnames:
        if c in cols:
            return c
    return None


def safe_top(df: pd.DataFrame, col: str, n: int = 3) -> pd.DataFrame:
    if col in df.columns:
        tmp = df.dropna(subset=[col])
        if tmp.empty:
            return pd.DataFrame()
        return tmp.nlargest(n, col)
    return pd.DataFrame()


# -------------------------------------------------------------------
# SESSION
# -------------------------------------------------------------------
st.session_state.setdefault("hospital_master", None)
st.session_state.setdefault("mapped_raw_columns", None)

# -------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------
tab1, tab2 = st.tabs(["Overview", "Application"])

# -------------------------------------------------------------------
# TAB 1: Overview
# -------------------------------------------------------------------
with tab1:
    st.markdown(
        '<div class="glow-card"><h2>About this Application</h2>'
        "<p>HospitalOps provides operational intelligence for hospital resource planning, equipment management, staffing, and patient flow analytics.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="glow-card"><h3>Purpose</h3>'
        "<p>Monitor KPIs, detect equipment shortages, benchmark hospital risk, and enable ML-driven operational forecasts.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            '<div class="glow-card"><h3>Capabilities</h3>'
            "<ul>"
            "<li>Dataset ingestion (default, sample, upload + mapping)</li>"
            "<li>Interactive filters and visualizations</li>"
            "<li>KPI monitoring</li>"
            "<li>ML regression predictions</li>"
            "<li>Automated insights</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            '<div class="glow-card"><h3>Business Impact</h3>'
            "<ul>"
            "<li>Faster triage and resource allocation</li>"
            "<li>Targeted equipment procurement</li>"
            "<li>Reduced operational bottlenecks</li>"
            "<li>Improved planning and budgeting</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="glow-card"><h3>Intended Users</h3>'
        "<p>Hospital administrators, operations teams, procurement, and finance.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### KPIs")

    # KPI placeholders (values will be computed in Application tab using filtered data)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown("<div class='metric-card'>High Patient Load<div class='kpi-value'>--</div></div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Avg Bed Availability<div class='kpi-value'>--</div></div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Equipment Shortage Score<div class='kpi-value'>--</div></div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Patients per Staff<div class='kpi-value'>--</div></div>", unsafe_allow_html=True)
    k5.markdown("<div class='metric-card'>Overall Risk Score<div class='kpi-value'>--</div></div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# TAB 2: Application
# -------------------------------------------------------------------
with tab2:
    st.header("Application")

    # -------------------------
    # Dataset selection modes
    # -------------------------
    st.subheader("Dataset Source")
    dataset_mode = st.radio(
        "Choose dataset mode",
        ["Default Dataset (full)", "Default Sample (20 rows)", "Upload + Manual Mapping"],
        horizontal=True,
    )

    df = None

    if dataset_mode == "Default Dataset (full)":
        try:
            df = pd.read_csv(DEFAULT_DATA_URL)
            df = remove_dup(df)
            st.session_state["hospital_master"] = df
            st.success("Default dataset loaded from URL.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Failed to load default dataset: {e}")

    elif dataset_mode == "Default Sample (20 rows)":
        try:
            df = pd.read_csv(DEFAULT_DATA_URL)
            df = remove_dup(df).head(20)
            st.session_state["hospital_master"] = df
            st.success("Sample dataset (20 rows) loaded.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Failed to load sample dataset: {e}")

    else:  # Upload + Mapping
        uploaded_file = st.file_uploader("Upload CSV for mapping", type=["csv"])
        if uploaded_file is not None:
            raw = pd.read_csv(uploaded_file)
            raw = remove_dup(raw)
            st.session_state["mapped_raw_columns"] = list(raw.columns)
            st.write("Uploaded file columns:", st.session_state["mapped_raw_columns"])

            st.markdown("#### Manual column mapping (map uploaded columns to expected schema)")

            expected_cols = [
                "Hospital Name",
                "Location",
                "Hospital Type",
                "Monthly Patients",
                "Staff Count",
                "Facilities Available",
                "Facilities Missing",
                "Ventilators Count",
                "ECG Machine Count",
                "X-Ray Machine Count",
                "MRI Scanner Count",
                "CT Scanner Count",
                "Dialysis Machine Count",
                "Infusion Pump Count",
                "Hospital Beds Count",
                "Patients Per Staff",
                "Equipment Shortage Score",
                "Facility Deficit Score",
                "Overall Risk Score",
            ]

            mapping = {}
            for col in expected_cols:
                mapping[col] = st.selectbox(f"Map to '{col}'", ["-- None --"] + st.session_state["mapped_raw_columns"], key=f"map_{col}")

            if st.button("Apply mapping"):
                mapped_df = pd.DataFrame()
                for col in expected_cols:
                    sel = mapping[col]
                    if sel != "-- None --":
                        mapped_df[col] = raw[sel]
                    else:
                        mapped_df[col] = np.nan
                st.session_state["hospital_master"] = mapped_df
                st.success("Mapping applied and dataset loaded.")
                st.dataframe(mapped_df.head())

    # stop if dataset not loaded
    df = st.session_state.get("hospital_master")
    if df is None:
        st.info("Load or map a dataset to continue.")
        st.stop()

    # normalize column names for internal usage
    df = df.copy()
    df.columns = df.columns.str.strip()
    df_internal = normalize_columns(df)

    # -------------------------
    # Filters (extended)
    # -------------------------
    st.subheader("Filters")
    with st.expander("Filter options", expanded=True):
        col_l, col_r = st.columns(2)
        with col_l:
            if "location" in df_internal.columns:
                locs = ["All"] + sorted(df_internal["location"].dropna().unique().tolist())
            else:
                locs = ["All"]
            selected_location = st.selectbox("Location", locs)

        with col_r:
            if "hospital_type" in df_internal.columns:
                types = ["All"] + sorted(df_internal["hospital_type"].dropna().unique().tolist())
            else:
                types = ["All"]
            selected_type = st.selectbox("Hospital Type", types)

        # numeric filters with safe defaults
        def numeric_slider_for(col, label):
            if col in df_internal.columns and pd.api.types.is_numeric_dtype(df_internal[col]):
                mn = float(np.nanmin(df_internal[col].values))
                mx = float(np.nanmax(df_internal[col].values))
                if np.isfinite(mn) and np.isfinite(mx) and mn != mx:
                    return st.slider(label, mn, mx, (mn, mx))
            return None

        eq_range = numeric_slider_for("equipment_shortage_score", "Equipment Shortage Score range")
        risk_range = numeric_slider_for("overall_risk_score", "Overall Risk Score range")
        pps_range = numeric_slider_for("patients_per_staff", "Patients Per Staff range")
        ventilator_range = numeric_slider_for("ventilators_count", "Ventilators Count range")

    # apply filters to working dataframe
    df_filtered = df_internal.copy()
    if selected_location != "All":
        if "location" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["location"] == selected_location]
    if selected_type != "All":
        if "hospital_type" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["hospital_type"] == selected_type]
    if eq_range is not None:
        df_filtered = df_filtered[
            (df_filtered["equipment_shortage_score"] >= eq_range[0]) & (df_filtered["equipment_shortage_score"] <= eq_range[1])
        ]
    if risk_range is not None:
        df_filtered = df_filtered[
            (df_filtered["overall_risk_score"] >= risk_range[0]) & (df_filtered["overall_risk_score"] <= risk_range[1])
        ]
    if pps_range is not None:
        df_filtered = df_filtered[
            (df_filtered["patients_per_staff"] >= pps_range[0]) & (df_filtered["patients_per_staff"] <= pps_range[1])
        ]
    if ventilator_range is not None:
        df_filtered = df_filtered[
            (df_filtered["ventilators_count"] >= ventilator_range[0]) & (df_filtered["ventilators_count"] <= ventilator_range[1])
        ]

    st.write("Filtered preview", df_filtered.head())

    # -------------------------
    # Compute KPI values dynamically from filtered data
    # -------------------------
    # prepare candidate column names we expect
    # mapping to internal names (snake_case)
    cname = lambda s: s.strip().lower().replace(" ", "_")
    col_map = {
        "hospital_name": get_safe_col(df_filtered, ["hospital_name", "hospital_name", "hospital"]),
        "monthly_patients": get_safe_col(df_filtered, ["monthly_patients", "monthly_patient", "monthly_patients_count"]),
        "hospital_beds_count": get_safe_col(df_filtered, ["hospital_beds_count", "hospital_beds"]),
        "equipment_shortage_score": get_safe_col(df_filtered, ["equipment_shortage_score"]),
        "patients_per_staff": get_safe_col(df_filtered, ["patients_per_staff"]),
        "facility_deficit_score": get_safe_col(df_filtered, ["facility_deficit_score"]),
        "overall_risk_score": get_safe_col(df_filtered, ["overall_risk_score"]),
        "ventilators_count": get_safe_col(df_filtered, ["ventilators_count"]),
    }

    # safe KPI computations
    def safe_mean(col):
        if col and col in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[col]):
            vals = df_filtered[col].dropna()
            if not vals.empty:
                return float(vals.mean())
        return None

    def safe_sum(col):
        if col and col in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[col]):
            vals = df_filtered[col].dropna()
            if not vals.empty:
                return float(vals.sum())
        return None

    # KPI: High Patient Load - show max monthly patients value or count of hospitals with monthly_patients > threshold
    k_high_patient = None
    if col_map["monthly_patients"] and col_map["monthly_patients"] in df_filtered.columns:
        mx = df_filtered[col_map["monthly_patients"]].dropna()
        if not mx.empty:
            k_high_patient = int(mx.max())

    # KPI: Avg Bed Availability -> mean beds count
    k_avg_beds = None
    if col_map["hospital_beds_count"] and col_map["hospital_beds_count"] in df_filtered.columns:
        k_avg_beds = int(safe_mean(col_map["hospital_beds_count"])) if safe_mean(col_map["hospital_beds_count"]) is not None else None

    # KPI: Equipment Shortage Score -> mean
    k_eq_short = safe_mean(col_map["equipment_shortage_score"])

    # KPI: Patients per Staff -> mean
    k_pps = safe_mean(col_map["patients_per_staff"])

    # KPI: Overall Risk Score -> mean
    k_risk = safe_mean(col_map["overall_risk_score"])

    # display KPI cards (values formatted)
    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    kc1_val = f"{k_high_patient:,}" if k_high_patient is not None else "--"
    kc2_val = f"{k_avg_beds:,}" if k_avg_beds is not None else "--"
    kc3_val = f"{k_eq_short:.2f}" if k_eq_short is not None else "--"
    kc4_val = f"{k_pps:.2f}" if k_pps is not None else "--"
    kc5_val = f"{k_risk:.2f}" if k_risk is not None else "--"

    kc1.markdown(f"<div class='metric-card'>High Patient Load<div class='kpi-value'>{kc1_val}</div></div>", unsafe_allow_html=True)
    kc2.markdown(f"<div class='metric-card'>Avg Bed Availability<div class='kpi-value'>{kc2_val}</div></div>", unsafe_allow_html=True)
    kc3.markdown(f"<div class='metric-card'>Equipment Shortage Score<div class='kpi-value'>{kc3_val}</div></div>", unsafe_allow_html=True)
    kc4.markdown(f"<div class='metric-card'>Patients per Staff<div class='kpi-value'>{kc4_val}</div></div>", unsafe_allow_html=True)
    kc5.markdown(f"<div class='metric-card'>Overall Risk Score<div class='kpi-value'>{kc5_val}</div></div>", unsafe_allow_html=True)

    # -------------------------
    # Charts & visualizations
    # -------------------------
    st.subheader("Charts & Visualizations")

    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_filtered.select_dtypes(exclude=[np.number]).columns.tolist()

    if numeric_cols:
        hist_col = st.selectbox("Histogram column", numeric_cols, index=0)
        st.plotly_chart(px.histogram(df_filtered, x=hist_col, nbins=30), use_container_width=True)
    else:
        st.info("No numeric columns available for charts.")

    if len(numeric_cols) >= 2:
        x_axis = st.selectbox("X axis", numeric_cols, key="x_axis")
        y_axis = st.selectbox("Y axis", numeric_cols, key="y_axis")
        color_col = categorical_cols[0] if categorical_cols else None
        st.plotly_chart(px.scatter(df_filtered, x=x_axis, y=y_axis, color=color_col), use_container_width=True)

    # -------------------------
    # ML Prediction (Regression)
    # -------------------------
    st.subheader("ML Prediction (Regression)")

    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns available to train ML models.")
    else:
        target_col = st.selectbox("Select target column for prediction", numeric_cols)
        feature_cols = [c for c in df_filtered.columns if c != target_col]

        X = df_filtered[feature_cols]
        y = df_filtered[target_col]

        test_size = st.slider("Test size", 0.1, 0.4, 0.2)

        if st.button("Train Regressor"):
            # separate numeric and categorical features
            num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
            cat_feats = X.select_dtypes(exclude=[np.number]).columns.tolist()

            transformers = []
            if num_feats:
                transformers.append(("num", StandardScaler(), num_feats))
            if cat_feats:
                transformers.append(("cat", OneHotEncoder(**safe_onehot_kwargs()), cat_feats))

            preproc = ColumnTransformer(transformers=transformers, remainder="drop")

            model = Pipeline([("preproc", preproc), ("model", RandomForestRegressor(n_estimators=150, random_state=42))])

            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
            model.fit(X_tr, y_tr)

            st.session_state["reg_pipe"] = model

            y_pred = model.predict(X_te)
            rmse = math.sqrt(np.mean((y_te - y_pred) ** 2))
            r2 = np.corrcoef(y_te, y_pred)[0, 1] ** 2 if len(y_te) > 1 else 0.0

            st.success(f"Model trained — RMSE: {rmse:.2f} | R²: {r2:.2f}")

            out_df = X_te.copy()
            out_df[f"Actual_{target_col}"] = y_te.values
            out_df[f"Predicted_{target_col}"] = y_pred
            st.dataframe(out_df)
            download_df(out_df, f"{target_col}_predictions.csv")

    # -------------------------
    # Automated Insights (dynamic with filters)
    # -------------------------
    st.subheader("Automated Insights")

    insights = []
    # operate on df_filtered (columns already normalized)
    df_i = df_filtered.copy()

    # ensure normalized columns set
    df_i.columns = df_i.columns.astype(str)

    # 1. Top hospitals by monthly patients
    top_patients = safe_top(df_i, "monthly_patients", n=5)
    if not top_patients.empty and "hospital_name" in df_i.columns:
        for _, row in top_patients.iterrows():
            insights.append(f"{row['hospital_name']} handles {int(row['monthly_patients'])} monthly patients (filtered view).")

    # 2. Highest equipment shortage
    top_equipment = safe_top(df_i, "equipment_shortage_score", n=5)
    if not top_equipment.empty and "hospital_name" in df_i.columns:
        for _, row in top_equipment.iterrows():
            insights.append(f"{row['hospital_name']} shows equipment shortage score {row['equipment_shortage_score']} (filtered view).")

    # 3. Highest patients per staff
    top_pps = safe_top(df_i, "patients_per_staff", n=5)
    if not top_pps.empty and "hospital_name" in df_i.columns:
        for _, row in top_pps.iterrows():
            insights.append(f"{row['hospital_name']} has {row['patients_per_staff']:.2f} patients per staff (filtered view).")

    # 4. Highest overall risk
    top_risk = safe_top(df_i, "overall_risk_score", n=5)
    if not top_risk.empty and "hospital_name" in df_i.columns:
        for _, row in top_risk.iterrows():
            insights.append(f"{row['hospital_name']} has overall risk score {row['overall_risk_score']} (filtered view).")

    # 5. Equipment totals summary (if ventilators_count present)
    if "ventilators_count" in df_i.columns and pd.api.types.is_numeric_dtype(df_i["ventilators_count"]):
        total_vents = int(df_i["ventilators_count"].sum())
        insights.append(f"Total ventilators in filtered view: {total_vents}")

    # Display insights
    if insights:
        ins_df = pd.DataFrame({"Insights": insights})
        st.dataframe(ins_df)
        download_df(ins_df, "automated_insights.csv")
    else:
        st.warning("No insights could be generated for the current filtered dataset. Make sure dataset contains expected columns (hospital_name, monthly_patients, equipment_shortage_score, patients_per_staff, overall_risk_score).")
