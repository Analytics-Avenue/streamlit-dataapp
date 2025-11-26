import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import math
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
# PAGE CONFIG & HIDE SIDEBAR
# ==========================================================
st.set_page_config(page_title="HospitalOps – Operations & Risk Lab", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebarNav"] {display:none;}
section[data-testid="stSidebar"] {display:none;}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================
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

# ==========================================================
# GLOBAL CSS – MARKETING LAB STANDARD
# ==========================================================
st.markdown("""
<style>
* { font-family: 'Inter', sans-serif; }

/* GLOBAL TEXT */
body, [class*="css"] { color:#000 !important; font-size:17px; }

/* MAIN HEADER */
.big-header {
    font-size: 34px !important;
    font-weight: 700 !important;
    color:#000 !important;
    margin-bottom:10px;
}

/* SECTION TITLE */
.section-title {
    font-size: 24px !important;
    font-weight: 600 !important;
    margin-top:26px;
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
    padding:18px;
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

/* KPI VALUE INSIDE METRIC CARD */
.kpi-value {
    display:block;
    font-size:20px;
    font-weight:700;
    margin-top:6px;
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

/* GENERIC METRIC CARD USED IN OVERVIEW */
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

/* TABLE OVERRIDE (INDEX-SAFE) */
.required-table th {
    background:#ffffff !important;
    color:#000 !important;
    font-size:17px !important;
    border-bottom:2px solid #000 !important;
}
.required-table td {
    color:#000 !important;
    font-size:15.5px !important;
    padding:8px !important;
    border-bottom:1px solid #dcdcdc !important;
}
.required-table tr:hover td {
    background:#f8f8f8 !important;
}

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
@keyframes fadeIn {
    from {opacity:0; transform:translateY(10px);}
    to {opacity:1; transform:translateY(0);}
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-header'>HospitalOps – Operations & Risk Lab</div>", unsafe_allow_html=True)
st.markdown("Unified view of hospital capacity, equipment, staffing and operational risk, with ML-backed risk prediction.")

# ==========================================================
# CONSTANTS / REQUIRED COLUMNS
# ==========================================================
DEFAULT_DATA_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/healthcare/healthcare_syn.csv"

REQUIRED_HOSP_COLS = [
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
    "Overall Risk Score"
]

# Data dictionary for tab 2
REQUIRED_DICT = {
    "Hospital Name": "Name / identifier of the hospital or medical center.",
    "Location": "City / region / geography where the hospital is located.",
    "Hospital Type": "Category of hospital (e.g., Government, Private, Multi-speciality, Trust).",
    "Monthly Patients": "Approximate number of patients handled per month.",
    "Staff Count": "Total doctors, nurses and support staff headcount.",
    "Facilities Available": "Key critical facilities currently available (ICU, OT, NICU, etc.).",
    "Facilities Missing": "Key critical facilities missing or not yet installed.",
    "Ventilators Count": "Number of ventilators installed and usable.",
    "ECG Machine Count": "Number of ECG machines available.",
    "X-Ray Machine Count": "Number of X-Ray units available.",
    "MRI Scanner Count": "Number of MRI scanners available.",
    "CT Scanner Count": "Number of CT scanners available.",
    "Dialysis Machine Count": "Number of dialysis machines installed.",
    "Infusion Pump Count": "Number of infusion pumps in working condition.",
    "Hospital Beds Count": "Total inpatient beds configured in the hospital.",
    "Patients Per Staff": "Ratio of patients to total staff (load per staff).",
    "Equipment Shortage Score": "Composite score indicating equipment gap vs demand (higher = worse).",
    "Facility Deficit Score": "Score reflecting missing facilities severity (higher = more deficit).",
    "Overall Risk Score": "Overall operational risk score combining load, equipment and facility gaps."
}

# ==========================================================
# HELPERS
# ==========================================================
def remove_dup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = df.columns.str.strip()
    return df

def download_df(df: pd.DataFrame, filename: str, label: str = "Download CSV"):
    bio = BytesIO()
    bio.write(df.to_csv(index=False).encode("utf-8"))
    bio.seek(0)
    st.download_button(label, bio, file_name=filename, mime="text/csv")

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

def get_safe_col(df: pd.DataFrame, candidates):
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None

def safe_mean(df, col):
    if col and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        vals = df[col].dropna()
        if not vals.empty:
            return float(vals.mean())
    return None

def safe_max(df, col):
    if col and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        vals = df[col].dropna()
        if not vals.empty:
            return float(vals.max())
    return None

def safe_onehot_kwargs():
    return {"handle_unknown": "ignore", "sparse_output": False}

# ==========================================================
# SESSION
# ==========================================================
st.session_state.setdefault("hospital_master", None)
st.session_state.setdefault("reg_pipe", None)

# ==========================================================
# TABS
# ==========================================================
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# ==========================================================
# TAB 1 – OVERVIEW
# ==========================================================
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <b>HospitalOps</b> brings together hospital capacity, equipment inventory, staffing and risk metrics
    into a single analytics workspace. It helps operations teams see where load is spiking, where machines are
    missing, and which hospitals are drifting into high-risk territory.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Purpose</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Track <b>monthly patient load</b> across locations<br>
        • Benchmark <b>beds and equipment</b> vs demand<br>
        • Quantify <b>equipment & facility deficits</b><br>
        • Compute an <b>Overall Risk Score</b> per hospital<br>
        • Enable <b>ML-based risk forecasting</b> from structural variables
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Prioritized <b>capex allocation</b> to high-risk hospitals<br>
        • Faster identification of <b>equipment shortages</b><br>
        • Smarter staffing & workload balancing<br>
        • Evidence-backed <b>budgeting & planning</b><br>
        • Clear story for <b>boards, donors, govt & regulators</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Key KPIs Tracked</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown("<div class='kpi'>High Patient Load</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Avg Bed Availability</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Equipment Shortage Score</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Patients per Staff</div>", unsafe_allow_html=True)
    k5.markdown("<div class='kpi'>Overall Risk Score</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Who Uses This</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    Hospital COOs, operations teams, state health departments, PMUs, NGOs and analytics teams who want a structured
    view of <b>hospital risk, capacity and equipment readiness</b>.
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# TAB 2 – IMPORTANT ATTRIBUTES (DATA DICTIONARY)
# ==========================================================
with tab2:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    dict_df = pd.DataFrame(
        [{"Attribute": k, "Description": v} for k, v in REQUIRED_DICT.items()]
    )

    st.markdown("""
        <style>
            .required-table th {
                background:#ffffff !important;
                color:#000 !important;
                font-size:17px !important;
                border-bottom:2px solid #000 !important;
            }
            .required-table td {
                color:#000 !important;
                font-size:15.5px !important;
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
            "Facility Deficit Score"
        ]
        for v in independent_vars:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Dependent Variable</div>', unsafe_allow_html=True)
        dependent_vars = ["Overall Risk Score"]
        for v in dependent_vars:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# ==========================================================
# TAB 3 – APPLICATION
# ==========================================================
with tab3:
    st.markdown('<div class="section-title">Step 1: Load dataset</div>', unsafe_allow_html=True)

    dataset_mode = st.radio(
        "Choose dataset mode",
        ["Default Dataset (full)",  "Upload + Manual Mapping"],
        horizontal=True,
    )

    df = None

    if dataset_mode == "Default Dataset (full)":
        try:
            df = pd.read_csv(DEFAULT_DATA_URL)
            df = remove_dup(df)
            st.session_state["hospital_master"] = df
            st.success("Default dataset loaded from URL.")
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load default dataset: {e}")

    else:  # Upload + Manual Mapping
        uploaded_file = st.file_uploader("Upload CSV for mapping", type=["csv"])
        if uploaded_file is not None:
            raw = pd.read_csv(uploaded_file)
            raw = remove_dup(raw)
            raw_cols = list(raw.columns)
            st.write("Uploaded file columns:", raw_cols)

            st.markdown("#### Manual column mapping (map uploaded columns to expected schema)")
            mapping = {}
            for col in REQUIRED_HOSP_COLS:
                mapping[col] = st.selectbox(
                    f"Map to '{col}'",
                    ["-- None --"] + raw_cols,
                    key=f"map_{col}"
                )

            if st.button("Apply mapping"):
                mapped_df = pd.DataFrame()
                for col in REQUIRED_HOSP_COLS:
                    sel = mapping[col]
                    if sel != "-- None --":
                        mapped_df[col] = raw[sel]
                    else:
                        mapped_df[col] = np.nan
                st.session_state["hospital_master"] = mapped_df
                st.success("Mapping applied and dataset loaded.")
                st.dataframe(mapped_df.head(10), use_container_width=True)

    df = st.session_state.get("hospital_master")

    if df is None:
        st.info("Load or map a dataset to continue.")
        st.stop()

    # Normalize for internal use
    df = df.copy()
    df.columns = df.columns.str.strip()
    df_internal = normalize_columns(df)

    st.markdown('<div class="section-title">Step 2: Filters & Preview</div>', unsafe_allow_html=True)

    # Filters
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

    df_filtered = df_internal.copy()
    if selected_location != "All" and "location" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["location"] == selected_location]
    if selected_type != "All" and "hospital_type" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["hospital_type"] == selected_type]

    st.markdown(f"""
    <div style='padding:8px; background:#eef4ff; border-radius:8px; font-weight:600; margin-bottom:8px;'>
    Filtered Rows: {len(df_filtered)} of {len(df_internal)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Data Preview (first 15 rows)")
    st.dataframe(df_filtered.head(15), use_container_width=True)
    download_df(df_filtered.head(500), "hospitalops_filtered_preview.csv", "Download filtered preview (first 500 rows)")

    if df_filtered.empty:
        st.warning("Filtered dataset is empty. Adjust filters above.")
        st.stop()

    # KPI Mapping
    col_map = {
        "hospital_name": get_safe_col(df_filtered, ["hospital_name", "hospital"]),
        "monthly_patients": get_safe_col(df_filtered, ["monthly_patients", "monthly_patient", "monthly_patients_count"]),
        "hospital_beds_count": get_safe_col(df_filtered, ["hospital_beds_count", "hospital_beds"]),
        "equipment_shortage_score": get_safe_col(df_filtered, ["equipment_shortage_score"]),
        "patients_per_staff": get_safe_col(df_filtered, ["patients_per_staff"]),
        "overall_risk_score": get_safe_col(df_filtered, ["overall_risk_score"]),
    }

    # KPI values
    k_high_patient = safe_max(df_filtered, col_map["monthly_patients"])
    k_avg_beds = safe_mean(df_filtered, col_map["hospital_beds_count"])
    k_eq_short = safe_mean(df_filtered, col_map["equipment_shortage_score"])
    k_pps = safe_mean(df_filtered, col_map["patients_per_staff"])
    k_risk = safe_mean(df_filtered, col_map["overall_risk_score"])

    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    kc1_val = f"{int(k_high_patient):,}" if k_high_patient is not None else "--"
    kc2_val = f"{int(k_avg_beds):,}" if k_avg_beds is not None else "--"
    kc3_val = f"{k_eq_short:.2f}" if k_eq_short is not None else "--"
    kc4_val = f"{k_pps:.2f}" if k_pps is not None else "--"
    kc5_val = f"{k_risk:.2f}" if k_risk is not None else "--"

    kc1.markdown(f"<div class='kpi'>High Patient Load<span class='kpi-value'>{kc1_val}</span></div>", unsafe_allow_html=True)
    kc2.markdown(f"<div class='kpi'>Avg Bed Availability<span class='kpi-value'>{kc2_val}</span></div>", unsafe_allow_html=True)
    kc3.markdown(f"<div class='kpi'>Equipment Shortage Score<span class='kpi-value'>{kc3_val}</span></div>", unsafe_allow_html=True)
    kc4.markdown(f"<div class='kpi'>Patients per Staff<span class='kpi-value'>{kc4_val}</span></div>", unsafe_allow_html=True)
    kc5.markdown(f"<div class='kpi'>Overall Risk Score<span class='kpi-value'>{kc5_val}</span></div>", unsafe_allow_html=True)

    # ==========================================================
    # CHARTS
    # ==========================================================
    st.markdown('<div class="section-title">Charts & Visualizations</div>', unsafe_allow_html=True)

    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_filtered.select_dtypes(exclude=[np.number]).columns.tolist()

    if numeric_cols:
        hist_col = st.selectbox("Histogram column", numeric_cols, index=0)
        fig_hist = px.histogram(df_filtered, x=hist_col, nbins=30, template="plotly_white")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No numeric columns available for histogram.")

    if len(numeric_cols) >= 2:
        col_x, col_y = st.columns(2)
        with col_x:
            x_axis = st.selectbox("X axis", numeric_cols, key="x_axis_hosp")
        with col_y:
            y_axis = st.selectbox("Y axis", numeric_cols, key="y_axis_hosp")
        color_col = categorical_cols[0] if categorical_cols else None
        fig_scatter = px.scatter(df_filtered, x=x_axis, y=y_axis, color=color_col, template="plotly_white")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ==========================================================
    # ML – REGRESSION (RISK OR ANY NUMERIC TARGET)
    # ==========================================================
    st.markdown('<div class="section-title">ML Prediction (Regression)</div>', unsafe_allow_html=True)

    if not numeric_cols:
        st.info("No numeric columns available to train ML models.")
    else:
        # Recommended target default = overall_risk_score if present
        default_target = "overall_risk_score" if "overall_risk_score" in numeric_cols else numeric_cols[0]
        target_col = st.selectbox(
            "Select target column for prediction",
            numeric_cols,
            index=numeric_cols.index(default_target) if default_target in numeric_cols else 0
        )
        feature_cols = [c for c in df_filtered.columns if c != target_col]

        X = df_filtered[feature_cols].copy()
        y = df_filtered[target_col].copy()

        test_size = st.slider("Test size (validation split)", 0.1, 0.4, 0.2)

        if st.button("Train Regressor"):
            # split numeric / categorical
            num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
            cat_feats = X.select_dtypes(exclude=[np.number]).columns.tolist()

            transformers = []
            if num_feats:
                transformers.append(("num", StandardScaler(), num_feats))
            if cat_feats:
                transformers.append(("cat", OneHotEncoder(**safe_onehot_kwargs()), cat_feats))

            preproc = ColumnTransformer(transformers=transformers, remainder="drop")

            model = Pipeline([
                ("preproc", preproc),
                ("model", RandomForestRegressor(n_estimators=150, random_state=42))
            ])

            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
            model.fit(X_tr, y_tr)

            st.session_state["reg_pipe"] = model

            y_pred = model.predict(X_te)
            rmse = math.sqrt(mean_squared_error(y_te, y_pred))
            r2 = r2_score(y_te, y_pred) if len(y_te) > 1 else 0.0

            st.success(f"Model trained — RMSE: {rmse:.2f} | R²: {r2:.2f}")

            out_df = X_te.copy()
            out_df[f"Actual_{target_col}"] = y_te.values
            out_df[f"Predicted_{target_col}"] = y_pred
            st.dataframe(out_df.head(20), use_container_width=True)
            download_df(out_df, f"{target_col}_predictions.csv", f"Download {target_col} prediction sample")

    # ==========================================================
    # AUTOMATED INSIGHTS
    # ==========================================================
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)

    insights = []
    df_i = df_filtered.copy()

    # Top hospitals by monthly patients
    if "hospital_name" in df_i.columns and "monthly_patients" in df_i.columns:
        top_pat = df_i.dropna(subset=["monthly_patients"]).nlargest(5, "monthly_patients")
        for _, r in top_pat.iterrows():
            insights.append(
                f"{r['hospital_name']} handles ~{int(r['monthly_patients'])} patients per month."
            )

    # Highest equipment shortage score
    if "hospital_name" in df_i.columns and "equipment_shortage_score" in df_i.columns:
        top_eq = df_i.dropna(subset=["equipment_shortage_score"]).nlargest(5, "equipment_shortage_score")
        for _, r in top_eq.iterrows():
            insights.append(
                f"{r['hospital_name']} shows an equipment shortage score of {round(r['equipment_shortage_score'],2)}."
            )

    # Highest patients per staff
    if "hospital_name" in df_i.columns and "patients_per_staff" in df_i.columns:
        top_pps = df_i.dropna(subset=["patients_per_staff"]).nlargest(5, "patients_per_staff")
        for _, r in top_pps.iterrows():
            insights.append(
                f"{r['hospital_name']} is operating at {round(r['patients_per_staff'],2)} patients per staff."
            )

    # Highest overall risk
    if "hospital_name" in df_i.columns and "overall_risk_score" in df_i.columns:
        top_risk = df_i.dropna(subset=["overall_risk_score"]).nlargest(5, "overall_risk_score")
        for _, r in top_risk.iterrows():
            insights.append(
                f"{r['hospital_name']} has an overall risk score of {round(r['overall_risk_score'],2)}."
            )

    # ML-based insights if model exists
    if st.session_state.get("reg_pipe") is not None:
        try:
            model = st.session_state["reg_pipe"]
            preds_all = model.predict(df_filtered[model.feature_names_in_])
            df_pred_temp = df_filtered.copy()
            df_pred_temp["ML_Prediction"] = preds_all

            top_pred = df_pred_temp.nlargest(5, "ML_Prediction")
            if "hospital_name" in df_pred_temp.columns:
                for _, r in top_pred.iterrows():
                    insights.append(
                        f"ML flags {r['hospital_name']} with predicted {round(r['ML_Prediction'],2)} for '{target_col}'."
                    )
            insights.append(
                f"Across current filters, ML predictions for '{target_col}' span from {round(df_pred_temp['ML_Prediction'].min(),2)} to {round(df_pred_temp['ML_Prediction'].max(),2)}."
            )
        except Exception as e:
            st.warning(f"ML insights unavailable: {e}")

    if insights:
        df_ins = pd.DataFrame({"Insights": insights})
        st.dataframe(df_ins, use_container_width=True)
        download_df(df_ins, "hospitalops_automated_insights.csv", "Download automated insights")
    else:
        st.warning("No insights generated. Dataset is missing expected columns.")
