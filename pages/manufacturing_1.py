import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import math
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Machine Failure & Predictive Maintenance Lab",
    layout="wide"
)

# ---------------------------------------------------------
# HIDE SIDEBAR
# ---------------------------------------------------------
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# GLOBAL CSS – MASTER UI
# ---------------------------------------------------------
st.markdown("""
<style>
* { font-family: 'Inter', sans-serif; }

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

/* CARD */
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

/* KPI CARDS */
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
.kpi-value {
    display:block;
    font-size:22px;
    font-weight:700;
    margin-top:8px;
}

/* VARIABLE BOXES */
.variable-box {
    padding:16px;
    border-radius:14px;
    background:white;
    border:1px solid #e5e5e5;
    box-shadow:0 2px 10px rgba(0,0,0,0.10);
    transition:0.25s ease;
    text-align:center;
    font-size:16.5px !important;
    font-weight:500 !important;
    color:#064b86 !important;
    margin-bottom:10px;
}
.variable-box:hover {
    transform:translateY(-5px);
    box-shadow:0 12px 22px rgba(6,75,134,0.18);
    border-color:#064b86;
}

/* BUTTONS */
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

/* TABLE OVERRIDE */
.required-table th {
    background:#ffffff !important;
    color:#000 !important;
    font-size:17px !important;
    border-bottom:2px solid #000 !important;
}
.required-table td {
    color:#000 !important;
    font-size:15.5px !important;
    padding:6px 8px !important;
    border-bottom:1px solid #dcdcdc !important;
}
.required-table tr:hover td {
    background:#f8f8f8 !important;
}

/* PAGE FADE */
.block-container { animation: fadeIn 0.5s ease; }
@keyframes fadeIn {
    from {opacity:0; transform:translateY(10px);}
    to {opacity:1; transform:translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HEADER + LOGO
# ---------------------------------------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

st.markdown(f"""
<div style="display:flex; align-items:center; margin-bottom:16px;">
    <img src="{logo_url}" width="60" style="margin-right:12px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:32px; font-weight:700;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:32px; font-weight:700;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='big-header'>Machine Failure & Predictive Maintenance Lab</div>", unsafe_allow_html=True)
st.write("Monitor machine health, detect failures early, and use ML to predict breakdown risk at scale.")

# ---------------------------------------------------------
# CONSTANTS / REQUIRED COLS + DICTIONARY
# ---------------------------------------------------------
REQUIRED_COLS = [
    "Timestamp", "Machine_ID", "Machine_Type", "Temperature", "Vibration",
    "RPM", "Load", "Run_Hours", "Temp_Anomaly", "Vib_Anomaly",
    "Load_Anomaly", "RPM_Anomaly", "Failure_Flag"
]

DATA_DICTIONARY = {
    "Timestamp": "Exact date-time when machine sensor readings were captured.",
    "Machine_ID": "Unique identifier for each machine or asset in the plant.",
    "Machine_Type": "Category of machine (e.g., CNC, Press, Motor, Conveyor).",
    "Temperature": "Real-time machine temperature in °C captured from internal sensors.",
    "Vibration": "Vibration level (e.g., mm/s) indicating mechanical balance and wear.",
    "RPM": "Rotational speed of shafts or motors, measured in rotations per minute.",
    "Load": "Machine load as a % of rated capacity at the time of reading.",
    "Run_Hours": "Cumulative operating hours since last overhaul / reset / maintenance.",
    "Temp_Anomaly": "Binary flag (0/1) indicating abnormal temperature patterns vs baseline.",
    "Vib_Anomaly": "Binary flag indicating abnormal vibration beyond safe thresholds.",
    "Load_Anomaly": "Binary anomaly indicator for unusual load behavior.",
    "RPM_Anomaly": "Binary anomaly indicator for RPM spikes/drops or unstable rotation.",
    "Failure_Flag": "Target variable: 1 if machine experienced a failure event, else 0."
}

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def download_df(df: pd.DataFrame, filename: str, label: str = "Download CSV"):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")


def read_csv_safe(url_or_file):
    """
    Read CSV safely and handle duplicate column names by appending suffixes.
    """
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


def prefer_column(df, base):
    """
    Prefer canonical column name if exists, else first duplicate variant.
    """
    for c in df.columns:
        if c == base:
            return c
    for c in df.columns:
        if c.startswith(base + "__dup"):
            return c
    return None


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# =========================================================
# TAB 1 – OVERVIEW
# =========================================================
with tab1:
    st.markdown("<div class='section-title'>Overview</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    This lab brings together machine sensor telemetry (temperature, vibration, RPM, load) and
    maintenance events to identify early failure signals, quantify breakdown risk, and support
    preventive maintenance schedules.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-title'>What This Lab Does</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Tracks <b>sensor signals</b> per machine over time<br>
        • Monitors <b>Temperature, Vibration, RPM, Load</b> and anomaly flags<br>
        • Calculates <b>failure incidence</b> across machines & machine types<br>
        • Trains ML models to predict <b>Failure_Flag</b> from sensor features<br>
        • Identifies high-risk machines for <b>targeted preventive maintenance</b>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='section-title'>Business Impact</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Reduced <b>unplanned downtime</b> & production loss<br>
        • Extended <b>machine lifespan</b> through targeted interventions<br>
        • Lower maintenance cost via <b>data-driven scheduling</b><br>
        • Prioritized attention to <b>high-risk assets</b><br>
        • Stronger <b>OEE (Overall Equipment Effectiveness)</b> story for management
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Key KPIs Tracked</div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Machines Tracked</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Average Temperature</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Average Vibration</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Failure Events</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Who Should Use This</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    Plant heads, maintenance engineers, reliability teams, data engineers and analysts who want a
    single workspace for <b>machine health monitoring, failure prediction, and downtime analytics</b>.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# TAB 2 – IMPORTANT ATTRIBUTES (DATA DICTIONARY)
# =========================================================
with tab2:
    st.markdown("<div class='section-title'>Required Columns – Data Dictionary</div>", unsafe_allow_html=True)

    dd_df = pd.DataFrame(
        [{"Attribute": k, "Description": v} for k, v in DATA_DICTIONARY.items()]
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
                padding:6px 8px !important;
                border-bottom:1px solid #dcdcdc !important;
            }
            .required-table tr:hover td {
                background:#f8f8f8 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.dataframe(
        dd_df.style.set_table_attributes('class="required-table"'),
        use_container_width=True
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-title'>Independent Variables</div>", unsafe_allow_html=True)
        independents = [
            "Machine_ID",
            "Machine_Type",
            "Temperature",
            "Vibration",
            "RPM",
            "Load",
            "Run_Hours",
            "Temp_Anomaly",
            "Vib_Anomaly",
            "Load_Anomaly",
            "RPM_Anomaly"
        ]
        for v in independents:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='section-title'>Dependent Variables</div>", unsafe_allow_html=True)
        dependents = ["Failure_Flag"]
        for v in dependents:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# =========================================================
# TAB 3 – APPLICATION
# =========================================================
with tab3:
    st.markdown("<div class='section-title'>Step 1: Load Dataset</div>", unsafe_allow_html=True)

    mode = st.radio(
        "Dataset option:",
        ["Default dataset (GitHub)", "Upload CSV", "Upload CSV + Manual Column Mapping"],
        horizontal=True
    )

    df = None
    DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/manufacturing/machine_failure_data.csv"

    # ---------------------- DEFAULT DATASET ----------------------
    if mode == "Default dataset (GitHub)":
        try:
            df = read_csv_safe(DEFAULT_URL)
            st.success("Default dataset loaded successfully from GitHub.")
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    # ---------------------- UPLOAD CSV ---------------------------
    elif mode == "Upload CSV":
        file = st.file_uploader("Upload CSV file", type=["csv"], key="upload_simple")
        if file:
            try:
                df = read_csv_safe(file)
                st.success("File uploaded.")
                st.dataframe(df.head(10), use_container_width=True)
            except Exception as e:
                st.error("Failed to read uploaded CSV: " + str(e))
                st.stop()
        else:
            st.stop()

    # ---------------------- UPLOAD + MAPPING ---------------------
    else:
        file = st.file_uploader("Upload CSV for mapping", type=["csv"], key="upload_map")
        if file:
            try:
                raw = read_csv_safe(file)
            except Exception as e:
                st.error("Failed to read uploaded CSV: " + str(e))
                st.stop()

            st.markdown("Preview (first 10 rows):")
            st.dataframe(raw.head(10), use_container_width=True)

            mapping = {}
            cols_list = list(raw.columns)
            for col in REQUIRED_COLS:
                mapping[col] = st.selectbox(f"Map → {col}", ["-- Select --"] + cols_list, key=f"map_{col}")

            if st.button("Apply Mapping", key="apply_map"):
                missing = [m for m in mapping if mapping[m] == "-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    rename_map = {mapping[k]: k for k in mapping}
                    df = raw.rename(columns=rename_map)
                    st.success("Mapping applied successfully.")
                    st.dataframe(df.head(10), use_container_width=True)
        else:
            st.stop()

    if df is None:
        st.stop()

    # ---------------------------------------------------------
    # CLEANUP & CANONICAL COLUMN ALIGNMENT
    # ---------------------------------------------------------
    df.columns = [str(c).strip() for c in df.columns]

    # Align common duplicate logical columns to canonical names where possible
    canonical_map = {}
    for base in ["Timestamp", "Machine_ID", "Machine_Type", "Temperature",
                 "Vibration", "RPM", "Load", "Run_Hours", "Failure_Flag"]:
        col_found = prefer_column(df, base)
        if col_found and col_found != base:
            canonical_map[col_found] = base
    if canonical_map:
        df = df.rename(columns=canonical_map)

    # Ensure Timestamp is datetime if present
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Ensure numeric columns exist and are numeric
    numeric_cols_base = ["Temperature", "Vibration", "RPM", "Load", "Run_Hours", "Failure_Flag"]
    for c in numeric_cols_base:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # If Failure_Flag missing, create default 0
    if "Failure_Flag" not in df.columns:
        df["Failure_Flag"] = 0

    # Replace NaNs in numeric sensor cols with median or 0
    for c in ["Temperature", "Vibration", "RPM", "Load", "Run_Hours"]:
        if c in df.columns:
            if df[c].notna().any():
                df[c] = df[c].fillna(df[c].median())
            else:
                df[c] = df[c].fillna(0)

    # ---------------------------------------------------------
    # STEP 2 – FILTERS & PREVIEW
    # ---------------------------------------------------------
    st.markdown("<div class='section-title'>Step 2: Filters & Preview</div>", unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    machine_ids = df["Machine_ID"].dropna().unique().tolist() if "Machine_ID" in df.columns else []
    machine_types = df["Machine_Type"].dropna().unique().tolist() if "Machine_Type" in df.columns else []

    with m1:
        sel_id = st.multiselect(
            "Machine ID",
            options=machine_ids,
            default=machine_ids[:5] if machine_ids else []
        )
    with m2:
        sel_type = st.multiselect(
            "Machine Type",
            options=machine_types,
            default=machine_types if machine_types else []
        )

    df_f = df.copy()
    if sel_id and "Machine_ID" in df_f.columns:
        df_f = df_f[df_f["Machine_ID"].isin(sel_id)]
    if sel_type and "Machine_Type" in df_f.columns:
        df_f = df_f[df_f["Machine_Type"].isin(sel_type)]

    st.markdown(f"""
    <div style='padding:8px; background:#eef4ff; border-radius:8px; font-weight:600; margin-bottom:8px;'>
    Filtered Rows: {len(df_f)} of {len(df)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Filtered Data Preview (first 10 rows)")
    st.dataframe(df_f.head(10), use_container_width=True)
    download_df(df_f.head(500), "filtered_machines.csv", "Download filtered preview (first 500 rows)")

    if df_f.empty:
        st.warning("Filtered dataset is empty. Adjust filters above.")
        st.stop()

    # ---------------------------------------------------------
    # KPIs (DYNAMIC)
    # ---------------------------------------------------------
    st.markdown("<div class='section-title'>KPIs (Dynamic)</div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    machines_count = int(df_f["Machine_ID"].nunique()) if "Machine_ID" in df_f.columns else 0
    avg_temp = df_f["Temperature"].mean() if "Temperature" in df_f.columns and df_f["Temperature"].notna().any() else None
    avg_vib = df_f["Vibration"].mean() if "Vibration" in df_f.columns and df_f["Vibration"].notna().any() else None
    failures = int(df_f["Failure_Flag"].sum()) if "Failure_Flag" in df_f.columns else 0

    # KPI FIX
    avg_temp_display = f"{avg_temp:.2f}" if avg_temp is not None else "--"
    avg_vib_display = f"{avg_vib:.2f}" if avg_vib is not None else "--"
    
    k1.markdown(
        f"<div class='kpi'>Machines<div class='kpi-value'>{machines_count}</div></div>",
        unsafe_allow_html=True
    )
    
    k2.markdown(
        f"<div class='kpi'>Avg Temperature<div class='kpi-value'>{avg_temp_display}</div></div>",
        unsafe_allow_html=True
    )
    
    k3.markdown(
        f"<div class='kpi'>Avg Vibration<div class='kpi-value'>{avg_vib_display}</div></div>",
        unsafe_allow_html=True
    )
    
    k4.markdown(
        f"<div class='kpi'>Failure Events<div class='kpi-value'>{failures}</div></div>",
        unsafe_allow_html=True
    )



   
    # ---------------------------------------------------------
    # CHARTS
    # ---------------------------------------------------------
    st.markdown("<div class='section-title'>Charts & Diagnostics</div>", unsafe_allow_html=True)

    # Temperature trend
    if "Timestamp" in df_f.columns and "Temperature" in df_f.columns:
        temp_ts = df_f.dropna(subset=["Timestamp"]).sort_values("Timestamp")
        if not temp_ts.empty:
            fig1 = px.line(
                temp_ts,
                x="Timestamp",
                y="Temperature",
                color="Machine_ID",
                title="Temperature Trend by Machine"
            )
            st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Timestamp or Temperature missing – skipping time-series chart.")

    # Vibration distribution by Machine_Type
    if "Machine_Type" in df_f.columns and "Vibration" in df_f.columns:
        vib_box = df_f.dropna(subset=["Machine_Type", "Vibration"])
        if not vib_box.empty:
            fig2 = px.box(
                vib_box,
                x="Machine_Type",
                y="Vibration",
                title="Vibration Distribution by Machine Type"
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Load vs Temperature scatter
    if "Load" in df_f.columns and "Temperature" in df_f.columns:
        scat = df_f.dropna(subset=["Load", "Temperature"])
        if not scat.empty:
            fig3 = px.scatter(
                scat,
                x="Load",
                y="Temperature",
                color="Failure_Flag" if "Failure_Flag" in scat.columns else None,
                title="Temperature vs Load (colored by Failure_Flag)"
            )
            st.plotly_chart(fig3, use_container_width=True)

    # ---------------------------------------------------------
    # ML – FAILURE PREDICTION
    # ---------------------------------------------------------
    st.markdown("<div class='section-title'>ML – Failure Prediction (RandomForest)</div>", unsafe_allow_html=True)

    ML_COLS = [c for c in ["Temperature", "Vibration", "RPM", "Load", "Run_Hours"] if c in df_f.columns]

    if len(df_f) >= 50 and len(ML_COLS) >= 2 and "Failure_Flag" in df_f.columns:
        X = df_f[ML_COLS].fillna(0)
        y = df_f["Failure_Flag"].astype(int).fillna(0)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X.values)

        strat = y.values if len(np.unique(y.values)) > 1 else None
        Xtr, Xte, ytr, yte = train_test_split(
            Xs, y.values,
            test_size=0.2,
            random_state=42,
            stratify=strat
        )

        model = RandomForestClassifier(n_estimators=150, random_state=42)

        with st.spinner("Training RandomForest failure prediction model..."):
            model.fit(Xtr, ytr)

        preds_prob = model.predict_proba(Xte)[:, 1]
        preds_class = model.predict(Xte)

        results = pd.DataFrame({
            "Actual_Failure": yte,
            "Predicted_Prob": preds_prob,
            "Predicted_Class": preds_class
        })

        st.success(f"Model trained on {len(Xtr)} rows, tested on {len(Xte)} rows.")
        st.dataframe(results.head(15), use_container_width=True)
        download_df(results, "machine_failure_predictions.csv", "Download ML prediction sample")

        # Feature importances
        fi_df = pd.DataFrame({
            "Feature": ML_COLS,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        fig_fi = px.bar(
            fi_df,
            x="Feature",
            y="Importance",
            title="Feature Importance – Failure Prediction Model",
            text="Importance"
        )
        fig_fi.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        st.plotly_chart(fig_fi, use_container_width=True)

    else:
        st.info("Not enough data or features for ML model. Need ≥ 50 rows, at least 2 features among [Temperature, Vibration, RPM, Load, Run_Hours], and Failure_Flag column.")

    # ---------------------------------------------------------
    # AUTOMATED INSIGHTS
    # ---------------------------------------------------------
    st.markdown("<div class='section-title'>Automated Insights</div>", unsafe_allow_html=True)

    insights_rows = []

    # 1) Highest failure-prone machine
    if "Machine_ID" in df_f.columns and "Failure_Flag" in df_f.columns:
        risk_series = df_f.groupby("Machine_ID")["Failure_Flag"].mean().sort_values(ascending=False)
        if not risk_series.empty:
            top_machine = risk_series.index[0]
            top_rate = risk_series.iloc[0]
            insights_rows.append({
                "Insight": "Highest failure-prone machine",
                "Entity": top_machine,
                "Metric": f"{top_rate*100:.2f}% failure rate",
                "Action": "Prioritize detailed inspection and preventive maintenance plan."
            })

    # 2) Machine type with highest average failure rate
    if "Machine_Type" in df_f.columns and "Failure_Flag" in df_f.columns:
        mt = df_f.groupby("Machine_Type")["Failure_Flag"].mean().sort_values(ascending=False)
        if not mt.empty:
            mt_top = mt.index[0]
            mt_val = mt.iloc[0]
            insights_rows.append({
                "Insight": "Most failure-prone machine type",
                "Entity": mt_top,
                "Metric": f"{mt_val*100:.2f}% failure rate",
                "Action": "Review design, duty cycles and maintenance regime for this machine type."
            })

    # 3) Highest average temperature machine
    if "Machine_ID" in df_f.columns and "Temperature" in df_f.columns:
        avg_temp_id = df_f.groupby("Machine_ID")["Temperature"].mean().sort_values(ascending=False)
        if not avg_temp_id.empty:
            mt_id = avg_temp_id.index[0]
            mt_val = avg_temp_id.iloc[0]
            insights_rows.append({
                "Insight": "Machine running at highest temperatures",
                "Entity": mt_id,
                "Metric": f"{mt_val:.2f} °C average",
                "Action": "Check cooling, lubrication and load profiles."
            })

    # 4) Overall failure rate
    if "Failure_Flag" in df_f.columns:
        overall_fail_rate = df_f["Failure_Flag"].mean()
        insights_rows.append({
            "Insight": "Overall failure rate in filtered view",
            "Entity": "All machines (filtered)",
            "Metric": f"{overall_fail_rate*100:.2f}%",
            "Action": "Use this as a baseline to track improvements after interventions."
        })

    if not insights_rows:
        st.info("No insights generated for the current filter selection.")
    else:
        insights_df = pd.DataFrame(insights_rows)
        st.dataframe(insights_df, use_container_width=True)
        download_df(insights_df, "predictive_maintenance_insights.csv", "Download insights CSV")
