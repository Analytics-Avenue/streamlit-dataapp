import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import math
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Driver Behaviour & Safety Analytics Lab", layout="wide")

# Hide default sidebar nav
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display:none;}
section[data-testid="stSidebar"] {display:none;}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header & Logo
# -------------------------
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

# -------------------------
# Global CSS – Marketing Lab standard
# -------------------------
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
    font-size:20px !important;
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
    padding:18px;
    border-radius:14px;
    background:white;
    border:1px solid #e5e5e5;
    box-shadow:0 2px 10px rgba(0,0,0,0.10);
    transition:0.25s ease;
    text-align:center;
    font-size:17.5px !important;
    font-weight:500 !important;
    color:#064b86 !important;
    margin-bottom:12px;
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

st.markdown("<div class='big-header'>Driver Behaviour & Safety Analytics Lab</div>", unsafe_allow_html=True)

# -------------------------
# Helpers & constants
# -------------------------

# TODO: update this if your file path is different
DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/transportation/driver_behaviour.csv"

REQUIRED_BEHAVIOR_COLS = [
    "driver_id",
    "trip_id",
    "overspeed_events",
    "harsh_brake_events",
    "rapid_accel_events",
    "idling_minutes",
    "night_driving_minutes",
    "distraction_score",
    "safety_score",
    "seatbelt_violation_count",
    "camera_distraction_alerts",
    "sharp_turn_events",
    "avg_following_distance_meters",
    "traffic_signal_violations",
    "fatigue_alerts",
    "driving_hours_this_week",
    "eco_score",
    "risk_category"
]

def to_float_series(s):
    return pd.to_numeric(s, errors="coerce")

def download_df(df, filename, label):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# ==========================================================
# TAB 1 - OVERVIEW
# ==========================================================
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <b>Purpose:</b><br><br>
    This lab focuses on <b>driver behaviour, road safety, and risk management</b>. It transforms raw telematics & event data
    into safety scores, risk categories, and actionable coaching insights at the <b>driver, trip and fleet</b> level.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">What This Lab Does</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Tracks <b>overspeed, harsh braking, rapid acceleration, sharp turns</b><br>
        • Quantifies <b>fatigue, distraction & seatbelt violations</b><br>
        • Builds a consolidated <b>safety_score & eco_score</b> per driver<br>
        • Segments drivers into <b>risk_category</b> for coaching & incentives<br>
        • Links weekly driving hours & night driving to <b>risk exposure</b>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Reduce accident probability & incident costs<br>
        • Lower insurance premiums via <b>data-backed driver programs</b><br>
        • Improve fuel usage with <b>eco-driving</b> behaviour<br>
        • Standardize safety benchmarks across locations & vendors<br>
        • Support <b>reward & penalty frameworks</b> with objective data
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Key Safety KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Average Safety Score</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>High-Risk Driver Share</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Overspeed & Harsh Brake Rate</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Eco Driving Index</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Who Should Use This</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    Fleet safety managers, transport heads, control-tower teams, HR & compliance, and analytics teams who need
    a <b>single view of driver risk</b> across routes, shifts and assets, with measurable impact on <b>accidents,
    compliance, fuel and brand reputation</b>.
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# TAB 2 - IMPORTANT ATTRIBUTES
# ==========================================================
with tab2:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    required_dict = {
        "driver_id": "Unique driver identifier.",
        "trip_id": "Trip identifier associated with this driving session.",
        "overspeed_events": "Count of overspeeding events in the trip.",
        "harsh_brake_events": "Count of harsh braking incidents.",
        "rapid_accel_events": "Count of rapid / aggressive accelerations.",
        "idling_minutes": "Total minutes the engine was idling.",
        "night_driving_minutes": "Minutes driven during night hours.",
        "distraction_score": "Score representing mobile/visual distraction level.",
        "safety_score": "Overall safety score (composite index).",
        "seatbelt_violation_count": "Number of times seatbelt violation was detected.",
        "camera_distraction_alerts": "Camera-based distraction alerts (e.g., phone usage).",
        "sharp_turn_events": "Count of sharp turns detected.",
        "avg_following_distance_meters": "Average following distance behind lead vehicle.",
        "traffic_signal_violations": "Count of traffic signal / red-light violations.",
        "fatigue_alerts": "Fatigue / drowsiness alerts raised.",
        "driving_hours_this_week": "Total driving hours logged in the current week.",
        "eco_score": "Eco-driving score (smoothness, speed, RPM, etc.).",
        "risk_category": "Categorical risk band (e.g., Low/Medium/High)."
    }

    req_df = pd.DataFrame(
        [{"Attribute": k, "Description": v} for k, v in required_dict.items()]
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
        req_df.style.set_table_attributes('class="required-table"'),
        use_container_width=True
    )

    # Independent vs Dependent variables
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        independents = [
            "overspeed_events",
            "harsh_brake_events",
            "rapid_accel_events",
            "idling_minutes",
            "night_driving_minutes",
            "seatbelt_violation_count",
            "camera_distraction_alerts",
            "sharp_turn_events",
            "avg_following_distance_meters",
            "traffic_signal_violations",
            "fatigue_alerts",
            "driving_hours_this_week"
        ]
        for v in independents:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        dependents = [
            "safety_score",
            "eco_score",
            "risk_category"
        ]
        for v in dependents:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# ==========================================================
# TAB 3 - APPLICATION
# ==========================================================
with tab3:
    st.markdown('<div class="section-title">Step 1: Load dataset</div>', unsafe_allow_html=True)

    df = None
    mode = st.radio(
        "Select Dataset Option:",
        ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"],
        horizontal=True
    )

    # -------------------------
    # Default dataset
    # -------------------------
    if mode == "Default dataset":
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            st.success("Default dataset loaded from GitHub.")
            st.dataframe(df.head(5), use_container_width=True)
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    # -------------------------
    # Upload CSV (with sample preview)
    # -------------------------
    elif mode == "Upload CSV":
        st.markdown("#### Sample structure (from standard Driver Behaviour dataset)", unsafe_allow_html=True)
        try:
            sample_df = pd.read_csv(DEFAULT_URL).head(5)
            st.dataframe(sample_df, use_container_width=True)
            sample_csv = sample_df.to_csv(index=False)
            st.download_button(
                "Download Sample CSV",
                sample_csv,
                "sample_driver_behaviour.csv",
                "text/csv"
            )
        except Exception as e:
            st.info(f"Sample CSV unavailable from GitHub: {e}")

        uploaded = st.file_uploader("Upload your Driver Behaviour dataset", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            st.success("File uploaded.")
            st.dataframe(df.head(5), use_container_width=True)

    # -------------------------
    # Upload + column mapping
    # -------------------------
    else:
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head(5), use_container_width=True)
            st.markdown("Map your columns to the required fields:", unsafe_allow_html=True)
            mapping = {}
            for req in REQUIRED_BEHAVIOR_COLS:
                mapping[req] = st.selectbox(
                    f"Map → {req}",
                    options=["-- Select --"] + list(raw.columns)
                )
            if st.button("Apply mapping"):
                missing = [k for k, v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    inv = {v: k for k, v in mapping.items()}
                    df = raw.rename(columns=inv)
                    st.success("Mapping applied.")
                    st.dataframe(df.head(5), use_container_width=True)

    if df is None:
        st.stop()

    # -------------------------
    # Validate required columns
    # -------------------------
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED_BEHAVIOR_COLS if c not in df.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.info("Use 'Upload CSV + Column mapping' if your column names differ.")
        st.stop()

    # -------------------------
    # Coerce numeric columns
    # -------------------------
    num_cols = [
        "overspeed_events",
        "harsh_brake_events",
        "rapid_accel_events",
        "idling_minutes",
        "night_driving_minutes",
        "distraction_score",
        "safety_score",
        "seatbelt_violation_count",
        "camera_distraction_alerts",
        "sharp_turn_events",
        "avg_following_distance_meters",
        "traffic_signal_violations",
        "fatigue_alerts",
        "driving_hours_this_week",
        "eco_score"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = to_float_series(df[c])

    # -------------------------
    # Filters
    # -------------------------
    st.markdown('<div class="section-title">Step 2: Filters & Preview</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        drivers = sorted(df["driver_id"].astype(str).unique().tolist())
        sel_drivers = st.multiselect("Driver ID", options=drivers, default=drivers[:10] if len(drivers) > 10 else drivers)
    with f2:
        risks = sorted(df["risk_category"].astype(str).unique().tolist())
        sel_risk = st.multiselect("Risk Category", options=risks, default=risks)
    with f3:
        try:
            s_min = float(df["safety_score"].min())
            s_max = float(df["safety_score"].max())
            safety_range = st.slider("Safety score range", float(s_min), float(s_max), (float(s_min), float(s_max)))
        except Exception:
            safety_range = None

    filt = df.copy()
    if sel_drivers:
        filt = filt[filt["driver_id"].astype(str).isin(sel_drivers)]
    if sel_risk:
        filt = filt[filt["risk_category"].astype(str).isin(sel_risk)]
    if safety_range is not None:
        filt = filt[
            (filt["safety_score"] >= safety_range[0]) &
            (filt["safety_score"] <= safety_range[1])
        ]

    if filt.empty:
        st.warning("Filters removed all rows. Showing full dataset.")
        filt = df.copy()

    st.markdown(f"""
    <div style='padding:8px; background:#eef4ff; border-radius:8px; font-weight:600;'>
    Filtered Rows: {len(filt)} of {len(df)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Preview (first 10 rows)")
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(500), "driver_behaviour_filtered_preview.csv", "Download filtered preview (max 500 rows)")

    # -------------------------
    # KPIs
    # -------------------------
    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    def safe_mean(col):
        try:
            return float(pd.to_numeric(filt[col], errors="coerce").mean())
        except Exception:
            return float("nan")

    total_trips = len(filt)
    avg_safety = safe_mean("safety_score")
    avg_eco = safe_mean("eco_score")
    avg_overspeed = safe_mean("overspeed_events")
    high_risk_share = (
        (filt["risk_category"].astype(str).str.lower() == "high").mean()
        if "risk_category" in filt.columns and len(filt) > 0
        else 0
    )

    k1.metric("Total Trips (rows)", f"{total_trips:,}")
    k2.metric("Avg Safety Score", f"{avg_safety:.2f}" if not math.isnan(avg_safety) else "N/A")
    k3.metric("Avg Eco Score", f"{avg_eco:.2f}" if not math.isnan(avg_eco) else "N/A")
    k4.metric("High Risk Driver Share", f"{high_risk_share*100:.1f}%" if total_trips > 0 else "N/A")

    # -------------------------
    # Charts & Diagnostics
    # -------------------------
    st.markdown('<div class="section-title">Charts & Diagnostics</div>', unsafe_allow_html=True)

    # 1) Risk category distribution
    if "risk_category" in filt.columns:
        rc = filt["risk_category"].astype(str).value_counts().reset_index()
        rc.columns = ["risk_category", "count"]
        if not rc.empty:
            fig_risk = px.bar(
                rc,
                x="risk_category",
                y="count",
                text="count",
                title="Driver count by risk category"
            )
            fig_risk.update_traces(textposition="outside")
            st.plotly_chart(fig_risk, use_container_width=True)

    # 2) Safety score by driver (top N)
    if "driver_id" in filt.columns and "safety_score" in filt.columns:
        ds = filt.groupby("driver_id")["safety_score"].mean().reset_index()
        ds = ds.dropna(subset=["safety_score"])
        if not ds.empty:
            ds = ds.sort_values("safety_score", ascending=False).head(25)
            fig_safe = px.bar(
                ds,
                x="driver_id",
                y="safety_score",
                text="safety_score",
                title="Average safety score by driver (top 25)"
            )
            fig_safe.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_safe.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_safe, use_container_width=True)

    # 3) Overspeed vs Safety scatter
    if set(["overspeed_events", "safety_score"]).issubset(filt.columns):
        sc_df = filt.dropna(subset=["overspeed_events", "safety_score"])
        if not sc_df.empty:
            fig_os = px.scatter(
                sc_df,
                x="overspeed_events",
                y="safety_score",
                color="risk_category" if "risk_category" in sc_df.columns else None,
                hover_data=["driver_id", "trip_id"],
                title="Overspeed events vs Safety score",
                labels={"overspeed_events": "Overspeed events", "safety_score": "Safety score"}
            )
            st.plotly_chart(fig_os, use_container_width=True)

    # 4) Harsh braking & rapid acceleration (stacked bar by driver)
    if set(["driver_id", "harsh_brake_events", "rapid_accel_events"]).issubset(filt.columns):
        hb = filt.groupby("driver_id")[["harsh_brake_events", "rapid_accel_events"]].sum().reset_index()
        hb = hb.sort_values("harsh_brake_events", ascending=False).head(20)
        if not hb.empty:
            fig_events = go.Figure(data=[
                go.Bar(
                    name="Harsh brake",
                    x=hb["driver_id"],
                    y=hb["harsh_brake_events"]
                ),
                go.Bar(
                    name="Rapid accel",
                    x=hb["driver_id"],
                    y=hb["rapid_accel_events"]
                )
            ])
            fig_events.update_layout(
                barmode="stack",
                title="Harsh brake & rapid acceleration count by driver (top 20)",
                xaxis_title="Driver",
                yaxis_title="Event count"
            )
            fig_events.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_events, use_container_width=True)

    # -------------------------
    # ML — Safety score regression (RandomForest)
    # -------------------------
    st.markdown('<div class="section-title">ML — Safety Score Regression (RandomForest)</div>', unsafe_allow_html=True)
    with st.expander("Train & evaluate model (requires >=80 rows with non-null safety_score)", expanded=False):
        ml_df = filt.dropna(subset=["safety_score"]).copy()

        feat_cols = [
            "overspeed_events",
            "harsh_brake_events",
            "rapid_accel_events",
            "idling_minutes",
            "night_driving_minutes",
            "seatbelt_violation_count",
            "camera_distraction_alerts",
            "sharp_turn_events",
            "avg_following_distance_meters",
            "traffic_signal_violations",
            "fatigue_alerts",
            "driving_hours_this_week",
            "eco_score",
            "risk_category"
        ]
        feat_cols = [c for c in feat_cols if c in ml_df.columns]

        if len(ml_df) < 80 or len(feat_cols) < 3:
            st.info("Not enough rows or features to train a reliable model (need at least ~80 rows).")
        else:
            X = ml_df[feat_cols].copy()
            y = ml_df["safety_score"].astype(float)

            cat_cols = [c for c in X.columns if X[c].dtype == "object"]
            num_cols_ml = [c for c in X.columns if c not in cat_cols]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols) if cat_cols else ("noop", "passthrough", []),
                    ("num", StandardScaler(), num_cols_ml) if num_cols_ml else ("noop2", "passthrough", [])
                ],
                remainder="drop"
            )

            try:
                X_t = preprocessor.fit_transform(X)
            except Exception as e:
                st.error("Preprocessing failed: " + str(e))
                X_t = None

            if X_t is not None:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_t, y, test_size=0.2, random_state=42
                )
                rf = RandomForestRegressor(n_estimators=200, random_state=42)
                with st.spinner("Training RandomForest regression model..."):
                    rf.fit(X_train, y_train)
                preds = rf.predict(X_test)

                rmse = math.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)
                st.write(f"Safety score regression — RMSE: {rmse:.2f}, R²: {r2:.3f}")

                res_df = pd.DataFrame({
                    "Actual_safety_score": y_test.reset_index(drop=True),
                    "Predicted_safety_score": preds
                })
                st.dataframe(res_df.head(20), use_container_width=True)
                download_df(res_df, "safety_score_predictions.csv", "Download safety prediction sample")

    # -------------------------
    # Automated Insights
    # -------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)

    insights_rows = []

    # 1) Driver with lowest safety score
    if "driver_id" in filt.columns and "safety_score" in filt.columns and not filt.empty:
        ds_full = filt.groupby("driver_id")["safety_score"].mean().reset_index()
        ds_full = ds_full.dropna(subset=["safety_score"])
        if not ds_full.empty:
            worst = ds_full.sort_values("safety_score", ascending=True).iloc[0]
            insights_rows.append({
                "Insight": "Driver with lowest average safety score",
                "Entity": worst["driver_id"],
                "Metric": f"{worst['safety_score']:.2f}",
                "Action": "Priority coaching & monitoring required."
            })

    # 2) Driver with best eco-driving
    if "driver_id" in filt.columns and "eco_score" in filt.columns and not filt.empty:
        eco = filt.groupby("driver_id")["eco_score"].mean().reset_index()
        eco = eco.dropna(subset=["eco_score"])
        if not eco.empty:
            best_eco = eco.sort_values("eco_score", ascending=False).iloc[0]
            insights_rows.append({
                "Insight": "Best eco-driving driver",
                "Entity": best_eco["driver_id"],
                "Metric": f"{best_eco['eco_score']:.2f}",
                "Action": "Use as benchmark profile for eco-driving playbooks."
            })

    # 3) Risk category with worst behaviour
    if "risk_category" in filt.columns and "safety_score" in filt.columns and not filt.empty:
        rc_agg = filt.groupby("risk_category")["safety_score"].mean().reset_index()
        rc_agg = rc_agg.dropna(subset=["safety_score"])
        if not rc_agg.empty:
            worst_band = rc_agg.sort_values("safety_score", ascending=True).iloc[0]
            insights_rows.append({
                "Insight": "Worst performing risk band",
                "Entity": worst_band["risk_category"],
                "Metric": f"Avg safety_score {worst_band['safety_score']:.2f}",
                "Action": "Target this band with stricter policies & training."
            })

    # 4) Most frequent critical event type
    event_cols = [
        "overspeed_events",
        "harsh_brake_events",
        "rapid_accel_events",
        "sharp_turn_events",
        "traffic_signal_violations",
        "fatigue_alerts"
    ]
    available_events = [c for c in event_cols if c in filt.columns]
    if available_events:
        totals = filt[available_events].sum(numeric_only=True)
        if not totals.empty:
            worst_event = totals.sort_values(ascending=False).index[0]
            insights_rows.append({
                "Insight": "Most frequent critical event type",
                "Entity": worst_event,
                "Metric": f"Total count: {int(totals[worst_event])}",
                "Action": "Design focused interventions to cut this event type."
            })

    # 5) Long driving hours correlation hint
    if set(["driving_hours_this_week", "safety_score"]).issubset(filt.columns):
        corr = np.corrcoef(
            pd.to_numeric(filt["driving_hours_this_week"], errors="coerce").fillna(0),
            pd.to_numeric(filt["safety_score"], errors="coerce").fillna(0)
        )[0, 1]
        insights_rows.append({
            "Insight": "Driving hours vs safety correlation",
            "Entity": "All filtered drivers",
            "Metric": f"Correlation ≈ {corr:.2f}",
            "Action": "If negative, tighten weekly hour limits to protect safety."
        })

    if not insights_rows:
        st.info("No insights generated for the current filter selection.")
    else:
        insights_df = pd.DataFrame(insights_rows)
        st.dataframe(insights_df, use_container_width=True)
        download_df(insights_df, "driver_behaviour_insights.csv", "Download insights")
