import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import math
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="PV Panel Degradation Intelligence Lab", layout="wide")

# Hide default sidebar
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display:none;}
section[data-testid="stSidebar"] {display:none;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER & LOGO
# ---------------------------------------------------
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

# ---------------------------------------------------
# GLOBAL CSS – Marketing Lab Standard
# ---------------------------------------------------
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

/* TABLE (index-safe renderer) */
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

/* PAGE FADE */
.block-container { animation: fadeIn 0.5s ease; }
@keyframes fadeIn { from {opacity:0; transform:translateY(10px);} to {opacity:1; transform:translateY(0);} }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-header'>PV Panel Degradation Intelligence Lab</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# CONSTANTS
# ---------------------------------------------------
# If this path is wrong in your repo, just update this URL
DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/Solar/pv_degradation.csv"

REQUIRED_PV_COLS = [
    "panel_id",
    "string_id",
    "timestamp",
    "irradiation_wm2",
    "temperature_c",
    "panel_voltage_v",
    "panel_current_a",
    "panel_efficiency_pct",
    "baseline_efficiency_pct",
    "degradation_pct",
    "performance_ratio",
    "soiling_factor_pct",
    "shading_loss_pct",
    "module_age_years",
    "installation_angle_deg",
    "panel_serial_number",
    "manufacturer",
    "underperformance_flag",
    "hotspot_detected_flag",
    "last_cleaned_date",
    "region"
]

NUM_COLS = [
    "irradiation_wm2",
    "temperature_c",
    "panel_voltage_v",
    "panel_current_a",
    "panel_efficiency_pct",
    "baseline_efficiency_pct",
    "degradation_pct",
    "performance_ratio",
    "soiling_factor_pct",
    "shading_loss_pct",
    "module_age_years",
    "installation_angle_deg"
]

def download_df(df, filename, label):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")

def safe_mean(series):
    return pd.to_numeric(series, errors="coerce").mean()

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# ===================================================
# TAB 1 – OVERVIEW
# ===================================================
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <b>Why this lab exists:</b><br><br>
    Utility-scale and rooftop PV plants silently lose efficiency every month because modules degrade,
    soil up, or suffer from hotspots and shading. Most plants only notice when annual energy yield
    drops or when an OEM audit calls it out.<br><br>
    This lab creates a <b>panel-level degradation cockpit</b> where operations, asset management,
    and engineering teams can see which strings are aging faster, which regions are underperforming,
    and how cleaning, irradiation and temperature link to real efficiency loss.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">What this Lab Does</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Tracks <b>panel_efficiency_pct vs baseline_efficiency_pct</b> over time<br>
        • Quantifies <b>degradation_pct</b> at panel, string, and region level<br>
        • Combines <b>soiling_factor_pct</b> & <b>shading_loss_pct</b> to surface avoidable loss<br>
        • Flags <b>underperformance_flag</b> and <b>hotspot_detected_flag</b> segments<br>
        • Links degradation to <b>irradiation_wm2</b> and <b>temperature_c</b> stress<br>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Increase plant AEP by focusing on <b>high-loss strings</b><br>
        • Reduce random cleaning & move to <b>data-driven wash cycles</b><br>
        • Detect <b>early hotspots</b> before they become safety & asset risks<br>
        • Support <b>warranty / performance-guarantee claims</b> with hard data<br>
        • Build long-term degradation curves for <b>OEM & material comparison</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Key Degradation KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Avg degradation_pct</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Share of underperforming panels</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Hotspot-affected share</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Region-wise performance_ratio spread</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Who Should Use This</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    • Asset owners & IPP performance teams<br>
    • O&M providers and site engineers<br>
    • OEM / EPC performance analytics teams<br>
    • Anyone who needs to justify cleaning, retrofits, or module replacement decisions with data
    </div>
    """, unsafe_allow_html=True)

# ===================================================
# TAB 2 – IMPORTANT ATTRIBUTES
# ===================================================
with tab2:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    required_dict = {
        "panel_id": "Unique identifier of the PV panel / module.",
        "string_id": "String / array identifier the panel belongs to.",
        "timestamp": "Reading timestamp for performance measurement.",
        "irradiation_wm2": "Solar irradiation at module plane (W/m²).",
        "temperature_c": "Module or cell temperature in °C.",
        "panel_voltage_v": "Panel output voltage (V).",
        "panel_current_a": "Panel output current (A).",
        "panel_efficiency_pct": "Actual module efficiency at this timestamp (%).",
        "baseline_efficiency_pct": "Reference / expected efficiency under similar conditions (%).",
        "degradation_pct": "Loss in efficiency vs baseline (%).",
        "performance_ratio": "PR at panel/string level (0–1 or 0–100%).",
        "soiling_factor_pct": "Loss attributed purely to soiling (%).",
        "shading_loss_pct": "Loss due to shading (%).",
        "module_age_years": "Age of module in years since commissioning.",
        "installation_angle_deg": "Tilt / angle of installation (degrees).",
        "panel_serial_number": "OEM serial number for traceability.",
        "manufacturer": "Module manufacturer name.",
        "underperformance_flag": "Flag indicating panel underperformance vs expected (0/1 or True/False).",
        "hotspot_detected_flag": "Flag if hotspot has been detected (0/1 or True/False).",
        "last_cleaned_date": "Last date the panel/string was cleaned.",
        "region": "Plant / geographic region label."
    }

    dict_df = pd.DataFrame(
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
        dict_df.style.set_table_attributes('class="required-table"'),
        use_container_width=True
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        independents = [
            "irradiation_wm2",
            "temperature_c",
            "panel_voltage_v",
            "panel_current_a",
            "soiling_factor_pct",
            "shading_loss_pct",
            "module_age_years",
            "installation_angle_deg",
            "manufacturer",
            "region",
            "timestamp",
            "last_cleaned_date"
        ]
        for v in independents:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        dependents = [
            "degradation_pct",
            "panel_efficiency_pct",
            "performance_ratio",
            "underperformance_flag",
            "hotspot_detected_flag"
        ]
        for v in dependents:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# ===================================================
# TAB 3 – APPLICATION
# ===================================================
with tab3:
    st.markdown('<div class="section-title">Step 1: Load dataset</div>', unsafe_allow_html=True)

    df = None
    mode = st.radio(
        "Select Dataset Option:",
        ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"],
        horizontal=True
    )

    # ----- Default dataset -----
    if mode == "Default dataset":
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            st.success("Default PV degradation dataset loaded from GitHub.")
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    # ----- Upload CSV with sample preview -----
    elif mode == "Upload CSV":
        st.markdown("#### Sample structure (from standard PV degradation dataset)", unsafe_allow_html=True)
        try:
            sample_df = pd.read_csv(DEFAULT_URL).head(5)
            st.dataframe(sample_df, use_container_width=True)
            sample_csv = sample_df.to_csv(index=False)
            st.download_button(
                "Download Sample CSV",
                sample_csv,
                "sample_pv_degradation.csv",
                "text/csv"
            )
        except Exception as e:
            st.info(f"Sample CSV unavailable from GitHub: {e}")

        uploaded = st.file_uploader("Upload your PV degradation dataset", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            st.success("File uploaded.")
            st.dataframe(df.head(10), use_container_width=True)

    # ----- Upload + mapping -----
    else:
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head(5), use_container_width=True)
            st.markdown("Map your columns to the required fields:", unsafe_allow_html=True)
            mapping = {}
            for req in REQUIRED_PV_COLS:
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
                    st.dataframe(df.head(10), use_container_width=True)

    if df is None:
        st.stop()

    # ---------------------------------------------------
    # VALIDATE REQUIRED COLUMNS
    # ---------------------------------------------------
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED_PV_COLS if c not in df.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.info("Use 'Upload CSV + Column mapping' if your column names differ.")
        st.stop()

    # ---------------------------------------------------
    # TYPE HANDLING
    # ---------------------------------------------------
    # Coerce numeric
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Dates
    for dcol in ["timestamp", "last_cleaned_date"]:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

    # Flags to numeric/bool-ish
    for flag_col in ["underperformance_flag", "hotspot_detected_flag"]:
        if flag_col in df.columns:
            df[flag_col] = df[flag_col].apply(lambda x: 1 if str(x).strip().lower() in ["1", "true", "yes", "y"] else 0)

    df = df.dropna(subset=["degradation_pct", "panel_efficiency_pct", "performance_ratio"])

    # ---------------------------------------------------
    # FILTERS
    # ---------------------------------------------------
    st.markdown("### Step 2: Filters")

    f1, f2, f3 = st.columns(3)
    with f1:
        region_sel = st.multiselect("Region", sorted(df["region"].dropna().unique().tolist()))
    with f2:
        manu_sel = st.multiselect("Manufacturer", sorted(df["manufacturer"].dropna().unique().tolist()) if "manufacturer" in df.columns else [])
    with f3:
        if "module_age_years" in df.columns:
            age_min = int(np.nanmin(df["module_age_years"]))
            age_max = int(np.nanmax(df["module_age_years"]))
        else:
            age_min, age_max = 0, 30
        age_range = st.slider("Module Age (years)", age_min, age_max, (age_min, age_max))

    filt = df.copy()
    if region_sel:
        filt = filt[filt["region"].isin(region_sel)]
    if manu_sel and "manufacturer" in filt.columns:
        filt = filt[filt["manufacturer"].isin(manu_sel)]
    if "module_age_years" in filt.columns:
        filt = filt[
            (filt["module_age_years"] >= age_range[0]) &
            (filt["module_age_years"] <= age_range[1])
        ]

    if filt.empty:
        st.warning("Filters removed all rows. Resetting to full dataset.")
        filt = df.copy()

    st.markdown(f"""
    <div style='padding:8px; background:#eef4ff; border-radius:8px; font-weight:600; margin-bottom:10px;'>
    Filtered Rows: {len(filt)} of {len(df)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Data Preview")
    st.dataframe(filt.head(20), use_container_width=True)
    download_df(filt.head(200), "pv_degradation_filtered_sample.csv", "Download filtered sample (top 200)")

    # ---------------------------------------------------
    # KPIs
    # ---------------------------------------------------
    st.markdown('<div class="section-title">Degradation KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    avg_deg = safe_mean(filt["degradation_pct"])
    avg_pr = safe_mean(filt["performance_ratio"])
    under_share = filt["underperformance_flag"].mean() if "underperformance_flag" in filt.columns and len(filt) > 0 else np.nan
    hotspot_share = filt["hotspot_detected_flag"].mean() if "hotspot_detected_flag" in filt.columns and len(filt) > 0 else np.nan

    k1.metric("Avg degradation_pct", f"{avg_deg:.2f} %" if not math.isnan(avg_deg) else "N/A")
    k2.metric("Avg performance_ratio", f"{avg_pr:.3f}" if not math.isnan(avg_pr) else "N/A")
    k3.metric("Underperforming share", f"{under_share*100:.1f} %" if not math.isnan(under_share) else "N/A")
    k4.metric("Hotspot-affected share", f"{hotspot_share*100:.1f} %" if not math.isnan(hotspot_share) else "N/A")

    # ---------------------------------------------------
    # CHARTS & DIAGNOSTICS
    # ---------------------------------------------------
    st.markdown('<div class="section-title">Charts & Diagnostics</div>', unsafe_allow_html=True)

    # 1) Degradation distribution
    if "degradation_pct" in filt.columns:
        hist_df = filt[["degradation_pct"]].dropna()
        if not hist_df.empty:
            fig_deg = px.histogram(
                hist_df,
                x="degradation_pct",
                nbins=40,
                title="Distribution of degradation_pct"
            )
            st.plotly_chart(fig_deg, use_container_width=True)

    # 2) Region-wise avg degradation
    if "region" in filt.columns:
        reg_df = filt.groupby("region")["degradation_pct"].mean().reset_index()
        if not reg_df.empty:
            fig_reg = px.bar(
                reg_df,
                x="region",
                y="degradation_pct",
                text="degradation_pct",
                title="Average degradation_pct by region"
            )
            fig_reg.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_reg.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_reg, use_container_width=True)

    # 3) Efficiency vs baseline
    if "panel_efficiency_pct" in filt.columns and "baseline_efficiency_pct" in filt.columns:
        eff_df = filt[["panel_efficiency_pct", "baseline_efficiency_pct"]].dropna()
        eff_df = eff_df.sample(min(len(eff_df), 500)) if len(eff_df) > 500 else eff_df
        if not eff_df.empty:
            fig_eff = px.scatter(
                eff_df,
                x="baseline_efficiency_pct",
                y="panel_efficiency_pct",
                title="Actual vs Baseline efficiency",
                labels={
                    "baseline_efficiency_pct": "Baseline efficiency (%)",
                    "panel_efficiency_pct": "Actual efficiency (%)"
                }
            )
            st.plotly_chart(fig_eff, use_container_width=True)

    # 4) Degradation vs module age by region
    if "module_age_years" in filt.columns:
        age_df = filt[["module_age_years", "degradation_pct", "region"]].dropna()
        age_df = age_df.sample(min(len(age_df), 800)) if len(age_df) > 800 else age_df
        if not age_df.empty:
            fig_age = px.scatter(
                age_df,
                x="module_age_years",
                y="degradation_pct",
                color="region",
                title="degradation_pct vs module_age_years by region",
                labels={"module_age_years": "Module age (years)", "degradation_pct": "Degradation (%)"}
            )
            st.plotly_chart(fig_age, use_container_width=True)

    # ---------------------------------------------------
    # ML – Degradation Regression
    # ---------------------------------------------------
    st.markdown('<div class="section-title">ML — Predict degradation_pct (RandomForest)</div>', unsafe_allow_html=True)

    with st.expander("Train & evaluate model (needs ~80+ rows with non-null degradation_pct)", expanded=False):
        ml_df = filt.dropna(subset=["degradation_pct"]).copy()

        feat_cols = [
            "irradiation_wm2",
            "temperature_c",
            "panel_voltage_v",
            "panel_current_a",
            "performance_ratio",
            "soiling_factor_pct",
            "shading_loss_pct",
            "module_age_years",
            "installation_angle_deg",
            "region",
            "manufacturer"
        ]
        feat_cols = [c for c in feat_cols if c in ml_df.columns]

        if len(ml_df) < 80 or len(feat_cols) < 3:
            st.info("Not enough rows or features to train a robust model (need at least ~80 rows and a few features).")
        else:
            X = ml_df[feat_cols].copy()
            y = ml_df["degradation_pct"].astype(float)

            cat_cols = [c for c in X.columns if X[c].dtype == "object"]
            num_cols_ml = [c for c in X.columns if c not in cat_cols]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols) if cat_cols else ("noop_cat", "passthrough", []),
                ] + (
                    [("num", StandardScaler(), num_cols_ml)] if num_cols_ml else []
                ),
                remainder="drop"
            )

            try:
                X_t = preprocessor.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)
                rf = RandomForestRegressor(n_estimators=200, random_state=42)
                with st.spinner("Training RandomForest regression model..."):
                    rf.fit(X_train, y_train)
                preds = rf.predict(X_test)

                from sklearn.metrics import mean_squared_error, r2_score
                rmse = math.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)

                st.write(f"Degradation regression — RMSE: {rmse:.3f} %, R²: {r2:.3f}")

                res_df = pd.DataFrame({
                    "Actual_degradation_pct": y_test.reset_index(drop=True),
                    "Predicted_degradation_pct": preds
                })
                st.dataframe(res_df.head(30), use_container_width=True)
                download_df(res_df, "pv_degradation_predictions.csv", "Download prediction sample")
            except Exception as e:
                st.error(f"Model training failed: {e}")

    # ---------------------------------------------------
    # AUTOMATED INSIGHTS
    # ---------------------------------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)

    insights_rows = []

    # Region with highest avg degradation
    if "region" in filt.columns and not filt.empty:
        reg = filt.groupby("region")["degradation_pct"].mean().reset_index()
        reg = reg.dropna(subset=["degradation_pct"])
        if not reg.empty:
            worst_reg = reg.sort_values("degradation_pct", ascending=False).iloc[0]
            insights_rows.append({
                "Insight": "Region with highest average degradation_pct",
                "Entity": worst_reg["region"],
                "Metric": f"{worst_reg['degradation_pct']:.2f} %",
                "Action": "Prioritize deeper diagnostics and cleaning strategy in this region."
            })

    # Manufacturer with worst degradation
    if "manufacturer" in filt.columns and not filt.empty:
        mdf = filt.groupby("manufacturer")["degradation_pct"].mean().reset_index()
        mdf = mdf.dropna(subset=["degradation_pct"])
        if not mdf.empty:
            worst_manu = mdf.sort_values("degradation_pct", ascending=False).iloc[0]
            insights_rows.append({
                "Insight": "Manufacturer with highest average degradation_pct",
                "Entity": worst_manu["manufacturer"],
                "Metric": f"{worst_manu['degradation_pct']:.2f} %",
                "Action": "Use this in OEM discussions and future procurement evaluation."
            })

    # Soiling-driven loss
    if "soiling_factor_pct" in filt.columns:
        avg_soiling = safe_mean(filt["soiling_factor_pct"])
        insights_rows.append({
            "Insight": "Average soiling-driven loss",
            "Entity": "All filtered panels",
            "Metric": f"{avg_soiling:.2f} %",
            "Action": "Check if cleaning intervals can be optimized to reduce this avoidable loss."
        })

    # Hotspot rate
    if "hotspot_detected_flag" in filt.columns:
        hotspot_share = filt["hotspot_detected_flag"].mean()
        insights_rows.append({
            "Insight": "Hotspot occurrence rate",
            "Entity": "All filtered panels",
            "Metric": f"{hotspot_share*100:.1f} %",
            "Action": "Inspect strings with recurring hotspots for wiring, connectors or shading issues."
        })

    # Underperforming share vs average degradation
    if "underperformance_flag" in filt.columns:
        under_df = filt[filt["underperformance_flag"] == 1]
        if not under_df.empty:
            avg_under_deg = safe_mean(under_df["degradation_pct"])
            insights_rows.append({
                "Insight": "Average degradation in underperforming panels",
                "Entity": "Underperformance flagged = 1",
                "Metric": f"{avg_under_deg:.2f} %",
                "Action": "Consider targeted cleaning, IV-curve testing or replacement for this segment."
            })

    if not insights_rows:
        st.info("No insights generated for the current filter selection.")
    else:
        insights_df = pd.DataFrame(insights_rows)
        st.dataframe(insights_df, use_container_width=True)
        download_df(insights_df, "pv_degradation_insights.csv", "Download insights")

