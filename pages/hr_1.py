import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
import math

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# BASIC CONFIG & SIDEBAR HIDING
# ---------------------------------------------------------
hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

st.set_page_config(
    page_title="Hiring Funnel Drop-Off Analysis",
    layout="wide"
)

# ---------------------------------------------------------
# GLOBAL CSS  (standardized UI system)
# ---------------------------------------------------------
st.markdown("""
<style>
* {
    color: #000000 !important;
    font-family: "Inter", sans-serif;
}

/* Top-level layout tweaks */
.block-container {
    padding-top: 12px;
    padding-bottom: 40px;
}

/* Logo container */
.app-header {
    display: flex;
    align-items: center;
    gap: 10px;
}
.app-title-main {
    font-size: 28px;
    font-weight: 800;
    margin: 0;
    padding: 0;
    color: #064b86 !important;
}
.app-title-sub {
    font-size: 13px;
    margin: 0;
    padding: 0;
    color: #444;
}

/* Generic glow card */
.card {
    padding: 16px 18px;
    border-radius: 12px;
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    transition: all 0.18s ease-in-out;
}
.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.16);
    border-color: rgba(6,75,134,0.35);
}
.card h3, .card h4 {
    margin-top: 0;
}
.card-left {
    text-align: left !important;
}

/* KPI cards row */
.kpi-row {
    display: flex;
    gap: 12px;
    margin-top: 10px;
}
.kpi-card {
    flex: 1;
    padding: 16px 18px;
    border-radius: 12px;
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.07);
    box-shadow: 0 3px 10px rgba(0,0,0,0.06);
    text-align: center;
    transition: all 0.18s ease-in-out;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 24px rgba(6,75,134,0.22);
    border-color: rgba(6,75,134,0.35);
}
.kpi-label {
    font-size: 13px;
    font-weight: 600;
    opacity: 0.8;
}
.kpi-value {
    margin-top: 6px;
    font-size: 20px;
    font-weight: 800;
    color: #064b86 !important;  /* KPI value in blue */
}

/* Variable / data dictionary chips */
.var-chip {
    border-radius: 999px;
    padding: 4px 10px;
    border: 1px solid rgba(0,0,0,0.12);
    display: inline-block;
    font-size: 12px;
    margin: 3px 4px 3px 0;
}
.var-chip span.name {
    font-weight: 600;
}
.var-chip span.role {
    font-size: 11px;
    opacity: 0.7;
}

/* Independent vars – blue text */
.var-indep span.name {
    color: #064b86 !important;
}

/* Section subtitles */
.section-subtitle {
    font-weight: 700;
    font-size: 15px;
    margin-top: 10px;
}

/* Hide Streamlit footer/menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOGO + TITLE
# ---------------------------------------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

st.markdown(
    f"""
<div class="app-header">
    <img src="{logo_url}" width="56">
    <div>
        <p class="app-title-main">Analytics Avenue & Advanced Analytics</p>
        <p class="app-title-sub">Hiring Funnel Drop-Off Analysis • TA & People Analytics</p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("")

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def download_df(df, filename, button_label="Download CSV", key=None):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(button_label, b, file_name=filename, mime="text/csv", key=key)

def read_csv_safe(url_or_file):
    """Read CSV and make duplicate columns unique with __dup suffix."""
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

def canonicalize_columns(df, expected_cols):
    """Rename dup versions back to canonical if matched logically."""
    def prefer_column(df_, base):
        for c_ in df_.columns:
            if c_ == base:
                return c_
        for c_ in df_.columns:
            if c_.startswith(base + "__dup"):
                return c_
        lower_map = {c_.lower(): c_ for c_ in df_.columns}
        if base.lower() in lower_map:
            return lower_map[base.lower()]
        return None

    rename = {}
    for base in expected_cols:
        found = prefer_column(df, base)
        if found and found != base:
            rename[found] = base
    if rename:
        df = df.rename(columns=rename)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def safe_to_datetime(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

# ---------------------------------------------------------
# EXPECTED / CANONICAL COLUMNS
# ---------------------------------------------------------
EXPECTED_COLS = [
    "Applicant_ID",
    "Apply_Date",
    "Source",
    "Role",
    "Stage_Apply",
    "Stage_Screen",
    "Stage_Interview",
    "Stage_Offer",
    "Stage_Join",
    "Current_Stage",
    "Days_in_Stage",
    "Total_Time_to_Hire_Days",
    "Offer_Accepted_Flag",
    "Screen_Score",
    "Resume_Score",
    "Recruiter_ID",
    "JD_Variant",
    "Channel",
    "Candidate_Response_Time_Hrs"
]

DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/hr/hiring_funnel_data.csv"


dict_rows = [
    {"Column": "Applicant_ID", "Type": "ID", "Role": "Identifier", "Description": "Unique candidate identifier"},
    {"Column": "Apply_Date", "Type": "DateTime", "Role": "Independent", "Description": "Date when candidate applied"},
    {"Column": "Source", "Type": "Categorical", "Role": "Independent", "Description": "Source of the candidate"},
    {"Column": "Role", "Type": "Categorical", "Role": "Independent", "Description": "Role / position applied for"},
    {"Column": "Stage_Apply", "Type": "Date/Flag", "Role": "Meta", "Description": "Application timestamp/flag"},
    {"Column": "Stage_Screen", "Type": "Date/Flag", "Role": "Meta", "Description": "Screening stage timestamp/flag"},
    {"Column": "Stage_Interview", "Type": "Date/Flag", "Role": "Meta", "Description": "Interview stage timestamp/flag"},
    {"Column": "Stage_Offer", "Type": "Date/Flag", "Role": "Meta", "Description": "Offer stage timestamp/flag"},
    {"Column": "Stage_Join", "Type": "Date/Flag", "Role": "Meta", "Description": "Joining stage timestamp/flag"},
    {"Column": "Current_Stage", "Type": "Categorical", "Role": "Independent", "Description": "Latest stage reached"},
    {"Column": "Days_in_Stage", "Type": "Numeric", "Role": "Independent", "Description": "Days spent in current stage"},
    {"Column": "Total_Time_to_Hire_Days", "Type": "Numeric", "Role": "Dependent", "Description": "Days from apply to join"},
    {"Column": "Offer_Accepted_Flag", "Type": "Binary", "Role": "Dependent", "Description": "1 if offer accepted"},
    {"Column": "Screen_Score", "Type": "Numeric", "Role": "Independent", "Description": "Screening score"},
    {"Column": "Resume_Score", "Type": "Numeric", "Role": "Independent", "Description": "Resume match score"},
    {"Column": "Recruiter_ID", "Type": "Categorical", "Role": "Independent", "Description": "Recruiter handling the case"},
    {"Column": "JD_Variant", "Type": "Categorical", "Role": "Independent", "Description": "Job description variant"},
    {"Column": "Channel", "Type": "Categorical", "Role": "Independent", "Description": "Channel bucket (direct/agency/etc.)"},
    {"Column": "Candidate_Response_Time_Hrs", "Type": "Numeric", "Role": "Independent", "Description": "Candidate response time"},
]

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab_overview, tab_dict, tab_app = st.tabs(["Overview", "Data Dictionary", "Application"])

# =========================================================
# TAB 1 — OVERVIEW
# =========================================================
with tab_overview:
    st.markdown("### Overview")

    st.markdown(
        """
        <div class="card card-left">
            <b>Purpose</b>: Identify where candidates drop off across the hiring journey 
            (Apply → Screen → Interview → Offer → Join), and use predictive analytics 
            to reduce leakage, speed up hiring, and improve candidate experience.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Capabilities")
        st.markdown(
            """
            <div class="card card-left">
            • End-to-end funnel conversion tracking by role, source, JD, channel<br>
            • Stage-wise bottleneck diagnosis & time-in-stage analytics<br>
            • Source, JD & channel performance benchmarking<br>
            • ML-based dropout risk scoring at candidate level<br>
            • Anomaly & cluster analysis of candidate behaviour patterns<br>
            • Scenario simulator for time-to-fill under different funnel strengths
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown("#### Business Impact")
        st.markdown(
            """
            <div class="card card-left">
            • Lower time-to-fill and fewer open-vacancy days<br>
            • Improved offer-accept rate and join ratio<br>
            • Reduced recruiter load via prioritized follow-ups<br>
            • Lower cost-per-hire through focused channels & JDs<br>
            • Data-backed hiring plans for TA, HRBP, and business leaders
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### KPIs (conceptual)")
    st.markdown(
        """
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-label">Applicants</div>
                <div class="kpi-value">—</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Screen Pass %</div>
                <div class="kpi-value">—</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Interview Rate</div>
                <div class="kpi-value">—</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Offer Rate</div>
                <div class="kpi-value">—</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Time-to-Fill (days)</div>
                <div class="kpi-value">—</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Who should use")
    st.markdown(
        """
        <div class="card card-left">
        <b>Users</b>: TA leads, recruiters, hiring managers, HRBP, and People Analytics teams.<br><br>
        <b>How</b>:<br>
        1) Load hiring funnel data (default or your CSV).<br>
        2) Filter by role, source, JD, channel, and dates.<br>
        3) Analyse drop-offs across stages and time-to-fill by role.<br>
        4) Use ML risk scores, anomalies, and clusters to prioritize follow-ups.<br>
        5) Run the scheduling simulator to estimate hiring timelines under different funnel strengths.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# TAB 2 — DATA DICTIONARY
# =========================================================
with tab_dict:
    st.markdown("### Required Columns & Data Dictionary")

    st.markdown("---")

    # ---- SPLIT VIEW: Independent (Left) vs Dependent (Right) ----
    st.markdown("### Variables by Role (Independent vs Dependent)")

    left, right = st.columns(2)

    # -------- LEFT: Independent Variables --------
    with left:
        st.markdown("#### Independent Variables")
        for _, row in dd_df[dd_df["Role"] == "Independent"].iterrows():
            st.markdown(
                f"""
                <div class='card card-left'>
                    <b>{row['Column']}</b>
                    </div>
                """,
                unsafe_allow_html=True,
            )

    # -------- RIGHT: Dependent Variables --------
    with right:
        st.markdown("#### Dependent Variables")
        for _, row in dd_df[dd_df["Role"] == "Dependent"].iterrows():
            st.markdown(
                f"""
                <div class='card card-left'>
                    <b>{row['Column']}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

    download_df(dd_df, "hiring_funnel_data_dictionary.csv", "Download Data Dictionary")

# =========================================================
# TAB 3 — APPLICATION
# =========================================================
with tab_app:
    st.header("Application")
    st.markdown("### Step 1 — Load dataset")

    mode = st.radio(
        "Dataset option:",
        ["Default dataset (GitHub URL)", "Upload CSV", "Upload CSV + Column mapping"],
        horizontal=True,
    )

    df = None

    if mode == "Default dataset (GitHub URL)":
        try:
            df = read_csv_safe(DEFAULT_URL)
            df = canonicalize_columns(df, EXPECTED_COLS)

            # simple auto-mapping for some common column names
            auto_map = {
                "Job_Role": "Role",
                "Applied_Date": "Apply_Date",
                "ApplyDate": "Apply_Date",
                "Apply_Timestamp": "Apply_Date",
                "Application_Date": "Apply_Date",
                "Screen_Date": "Stage_Screen",
                "Interview_Date": "Stage_Interview",
                "Offer_Date": "Stage_Offer",
                "Join_Date": "Stage_Join",
            }
            rename_dict = {old: new for old, new in auto_map.items() if old in df.columns}
            if rename_dict:
                df = df.rename(columns=rename_dict)

            # parse dates
            for c in ["Apply_Date", "Stage_Screen", "Stage_Interview", "Stage_Offer", "Stage_Join"]:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")

            # Offer_Accepted_Flag to 0/1 if text
            if "Offer_Accepted_Flag" in df.columns:
                df["Offer_Accepted_Flag"] = df["Offer_Accepted_Flag"].astype(str).str.lower().map(
                    {
                        "accepted": 1, "yes": 1, "1": 1, "joined": 1, "join": 1,
                        "declined": 0, "rejected": 0, "no": 0, "0": 0
                    }
                ).fillna(0).astype(int)

            # derive Total_Time_to_Hire_Days if join is present
            if "Apply_Date" in df.columns and "Stage_Join" in df.columns:
                df["Total_Time_to_Hire_Days"] = (df["Stage_Join"] - df["Apply_Date"]).dt.days

            # derive Current_Stage if missing
            if "Current_Stage" not in df.columns:
                def get_stage(row):
                    if "Stage_Join" in row and pd.notna(row.get("Stage_Join")):
                        return "Join"
                    if "Stage_Offer" in row and pd.notna(row.get("Stage_Offer")):
                        return "Offer"
                    if "Stage_Interview" in row and pd.notna(row.get("Stage_Interview")):
                        return "Interview"
                    if "Stage_Screen" in row and pd.notna(row.get("Stage_Screen")):
                        return "Screen"
                    if "Apply_Date" in row and pd.notna(row.get("Apply_Date")):
                        return "Apply"
                    return "Apply"
                df["Current_Stage"] = df.apply(get_stage, axis=1)

            st.success("Default dataset loaded from GitHub.")
            st.dataframe(df.head(8), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load default dataset: {e}")
            st.stop()

    elif mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"], key="upload_simple")
        if uploaded:
            try:
                df = read_csv_safe(uploaded)
                df = canonicalize_columns(df, EXPECTED_COLS)
                st.success("File uploaded.")
                st.dataframe(df.head(8), use_container_width=True)
            except Exception as e:
                st.error("Failed to read uploaded CSV: " + str(e))
                st.stop()
        else:
            st.stop()

    else:
        uploaded = st.file_uploader("Upload CSV to map columns", type=["csv"], key="upload_map")
        if uploaded:
            try:
                raw = read_csv_safe(uploaded)
            except Exception as e:
                st.error("Failed to read uploaded CSV: " + str(e))
                st.stop()

            st.markdown("Preview (first 8 rows):")
            st.dataframe(raw.head(8), use_container_width=True)

            st.markdown("Map your CSV columns to the expected funnel schema (map at least the important ones).")
            mapping = {}
            cols_list = list(raw.columns)
            for key in EXPECTED_COLS:
                mapping[key] = st.selectbox(f"Map → {key}", ["-- Skip --"] + cols_list, key=f"map_{key}")

            if st.button("Apply mapping and continue", key="apply_map_btn"):
                rename_map = {v: k for k, v in mapping.items() if v != "-- Skip --"}
                if not rename_map:
                    st.error("You must map at least one column.")
                    st.stop()
                df = raw.rename(columns=rename_map)
                df = canonicalize_columns(df, EXPECTED_COLS)
                st.success("Mapping applied.")
                st.dataframe(df.head(8), use_container_width=True)
        else:
            st.stop()

    if df is None:
        st.stop()

    # basic cleanup
    df.columns = [str(c).strip() for c in df.columns]

    # ensure Apply_Date-like column
    if "Apply_Date" in df.columns:
        df = safe_to_datetime(df, "Apply_Date")

    # numeric conversions
    for c in ["Total_Time_to_Hire_Days", "Screen_Score", "Resume_Score",
              "Days_in_Stage", "Candidate_Response_Time_Hrs"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    st.markdown("### Column diagnostics")
    diag = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "non_null_count": [int(df[c].notna().sum()) for c in df.columns]
    })
    st.dataframe(diag, use_container_width=True)

    download_df(df.head(1000), "raw_hiring_data_sample.csv", "Download raw (1000 rows)", key="raw_sample")

    # =====================================================
    # STEP 2 — FILTERS & EDA
    # =====================================================
    st.markdown("---")
    st.markdown("### Step 2 — Filters & EDA")

    # Date column detection
    date_col = None
    # prefer Apply_Date
    if "Apply_Date" in df.columns:
        date_col = "Apply_Date"
    else:
        # fallback: first datetime
        for c in df.columns:
            if np.issubdtype(df[c].dtype, np.datetime64):
                date_col = c
                break

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        min_dt = df[date_col].min()
        max_dt = df[date_col].max()
        if pd.isna(min_dt) or pd.isna(max_dt):
            min_dt = max_dt = None
    else:
        min_dt = max_dt = None

    if min_dt is not None and max_dt is not None:
        date_range = st.slider(
            "Apply date range",
            min_value=min_dt.to_pydatetime(),
            max_value=max_dt.to_pydatetime(),
            value=(min_dt.to_pydatetime(), max_dt.to_pydatetime()),
            format="YYYY-MM-DD"
        )
    else:
        date_range = None

    # filter columns
    # guess role, source, JD, channel columns
    role_col = None
    source_col = None
    jd_col = None
    channel_col = None

    for c in df.columns:
        cname = c.lower()
        if role_col is None and cname in ["role", "job_role", "position"]:
            role_col = c
        if source_col is None and "source" in cname:
            source_col = c
        if jd_col is None and ("jd" in cname or "variant" in cname):
            jd_col = c
        if channel_col is None and "channel" in cname:
            channel_col = c

    f1, f2, f3, f4 = st.columns(4)
    if role_col:
        roles = sorted(df[role_col].dropna().unique().tolist())
        sel_roles = f1.multiselect("Role", roles, default=roles[:5] if len(roles) > 5 else roles)
    else:
        sel_roles = []
    if source_col:
        srcs = sorted(df[source_col].dropna().unique().tolist())
        sel_sources = f2.multiselect("Source", srcs, default=srcs[:5] if len(srcs) > 5 else srcs)
    else:
        sel_sources = []
    if jd_col:
        jds = sorted(df[jd_col].dropna().unique().tolist())
        sel_jds = f3.multiselect("JD Variant", jds, default=jds[:3] if len(jds) > 3 else jds)
    else:
        sel_jds = []
    if channel_col:
        chs = sorted(df[channel_col].dropna().unique().tolist())
        sel_chs = f4.multiselect("Channel", chs, default=chs[:3] if len(chs) > 3 else chs)
    else:
        sel_chs = []

    # apply filters
    filt = df.copy()
    if date_col and date_range is not None:
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filt = filt[(filt[date_col] >= start_dt) & (filt[date_col] <= end_dt)]

    if role_col and sel_roles:
        filt = filt[filt[role_col].isin(sel_roles)]
    if source_col and sel_sources:
        filt = filt[filt[source_col].isin(sel_sources)]
    if jd_col and sel_jds:
        filt = filt[filt[jd_col].isin(sel_jds)]
    if channel_col and sel_chs:
        filt = filt[filt[channel_col].isin(sel_chs)]

    st.markdown("#### Filtered data preview")
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(1000), "filtered_hiring_funnel.csv", "Download filtered (1000 rows)", key="dl_filtered")

    # =====================================================
    # FUNNEL METRICS
    # =====================================================
    st.markdown("### Funnel Metrics")

    # compute stage counts
    def infer_stage_counts(df_):
        # if Current_Stage present
        if "Current_Stage" in df_.columns:
            return df_["Current_Stage"].value_counts().to_dict()
        counts = {}
        # fallback: presence of stage timestamps
        counts["Apply"] = len(df_)
        for stage, col in [
            ("Screen", "Stage_Screen"),
            ("Interview", "Stage_Interview"),
            ("Offer", "Stage_Offer"),
            ("Join", "Stage_Join"),
        ]:
            if col in df_.columns:
                counts[stage] = int(df_[col].notna().sum())
            else:
                counts[stage] = 0
        return counts

    stage_counts = infer_stage_counts(filt)
    funnel_order = ["Apply", "Screen", "Interview", "Offer", "Join"]
    funnel_df = pd.DataFrame({
        "Stage": funnel_order,
        "Count": [stage_counts.get(s, 0) for s in funnel_order]
    })

    # funnel chart
    fig_funnel = px.bar(
        funnel_df, x="Stage", y="Count", text="Count",
        title="Applicants by Funnel Stage"
    )
    fig_funnel.update_traces(marker_color="#064b86")
    fig_funnel.update_layout(title_x=0.02)
    st.plotly_chart(fig_funnel, use_container_width=True)

    # conversions
    st.markdown("#### Stage-to-Stage Conversion")
    conv_rows = []
    prev_stage = None
    prev_cnt = None
    for s in funnel_order:
        cnt = funnel_df.loc[funnel_df["Stage"] == s, "Count"].values[0]
        if prev_stage is not None:
            rate = cnt / prev_cnt if prev_cnt and prev_cnt > 0 else 0
            conv_rows.append({
                "From": prev_stage,
                "To": s,
                "From_Count": prev_cnt,
                "To_Count": cnt,
                "Conv_Rate": rate
            })
        prev_stage = s
        prev_cnt = cnt

    conv_df = pd.DataFrame(conv_rows)
    if not conv_df.empty:
        conv_df["Conv_Rate_%"] = (conv_df["Conv_Rate"] * 100).round(2)
        st.dataframe(conv_df[["From", "To", "From_Count", "To_Count", "Conv_Rate_%"]], use_container_width=True)

        fig_conv = px.bar(
            conv_df,
            x="From",
            y="Conv_Rate",
            text=conv_df["Conv_Rate"].apply(lambda x: f"{x*100:.1f}%"),
            title="Stage-to-Stage Conversion Rate"
        )
        fig_conv.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_conv, use_container_width=True)
    else:
        st.info("Not enough data to compute conversion rates.")

    # high-level KPIs from filtered data
    st.markdown("#### KPI Snapshot (Filtered)")
    total_apps = len(filt)
    screen_cnt = stage_counts.get("Screen", 0)
    intr_cnt = stage_counts.get("Interview", 0)
    offer_cnt = stage_counts.get("Offer", 0)
    join_cnt = stage_counts.get("Join", 0)
    ttH = filt["Total_Time_to_Hire_Days"].mean() if "Total_Time_to_Hire_Days" in filt.columns else None

    def pct(num, den):
        return (num/den*100.0) if den and den > 0 else 0.0

    k_apps = total_apps
    k_screen = pct(screen_cnt, total_apps)
    k_intr = pct(intr_cnt, total_apps)
    k_offer = pct(offer_cnt, total_apps)
    k_tth = ttH if ttH is not None and not math.isnan(ttH) else None

    st.markdown(
        f"""
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-label">Applicants</div>
                <div class="kpi-value">{k_apps:,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Screen Pass %</div>
                <div class="kpi-value">{k_screen:.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Interview Rate</div>
                <div class="kpi-value">{k_intr:.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Offer Rate</div>
                <div class="kpi-value">{pct(offer_cnt, total_apps):.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Avg Time-to-Hire</div>
                <div class="kpi-value">{k_tth:.1f} days</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # CORE CHARTS
    # =====================================================
    st.markdown("### Visualisations")

    # Applications over time
    if date_col and date_col in filt.columns:
        apps_ts = filt.set_index(date_col).resample("D").size().reset_index(name="Applications")
        if not apps_ts.empty:
            fig_apps = px.line(apps_ts, x=date_col, y="Applications", title="Daily Applications (filtered)", markers=True)
            st.plotly_chart(fig_apps, use_container_width=True)

    # Source vs current stage
    st.markdown("#### Source vs Current Stage")
    if source_col and "Current_Stage" in filt.columns:
        src_stage = filt.groupby([source_col, "Current_Stage"]).size().reset_index(name="count")
        fig_src = px.bar(
            src_stage,
            x=source_col,
            y="count",
            color="Current_Stage",
            title="Source vs Current Stage",
            text="count"
        )
        st.plotly_chart(fig_src, use_container_width=True)
    else:
        st.info("Source or Current_Stage column missing for this view.")

    # Time-to-hire distribution
    st.markdown("#### Time-to-Hire Distribution")
    if "Total_Time_to_Hire_Days" in filt.columns:
        fig_hist = px.histogram(filt, x="Total_Time_to_Hire_Days", nbins=30, title="Total Time-to-Hire (days)")
        st.plotly_chart(fig_hist, use_container_width=True)
        if role_col:
            fig_box = px.box(filt, x=role_col, y="Total_Time_to_Hire_Days", title="Time-to-Hire by Role")
            st.plotly_chart(fig_box, use_container_width=True)

    # Cohort overview
    st.markdown("#### Cohort Overview (Monthly)")
    if date_col:
        cohort = filt.copy()
        cohort["Month"] = cohort[date_col].dt.to_period("M").dt.to_timestamp()
        if "Applicant_ID" not in cohort.columns:
            cohort["Applicant_ID"] = range(1, len(cohort) + 1)
        if "Stage_Join" not in cohort.columns:
            cohort["Stage_Join"] = pd.NaT

        grp = cohort.groupby("Month").agg(
            Applicants=("Applicant_ID", "count"),
            Joins=("Stage_Join", lambda x: x.notna().sum())
        ).reset_index()
        if not grp.empty:
            grp["Join_Rate_%"] = grp.apply(lambda r: (r["Joins"] / r["Applicants"] * 100.0) if r["Applicants"] else 0.0, axis=1)
            st.dataframe(grp, use_container_width=True)
            download_df(grp, "hiring_cohort_summary.csv", "Download cohort summary", key="dl_cohort")
        else:
            st.info("Not enough data for cohort summary.")

    # =====================================================
    # STEP 3 — AUTO ML & DROP-OFF PREDICTION
    # =====================================================
    st.markdown("---")
    st.markdown("### Step 3 — AutoML: Drop-off Risk Prediction")

    # Build binary target: Dropped (1) vs Joined (0)
    target = None
    if "Stage_Join" in filt.columns:
        dropped_flag = filt["Stage_Join"].isna().astype(int)
        target = dropped_flag
    elif "Offer_Accepted_Flag" in filt.columns:
        # treat non-accepted as drop
        target = (1 - filt["Offer_Accepted_Flag"].astype(int)).clip(0, 1)
    else:
        target = None

    if target is None or target.nunique() < 2 or len(filt) < 80:
        st.info("Not enough labelled data for drop-off prediction (need Stage_Join or Offer_Accepted_Flag with 0/1 and ≥80 rows).")
    else:
        # Features
        feature_candidates = [
            "Screen_Score",
            "Resume_Score",
            "Days_in_Stage",
            "Total_Time_to_Hire_Days",
            "Candidate_Response_Time_Hrs"
        ]
        cat_candidates = []
        if source_col: cat_candidates.append(source_col)
        if role_col: cat_candidates.append(role_col)
        if jd_col: cat_candidates.append(jd_col)
        if channel_col: cat_candidates.append(channel_col)
        if "Current_Stage" in filt.columns: cat_candidates.append("Current_Stage")

        used_numeric = [c for c in feature_candidates if c in filt.columns]
        used_cats = [c for c in cat_candidates if c in filt.columns]

        if len(used_numeric) == 0 and len(used_cats) == 0:
            st.info("No suitable feature columns found for ML.")
        else:
            base_X = pd.DataFrame(index=filt.index)

            for c in used_numeric:
                base_X[c] = pd.to_numeric(filt[c], errors="coerce")

            for c in used_cats:
                dummies = pd.get_dummies(filt[c].astype(str), prefix=c, drop_first=True)
                base_X = pd.concat([base_X, dummies], axis=1)

            X = base_X.fillna(0)
            y = target.values

            # Train/test split
            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                X, y, X.index, test_size=0.25, random_state=42, stratify=y if y.sum() > 0 else None
            )

            # Preprocess: impute + scale
            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()

            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)

            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_test_scaled = scaler.transform(X_test_imp)

            model_choice = st.radio(
                "Select AutoML model(s)",
                ["Logistic Regression", "Random Forest", "Both (compare)"],
                horizontal=True,
            )

            results = []

            # Logistic Regression
            if model_choice in ["Logistic Regression", "Both (compare)"]:
                log_reg = LogisticRegression(max_iter=1000)
                log_reg.fit(X_train_scaled, y_train)
                prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]
                pred_lr = (prob_lr >= 0.5).astype(int)
                acc_lr = accuracy_score(y_test, pred_lr)
                auc_lr = roc_auc_score(y_test, prob_lr) if len(np.unique(y_test)) == 2 else np.nan
                results.append(("Logistic Regression", acc_lr, auc_lr, prob_lr, log_reg))

            # RandomForest
            if model_choice in ["Random Forest", "Both (compare)"]:
                rf = RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    max_depth=None,
                    min_samples_split=4,
                    min_samples_leaf=2,
                )
                rf.fit(X_train_imp, y_train)  # RF can work on unscaled
                prob_rf = rf.predict_proba(X_test_imp)[:, 1]
                pred_rf = (prob_rf >= 0.5).astype(int)
                acc_rf = accuracy_score(y_test, pred_rf)
                auc_rf = roc_auc_score(y_test, prob_rf) if len(np.unique(y_test)) == 2 else np.nan
                results.append(("Random Forest", acc_rf, auc_rf, prob_rf, rf))

            # Show comparison
            st.markdown("#### Model performance (test set)")
            perf_rows = []
            for name, acc_, auc_, _, _ in results:
                perf_rows.append({
                    "Model": name,
                    "Accuracy": round(acc_, 3),
                    "ROC AUC": round(auc_, 3) if not math.isnan(auc_) else np.nan
                })
            if perf_rows:
                st.dataframe(pd.DataFrame(perf_rows), use_container_width=True)

            # choose best model by AUC then accuracy
            best_name, best_model, best_probs = None, None, None
            if results:
                best = sorted(
                    results,
                    key=lambda r: (0 if math.isnan(r[2]) else r[2], r[1]),
                    reverse=True
                )[0]
                best_name, best_acc, best_auc, best_probs, best_model = best
                st.success(f"Best model (by AUC then Accuracy): {best_name} | Acc: {best_acc:.3f}, AUC: {best_auc:.3f}")

            # Build risk table
            if best_model is not None:
                full_imp = imputer.transform(X)
                if isinstance(best_model, LogisticRegression):
                    full_scale = scaler.transform(full_imp)
                    full_probs = best_model.predict_proba(full_scale)[:, 1]
                elif isinstance(best_model, RandomForestClassifier):
                    full_probs = best_model.predict_proba(full_imp)[:, 1]
                else:
                    full_probs = best_probs

                risk_df = pd.DataFrame(index=filt.index)
                if "Applicant_ID" in filt.columns:
                    risk_df["Applicant_ID"] = filt["Applicant_ID"]
                if role_col:
                    risk_df[role_col] = filt[role_col]
                if source_col:
                    risk_df[source_col] = filt[source_col]
                risk_df["Dropoff_Risk_Prob"] = full_probs
                risk_df["Dropoff_Risk_Bucket"] = pd.cut(
                    full_probs,
                    bins=[0, 0.3, 0.6, 1.0],
                    labels=["Low", "Medium", "High"],
                    include_lowest=True
                )

                st.markdown("#### Candidate-level Drop-off Risk (top 100 by risk)")
                top_risk = risk_df.sort_values("Dropoff_Risk_Prob", ascending=False).head(100)
                st.dataframe(top_risk, use_container_width=True)
                download_df(risk_df, "candidate_dropoff_risk_scores.csv", "Download full risk scores", key="dl_risk")

                # Feature importance for RF
                if isinstance(best_model, RandomForestClassifier):
                    feat_imp = pd.DataFrame({
                        "Feature": X.columns,
                        "Importance": best_model.feature_importances_
                    }).sort_values("Importance", ascending=False).head(20)
                    st.markdown("#### Top features (Random Forest importance)")
                    fig_imp = px.bar(feat_imp, x="Importance", y="Feature", orientation="h")
                    st.plotly_chart(fig_imp, use_container_width=True)

    # =====================================================
    # STEP 4 — SCHEDULING SIMULATOR
    # =====================================================
    st.markdown("---")
    st.markdown("### Step 4 — Scheduling Simulator (Time-to-Fill)")

    apply_cnt = funnel_df.loc[funnel_df["Stage"] == "Apply", "Count"].iloc[0] if not funnel_df.empty else 0
    join_cnt = funnel_df.loc[funnel_df["Stage"] == "Join", "Count"].iloc[0] if not funnel_df.empty else 0
    base_conv = join_cnt / apply_cnt if apply_cnt else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        daily_apps = st.number_input("Expected applicants per day", min_value=1, value=20, step=1)
    with c2:
        target_joins = st.number_input("Target hires (joins)", min_value=1, value=50, step=1)
    with c3:
        conv_factor = st.slider("Conversion uplift factor", 0.5, 1.5, 1.0, 0.05)

    eff_conv = base_conv * conv_factor if base_conv else 0

    if eff_conv <= 0:
        st.info("Base apply→join conversion could not be computed (no data). Simulator will skip.")
    else:
        expected_days = target_joins / (daily_apps * eff_conv)
        expected_weeks = expected_days / 7
        st.markdown(
            f"""
            <div class="card card-left">
            With <b>{daily_apps}</b> applicants/day and an effective apply→join conversion of 
            <b>{eff_conv*100:.1f}%</b>, you need approximately 
            <b>{expected_days:.1f} days</b> (~<b>{expected_weeks:.1f} weeks</b>) to close 
            <b>{target_joins}</b> hires for the current funnel behaviour.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =====================================================
    # STEP 5 — ANOMALY DETECTION & CLUSTERING
    # =====================================================
    st.markdown("---")
    st.markdown("### Step 5 — Anomaly Detection & Clustering")

    numeric_for_anomaly = [c for c in ["Days_in_Stage", "Total_Time_to_Hire_Days", "Screen_Score", "Resume_Score", "Candidate_Response_Time_Hrs"] if c in filt.columns]
    if len(numeric_for_anomaly) < 2 or len(filt) < 50:
        st.info("Not enough numeric fields or rows for anomaly detection / clustering (need ≥2 numeric features and ≥50 rows).")
    else:
        base_anom = filt[numeric_for_anomaly].copy()
        base_anom = base_anom.replace([np.inf, -np.inf], np.nan).fillna(base_anom.median())

        # Isolation Forest
        iso = IsolationForest(contamination=0.05, random_state=42)
        iso.fit(base_anom.values)
        scores = iso.decision_function(base_anom.values)
        preds = iso.predict(base_anom.values)  # -1 anomaly, 1 normal

        filt["_Anomaly_Score"] = scores
        filt["_Is_Anomaly"] = np.where(preds == -1, 1, 0)

        anom_df = filt[filt["_Is_Anomaly"] == 1].copy()
        st.markdown("#### Anomalous candidates (top 50 by severity)")
        if "Applicant_ID" not in anom_df.columns:
            anom_df["Applicant_ID"] = range(1, len(anom_df) + 1)
        anom_df = anom_df.sort_values("_Anomaly_Score").head(50)
        cols_show = ["Applicant_ID", "_Anomaly_Score"] + numeric_for_anomaly
        cols_show = [c for c in cols_show if c in anom_df.columns]
        st.dataframe(anom_df[cols_show], use_container_width=True)
        download_df(anom_df[cols_show], "hiring_anomalies.csv", "Download anomalies", key="dl_anom")

        # Clustering
        st.markdown("#### Behavioural Clusters (KMeans)")
        k = st.slider("Number of clusters (K)", 2, 6, 3, 1)
        scaler_cl = StandardScaler()
        X_cl = scaler_cl.fit_transform(base_anom.values)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_cl)
        filt["_Cluster"] = labels

        cluster_summary = filt.groupby("_Cluster").agg(
            Applicants=("Applicant_ID" if "Applicant_ID" in filt.columns else filt.columns[0], "count"),
            Avg_Days_in_Stage=("Days_in_Stage", "mean") if "Days_in_Stage" in filt.columns else ("_Cluster", "size"),
            Avg_Time_to_Hire=("Total_Time_to_Hire_Days", "mean") if "Total_Time_to_Hire_Days" in filt.columns else ("_Cluster", "size"),
            Avg_Screen_Score=("Screen_Score", "mean") if "Screen_Score" in filt.columns else ("_Cluster", "size"),
            Drop_Rate=("Stage_Join", lambda x: x.isna().mean()) if "Stage_Join" in filt.columns else ("_Cluster", lambda x: 0.0)
        ).reset_index()
        st.dataframe(cluster_summary, use_container_width=True)
        download_df(cluster_summary, "hiring_cluster_summary.csv", "Download cluster summary", key="dl_cluster")

    # Final statement
    st.markdown(
        """
        <div class="card card-left">
        Done. Use funnel metrics, ML-based risk scores, anomalies, and behavioural clusters
        to focus recruiter effort where it actually moves the needle.
        </div>
        """,
        unsafe_allow_html=True,
    )
