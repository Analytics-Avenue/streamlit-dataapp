# app_hiring_funnel.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO, StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import warnings
import math
warnings.filterwarnings("ignore")

# -------------------------
# CONFIG & HEADER (left aligned title)
# -------------------------
st.set_page_config(page_title="Hiring Funnel Drop-Off Analysis", layout="wide", page_icon="🧭")
# hide default Streamlit menu and footer if you want (comment if not desired)
st.markdown("""<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>""", unsafe_allow_html=True)

# CSS - hover glow and left aligned title
st.markdown("""
<style>
.container-left { display: flex; align-items: center; gap:12px; }
.app-title { font-size: 30px; font-weight: 700; color:#064b86; text-align: left; margin:0; padding:0; }
.app-sub { font-size: 14px; color:#444; margin:0; padding:0 0 10px 0; text-align: left; }
.card { padding: 18px; border-radius: 12px; background: #ffffff; border: 1px solid #e6eef6; box-shadow: 0 2px 6px rgba(0,0,0,0.04); transition: all .18s ease; }
.card:hover { transform: translateY(-6px); box-shadow: 0 8px 30px rgba(6,75,134,0.18); border-color:#0a4f8a; }
.kpi { padding: 20px; border-radius: 12px; text-align:center; font-weight:700; color:#064b86; font-size:20px; background:white; border:1px solid #e6eef6; transition:all .18s ease; }
.kpi:hover { transform: translateY(-6px); box-shadow: 0 8px 30px rgba(6,75,134,0.14); border-color:#064b86; }
.left-align-card { text-align: left; }
.small { font-size:13px; color:#666; }
.badge {display:inline-block; padding:6px 10px; border-radius:8px; background:#eef6ff; color:#064b86; font-weight:600; margin-right:6px;}
</style>
""", unsafe_allow_html=True)

# Header area: logo + left aligned title + subtitle
LOGO_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
col_logo, col_title = st.columns([1, 8], gap="small")
with col_logo:
    st.image(LOGO_URL, width=64)
with col_title:
    st.markdown('<div class="container-left"><div><h1 class="app-title">Hiring Funnel Drop-Off Analysis</h1>'
                '<div class="app-sub">Track apply → screen → interview → offer → join funnel, find conversion bottlenecks, and improve hiring velocity.</div>'
                '</div></div>', unsafe_allow_html=True)

# -------------------------
# Utility functions
# -------------------------
REQUIRED_COLS = [
    "Applicant_ID", "Apply_Date", "Channel", "Position", "Stage", "Stage_Date",
    "Screened_Flag", "Interviewed_Flag", "Offered_Flag", "Joined_Flag",
    "Time_To_Hire_Days", "Resume_Score", "Recruiter_Response_Hours"
]

def read_csv_safe(src):
    """Read CSV from URL or file-like object. Make duplicate columns unique by suffixing."""
    df = pd.read_csv(src)
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

def download_df(df, filename, label="Download CSV"):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")

def try_parse_dates(df, col):
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        except:
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
    return df

def map_common_columns(df):
    """Attempt to map likely column names to canonical required names."""
    mapping = {}
    lower_map = {c.lower(): c for c in df.columns}
    # map common synonyms
    synonyms = {
        "applicant_id": ["applicant_id","id","candidate_id","candidate"],
        "apply_date": ["apply_date","application_date","applied_on","applied_date","created_at"],
        "channel": ["channel","source","referral_source"],
        "position": ["position","job","role","job_title"],
        "stage": ["stage","current_stage","status"],
        "stage_date": ["stage_date","status_date","updated_at"],
        "screened_flag": ["screened_flag","screened","screened?"],
        "interviewed_flag": ["interviewed_flag","interviewed","interviewed?"],
        "offered_flag": ["offered_flag","offered","offered?"],
        "joined_flag": ["joined_flag","joined","joined?","hired"],
        "time_to_hire_days": ["time_to_hire_days","time_to_hire","days_to_hire","time_to_offer"],
        "resume_score": ["resume_score","screen_score","profile_score"],
        "recruiter_response_hours": ["recruiter_response_hours","response_hours","response_time_hours"]
    }
    for canon, cands in synonyms.items():
        for cand in cands:
            for col in df.columns:
                if col.lower() == cand:
                    mapping[col] = canon.replace("_", " ").title().replace(" ", "_")
                    break
                # also try substring
                if cand in col.lower():
                    mapping[col] = canon.replace("_", " ").title().replace(" ", "_")
                    break
            if any(v==canon.replace("_", " ").title().replace(" ", "_") for v in mapping.values()):
                break
    # apply mapping if any
    if mapping:
        df = df.rename(columns=mapping)
    # canonicalize further by lower-case keys to required style
    # Finally, try to coerce to exact required names if matches exist
    col_map = {}
    for req in REQUIRED_COLS:
        for col in df.columns:
            if col.lower() == req.lower():
                col_map[col] = req
                break
    if col_map:
        df = df.rename(columns=col_map)
    return df

def safe_numeric(df, col, fill_median=True):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if fill_median:
            if df[col].notna().any():
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)
    return df

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["Overview", "Application"])

# -------------------------
# Overview Tab (left-aligned content inside cards)
# -------------------------
with tabs[0]:
    st.markdown('<div class="card left-align-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin:0;'>Purpose</h3>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Detect drop-offs across Apply → Screen → Interview → Offer → Join, and improve conversion with prioritized interventions.</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    left, right = st.columns([1,1], gap="large")
    with left:
        st.markdown('<div class="card left-align-card">', unsafe_allow_html=True)
        st.markdown("### Capabilities", unsafe_allow_html=True)
        st.markdown("""
        • Funnel conversion analysis and bottleneck identification<br>
        • Stage-level conversion rates and time-in-stage metrics<br>
        • A/B analysis on JD and channel performance<br>
        • Predictive drop-off models and candidate propensity scoring<br>
        • Automated insights and prioritized candidate lists
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card left-align-card">', unsafe_allow_html=True)
        st.markdown("### Business Impact", unsafe_allow_html=True)
        st.markdown("""
        • Reduced time-to-hire and improved offer-to-join ratios<br>
        • Lower cost-per-hire through better channel allocation<br>
        • Improved candidate experience with quicker recruiter responses<br>
        • Reduced ghosting and improved acceptance rates
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    st.markdown("## KPIs")
    kcols = st.columns(5)
    k_labels = ["Applicants (Period)", "Apply→Interview %", "Interview→Offer %", "Offer→Join %", "Median Time-to-Hire (days)"]
    for col, lbl in zip(kcols, k_labels):
        col.markdown(f"<div class='kpi'>{lbl}</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### Who should use & How to use", unsafe_allow_html=True)
    st.markdown('<div class="card left-align-card">', unsafe_allow_html=True)
    st.markdown("""
    • Talent Acquisition Leads, Hiring Managers, Recruiters, People Analytics teams.<br><br>
    <b>How</b>: Upload hiring events or use default repo CSV. Filter by date, position, or channel. Inspect funnel charts, run predictive drop-off scoring, export prioritized candidate lists and insights.
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Application Tab
# -------------------------
with tabs[1]:
    st.markdown("<div style='display:flex; justify-content:space-between; align-items:center;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin:0; text-align:left;'>Application</h3>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Step 1 — Load dataset (three options)", unsafe_allow_html=True)
    mode = st.radio("Dataset option:", ["Default dataset (set DEFAULT_URL inside code)", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)

    df = None
    DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/hr/hiring_funnel_data.csv"  # <<--- leave blank or paste your raw GitHub CSV raw URL here if you want default mode to work

    try:
        if mode == "Default dataset (set DEFAULT_URL inside code)":
            if DEFAULT_URL:
                df = read_csv_safe(DEFAULT_URL)
                st.success("Default dataset loaded from DEFAULT_URL.")
            else:
                st.info("DEFAULT_URL is empty. Either set DEFAULT_URL variable in the script or use Upload CSV options.")
        elif mode == "Upload CSV":
            uploaded = st.file_uploader("Upload your hiring funnel CSV", type=["csv"], key="upload1")
            if uploaded is not None:
                df = read_csv_safe(uploaded)
                st.success("File uploaded.")
        else:
            uploaded2 = st.file_uploader("Upload for manual column mapping", type=["csv"], key="upload2")
            if uploaded2 is not None:
                raw = read_csv_safe(uploaded2)
                st.write("Preview (first 5 rows):")
                st.dataframe(raw.head())
                st.markdown("Map your columns to required fields (map only those you have).")
                mapping = {}
                cols_list = list(raw.columns)
                for req in REQUIRED_COLS:
                    mapping[req] = st.selectbox(f"Map → {req}", ["-- Skip --"] + cols_list, key=f"map_{req}")
                if st.button("Apply mapping and load"):
                    rename_map = {}
                    for req, sel in mapping.items():
                        if sel != "-- Skip --":
                            rename_map[sel] = req
                    df = raw.rename(columns=rename_map)
                    st.success("Mapping applied.")
    except Exception as e:
        st.error("Failed to load dataset: " + str(e))

    if df is None:
        st.stop()

    # basic cleaning & mapping attempts
    df.columns = [str(c).strip() for c in df.columns]
    df = map_common_columns(df)

    # Try common candidate id and date conversions
    # prefer Applicant_ID or fallback
    if "Applicant_ID" not in df.columns:
        # try typical id columns
        possible_id_cols = [c for c in df.columns if 'id' in c.lower() and len(df[c].astype(str).unique())>10]
        if possible_id_cols:
            df = df.rename(columns={possible_id_cols[0]: "Applicant_ID"})

    # Date parsing: Apply_Date and Stage_Date if present
    df = try_parse_dates(df, "Apply_Date")
    df = try_parse_dates(df, "Stage_Date")

    # Normalize stage flags if present as text
    for flag in ["Screened_Flag","Interviewed_Flag","Offered_Flag","Joined_Flag"]:
        if flag in df.columns and df[flag].dtype == object:
            df[flag] = df[flag].str.strip().str.lower().map({"yes":1,"y":1,"true":1,"1":1,"no":0,"n":0,"false":0,"0":0}).fillna(0).astype(int)

    # numeric sanitization
    for col in ["Time_To_Hire_Days","Resume_Score","Recruiter_Response_Hours"]:
        df = safe_numeric(df, col, fill_median=True)

    # Basic derived columns if not present
    if "Apply_Date" in df.columns and "Stage_Date" in df.columns:
        try:
            df["Days_To_Stage"] = (df["Stage_Date"] - df["Apply_Date"]).dt.total_seconds() / 86400.0
        except:
            df["Days_To_Stage"] = np.nan

    # Provide a preview and download sample
    st.markdown("### Dataset preview")
    st.dataframe(df.head(10), use_container_width=True)
    download_df(df.head(200), "sample_preview.csv", label="Download preview (CSV)")

    # -------------------------
    # Filters: Date range (date_input), Position, Channel
    # -------------------------
    st.markdown("### Filters")
    fcol1, fcol2, fcol3, fcol4 = st.columns([2,2,2,1])

    # Date filter uses date_input to avoid slider type issues
    if "Apply_Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Apply_Date"]):
        min_date = df["Apply_Date"].min().date()
        max_date = df["Apply_Date"].max().date()
        date_range = fcol1.date_input("Apply Date range", value=(min_date, max_date))
    else:
        date_range = None

    positions = sorted(df["Position"].dropna().unique()) if "Position" in df.columns else []
    channels = sorted(df["Channel"].dropna().unique()) if "Channel" in df.columns else []

    sel_pos = fcol2.multiselect("Position", options=positions, default=positions if positions else [])
    sel_channel = fcol3.multiselect("Channel", options=channels, default=channels if channels else [])

    # apply filters
    filt = df.copy()
    if date_range and "Apply_Date" in filt.columns:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filt = filt[(filt["Apply_Date"] >= start_date) & (filt["Apply_Date"] <= end_date)]
    if sel_pos:
        if "Position" in filt.columns:
            filt = filt[filt["Position"].isin(sel_pos)]
    if sel_channel:
        if "Channel" in filt.columns:
            filt = filt[filt["Channel"].isin(sel_channel)]

    st.markdown("Filtered preview (first 10 rows)")
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(200), "filtered_funnel_preview.csv", label="Download filtered preview")

    # -------------------------
    # Funnel EDA & Charts (many charts, varied)
    # -------------------------
    st.markdown("## Exploratory Data Analysis (EDA)")

    # Basic counts and funnel conversion
    st.markdown("### Funnel Overview")
    # compute counts per stage
    try:
        # If flags present compute funnel counts by stage
        total_applicants = len(filt)
        screened = int(filt["Screened_Flag"].sum()) if "Screened_Flag" in filt.columns else np.nan
        interviewed = int(filt["Interviewed_Flag"].sum()) if "Interviewed_Flag" in filt.columns else np.nan
        offered = int(filt["Offered_Flag"].sum()) if "Offered_Flag" in filt.columns else np.nan
        joined = int(filt["Joined_Flag"].sum()) if "Joined_Flag" in filt.columns else np.nan
        funnel_df = pd.DataFrame({
            "Stage":["Applied","Screened","Interviewed","Offered","Joined"],
            "Count":[total_applicants, screened, interviewed, offered, joined]
        })
        fig_funnel = px.funnel(funnel_df, x='Count', y='Stage', title="Hiring Funnel (counts)")
        st.plotly_chart(fig_funnel, use_container_width=True)
    except Exception as e:
        st.info("Could not build funnel chart: " + str(e))

    # Conversion rates by stage as bar chart
    st.markdown("### Stage Conversion Rates")
    try:
        conv = {}
        if not np.isnan(screened):
            conv["Apply->Screen"] = screened / total_applicants if total_applicants>0 else np.nan
        if not np.isnan(interviewed):
            conv["Apply->Interview"] = interviewed / total_applicants if total_applicants>0 else np.nan
        if not np.isnan(offered):
            conv["Interview->Offer"] = offered / max(1, interviewed) if interviewed else np.nan
        if not np.isnan(joined):
            conv["Offer->Join"] = joined / max(1, offered) if offered else np.nan
        conv_df = pd.DataFrame({"Conversion": conv}).transpose().reset_index().rename(columns={"index":"Metric"})
        fig_conv = px.bar(conv_df, x="Metric", y="Conversion", text=conv_df["Conversion"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "N/A"), title="Stage conversion rates")
        fig_conv.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_conv, use_container_width=True)
    except Exception as e:
        st.info("Could not build conversion chart: " + str(e))

    # Time-to-hire distribution
    st.markdown("### Time to Hire distribution")
    if "Time_To_Hire_Days" in filt.columns and filt["Time_To_Hire_Days"].notna().any():
        fig_tt = px.histogram(filt, x="Time_To_Hire_Days", nbins=30, title="Distribution of Time-to-hire (days)")
        st.plotly_chart(fig_tt, use_container_width=True)
    else:
        st.info("Time_To_Hire_Days column missing or empty — skipping histogram.")

    # Applications by Channel over time (area chart)
    st.markdown("### Applications by Channel (time series)")
    if "Apply_Date" in filt.columns and "Channel" in filt.columns:
        ts = filt.groupby([pd.Grouper(key="Apply_Date", freq="W"), "Channel"]).size().reset_index(name="count")
        fig_ch = px.area(ts, x="Apply_Date", y="count", color="Channel", title="Weekly applications by channel")
        st.plotly_chart(fig_ch, use_container_width=True)
    else:
        st.info("Apply_Date or Channel missing — skipping time series.")

    # Top positions by drop-off rate (bar)
    st.markdown("### Top Positions by Drop-off (Apply->Interview)")
    if "Position" in filt.columns and "Interviewed_Flag" in filt.columns:
        pos = filt.groupby("Position").agg(total=("Applicant_ID","count"), interviewed=("Interviewed_Flag","sum")).reset_index()
        pos["Apply_Interview_Rate"] = pos["interviewed"] / pos["total"]
        pos = pos.sort_values("Apply_Interview_Rate")
        fig_pos = px.bar(pos.tail(20), x="Position", y="Apply_Interview_Rate", title="Apply->Interview rate by position (last 20)")
        fig_pos.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_pos, use_container_width=True)
    else:
        st.info("Position or Interviewed_Flag missing — skipping position drop-off chart.")

    # Recruiter response time vs join probability scatter
    st.markdown("### Recruiter response vs Join probability")
    if "Recruiter_Response_Hours" in filt.columns and "Joined_Flag" in filt.columns:
        agg = filt.groupby("Applicant_ID").agg(resp_hours=("Recruiter_Response_Hours","mean"), joined=("Joined_Flag","max")).reset_index()
        fig_sc = px.scatter(agg, x="resp_hours", y="joined", title="Avg recruiter response hours vs joined (0/1)", trendline="ols")
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("Recruiter_Response_Hours or Joined_Flag missing — skipping scatter.")

    # Heatmap: channel vs position conversion
    st.markdown("### Channel x Position — Interview rate heatmap")
    if "Channel" in filt.columns and "Position" in filt.columns and "Interviewed_Flag" in filt.columns:
        heat = filt.groupby(["Channel","Position"]).agg(total=("Applicant_ID","count"), interviewed=("Interviewed_Flag","sum")).reset_index()
        heat["Interview_Rate"] = heat["interviewed"] / heat["total"]
        pivot = heat.pivot(index="Channel", columns="Position", values="Interview_Rate").fillna(0)
        fig_heat = px.imshow(pivot, text_auto=".0%", aspect="auto", title="Interview rate (Channel x Position)")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Channel/Position/Interviewed_Flag missing — skipping heatmap.")

    # Funnel timeline: median days to each stage by position
    st.markdown("### Median days to stage by position")
    if "Days_To_Stage" in filt.columns and "Position" in filt.columns:
        med = filt.groupby("Position")["Days_To_Stage"].median().reset_index().sort_values("Days_To_Stage")
        fig_med = px.bar(med.head(20), x="Position", y="Days_To_Stage", title="Median days to stage (top 20 positions)")
        st.plotly_chart(fig_med, use_container_width=True)
    else:
        st.info("Days_To_Stage or Position missing — skipping median timeline chart.")

    # Cohort analysis: apply week cohort retention to interview stage
    st.markdown("### Weekly Cohort — Apply to Interview retention (cohort heatmap)")
    if "Apply_Date" in filt.columns and "Interviewed_Flag" in filt.columns and "Applicant_ID" in filt.columns:
        cohort = filt.copy()
        cohort["Apply_Week"] = cohort["Apply_Date"].dt.to_period("W").apply(lambda r: r.start_time)
        cohort_group = cohort.groupby(["Apply_Week", "Applicant_ID"]).agg(interviewed=("Interviewed_Flag","max")).reset_index()
        cohort_pivot = cohort_group.groupby(["Apply_Week"])["interviewed"].mean().reset_index()
        fig_cohort = px.line(cohort_pivot, x="Apply_Week", y="interviewed", title="Weekly avg Apply->Interview conversion")
        st.plotly_chart(fig_cohort, use_container_width=True)
    else:
        st.info("Need Apply_Date and Interviewed_Flag and Applicant_ID for cohort analysis.")

    # Treemap of channels and positions (volume)
    st.markdown("### Applications breakdown (treemap)")
    if "Channel" in filt.columns and "Position" in filt.columns:
        treedf = filt.groupby(["Channel","Position"]).size().reset_index(name="count")
        fig_tree = px.treemap(treedf, path=["Channel","Position"], values="count", title="Applications by Channel & Position")
        st.plotly_chart(fig_tree, use_container_width=True)
    else:
        st.info("Channel/Position missing — skipping treemap.")

    # Box plot: resume score by stage
    st.markdown("### Resume score distribution by stage")
    if "Resume_Score" in filt.columns and "Stage" in filt.columns:
        fig_box = px.box(filt, x="Stage", y="Resume_Score", title="Resume score by current stage")
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("Resume_Score or Stage missing — skipping box plot.")

    # Bar: top drop-off reasons if a reason column exists
    st.markdown("### Drop-off reasons (if present)")
    reason_cols = [c for c in filt.columns if "reason" in c.lower()]
    if reason_cols:
        reason = filt[reason_cols[0]].value_counts().reset_index().rename(columns={"index":"reason", reason_cols[0]:"count"})
        fig_reason = px.bar(reason.head(25), x="reason", y="count", title="Top drop-off reasons")
        st.plotly_chart(fig_reason, use_container_width=True)
    else:
        st.info("No explicit reason column found; skipping drop-off reasons chart.")

    # Scatter: resume_score vs recruiter_response_hours colored by final joined
    st.markdown("### Resume score vs Recruiter response (colored by join)")
    if "Resume_Score" in filt.columns and "Recruiter_Response_Hours" in filt.columns and "Joined_Flag" in filt.columns:
        fig_r = px.scatter(filt, x="Resume_Score", y="Recruiter_Response_Hours", color="Joined_Flag",
                           title="Resume score vs Response hrs (joined colored)", trendline="ols")
        st.plotly_chart(fig_r, use_container_width=True)
    else:
        st.info("Resume_Score or Recruiter_Response_Hours or Joined_Flag missing — skipping scatter.")

    # Add a compact EDA table of basic metrics
    st.markdown("### EDA summary table")
    eda_rows = []
    eda_rows.append({"Metric":"Total applicants", "Value": int(len(filt))})
    if "Screened_Flag" in filt.columns:
        eda_rows.append({"Metric":"Total screened", "Value": int(filt["Screened_Flag"].sum())})
    if "Interviewed_Flag" in filt.columns:
        eda_rows.append({"Metric":"Total interviewed", "Value": int(filt["Interviewed_Flag"].sum())})
    if "Offered_Flag" in filt.columns:
        eda_rows.append({"Metric":"Total offered", "Value": int(filt["Offered_Flag"].sum())})
    if "Joined_Flag" in filt.columns:
        eda_rows.append({"Metric":"Total joined", "Value": int(filt["Joined_Flag"].sum())})
    if "Time_To_Hire_Days" in filt.columns and filt["Time_To_Hire_Days"].notna().any():
        eda_rows.append({"Metric":"Median time-to-hire (days)", "Value": float(filt["Time_To_Hire_Days"].median())})
    eda_summary = pd.DataFrame(eda_rows)
    st.dataframe(eda_summary, use_container_width=True)
    download_df(eda_summary, "eda_summary.csv", label="Download EDA summary")

    # -------------------------
    # ML: 4 Models (classification: predict drop-off before interview or probability to join)
    # We'll build a model to predict "Joined_Flag" if present; else predict Interviewed_Flag or Screened_Flag
    # -------------------------
    st.markdown("## Machine Learning — Predict drop-off / Join probability")
    target_col = None
    if "Joined_Flag" in filt.columns and filt["Joined_Flag"].nunique()>1:
        target_col = "Joined_Flag"
    elif "Interviewed_Flag" in filt.columns and filt["Interviewed_Flag"].nunique()>1:
        target_col = "Interviewed_Flag"
    elif "Screened_Flag" in filt.columns and filt["Screened_Flag"].nunique()>1:
        target_col = "Screened_Flag"
    else:
        st.info("Not enough labeled flag columns for supervised ML (need one of Joined_Flag/Interviewed_Flag/Screened_Flag with >1 unique values).")
    if target_col:
        st.markdown(f"### Target chosen for prediction: **{target_col}**")
        # Build feature set (simple, robust)
        feature_cols = []
        # numeric features if present
        for c in ["Resume_Score", "Recruiter_Response_Hours", "Time_To_Hire_Days"]:
            if c in filt.columns:
                feature_cols.append(c)
        # encode channel and position as categorical one-hot (top N)
        if "Channel" in filt.columns:
            top_channels = filt["Channel"].value_counts().nlargest(8).index.tolist()
            for ch in top_channels:
                filt[f"channel_{ch}"] = (filt["Channel"]==ch).astype(int)
                feature_cols.append(f"channel_{ch}")
        if "Position" in filt.columns:
            top_pos = filt["Position"].value_counts().nlargest(12).index.tolist()
            for p in top_pos:
                coln = f"pos_{p}"
                filt[coln] = (filt["Position"]==p).astype(int)
                feature_cols.append(coln)

        # drop rows with NaN in features or target
        model_df = filt[[target_col] + feature_cols].dropna()
        if len(model_df) < 30:
            st.info("Not enough rows after feature selection for reliable ML (need >=30 rows).")
        else:
            X = model_df[feature_cols].values
            y = model_df[target_col].astype(int).values
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
                "Naive Bayes": GaussianNB()
            }
            results_all = {}
            for name, m in models.items():
                with st.spinner(f"Training {name}..."):
                    try:
                        m.fit(Xtr, ytr)
                        if hasattr(m, "predict_proba"):
                            probs = m.predict_proba(Xte)[:,1]
                            preds = (probs >= 0.5).astype(int)
                        else:
                            preds = m.predict(Xte)
                            # produce pseudo-prob using decision_function if present, else fallback to preds
                            try:
                                probs = m.decision_function(Xte)
                                probs = (probs - probs.min()) / (probs.max() - probs.min())
                            except:
                                probs = preds
                        # metrics
                        acc = accuracy_score(yte, preds)
                        prec = precision_score(yte, preds, zero_division=0)
                        rec = recall_score(yte, preds, zero_division=0)
                        f1 = f1_score(yte, preds, zero_division=0)
                        try:
                            auc = roc_auc_score(yte, probs)
                        except:
                            auc = math.nan
                        results_all[name] = {
                            "model": m,
                            "accuracy": acc,
                            "precision": prec,
                            "recall": rec,
                            "f1": f1,
                            "auc": auc,
                            "y_test": yte,
                            "preds": preds,
                            "probs": probs,
                            "X_test": Xte
                        }
                    except Exception as e:
                        st.warning(f"{name} failed: {e}")

            # Display model comparison table
            comp_rows = []
            for name, res in results_all.items():
                comp_rows.append({
                    "Model": name,
                    "Accuracy": f"{res['accuracy']:.3f}",
                    "Precision": f"{res['precision']:.3f}",
                    "Recall": f"{res['recall']:.3f}",
                    "F1": f"{res['f1']:.3f}",
                    "ROC_AUC": f"{res['auc']:.3f}" if not math.isnan(res['auc']) else "N/A"
                })
            comp_df = pd.DataFrame(comp_rows)
            st.markdown("### Model performance comparison")
            st.dataframe(comp_df, use_container_width=True)
            download_df(comp_df, "ml_model_comparison.csv", label="Download model comparison")

            # Show detailed predictions for best model by AUC or F1
            # choose best by AUC if available else F1
            best_name = None
            best_score = -1
            for name, res in results_all.items():
                score = res['auc'] if not math.isnan(res['auc']) else res['f1']
                if score > best_score:
                    best_score = score
                    best_name = name
            if best_name:
                st.markdown(f"### Best model: {best_name} (export actual vs predicted + features)")
                best = results_all[best_name]
                # create table: original features for test set -> need to find corresponding rows in model_df
                # We lost original index after dropna; reconstruct with an index mapping
                test_mask = model_df.index.to_numpy()
                # We used train_test_split on arrays; to get test indices, re-split indices similarly
                indices = np.arange(len(model_df))
                _, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=model_df[target_col] if model_df[target_col].nunique()>1 else None)
                test_idx = sorted(test_idx)
                model_features_df = model_df.iloc[test_idx].reset_index(drop=True)
                out = model_features_df.copy()
                out["Actual"] = best["y_test"]
                out["Predicted_Label"] = best["preds"]
                out["Predicted_Prob"] = best["probs"]
                st.dataframe(out.head(50), use_container_width=True)
                download_df(out, "ml_actual_vs_predicted.csv", label="Download actual vs predicted")

            # Feature importances (if available)
            st.markdown("### Feature importances / coefficients (where applicable)")
            fi_rows = []
            for name, res in results_all.items():
                model_obj = res["model"]
                if hasattr(model_obj, "feature_importances_"):
                    imps = model_obj.feature_importances_
                    fi_rows.append(pd.DataFrame({"feature": feature_cols, "importance": imps}).sort_values("importance", ascending=False).head(20).assign(model=name))
                elif hasattr(model_obj, "coef_"):
                    coef = model_obj.coef_.ravel()
                    fi_rows.append(pd.DataFrame({"feature": feature_cols, "coef": coef}).sort_values("coef", key=abs, ascending=False).head(20).assign(model=name))
            if fi_rows:
                fi_df = pd.concat(fi_rows, ignore_index=True)
                st.dataframe(fi_df, use_container_width=True)
                download_df(fi_df, "feature_importances.csv", label="Download feature importances")
            else:
                st.info("No feature importance / coefficients available for trained models.")

    # -------------------------
    # Automated Insights: generate based on EDA & ML results
    # -------------------------
    st.markdown("## Automated Insights")
    insights = []

    # Basic funnel insights
    try:
        if total_applicants > 0:
            insights.append({
                "Insight_Type":"Overall applicants",
                "Detail": f"{total_applicants} applicants in selected period"
            })
        # stage percentages
        if not np.isnan(screened):
            insights.append({"Insight_Type":"Screened rate", "Detail": f"{(screened/total_applicants*100):.1f}% of applicants were screened" if total_applicants>0 else "N/A"})
        if not np.isnan(interviewed):
            insights.append({"Insight_Type":"Interviewed rate", "Detail": f"{(interviewed/total_applicants*100):.1f}% of applicants reached interview" if total_applicants>0 else "N/A"})
        if not np.isnan(offered):
            insights.append({"Insight_Type":"Offered rate", "Detail": f"{(offered/max(1,interviewed)*100):.1f}% of interviewed got offers" if interviewed else "N/A"})
        if not np.isnan(joined):
            insights.append({"Insight_Type":"Joined rate", "Detail": f"{(joined/max(1,offered)*100):.1f}% of offered joined" if offered else "N/A"})
    except:
        pass

    # Channel insights
    if "Channel" in filt.columns and "Interviewed_Flag" in filt.columns:
        ch_stats = filt.groupby("Channel").agg(apps=("Applicant_ID","count"), interviewed=("Interviewed_Flag","sum")).reset_index()
        ch_stats["int_rate"] = ch_stats["interviewed"] / ch_stats["apps"]
        ch_stats = ch_stats.sort_values("int_rate", ascending=False)
        best_ch = ch_stats.iloc[0]
        insights.append({"Insight_Type":"Top channel by interview rate", "Detail": f"{best_ch['Channel']} — {best_ch['int_rate']:.1%} (interview rate)"})

    # Position insights
    if "Position" in filt.columns and "Interviewed_Flag" in filt.columns:
        pos_stats = filt.groupby("Position").agg(apps=("Applicant_ID","count"), interviewed=("Interviewed_Flag","sum")).reset_index()
        pos_stats["int_rate"] = pos_stats["interviewed"] / pos_stats["apps"]
        pos_worst = pos_stats.sort_values("int_rate").head(1)
        if not pos_worst.empty:
            p = pos_worst.iloc[0]
            insights.append({"Insight_Type":"Position with low apply->interview rate", "Detail": f"{p['Position']} — {p['int_rate']:.1%} (interview rate, low) - consider JD or screening changes"})

    # Recruiter responsiveness insight
    if "Recruiter_Response_Hours" in filt.columns and "Joined_Flag" in filt.columns:
        resp = filt.groupby(pd.cut(filt["Recruiter_Response_Hours"], bins=[-0.1,2,8,24,9999])).agg(join_rate=("Joined_Flag","mean")).reset_index()
        worst_bucket = resp.sort_values("join_rate").iloc[0]
        insights.append({"Insight_Type":"Response time impact", "Detail": f"Candidates with average recruiter response {worst_bucket['Recruiter_Response_Hours']} hrs have lower join rate ({worst_bucket['join_rate']:.1%}). Improve response times."})

    # ML-based priority list (if model exists)
    if target_col and 'results_all' in locals() and results_all:
        # pick best model (re-using selection logic)
        best_name = None
        best_score = -1
        for name, res in results_all.items():
            score = res['auc'] if not math.isnan(res['auc']) else res['f1']
            if score > best_score:
                best_score = score
                best_name = name
        if best_name:
            insights.append({"Insight_Type":"Model suggestion", "Detail": f"Best predictive model: {best_name}. Use to score new applicants and prioritize outreach."})

    # Show insights table and download
    ins_df = pd.DataFrame(insights)
    if ins_df.empty:
        st.info("No automated insights generated for the current filter.")
    else:
        st.dataframe(ins_df, use_container_width=True)
        download_df(ins_df, "automated_insights_hiring.csv", label="Download automated insights")

    st.markdown("### Done — export insights, prioritize top-risk candidates, tidy up job descriptions and recruiter SLAs. You’re welcome.")

