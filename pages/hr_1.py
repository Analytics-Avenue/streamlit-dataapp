import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO, StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score
import warnings
import textwrap
import math
warnings.filterwarnings("ignore")

hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(page_title="Hiring Funnel Drop-Off Analysis", layout="wide")

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
st.title("Hiring Funnel Drop-Off Analysis")

# ---------------------------------------------------------
# GLOBAL CSS: hover glow + card styles + left aligned cards inside overview
# ---------------------------------------------------------
st.markdown("""
<style>
/* Hide default Streamlit menu and footer (optional) */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Generic card */
.card {
    padding: 18px;
    border-radius: 12px;
    background: #ffffff;
    border: 1px solid rgba(6,75,134,0.06);
    box-shadow: 0 2px 6px rgba(6,75,134,0.06);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 10px 30px rgba(6,75,134,0.18);
    border-color: rgba(6,75,134,0.18);
}

/* KPI label-only big */
.kpi {
    padding: 22px;
    border-radius: 12px;
    background: linear-gradient(180deg, #ffffff 0%, #fbfbff 100%);
    border: 1px solid rgba(6,75,134,0.06);
    text-align: center;
    font-weight:700;
    font-size:20px;
    color:#064b86;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.kpi:hover {
    transform: translateY(-6px);
    box-shadow: 0 8px 20px rgba(6,75,134,0.16);
}

/* left-aligned content inside cards */
.card .left {
    text-align: left;
}

/* responsive spacer for KPI row */
.kpi-row { display:flex; gap:12px; align-items:stretch; }
.kpi-row > div { flex:1; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def download_df(df, filename, button_label="Download CSV", key=None):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(button_label, b, file_name=filename, mime="text/csv", key=key)

def read_csv_safe(url_or_file):
    """
    Read CSV from URL or file-like. If duplicate columns exist, make them unique
    by appending suffixes: col, col__dup1, col__dup2...
    Returns dataframe with stripped column names.
    """
    # Use pandas to read
    if isinstance(url_or_file, (str,)):
        df = pd.read_csv(url_or_file)
    else:
        # file-like from uploader
        url_or_file.seek(0)
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

def ensure_datetime(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def prefer_column(df, base):
    """
    pick first exact match, otherwise first col that startswith base + '__dup'
    returns None if not found
    """
    for c in df.columns:
        if c == base:
            return c
    for c in df.columns:
        if c.startswith(base + "__dup"):
            return c
    # try lower-case fuzzy match
    lower_map = {c.lower(): c for c in df.columns}
    if base.lower() in lower_map:
        return lower_map[base.lower()]
    return None

def canonicalize_columns(df, mapping_expected):
    """
    mapping_expected: list of canonical column names
    It will rename duplicates found with __dup suffix back to canonical if logical match.
    """
    rename = {}
    for base in mapping_expected:
        found = prefer_column(df, base)
        if found and found != base:
            rename[found] = base
    if rename:
        df = df.rename(columns=rename)
    df.columns = [str(c).strip() for c in df.columns]
    return df

# ---------------------------------------------------------
# CONFIG: Default GitHub URL for hiring funnel dataset (user can override)
# ---------------------------------------------------------
DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/hr/hiring_funnel_data.csv"
# Note: replace DEFAULT_URL with your actual repo raw URL if different.

# ---------------------------------------------------------
# TABS: Overview + Application
# ---------------------------------------------------------
tabs = st.tabs(["Overview", "Application"])

# -----------------------
# OVERVIEW TAB
# -----------------------
with tabs[0]:
    st.markdown("## Overview")

    st.markdown("""
    <div class="card left">
      <strong>Purpose</strong>: Understand where candidates drop off in the hiring funnel (Apply → Screen → Interview → Offer → Join). Use data to speed up hiring, improve candidate experience, and reduce cost-per-hire.
    </div>
    """, unsafe_allow_html=True)

    # Capabilities / Business Impact grid
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### Capabilities")
        st.markdown("""
        <div class="card left">
          • Funnel conversion measurement (stage-level)<br>
          • Time-in-stage and bottleneck identification<br>
          • Candidate source & JD performance analysis<br>
          • A/B style experiment results for job descriptions<br>
          • Predictive modeling for applicant dropout risk
        </div>
        """, unsafe_allow_html=True)
    with col_right:
        st.markdown("### Business impact")
        st.markdown("""
        <div class="card left">
          • Faster time-to-hire and lower vacancy days<br>
          • Better candidate experience & improved offer-accept rates<br>
          • Reduced recruiter workload via prioritized follow-ups<br>
          • Lower cost-per-hire through focused sourcing
        </div>
        """, unsafe_allow_html=True)

    # KPI cards (label-only, five across — responsive)
    st.markdown("### KPIs")
    # create 5 KPI columns (use st.columns)
    kp_cols = st.columns(5)
    labels = ["Applicants", "Screen Pass %", "Interview Rate", "Offer Rate", "Time-to-Fill (days)"]
    for col, lab in zip(kp_cols, labels):
        col.markdown(f"<div class='kpi'>{lab}</div>", unsafe_allow_html=True)

    st.markdown("### Who should use this app & How")
    st.markdown("""
    <div class="card left">
      <strong>Who</strong>: TA Leads, Recruiters, Hiring Managers, People Analytics teams.<br><br>
      <strong>How</strong>: 1) Load your hiring funnel CSV (or use default). 2) Filter by role / source / date range. 3) Inspect stage drop-offs and candidate timelines. 4) Download predicted drop-risk list for action.
    </div>
    """, unsafe_allow_html=True)

# -----------------------
# APPLICATION TAB — Part 1 (data loading + mapping)
# -----------------------
with tabs[1]:
    st.header("Application")
    st.markdown("### Step 1 — Load dataset")

    mode = st.radio("Dataset option:", ["Default dataset (GitHub URL)", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)
    df = None

    # Canonical columns expected for hiring funnel dataset
    EXPECTED_COLS = [
        "Applicant_ID", "Apply_Date", "Source", "Role", "Stage_Apply", "Stage_Screen",
        "Stage_Interview", "Stage_Offer", "Stage_Join", "Current_Stage", "Days_in_Stage",
        "Total_Time_to_Hire_Days", "Offer_Accepted_Flag", "Screen_Score", "Resume_Score",
        "Recruiter_ID", "JD_Variant", "Channel", "Candidate_Response_Time_Hrs"
    ]

    if mode == "Default dataset (GitHub URL)":
        try:
            df = read_csv_safe(DEFAULT_URL)
            df = canonicalize_columns(df, EXPECTED_COLS)

           # ===== AUTO MAP YOUR ACTUAL COLUMNS TO EXPECTED =====

            auto_map = {
                "Applicant_ID": "Applicant_ID",
                "Job_Role": "Role",
                "Source": "Source",
                "Applied_Date": "Apply_Date",
            
                # stage timestamps
                "Screen_Date": "Stage_Screen",
                "Interview_Date": "Stage_Interview",
                "Offer_Date": "Stage_Offer",
                "Join_Date": "Stage_Join",
            
                # results / status
                "Screen_Result": "Screen_Score",
                "Interview_Result": "Interview_Result",  # Not used by funnel, but kept
                "Offer_Status": "Offer_Accepted_Flag",
                "Joined": "Joined"
            }
            
            # apply mapping ONLY if column exists
            rename_dict = {}
            for old, new in auto_map.items():
                if old in df.columns:
                    rename_dict[old] = new
            
            df = df.rename(columns=rename_dict)
            
            # ===== CLEANUP FIELDS =====
            
            # parse all date columns
            date_like = ["Apply_Date","Stage_Screen","Stage_Interview","Stage_Offer","Stage_Join"]
            for c in date_like:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
            
            # Offer_Accepted_Flag → convert to 0/1
            if "Offer_Accepted_Flag" in df.columns:
                df["Offer_Accepted_Flag"] = df["Offer_Accepted_Flag"].astype(str).str.lower()
                df["Offer_Accepted_Flag"] = df["Offer_Accepted_Flag"].map({
                    "accepted": 1, "yes": 1, "offered": 1, "1": 1, "join": 1, "joined": 1,
                    "rejected": 0, "declined": 0, "no": 0, "0": 0
                }).fillna(0).astype(int)
            
            # derive Total_Time_to_Hire_Days
            if "Apply_Date" in df.columns and "Stage_Join" in df.columns:
                df["Total_Time_to_Hire_Days"] = (df["Stage_Join"] - df["Apply_Date"]).dt.days
            
            # create Current_Stage
            stage_cols = ["Stage_Screen","Stage_Interview","Stage_Offer","Stage_Join"]
            def get_stage(row):
                for col, stage in reversed([
                    ("Stage_Join", "Join"),
                    ("Stage_Offer", "Offer"),
                    ("Stage_Interview", "Interview"),
                    ("Stage_Screen", "Screen"),
                    ("Apply_Date", "Apply")
                ]):
                    if col in row and pd.notna(row[col]):
                        return stage
                return "Apply"
            
            df["Current_Stage"] = df.apply(get_stage, axis=1)

            st.success("Loaded default dataset from DEFAULT_URL (raw GitHub).")
            st.dataframe(df.head(8))
        except Exception as e:
            st.error("Failed to load default dataset. Check DEFAULT_URL or network. Error: " + str(e))
            st.stop()

    elif mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"], key="upload_simple")
        if uploaded is not None:
            try:
                df = read_csv_safe(uploaded)
                df = canonicalize_columns(df, EXPECTED_COLS)
                st.success("File uploaded.")
                st.dataframe(df.head(8))
            except Exception as e:
                st.error("Failed to read uploaded CSV: " + str(e))
                st.stop()
        else:
            st.info("Upload a CSV to continue.")
            st.stop()

    else:  # Upload + mapping
        uploaded = st.file_uploader("Upload CSV to map columns", type=["csv"], key="upload_map")
        if uploaded is not None:
            try:
                raw = read_csv_safe(uploaded)
            except Exception as e:
                st.error("Failed to read uploaded CSV: " + str(e))
                st.stop()

            st.markdown("Preview (first 8 rows):")
            st.dataframe(raw.head(8))

            st.markdown("Map your CSV columns to the expected funnel schema (map at least the required ones).")
            mapping = {}
            cols_list = list(raw.columns)
            # show mapping controls — only show the most relevant expected fields first
            map_keys = EXPECTED_COLS  # user can map all
            for key in map_keys:
                mapping[key] = st.selectbox(f"Map → {key}", ["-- Skip --"] + cols_list, key=f"map_{key}")

            if st.button("Apply mapping and continue", key="apply_map_btn"):
                # Build rename dict for mapped fields (only those not skipped)
                rename_map = {}
                for k, v in mapping.items():
                    if v != "-- Skip --":
                        rename_map[v] = k
                if len(rename_map) == 0:
                    st.error("You must map at least one column to proceed.")
                    st.stop()
                df = raw.rename(columns=rename_map)
                df = canonicalize_columns(df, EXPECTED_COLS)
                st.success("Mapping applied. Preview of mapped dataset:")
                st.dataframe(df.head(8))
        else:
            st.info("Upload a CSV to map columns.")
            st.stop()

    # Basic safety: if df is still None stop
    if df is None:
        st.error("No dataset loaded.")
        st.stop()

    # Normalize column whitespace
    df.columns = [str(c).strip() for c in df.columns]

    # Ensure apply date / date columns exist and convert
    # We'll try common names and prefer mapped canonical columns
    date_cols_try = ["Apply_Date", "ApplyDate", "Application_Date", "Apply Timestamp", "Apply_Timestamp"]
    for d in ["Apply_Date", "ApplyDate"]:
        if d in df.columns:
            df = ensure_datetime(df, d)

    # Also try generic candidate date columns (if they exist under other names)
    for cand in ["Apply_Date", "ApplyDate", "Apply Timestamp", "Application_Date", "Apply_Timestamp"]:
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand], errors="coerce")

    # Standardize small but important fields if present
    if "Offer_Accepted_Flag" in df.columns:
        df["Offer_Accepted_Flag"] = pd.to_numeric(df["Offer_Accepted_Flag"], errors="coerce").fillna(0).astype(int)

    if "Total_Time_to_Hire_Days" in df.columns:
        df["Total_Time_to_Hire_Days"] = pd.to_numeric(df["Total_Time_to_Hire_Days"], errors="coerce")

    # Provide quick column diagnostics for user to inspect and continue
    st.markdown("### Column diagnostics (first 40 columns shown)")
    diag = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "non_null_count": [int(df[c].notna().sum()) for c in df.columns]
    })
    st.dataframe(diag.head(200), use_container_width=True)

    st.info("If the important columns are missing or wrongly named, re-run with 'Upload CSV + Column mapping' mode and map manually.")

    # End of Part 1
    # Part 2 of 3
    # ---------- EDA, Date Slicer, Filters, Charts, Funnel Metrics ----------
    
    import datetime
    
    # --- Safety: require df from Part1
    try:
        df
    except NameError:
        st.error("Dataframe `df` not found. Make sure Part 1 was pasted and executed.")
        st.stop()
    
    st.markdown("---")
    st.markdown("## Step 2 — Filters & EDA (exploratory)")
    
    # -----------------------
    # Date slicer (safe)
    # -----------------------
    # prefer canonical Apply_Date
    date_col_candidates = [c for c in df.columns if c.lower().startswith("apply") or "date" in c.lower() or "time" in c.lower()]
    date_col = None
    for cand in ["Apply_Date", "ApplyDate", "Apply Timestamp", "Application_Date", "Apply_Timestamp"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        # fallback: pick first datetime-like column
        for c in df.columns:
            if np.issubdtype(df[c].dtype, np.datetime64):
                date_col = c
                break
    
    if date_col is None:
        st.warning("No date-like column detected. Date slicer will be disabled. If you have an Apply/Date column, map it in upload+map mode.")
        min_dt = max_dt = None
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        min_ts = df[date_col].min()
        max_ts = df[date_col].max()
        if pd.isna(min_ts) or pd.isna(max_ts):
            st.warning(f"Column {date_col} contains no parseable timestamps. Date slicer disabled.")
            min_dt = max_dt = None
        else:
            # convert to python datetimes (consistent types)
            min_dt = pd.to_datetime(min_ts).to_pydatetime()
            max_dt = pd.to_datetime(max_ts).to_pydatetime()
            # show a friendly date slider (datetime objects)
            date_range = st.slider("Select date range", min_value=min_dt, max_value=max_dt, value=(min_dt, max_dt), format="YYYY-MM-DD")
            st.write(f"Showing data between **{date_range[0].date()}** and **{date_range[1].date()}**")
    
    # -----------------------
    # Additional filters: Role, Source, JD Variant, Channel
    # -----------------------
    cols_for_filters = {
        "Role": [c for c in df.columns if c.lower() in ("role","job","position")],
        "Source": [c for c in df.columns if "source" in c.lower()],
        "JD_Variant": [c for c in df.columns if "jd" in c.lower() or "variant" in c.lower()],
        "Channel": [c for c in df.columns if "channel" in c.lower()]
    }
    # pick first available column for each
    role_col = cols_for_filters["Role"][0] if cols_for_filters["Role"] else None
    source_col = cols_for_filters["Source"][0] if cols_for_filters["Source"] else None
    jd_col = cols_for_filters["JD_Variant"][0] if cols_for_filters["JD_Variant"] else None
    channel_col = cols_for_filters["Channel"][0] if cols_for_filters["Channel"] else None
    
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        if role_col:
            selected_roles = st.multiselect("Role", options=sorted(df[role_col].dropna().unique().tolist()), default=sorted(df[role_col].dropna().unique().tolist())[:5])
        else:
            selected_roles = []
    with f2:
        if source_col:
            selected_sources = st.multiselect("Source", options=sorted(df[source_col].dropna().unique().tolist()), default=sorted(df[source_col].dropna().unique().tolist())[:5])
        else:
            selected_sources = []
    with f3:
        if jd_col:
            selected_jd = st.multiselect("JD Variant", options=sorted(df[jd_col].dropna().unique().tolist()), default=sorted(df[jd_col].dropna().unique().tolist())[:3])
        else:
            selected_jd = []
    with f4:
        if channel_col:
            selected_channel = st.multiselect("Channel", options=sorted(df[channel_col].dropna().unique().tolist()), default=sorted(df[channel_col].dropna().unique().tolist())[:3])
        else:
            selected_channel = []
    
    # -----------------------
    # Apply filters to create filt (filtered dataset)
    # -----------------------
    filt = df.copy()
    if date_col and min_dt and max_dt:
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filt = filt[(filt[date_col] >= start_dt) & (filt[date_col] <= end_dt)]
    
    if role_col and selected_roles:
        filt = filt[filt[role_col].isin(selected_roles)]
    if source_col and selected_sources:
        filt = filt[filt[source_col].isin(selected_sources)]
    if jd_col and selected_jd:
        filt = filt[filt[jd_col].isin(selected_jd)]
    if channel_col and selected_channel:
        filt = filt[filt[channel_col].isin(selected_channel)]
    
    st.markdown("### Filtered data preview")
    st.dataframe(filt.head(8), use_container_width=True)
    download_df(filt.head(500), "filtered_hiring_funnel_preview.csv", "Download filtered preview", key="dl_preview")
    
    # -----------------------
    # Funnel stage normalization & counts
    # -----------------------
    # Define canonical stage columns if exist else try to infer
    stage_cols = []
    for cand in ["Stage_Apply","Stage_Screen","Stage_Interview","Stage_Offer","Stage_Join","Current_Stage","Stage_Apply_Date"]:
        if cand in filt.columns:
            stage_cols.append(cand)
    # We'll compute counts per logical stage by examining a Current_Stage column if exists,
    # else fallback to boolean flags or stage columns presence.
    if "Current_Stage" in filt.columns:
        stage_counts = filt["Current_Stage"].value_counts().to_dict()
    else:
        # try to infer using presence of Stage_* columns that are numeric or flags
        stages_guess = []
        for s in ["Apply","Screen","Interview","Offer","Join"]:
            # look for columns that include stage name
            found = None
            for c in filt.columns:
                if s.lower() in c.lower() and ("stage" in c.lower() or s.lower() in c.lower()):
                    found = c
                    break
            if found:
                stages_guess.append((s, found))
        # If none found, we'll try to compute by presence of non-null timestamps per stage
        # else fallback to counts of applicants as apply-stage
        if stages_guess:
            stage_counts = {}
            for s, col in stages_guess:
                stage_counts[s] = int(filt[col].notna().sum())
        else:
            stage_counts = {"Apply": len(filt), "Screen": int(filt.get("Stage_Screen", pd.Series()).notna().sum() if "Stage_Screen" in filt.columns else 0),
                            "Interview": int(filt.get("Stage_Interview", pd.Series()).notna().sum() if "Stage_Interview" in filt.columns else 0),
                            "Offer": int(filt.get("Stage_Offer", pd.Series()).notna().sum() if "Stage_Offer" in filt.columns else 0),
                            "Join": int(filt.get("Stage_Join", pd.Series()).notna().sum() if "Stage_Join" in filt.columns else 0)}
    
    # Make funnel dataframe in canonical order
    funnel_order = ["Apply","Screen","Interview","Offer","Join"]
    funnel_df = pd.DataFrame({
        "Stage": funnel_order,
        "Count": [stage_counts.get(s, 0) for s in funnel_order]
    })
    
    # -----------------------
    # Show funnel (bar with decreasing order)
    # -----------------------
    st.markdown("### Funnel — Stage counts")
    fig_funnel = px.bar(funnel_df, x="Stage", y="Count", text="Count", title="Applicants by Funnel Stage", labels={"Count":"Count","Stage":"Funnel Stage"})
    fig_funnel.update_traces(marker_color="#064b86")
    fig_funnel.update_layout(title_x=0.02)
    st.plotly_chart(fig_funnel, use_container_width=True)
    
    # -----------------------
    # Stage-to-stage conversion rates
    # -----------------------
    st.markdown("### Stage-to-stage conversion rates")
    # compute conversion pairwise
    conv_rows = []
    prev = None
    prev_count = None
    for s in funnel_order:
        cnt = funnel_df.loc[funnel_df["Stage"]==s, "Count"].values[0]
        if prev is not None:
            rate = cnt / prev_count if prev_count and prev_count>0 else 0
            conv_rows.append({"From":prev,"To":s,"From_Count":prev_count,"To_Count":cnt,"Conv_Rate":rate})
        prev = s
        prev_count = cnt
    conv_df = pd.DataFrame(conv_rows)
    if conv_df.empty:
        st.info("Insufficient stage data to compute conversion rates.")
    else:
        conv_df["Conv_Rate_Pct"] = (conv_df["Conv_Rate"]*100).round(2).astype(str) + "%"
        st.dataframe(conv_df[["From","To","From_Count","To_Count","Conv_Rate_Pct"]], use_container_width=True)
        # conversion chart
        fig_conv = px.bar(conv_df, x="From", y="Conv_Rate", text=conv_df["Conv_Rate"].apply(lambda x: f"{x*100:.1f}%"), title="Conversion rate from stage to stage", labels={"Conv_Rate":"Conversion Rate"})
        fig_conv.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_conv, use_container_width=True)
    
    # -----------------------
    # Applications over time (line) — non-repetitive chart
    # -----------------------
    st.markdown("### Applications over time")
    if date_col and (date_col in filt.columns):
        apps_ts = filt.set_index(date_col).resample("D").size().reset_index(name="Applications")
        if not apps_ts.empty:
            fig_apps = px.line(apps_ts, x=date_col, y="Applications", title="Daily Applications (filtered)", markers=True)
            st.plotly_chart(fig_apps, use_container_width=True)
        else:
            st.info("No time-series data available after filtering.")
    else:
        st.info("Date column not available — skipping time-series chart.")
    
    # -----------------------
    # Source vs Stage stacked bar (non-repetitive) 
    # -----------------------
    st.markdown("### Source vs Current Stage (stacked)")
    if source_col and "Current_Stage" in filt.columns:
        src_stage = filt.groupby([source_col, "Current_Stage"]).size().reset_index(name="count")
        fig_src_stage = px.bar(src_stage, x=source_col, y="count", color="Current_Stage", title="Source vs Current Stage", text="count")
        st.plotly_chart(fig_src_stage, use_container_width=True)
    else:
        st.info("Source or Current_Stage column missing — skipping source-stage stacked chart.")
    
    # -----------------------
    # Time to hire distribution + boxplot by role
    # -----------------------
    st.markdown("### Time-to-hire distribution")
    if "Total_Time_to_Hire_Days" in filt.columns:
        fig_hist = px.histogram(filt, x="Total_Time_to_Hire_Days", nbins=40, title="Distribution of Total Time to Hire (days)")
        st.plotly_chart(fig_hist, use_container_width=True)
        # box by role
        if role_col:
            fig_box = px.box(filt, x=role_col, y="Total_Time_to_Hire_Days", title="Time to Hire by Role")
            st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("Total_Time_to_Hire_Days column missing — skipping time-to-hire charts.")
    
    # -----------------------
    # Screening score vs outcome scatter + trend (no statsmodels dependency)
    # -----------------------
    st.markdown("### Screening score vs Offer acceptance")
    if "Screen_Score" in filt.columns and "Offer_Accepted_Flag" in filt.columns:
        # jitter a little
        tmp = filt[[c for c in ["Screen_Score","Offer_Accepted_Flag","Total_Time_to_Hire_Days"] if c in filt.columns]].dropna()
        if not tmp.empty:
            fig_ss = px.scatter(tmp, x="Screen_Score", y="Offer_Accepted_Flag", title="Screen Score vs Offer Acceptance (0/1)", labels={"Offer_Accepted_Flag":"Offer Accepted"})
            st.plotly_chart(fig_ss, use_container_width=True)
    else:
        st.info("Screen_Score or Offer_Accepted_Flag missing — skipping scatter.")

    # -----------------------
    # Cohort table: Monthly applicants and join rate
    # -----------------------
    st.markdown("### Cohort overview (monthly)")
    if date_col:
        cohort = filt.copy()
        cohort["apply_month"] = cohort[date_col].dt.to_period("M").dt.to_timestamp()
        cohort_grp = cohort.groupby("apply_month").agg(
            applicants=("Applicant_ID" if "Applicant_ID" in cohort.columns else cohort.columns[0],"count"),
            joins=("Stage_Join" if "Stage_Join" in cohort.columns else "Applicant_ID", lambda x: cohort.loc[x.index, "Stage_Join"].notna().sum() if "Stage_Join" in cohort.columns else 0)
        ).reset_index()
        # I used safe aggregations; if Stage_Join is missing joins will be zero
        if not cohort_grp.empty:
            cohort_grp["join_rate"] = cohort_grp.apply(lambda r: (r["joins"]/r["applicants"] if r["applicants"]>0 else 0), axis=1)
            cohort_grp["join_rate_pct"] = (cohort_grp["join_rate"]*100).round(2).astype(str) + "%"
            st.dataframe(cohort_grp.head(24), use_container_width=True)
            download_df(cohort_grp, "cohort_monthly_summary.csv", "Download cohort summary", key="dl_cohort")
        else:
            st.info("Cohort aggregation empty — not enough date data.")
    else:
        st.info("No date column — cohort table not available.")
    
    # -----------------------
    # Summary EDA export
    # -----------------------
    st.markdown("### Export — EDA dataset & summary")
    # prepare summary stats
    try:
        eda_num = filt.select_dtypes(include=[np.number]).describe().T.reset_index().rename(columns={"index":"metric"})
        eda_cat = pd.DataFrame({
            "column": [c for c in filt.columns if filt[c].dtype == "object"],
            "unique_count": [filt[c].nunique() for c in filt.columns if filt[c].dtype == "object"]
        })
        download_df(eda_num, "eda_numeric_summary.csv", "Download numeric summary", key="dl_eda_num")
        download_df(eda_cat, "eda_categorical_summary.csv", "Download categorical summary", key="dl_eda_cat")
    except Exception as e:
        st.info("Failed to create EDA summaries: " + str(e))
    
