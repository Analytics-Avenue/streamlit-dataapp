import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime, timedelta, date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Page config & CSS
# ---------------------------
st.set_page_config(page_title="Absenteeism Prediction & Workforce Planning", layout="wide", page_icon="📈")

# Hide default Streamlit nav + force left aligned title
st.markdown("""<style>[data-testid="stSidebarNav"]{display:none;}</style>""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Card + KPI hover glow */
.card {
    padding: 18px;
    border-radius: 12px;
    background: #ffffff;
    border: 1px solid #e6e9ee;
    transition: transform .18s ease, box-shadow .18s ease;
    box-shadow: 0 2px 6px rgba(2,6,23,0.06);
}
.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 12px 40px rgba(6,75,134,0.18);
    border-color: rgba(6,75,134,0.5);
}
.kpi {
    padding: 22px;
    border-radius: 12px;
    background: #ffffff;
    border: 1px solid #e8eef6;
    font-size: 22px;
    font-weight: 700;
    color: #064b86;
    text-align: center;
    transition: transform .18s ease, box-shadow .18s ease;
}
.kpi:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(6,75,134,0.15);
}
.left-card { text-align: left; }
.small { font-size:13px; color:#666; }
.title-left { text-align: left; font-weight:800; font-size:28px; color:#064b86; margin:0; padding:0; }
.subtitle-left { text-align:left; color:#555; margin-top:4px; margin-bottom:12px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Utilities
# ---------------------------
def download_df(df, filename, label="Download CSV"):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")

def read_csv_safe(url_or_file):
    """
    Read CSV from URL or file-like. If duplicate columns exist, make them unique by appending suffixes.
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

# ---------------------------
# App header
# ---------------------------
st.markdown('<div style="display:flex; align-items:center; justify-content:space-between;">'
            '<div><h1 class="title-left">Absenteeism Prediction & Workforce Planning</h1>'
            '<div class="subtitle-left">Forecast absenteeism, optimize workforce allocation, and reduce overtime costs.</div></div>'
            '<div style="text-align:right;"><img src="https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png" width=72></div>'
            '</div>', unsafe_allow_html=True)

# ---------------------------
# Tabs: Overview & Application
# ---------------------------
tabs = st.tabs(["Overview", "Application"])

# ---------------------------
# Overview tab
# ---------------------------
with tabs[0]:
    st.markdown("### Overview")
    st.markdown("""
    <div class="card left-card">
    <b>Purpose</b>: Predict absenteeism, identify patterns by shift/role/season, and provide staffing recommendations to reduce disruption.<br><br>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Capabilities")
        st.markdown("""
        <div class='card left-card'>
        • Absence pattern detection (time, role, shift).<br>
        • Predictive models for absenteeism probability (per employee).<br>
        • Workforce planning suggestions (roster adjustments, backup staffing).<br>
        • Drill-down EDA and exportable predictions.<br>
        • Automated insights: high-risk employees, peak months, shift hotspots.
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("#### Business impact")
        st.markdown("""
        <div class='card left-card'>
        • Reduced overtime and staffing costs.<br>
        • Better roster adherence and fewer surprises.<br>
        • Faster response to absenteeism spikes.<br>
        • Improved employee wellbeing via balanced workload.
        </div>
        """, unsafe_allow_html=True)

    # KPIs 5-in-row label-only with hover glow
    st.markdown("### Key Performance Indicators")
    kcols = st.columns(5)
    labels = ["Total Employees","Avg Workload","Avg Past Absences","Avg Hours Lost","Absence Rate"]
    for col,label in zip(kcols,labels):
        col.markdown(f"<div class='kpi'>{label}</div>", unsafe_allow_html=True)

    st.markdown("### Who should use & How")
    st.markdown("""
    <div class='card left-card'>
    <b>Who</b>: HR leads, Shift planners, Plant managers, Workforce analysts.<br><br>
    <b>How</b>: 1) Use default/example dataset or upload your CSV. 2) Filter by date/department/shift. 3) Inspect high-risk employees & automated insights. 4) Export predictions and adjust rosters.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# Application tab
# ---------------------------
with tabs[1]:
    st.header("Application")
    st.markdown("### Data Source — choose one option")

    data_mode = st.radio("Dataset option:", ["Use example dataset (generated)", "Upload CSV", "Upload CSV + Map columns"], horizontal=True)

    df = None
    # Columns expected for mapping mode
    EXPECTED_COLS = [
        "Date","Employee_ID","Department","Role","Shift","Workload_Index","Past_Absence_Count",
        "Absent_Flag","Hours_Lost","Absence_Reason","Seasonal_Factor","Absenteeism_Probability"
    ]
    
  if data_mode == "Use default data":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/hr/absenteeism_dataset.csv"
    
        try:
            df = read_csv_safe(DEFAULT_URL)
            st.success("Default dataset loaded from GitHub.")
            st.dataframe(df.head())
        except Exception as e:
            st.error("Failed to load dataset from GitHub raw URL. Error: " + str(e))
            st.stop()


    elif data_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            try:
                df = read_csv_safe(uploaded)
                st.success("CSV uploaded.")
                st.dataframe(df.head())
            except Exception as e:
                st.error("Failed to read uploaded CSV: " + str(e))
                st.stop()
        else:
            st.stop()

    else:  # Upload + mapping
        uploaded = st.file_uploader("Upload CSV for mapping", type=["csv"], key="map_upload")
        if uploaded:
            try:
                raw = read_csv_safe(uploaded)
            except Exception as e:
                st.error("Failed to read CSV: " + str(e))
                st.stop()
            st.write("Preview of uploaded file:")
            st.dataframe(raw.head())
            st.markdown("Map your columns (map at least Date, Employee_ID, Absent_Flag).")
            mapping = {}
            cols_list = list(raw.columns)
            for required in EXPECTED_COLS:
                mapping[required] = st.selectbox(f"Map → {required}", ["-- Select --"] + cols_list, key=f"map_{required}")
            if st.button("Apply mapping"):
                missing = [k for k,v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Please map required fields. Missing: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v:k for k,v in mapping.items()})
                    st.success("Mapping applied.")
                    st.dataframe(df.head())
        else:
            st.stop()

    if df is None:
        st.stop()

    # ---------- Standardize & cleanup ----------
    # strip column whitespace & handle duplicates
    df.columns = [str(c).strip() for c in df.columns]

    # helper to pick preference if duplicates exist
    def prefer_col(df, base):
        for c in df.columns:
            if c == base:
                return c
        for c in df.columns:
            if c.startswith(base + "__dup"):
                return c
        # try case-insensitive match
        for c in df.columns:
            if c.lower() == base.lower():
                return c
        return None

    # canonical rename map if uploaded file uses slightly different names
    canonical_map = {}
    for base in EXPECTED_COLS:
        col_found = prefer_col(df, base)
        if col_found and col_found != base:
            canonical_map[col_found] = base
    if canonical_map:
        df = df.rename(columns=canonical_map)

    # Ensure Date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        st.error("No Date column found after mapping. App needs a Date column.")
        st.stop()

    # Convert numeric fields safely
    for numc in ["Workload_Index","Past_Absence_Count","Absent_Flag","Hours_Lost","Seasonal_Factor","Absenteeism_Probability"]:
        if numc in df.columns:
            df[numc] = pd.to_numeric(df[numc], errors="coerce")

    # Fill missing sensible defaults
    if "Absent_Flag" not in df.columns:
        df["Absent_Flag"] = 0
    else:
        df["Absent_Flag"] = df["Absent_Flag"].fillna(0).astype(int)

    for c in ["Workload_Index","Past_Absence_Count","Hours_Lost"]:
        if c in df.columns:
            if df[c].notna().any():
                df[c] = df[c].fillna(df[c].median())
            else:
                df[c] = df[c].fillna(0)

    # ---------- Filters ----------
    st.markdown("### Filters & Preview")

    # Date slicer (use date_input to avoid type mismatch)
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date_sel, end_date_sel = date_range
    else:
        start_date_sel = date_range
        end_date_sel = date_range

    # Departments & Roles & Shifts
    depts = list(df["Department"].dropna().unique()) if "Department" in df.columns else []
    roles = list(df["Role"].dropna().unique()) if "Role" in df.columns else []
    shifts = list(df["Shift"].dropna().unique()) if "Shift" in df.columns else []

    c1,c2,c3 = st.columns([2,2,2])
    with c1:
        sel_dept = st.multiselect("Department", options=depts, default=depts if depts else [])
    with c2:
        sel_role = st.multiselect("Role", options=roles, default=roles if roles else [])
    with c3:
        sel_shift = st.multiselect("Shift", options=shifts, default=shifts if shifts else [])

    filt = df[(df["Date"]>=pd.to_datetime(start_date_sel)) & (df["Date"]<=pd.to_datetime(end_date_sel))]
    if sel_dept:
        filt = filt[filt["Department"].isin(sel_dept)]
    if sel_role:
        filt = filt[filt["Role"].isin(sel_role)]
    if sel_shift:
        filt = filt[filt["Shift"].isin(sel_shift)]

    st.markdown("Preview (first 10 rows)")
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.reset_index(drop=True), "filtered_absenteeism.csv", label="Download filtered data")

    # ---------- Dynamic KPIs (numeric) ----------
    st.markdown("### KPIs (dynamic)")
    kcols = st.columns(5)
    total_emp = int(filt["Employee_ID"].nunique()) if "Employee_ID" in filt.columns else 0
    avg_workload = round(float(filt["Workload_Index"].mean()),2) if "Workload_Index" in filt.columns and filt["Workload_Index"].notna().any() else "N/A"
    avg_past = round(float(filt["Past_Absence_Count"].mean()),2) if "Past_Absence_Count" in filt.columns else "N/A"
    avg_hours_lost = round(float(filt["Hours_Lost"].mean()),2) if "Hours_Lost" in filt.columns else "N/A"
    absence_rate = round(float(filt["Absent_Flag"].mean())*100,2) if "Absent_Flag" in filt.columns else "N/A"

    kcols[0].markdown(f"<div class='kpi'>Total Employees<br><div class='small'>{total_emp}</div></div>", unsafe_allow_html=True)
    kcols[1].markdown(f"<div class='kpi'>Avg Workload<br><div class='small'>{avg_workload}</div></div>", unsafe_allow_html=True)
    kcols[2].markdown(f"<div class='kpi'>Avg Past Absences<br><div class='small'>{avg_past}</div></div>", unsafe_allow_html=True)
    kcols[3].markdown(f"<div class='kpi'>Avg Hours Lost<br><div class='small'>{avg_hours_lost}</div></div>", unsafe_allow_html=True)
    kcols[4].markdown(f"<div class='kpi'>Absence Rate<br><div class='small'>{absence_rate}%</div></div>", unsafe_allow_html=True)

    # ---------- EDA (lots of different charts, not repeating previous ones) ----------
    st.markdown("## Exploratory Data Analysis")

    # 1) Absence rate over time (weekly)
    if "Date" in filt.columns and "Absent_Flag" in filt.columns:
        weekly = filt.set_index("Date").resample("W").agg({"Absent_Flag":"mean"}).reset_index()
        fig_w = px.line(weekly, x="Date", y="Absent_Flag", title="Weekly Absence Rate (mean fraction)", labels={"Absent_Flag":"Absence Rate"})
        st.plotly_chart(fig_w, use_container_width=True)

    # 2) Absence by department (bar)
    if "Department" in filt.columns and "Absent_Flag" in filt.columns:
        by_dept = filt.groupby("Department")["Absent_Flag"].mean().reset_index().sort_values("Absent_Flag", ascending=False)
        fig_dept = px.bar(by_dept, x="Department", y="Absent_Flag", title="Absence Rate by Department")
        st.plotly_chart(fig_dept, use_container_width=True)

    # 3) Shift vs Absence (stacked bar)
    if "Shift" in filt.columns and "Absent_Flag" in filt.columns:
        shift_table = filt.groupby(["Shift","Absent_Flag"]).size().reset_index(name="count")
        fig_shift = px.histogram(filt, x="Shift", color="Absent_Flag", barmode="group", title="Shift vs Absence counts")
        st.plotly_chart(fig_shift, use_container_width=True)

    # 4) Workload distribution split by Absent_Flag
    if "Workload_Index" in filt.columns and "Absent_Flag" in filt.columns:
        fig_wl = px.box(filt, x="Absent_Flag", y="Workload_Index", title="Workload distribution (by absent flag)", labels={"Absent_Flag":"Absent Flag"})
        st.plotly_chart(fig_wl, use_container_width=True)

    # 5) Hours lost pie
    if "Hours_Lost" in filt.columns:
        pie = filt["Hours_Lost"].value_counts().reset_index()
        pie.columns = ["Hours_Lost","count"]
        fig_pie = px.pie(pie, values="count", names="Hours_Lost", title="Hours lost distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

    # 6) Top 10 employees by absence rate (bar)
    if "Employee_ID" in filt.columns and "Absent_Flag" in filt.columns:
        emp_rate = filt.groupby("Employee_ID")["Absent_Flag"].mean().reset_index().sort_values("Absent_Flag", ascending=False).head(20)
        fig_emp = px.bar(emp_rate, x="Employee_ID", y="Absent_Flag", title="Top employees by absence rate (top 20)")
        st.plotly_chart(fig_emp, use_container_width=True)

    # 7) Heatmap: weekday vs hour (if time-of-day exists) — we only have date; show weekday vs absence rate
    filt["Weekday"] = filt["Date"].dt.day_name()
    wk = filt.groupby("Weekday")["Absent_Flag"].mean().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).reset_index()
    if not wk["Absent_Flag"].isna().all():
        fig_week = px.bar(wk, x="Weekday", y="Absent_Flag", title="Absence rate by weekday")
        st.plotly_chart(fig_week, use_container_width=True)

    # 8) Correlation matrix for numeric fields
    numeric_for_corr = [c for c in ["Workload_Index","Past_Absence_Count","Hours_Lost","Absenteeism_Probability","Absent_Flag"] if c in filt.columns]
    if len(numeric_for_corr) >= 2:
        corr = filt[numeric_for_corr].corr()
        fig_corr = px.imshow(corr, text_auto=True, title="Correlation matrix (numeric fields)")
        st.plotly_chart(fig_corr, use_container_width=True)

    # 9) Rolling cumulative absences (cumulative sum) by department
    if "Date" in filt.columns and "Department" in filt.columns and "Absent_Flag" in filt.columns:
        cum = filt.sort_values("Date").groupby(["Department"]).apply(lambda x: x.set_index("Date")["Absent_Flag"].resample("W").sum().cumsum().reset_index()).reset_index(level=0)
        # cum will have index Department; flatten
        try:
            cum = cum.reset_index().rename(columns={"level_1":"idx"})
            # simple way: plot department-wise small multiples
            dept_list = filt["Department"].unique().tolist()
            for d in dept_list:
                series = filt[filt["Department"]==d].set_index("Date").resample("W")["Absent_Flag"].sum().cumsum().reset_index()
                fig_c = px.line(series, x="Date", y="Absent_Flag", title=f"Cumulative absences - {d}")
                st.plotly_chart(fig_c, use_container_width=True)
        except Exception:
            pass

    # 10) Distribution of Past_Absence_Count
    if "Past_Absence_Count" in filt.columns:
        fig_hist = px.histogram(filt, x="Past_Absence_Count", nbins=15, title="Past absence count distribution")
        st.plotly_chart(fig_hist, use_container_width=True)

    # ---------- ML section ----------
    st.markdown("## Machine Learning — Absenteeism Prediction")
    st.markdown("We train 4 models (Logistic Regression, Random Forest, GradientBoosting, KNN). Output: Actual vs Predicted probabilities + downloadable table.")

    # Prepare features & label
    # Pick sensible features if available
    candidate_features = ["Workload_Index","Past_Absence_Count","Hours_Lost","Seasonal_Factor"]
    cat_features = []
    if "Shift" in filt.columns:
        cat_features.append("Shift")
    if "Department" in filt.columns:
        cat_features.append("Department")
    if "Role" in filt.columns:
        cat_features.append("Role")

    # Build X and y carefully
    # create a working copy
    model_df = filt.copy().reset_index(drop=True)

    # Ensure categorical features exist as strings
    for cf in cat_features:
        model_df[cf] = model_df[cf].astype(str)

    # Minimal required features: need at least 2 numeric features plus label
    numeric_available = [c for c in candidate_features if c in model_df.columns]
    if len(model_df) >= 80 and len(numeric_available) >= 2 and "Absent_Flag" in model_df.columns:
        # quick preprocessing: one-hot for cat, fillna for numeric
        X_num = model_df[numeric_available].fillna(0)
        X_cat = pd.get_dummies(model_df[cat_features].astype(str), drop_first=True) if cat_features else pd.DataFrame(index=model_df.index)
        X = pd.concat([X_num, X_cat], axis=1)
        y = model_df["Absent_Flag"].astype(int)

        # Standardize numeric part
        scaler = StandardScaler()
        X_scaled = X.copy()
        if numeric_available:
            X_scaled[numeric_available] = scaler.fit_transform(X_scaled[numeric_available])

        # train/test split (stratify if possible)
        strat = y if y.nunique()>1 else None
        Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=strat)

        models = {
            "LogisticRegression": LogisticRegression(max_iter=500),
            "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=7)
        }

        results_frames = []
        model_metrics = []

        for name, m in models.items():
            try:
                m.fit(Xtr, ytr)
                prob = m.predict_proba(Xte)[:,1] if hasattr(m, "predict_proba") else m.predict(Xte)
                pred_label = (prob >= 0.5).astype(int) if hasattr(m, "predict_proba") else prob
                auc = roc_auc_score(yte, prob) if len(np.unique(yte))>1 and hasattr(m, "predict_proba") else None
                acc = accuracy_score(yte, pred_label)
                f1 = f1_score(yte, pred_label, zero_division=0)
                model_metrics.append({"Model":name, "AUC": round(float(auc),3) if auc else "N/A", "Accuracy": round(float(acc),3), "F1": round(float(f1),3)})
                tmp = pd.DataFrame({
                    "Model": name,
                    "Actual": yte.values,
                    "Predicted_Prob": prob,
                })
                results_frames.append(tmp)
            except Exception as e:
                model_metrics.append({"Model":name, "AUC":"Err", "Accuracy":"Err", "F1":"Err"})
                st.warning(f"Model {name} failed: {e}")

        # metrics summary
        st.markdown("### Model performance summary")
        st.table(pd.DataFrame(model_metrics).set_index("Model"))

        # show predictions table (concatenate)
        st.markdown("### Sample predictions (Actual vs Predicted Probabilities)")
        predictions_df = pd.concat(results_frames, ignore_index=True)
        st.dataframe(predictions_df.head(200))
        download_df(predictions_df, "absenteeism_model_predictions.csv", label="Download model predictions (Actual vs Predicted)")

        # feature importance for tree models
        st.markdown("### Feature importance (Random Forest & Gradient Boosting)")
        for name in ["RandomForest","GradientBoosting"]:
            if name in models:
                mod = models[name]
                if hasattr(mod, "feature_importances_"):
                    fi = pd.Series(mod.feature_importances_, index=Xtr.columns).sort_values(ascending=False).reset_index()
                    fi.columns = ["feature","importance"]
                    st.write(f"Top features — {name}")
                    st.dataframe(fi.head(20))
                    fig_fi = px.bar(fi.head(15), x="feature", y="importance", title=f"{name} — feature importances")
                    st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Not enough data (need >=80 rows) or not enough numeric features (at least 2 among Workload_Index/Past_Absence_Count/Hours_Lost/Seasonal_Factor) to train ML models.")

    # ---------- Automated Insights ----------
    st.markdown("## Automated Insights")
    insights = []
    # Top risk employees
    if "Employee_ID" in filt.columns and "Absent_Flag" in filt.columns:
        emp_risk = filt.groupby("Employee_ID")["Absent_Flag"].mean().reset_index().sort_values("Absent_Flag", ascending=False).head(10)
        insights.append({"Insight":"Top 10 employees by absence rate", "Details": emp_risk.to_dict(orient="records")})

    # Department risk
    if "Department" in filt.columns and "Absent_Flag" in filt.columns:
        dept_risk = filt.groupby("Department")["Absent_Flag"].mean().reset_index().sort_values("Absent_Flag", ascending=False)
        top_dept = dept_risk.iloc[0].to_dict()
        insights.append({"Insight":"Department with highest absence rate", "Details": top_dept})

    # Peak months
    if "Date" in filt.columns and "Absent_Flag" in filt.columns:
        months = filt.set_index("Date").resample("M").agg({"Absent_Flag":"mean"}).reset_index()
        if not months.empty:
            peak = months.sort_values("Absent_Flag", ascending=False).iloc[0]
            insights.append({"Insight":"Peak month for absence rate", "Details": {"Month":peak["Date"].strftime("%Y-%m"), "Rate": round(float(peak["Absent_Flag"]),3)}})

    # Average hours lost
    if "Hours_Lost" in filt.columns:
        insights.append({"Insight":"Avg hours lost per record", "Details": round(float(filt["Hours_Lost"].mean()),2)})

    # Workload correlation
    if "Workload_Index" in filt.columns and "Absent_Flag" in filt.columns:
        corr = filt["Workload_Index"].corr(filt["Absent_Flag"])
        insights.append({"Insight":"Correlation Workload vs Absence", "Details": round(float(corr),3) if not np.isnan(corr) else "N/A"})

    # Convert to display table-friendly format
    ins_display = []
    for i,ins in enumerate(insights):
        ins_display.append({"Insight": ins["Insight"], "Details": str(ins["Details"])[:500]})

    ins_df = pd.DataFrame(ins_display)
    if ins_df.empty:
        st.info("No automated insights generated for the selected filter.")
    else:
        st.dataframe(ins_df, use_container_width=True)
        download_df(pd.DataFrame(insights), "automated_insights_absenteeism.csv", label="Download automated insights")

    st.markdown("### Done — export insights or predictions and go fix the rota before Monday ghosts you again.")
