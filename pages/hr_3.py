# app_skill_gap_streamlit_standardized.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PAGE CONFIG (must be early)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Skill Gap & Training Needs — Analytics",
    layout="wide",
    page_icon="📊"
)

# ---------------------------------------------------------
# HIDE SIDEBAR / NAV
# ---------------------------------------------------------
HIDE_SIDEBAR = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(HIDE_SIDEBAR, unsafe_allow_html=True)

# ---------------------------------------------------------
# COMPANY HEADER
# ---------------------------------------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display: flex; align-items: center;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0; padding:0;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0; padding:0;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='margin-bottom:0.2rem; text-align:left;'>Skill Gap & Training Needs — Analytics</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='margin-top:0; color:#555; text-align:left;'>Identify skill gaps, prioritise training, and quantify ROI at employee × skill level.</p>",
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# GLOBAL CSS (cards, KPIs, fade, pure-black text)
# ---------------------------------------------------------
st.markdown("""
<style>
body, .markdown-text-container, .stMarkdown, .stText, .stDataFrame { color: #000000; }

/* Card */
.card {
    padding: 18px;
    border-radius: 12px;
    background: #ffffff;
    border: 1px solid #ececec;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    text-align: left;
}
.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 10px 30px rgba(6,75,134,0.16);
    border-color: #064b86;
}

/* KPI */
.kpi {
    padding: 24px;
    border-radius: 12px;
    background: #ffffff;
    border: 1px solid #e6eef7;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    transition: transform .18s ease, box-shadow .18s ease;
    text-align: center;
}
.kpi-main {
    font-weight: 800;
    font-size: 22px;
    color: #064b86;  /* KPI number/value blue */
}
.kpi-label {
    font-size: 12px;
    font-weight: 600;
    color: #333333;
    margin-top: 6px;
}

/* Simple fade-in */
.fade {
    animation: fadeIn .25s ease-in-out;
}
@keyframes fadeIn {
    from {opacity:0; transform: translateY(4px);}
    to {opacity:1; transform: translateY(0);}
}

.small { font-size:13px; color:#666; line-height:1.3; }
.left-align { text-align:left; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------
DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/hr/skill_gap_dataset.csv"

def read_csv_safe(url_or_file):
    """
    Read CSV from URL or file-like. Make duplicate column names unique by suffixing __dupN.
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

def download_df(df, filename, button_label="Download CSV", key=None):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(button_label, b, file_name=filename, mime="text/csv", key=key)

def prefer_column(df, base):
    """
    Return the first matching column equal to base or base__dupX if present.
    """
    for c in df.columns:
        if c == base:
            return c
    for c in df.columns:
        if c.startswith(base + "__dup"):
            return c
    return None

# Canonical schema
EXPECTED_COLS = [
    "Employee_ID","Department","Role","Tenure_Years","Annual_Salary",
    "Skill","Required_Level","Current_Level","Skill_Gap",
    "Recommend_Training_Flag","Recommended_Training","Training_Count","Last_Training_Date","Last_Training_Type",
    "Training_Cost_USD","Predicted_Improvement_pts","Predicted_Level",
    "Estimated_Yearly_Productivity_Value_Gain_USD","Estimated_ROI_USD",
    "Performance_Score","Predicted_Performance_Score","Data_Generated_On"
]

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tabs = st.tabs(["Overview", "Application", "Data Dictionary"])

# ---------------------------------------------------------
# OVERVIEW TAB
# ---------------------------------------------------------
with tabs[0]:
    st.markdown("### Overview")
    st.markdown("""
    <div class='card fade left-align'>
        <b>Purpose</b>: Identify current vs required skill levels across the workforce, 
        prioritize training by impact, and estimate ROI from upskilling interventions.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Capabilities")
        st.markdown("""
        <div class='card fade left-align'>
        • Employee × Skill matrix analysis<br>
        • Skill gap detection & prioritized training queues<br>
        • Predict who should be trained and expected improvement<br>
        • Cohort clustering & PCA views for L&D planning<br>
        • Download-ready reports & ML outputs for LMS / HR dashboards
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("#### Business Impact")
        st.markdown("""
        <div class='card fade left-align'>
        • Higher training ROI via targeted interventions<br>
        • Faster time-to-competency for critical roles<br>
        • Improved performance & productivity outcomes<br>
        • Data-backed budget allocation & training roadmap
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.markdown("""
    <div class='kpi fade'>
        <div class='kpi-main'>Employees Tracked</div>
    </div>
    """, unsafe_allow_html=True)
    k2.markdown("""
    <div class='kpi fade'>
        <div class='kpi-main'>Skills Covered</div>
    </div>
    """, unsafe_allow_html=True)
    k3.markdown("""
    <div class='kpi fade'>
        <div class='kpi-main'>Avg Skill Gap</div>
    </div>
    """, unsafe_allow_html=True)
    k4.markdown("""
    <div class='kpi fade'>
        <div class='kpi-main'>% Recommend Training</div>
    </div>
    """, unsafe_allow_html=True)
    k5.markdown("""
    <div class='kpi fade'>
        <div class='kpi-main'>Estimated ROI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Who should use & How")
    st.markdown("""
    <div class='card fade left-align'>
        <b>Who</b>: L&D leads, HR analytics teams, Ops heads, Plant managers.<br><br>
        <b>How</b>:<br>
        1) Load the dataset (default or your own CSV).<br>
        2) Filter by department, role, skill, and employee.<br>
        3) Use charts, ML outputs, and insights to design training plan.<br>
        4) Export filtered views and ML predictions to plug into LMS / HRIS.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# APPLICATION TAB
# ---------------------------------------------------------
with tabs[1]:
    st.header("Application")
    st.markdown("#### Step 1 — Load dataset")

    load_mode = st.radio(
        "Data source:",
        ["Default dataset (GitHub URL)", "Upload CSV", "Upload CSV + Column mapping"],
        horizontal=True
    )

    df = None

    # ---------------- DATA LOADING ----------------
    if load_mode == "Default dataset (GitHub URL)":
        try:
            df = read_csv_safe(DEFAULT_URL)
            st.success("Loaded default dataset from GitHub (DEFAULT_URL).")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error("Failed to load default dataset. Error: " + str(e))
            st.stop()

    elif load_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            try:
                df = read_csv_safe(uploaded)
                st.success("Uploaded CSV read successfully.")
                st.dataframe(df.head(), use_container_width=True)
            except Exception as e:
                st.error("Failed to read uploaded CSV: " + str(e))
                st.stop()
        else:
            st.stop()

    else:  # Upload + mapping
        uploaded = st.file_uploader("Upload CSV to map columns", type=["csv"], key="map_upload")
        if uploaded:
            try:
                raw = read_csv_safe(uploaded)
            except Exception as e:
                st.error("Failed to read CSV: " + str(e))
                st.stop()

            st.markdown("Preview (first 5 rows):")
            st.dataframe(raw.head(), use_container_width=True)

            st.markdown(
                "Map your columns to the expected schema (at minimum: Employee_ID, Skill, Current_Level, Required_Level)."
            )
            mapping = {}
            cols_list = list(raw.columns)
            for col in EXPECTED_COLS:
                mapping[col] = st.selectbox(
                    f"Map → {col}",
                    options=["-- Select --"] + cols_list,
                    key=f"map_{col}"
                )

            if st.button("Apply mapping"):
                missing = [k for k,v in mapping.items()
                           if v=="-- Select --" and k in ["Employee_ID","Skill","Current_Level","Required_Level"]]
                if missing:
                    st.error("Please map at minimum: " + ", ".join(missing))
                else:
                    rename_map = {mapping[k]: k for k in mapping if mapping[k]!="-- Select --"}
                    df = raw.rename(columns=rename_map)
                    st.success("Mapping applied.")
                    st.dataframe(df.head(), use_container_width=True)
        else:
            st.stop()

    if df is None:
        st.stop()

    # --------------- CLEANING & CANONICALIZATION ---------------
    df.columns = [str(c).strip() for c in df.columns]

    canonical_map = {}
    for base in ["Employee_ID","Skill","Required_Level","Current_Level",
                 "Recommend_Training_Flag","Performance_Score"]:
        col_found = prefer_column(df, base)
        if col_found and col_found != base:
            canonical_map[col_found] = base
    if canonical_map:
        df = df.rename(columns=canonical_map)

    # numeric conversions
    num_cols = [
        "Required_Level","Current_Level","Skill_Gap","Training_Cost_USD",
        "Predicted_Improvement_pts","Predicted_Level",
        "Estimated_Yearly_Productivity_Value_Gain_USD","Estimated_ROI_USD",
        "Performance_Score","Predicted_Performance_Score",
        "Tenure_Years","Annual_Salary","Training_Count"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # skill gap & flag
    if "Skill_Gap" not in df.columns and "Required_Level" in df.columns and "Current_Level" in df.columns:
        df["Skill_Gap"] = df["Required_Level"] - df["Current_Level"]
    if "Skill_Gap" in df.columns:
        df["Skill_Gap"] = df["Skill_Gap"].fillna(0)

    if "Recommend_Training_Flag" in df.columns:
        df["Recommend_Training_Flag"] = pd.to_numeric(
            df["Recommend_Training_Flag"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        df["Recommend_Training_Flag"] = 0
        if "Skill_Gap" in df.columns:
            df["Recommend_Training_Flag"] = (df["Skill_Gap"] >= 10).astype(int)

    if "Last_Training_Date" in df.columns:
        df["Last_Training_Date"] = pd.to_datetime(df["Last_Training_Date"], errors="coerce")

    if "Department" not in df.columns:
        df["Department"] = "Unknown"
    if "Role" not in df.columns:
        df["Role"] = "Unknown"



    # --------------- DIAGNOSTICS ---------------
    st.markdown("#### Dataset diagnostics")
    st.dataframe(
        pd.DataFrame({
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "non_null_count": [int(df[c].notna().sum()) for c in df.columns]
        }),
        use_container_width=True
    )

    # store for Data Dictionary tab
    st.session_state["skillgap_df"] = df.copy()

    st.markdown("#### Sample preview")
    st.dataframe(df.head(10), use_container_width=True)
    download_df(df.head(200), "skill_gap_preview.csv", "Download sample preview")

    # --------------- FILTERS ---------------
    st.markdown("### Step 2 — Filters")

    f1, f2, f3, f4 = st.columns([2,2,2,1])
    departments = sorted(df["Department"].dropna().unique().tolist())
    roles = sorted(df["Role"].dropna().unique().tolist())
    skills = sorted(df["Skill"].dropna().unique().tolist()) if "Skill" in df.columns else []
    employees = sorted(df["Employee_ID"].dropna().unique().tolist()) if "Employee_ID" in df.columns else []

    sel_depts = f1.multiselect("Department", options=departments, default=departments if departments else [])
    sel_roles = f2.multiselect("Role", options=roles, default=roles if roles else [])
    sel_skills = f3.multiselect("Skill", options=skills, default=skills if skills else [])
    sel_emp = f4.multiselect("Employee", options=employees, default=[])

    filt = df.copy()
    if sel_depts:
        filt = filt[filt["Department"].isin(sel_depts)]
    if sel_roles:
        filt = filt[filt["Role"].isin(sel_roles)]
    if sel_skills and "Skill" in filt.columns:
        filt = filt[filt["Skill"].isin(sel_skills)]
    if sel_emp and "Employee_ID" in filt.columns:
        filt = filt[filt["Employee_ID"].isin(sel_emp)]

    st.markdown(f"Filtered rows: **{len(filt):,}**")
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.reset_index(drop=True), "skill_gap_filtered.csv", "Download filtered dataset")

    # --------------- KPIs (DYNAMIC) ---------------
    st.markdown("### Step 3 — KPIs (dynamic)")
    kk1,kk2,kk3,kk4,kk5 = st.columns(5)

    total_employees = int(filt["Employee_ID"].nunique()) if "Employee_ID" in filt.columns else 0
    total_skills = int(filt["Skill"].nunique()) if "Skill" in filt.columns else 0
    avg_gap_val = float(filt["Skill_Gap"].mean()) if "Skill_Gap" in filt.columns and len(filt)>0 else 0.0
    pct_recommend = float(filt["Recommend_Training_Flag"].mean()*100) if "Recommend_Training_Flag" in filt.columns and len(filt)>0 else 0.0
    est_roi_sum = float(filt["Estimated_ROI_USD"].sum()) if "Estimated_ROI_USD" in filt.columns else 0.0

    kk1.markdown(f"""
    <div class='kpi fade'>
        <div class='kpi-main'>{total_employees}</div>
        <div class='kpi-label'>Employees Tracked</div>
    </div>
    """, unsafe_allow_html=True)
    kk2.markdown(f"""
    <div class='kpi fade'>
        <div class='kpi-main'>{total_skills}</div>
        <div class='kpi-label'>Skills Covered</div>
    </div>
    """, unsafe_allow_html=True)
    kk3.markdown(f"""
    <div class='kpi fade'>
        <div class='kpi-main'>{avg_gap_val:.2f}</div>
        <div class='kpi-label'>Avg Skill Gap</div>
    </div>
    """, unsafe_allow_html=True)
    kk4.markdown(f"""
    <div class='kpi fade'>
        <div class='kpi-main'>{pct_recommend:.1f}%</div>
        <div class='kpi-label'>% Recommend Training</div>
    </div>
    """, unsafe_allow_html=True)
    kk5.markdown(f"""
    <div class='kpi fade'>
        <div class='kpi-main'>${est_roi_sum:,.0f}</div>
        <div class='kpi-label'>Estimated ROI (sum)</div>
    </div>
    """, unsafe_allow_html=True)

    # --------------- EDA / CHARTS ---------------
    st.markdown("### Step 4 — Exploratory Analysis")

    c1, c2 = st.columns(2)
    with c1:
        if "Skill_Gap" in filt.columns:
            fig_gap = px.histogram(
                filt, x="Skill_Gap", nbins=30,
                title="Skill Gap Distribution", marginal="box"
            )
            st.plotly_chart(fig_gap, use_container_width=True)
        else:
            st.info("Skill_Gap column not available for histogram.")
    with c2:
        if "Skill_Gap" in filt.columns and "Skill" in filt.columns:
            grp_skill = (
                filt.groupby("Skill")["Skill_Gap"]
                .mean()
                .reset_index()
                .sort_values("Skill_Gap", ascending=False)
                .head(20)
            )
            fig_bar = px.bar(
                grp_skill, x="Skill", y="Skill_Gap",
                title="Top 20 Skills by Avg Gap", labels={"Skill_Gap":"Avg Gap"}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Skill_Gap or Skill column not available.")

    st.markdown("#### Skill matrix heatmap (sample)")
    try:
        if "Employee_ID" in filt.columns and "Skill" in filt.columns and "Current_Level" in filt.columns:
            piv_sample = filt.sample(min(len(filt), 2000), random_state=42) if len(filt) > 2000 else filt
            heat = piv_sample.pivot_table(
                index="Employee_ID",
                columns="Skill",
                values="Current_Level",
                aggfunc="mean"
            ).fillna(0)
            if heat.shape[0] > 0 and heat.shape[1] > 0:
                fig_heat = px.imshow(
                    heat,
                    aspect="auto",
                    color_continuous_scale="Viridis",
                    title="Employee × Skill (Current Level)"
                )
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info("Not enough data for heatmap.")
        else:
            st.info("Need Employee_ID, Skill, Current_Level for heatmap.")
    except Exception as e:
        st.info("Heatmap could not be generated: " + str(e))

    st.markdown("#### Current Level by Role")
    if "Current_Level" in filt.columns and "Role" in filt.columns:
        try:
            fig_box = px.box(
                filt, x="Role", y="Current_Level",
                title="Current Skill Level by Role"
            )
            st.plotly_chart(fig_box, use_container_width=True)
        except Exception as e:
            st.info("Box plot error: " + str(e))
    else:
        st.info("Current_Level or Role column missing.")

    st.markdown("#### Predicted improvement vs training cost")
    if "Predicted_Improvement_pts" in filt.columns and "Training_Cost_USD" in filt.columns:
        try:
            fig_sc = px.scatter(
                filt,
                x="Training_Cost_USD",
                y="Predicted_Improvement_pts",
                color="Department",
                hover_data=["Employee_ID","Skill","Recommended_Training"] if "Recommended_Training" in filt.columns else ["Employee_ID","Skill"],
                title="Training Cost vs Predicted Improvement"
            )
            st.plotly_chart(fig_sc, use_container_width=True)
        except Exception as e:
            st.info("Scatter plot error: " + str(e))
    else:
        st.info("Predicted_Improvement_pts or Training_Cost_USD missing.")

    # --------------- ML: PCA, KMEANS, CLASSIFIER, REGRESSOR ---------------
    st.markdown("### Step 5 — Machine Learning")

    # Employee-skill matrix for PCA / KMeans
    employee_matrix = None
    try:
        if "Employee_ID" in filt.columns and "Skill" in filt.columns and "Current_Level" in filt.columns:
            pivot_emp = filt.pivot_table(
                index="Employee_ID",
                columns="Skill",
                values="Current_Level",
                aggfunc="mean"
            ).fillna(0)
            pivot_emp.columns = [str(c) for c in pivot_emp.columns]
            employee_matrix = pivot_emp.copy()

            st.markdown("#### PCA — Employee skill profiles")
            if employee_matrix.shape[0] >= 2 and employee_matrix.shape[1] >= 2:
                scaler_emp = StandardScaler()
                X_emp = scaler_emp.fit_transform(employee_matrix.values)
                pca = PCA(n_components=min(5, X_emp.shape[1]))
                pcs = pca.fit_transform(X_emp)
                pc_df = pd.DataFrame(pcs[:, :2], columns=["PC1","PC2"], index=employee_matrix.index).reset_index()
                fig_pca = px.scatter(
                    pc_df, x="PC1", y="PC2",
                    hover_name="Employee_ID",
                    title="PCA of Employee Skill Profiles"
                )
                st.plotly_chart(fig_pca, use_container_width=True)
                st.markdown(f"Explained variance (first 2 PCs): **{pca.explained_variance_ratio_[:2].sum():.2f}**")
            else:
                st.info("Not enough employee-skill columns for PCA.")
        else:
            st.info("Need Employee_ID, Skill, Current_Level for PCA & clustering.")
    except Exception as e:
        st.info("PCA error: " + str(e))

    st.markdown("#### KMeans clustering — Employee cohorts")
    try:
        if employee_matrix is not None and employee_matrix.shape[0] >= 3:
            n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3)
            scaler_emp = StandardScaler()
            X_emp = scaler_emp.fit_transform(employee_matrix.values)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_emp)
            cl_df = pd.DataFrame({"Employee_ID": employee_matrix.index, "Cluster": clusters})
            st.write("Cluster sizes:")
            st.dataframe(cl_df["Cluster"].value_counts().sort_index().to_frame("Size"))

            pca2 = PCA(n_components=2)
            pcs2 = pca2.fit_transform(X_emp)
            pc_cl = pd.DataFrame(pcs2, columns=["PC1","PC2"])
            pc_cl["Cluster"] = clusters.astype(str)
            st.plotly_chart(
                px.scatter(pc_cl, x="PC1", y="PC2", color="Cluster",
                           title="KMeans clusters (PCA projection)"),
                use_container_width=True
            )
        else:
            st.info("Not enough employees for clustering.")
    except Exception as e:
        st.info("KMeans error: " + str(e))

    st.markdown("#### Classification — Predict Recommend_Training_Flag")
    try:
        if "Recommend_Training_Flag" in filt.columns and filt["Recommend_Training_Flag"].nunique() > 1:
            clf_features = [c for c in ["Current_Level","Required_Level","Skill_Gap","Tenure_Years","Annual_Salary"] if c in filt.columns]
            if len(clf_features) >= 1:
                data_clf = filt[clf_features + ["Recommend_Training_Flag"]].dropna()
                sample_frac = st.slider(
                    "Sample fraction for classifier (for speed)",
                    min_value=0.05,
                    max_value=1.0,
                    value=1.0
                )
                if sample_frac < 1.0 and len(data_clf) > 0:
                    data_clf = data_clf.sample(frac=sample_frac, random_state=42)

                if len(data_clf) >= 20:
                    Xc = data_clf[clf_features].fillna(0)
                    yc = data_clf["Recommend_Training_Flag"].astype(int)
                    X_train, X_test, y_train, y_test = train_test_split(
                        Xc, yc, test_size=0.2, random_state=42,
                        stratify=yc if yc.nunique()>1 else None
                    )
                    clf = RandomForestClassifier(n_estimators=150, random_state=42)
                    clf.fit(X_train, y_train)
                    prob = clf.predict_proba(X_test)[:,1]
                    pred_label = (prob >= 0.5).astype(int)
                    acc = accuracy_score(y_test, pred_label)
                    try:
                        auc = roc_auc_score(y_test, prob)
                        st.write(f"Classifier Accuracy: **{acc:.3f}** | ROC AUC: **{auc:.3f}**")
                    except Exception:
                        st.write(f"Classifier Accuracy: **{acc:.3f}**")

                    fi = pd.DataFrame({
                        "feature": clf_features,
                        "importance": clf.feature_importances_
                    }).sort_values("importance", ascending=False)
                    st.markdown("Feature importances (Classifier):")
                    st.dataframe(fi, use_container_width=True)

                    preds_df = X_test.reset_index(drop=True).copy()
                    preds_df["Actual"] = y_test.reset_index(drop=True)
                    preds_df["Pred_Prob"] = prob
                    preds_df["Pred_Label"] = pred_label
                    st.markdown("Sample classifier predictions:")
                    st.dataframe(preds_df.head(10), use_container_width=True)
                    download_df(preds_df, "classifier_predictions_skillgap.csv", "Download classifier predictions")
                else:
                    st.info("Not enough rows after sampling to run classifier (need ≥ 20).")
            else:
                st.info("Not enough numeric features for classifier.")
        else:
            st.info("Recommend_Training_Flag missing or constant — cannot run classifier.")
    except Exception as e:
        st.info("Classifier error: " + str(e))

    st.markdown("#### Regression — Predict Performance_Score")
    try:
        if "Performance_Score" in filt.columns and "Employee_ID" in filt.columns:
            emp_feats = filt.groupby("Employee_ID").agg(
                Avg_Current_Level=("Current_Level","mean"),
                Avg_Required_Level=("Required_Level","mean"),
                Avg_Gap=("Skill_Gap","mean"),
                Training_Count=("Training_Count","sum") if "Training_Count" in filt.columns else ("Performance_Score","size"),
                Annual_Salary=("Annual_Salary","mean") if "Annual_Salary" in filt.columns else ("Performance_Score","mean"),
                Tenure_Years=("Tenure_Years","mean") if "Tenure_Years" in filt.columns else ("Performance_Score","mean"),
                Performance_Score=("Performance_Score","mean")
            ).reset_index()
            emp_feats = emp_feats.fillna(0)
            emp_feats = emp_feats[~emp_feats["Performance_Score"].isna()]

            if len(emp_feats) >= 10:
                reg_features = ["Avg_Current_Level","Avg_Required_Level","Avg_Gap","Training_Count","Annual_Salary","Tenure_Years"]
                Xr = emp_feats[reg_features]
                yr = emp_feats["Performance_Score"]
                Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)

                rfr = RandomForestRegressor(n_estimators=150, random_state=42)
                rfr.fit(Xtr, ytr)
                preds_r = rfr.predict(Xte)
                rmse = float(np.sqrt(mean_squared_error(yte, preds_r)))
                r2 = float(r2_score(yte, preds_r))
                st.write(f"Regression — RMSE: **{rmse:.3f}**, R²: **{r2:.3f}**")

                out_reg = Xte.reset_index(drop=True).copy()
                out_reg["Actual_Perf"] = yte.reset_index(drop=True)
                out_reg["Pred_Perf"] = preds_r
                st.dataframe(out_reg.head(10), use_container_width=True)
                download_df(out_reg, "regression_predictions_skillgap.csv", "Download regression predictions")

                fi_r = pd.DataFrame({
                    "feature": reg_features,
                    "importance": rfr.feature_importances_
                }).sort_values("importance", ascending=False)
                st.markdown("Feature importances (Regressor):")
                st.dataframe(fi_r, use_container_width=True)
            else:
                st.info("Not enough employees with Performance_Score to train regression (need ≥ 10).")
        else:
            st.info("Performance_Score or Employee_ID missing — cannot run regression.")
    except Exception as e:
        st.info("Regression error: " + str(e))

    # --------------- AUTOMATED INSIGHTS ---------------
    st.markdown("### Step 6 — Automated Insights")
    insights = []
    try:
        if "Skill_Gap" in filt.columns and "Skill" in filt.columns:
            top_skill_gaps = (
                filt.groupby("Skill")["Skill_Gap"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
            )
            for skill, gap in top_skill_gaps.items():
                insights.append({
                    "Insight_Type": "Top Skill Gap",
                    "Skill": skill,
                    "Avg_Gap": round(float(gap),2)
                })

        if "Skill_Gap" in filt.columns and "Employee_ID" in filt.columns:
            emp_gap = (
                filt.groupby("Employee_ID")["Skill_Gap"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
            )
            for emp, gap in emp_gap.items():
                insights.append({
                    "Insight_Type": "High Gap Employee",
                    "Employee_ID": emp,
                    "Avg_Gap": round(float(gap),2)
                })

        if "Recommended_Training" in filt.columns and "Recommend_Training_Flag" in filt.columns:
            top_recs = (
                filt[filt["Recommend_Training_Flag"]==1]["Recommended_Training"]
                .value_counts()
                .head(10)
            )
            for rec, cnt in top_recs.items():
                insights.append({
                    "Insight_Type": "Recommended Training",
                    "Training": rec,
                    "Count": int(cnt)
                })

        if "Estimated_ROI_USD" in filt.columns:
            tot_roi = round(float(filt["Estimated_ROI_USD"].sum()),2)
            insights.append({
                "Insight_Type": "ROI Summary",
                "Estimated_Total_ROI_USD": tot_roi
            })
    except Exception as e:
        insights.append({"Insight_Type":"Insight Error", "Message": str(e)})

    insights_df = pd.DataFrame(insights).fillna("")
    if insights_df.empty:
        st.info("No automated insights available for this filter.")
    else:
        st.dataframe(insights_df, use_container_width=True)
        download_df(insights_df, "automated_insights_skillgap.csv", "Download automated insights")

    # --------------- EXPORT EMPLOYEE MATRIX ---------------
    st.markdown("### Step 7 — Exports")
    col_a, col_b = st.columns(2)
    with col_a:
        col_a.markdown("Filtered dataset export:")
        download_df(filt.reset_index(drop=True), "skill_gap_filtered.csv", "Download filtered dataset (full)")
    with col_b:
        col_b.markdown("Employee skill matrix (for external ML / BI):")
        if 'employee_matrix' in locals() and employee_matrix is not None:
            download_df(employee_matrix.reset_index(), "employee_skill_matrix.csv", "Download employee skill matrix")
        else:
            col_b.info("Employee matrix not available (need Employee_ID × Skill × Current_Level).")

# ---------------------------------------------------------
# DATA DICTIONARY TAB
# ---------------------------------------------------------
with tabs[2]:
    st.markdown("### Data Dictionary")
    base_df = st.session_state.get("skillgap_df", None)

    if base_df is None:
        st.info("No dataset loaded yet. Load a dataset from the Application tab first.")
    else:
        dd = pd.DataFrame({
            "Column": base_df.columns,
            "Dtype": [str(base_df[c].dtype) for c in base_df.columns],
            "Non-null Count": [int(base_df[c].notna().sum()) for c in base_df.columns]
        })

        # simple guessed descriptions for known fields
        desc_map = {
            "Employee_ID": "Unique employee identifier",
            "Department": "Employee department / function",
            "Role": "Employee role / designation",
            "Tenure_Years": "Years of tenure in organisation",
            "Annual_Salary": "Annual salary approximation (numeric)",
            "Skill": "Skill name / competency label",
            "Required_Level": "Required skill level for role",
            "Current_Level": "Current assessed skill level",
            "Skill_Gap": "Required_Level - Current_Level",
            "Recommend_Training_Flag": "1 if training recommended, else 0",
            "Recommended_Training": "Training / course name recommended",
            "Training_Count": "Number of trainings attended",
            "Last_Training_Date": "Most recent training completion date",
            "Last_Training_Type": "Type/category of last training",
            "Training_Cost_USD": "Training cost estimate in USD",
            "Predicted_Improvement_pts": "Expected improvement in skill points",
            "Predicted_Level": "Predicted post-training skill level",
            "Estimated_Yearly_Productivity_Value_Gain_USD": "Yearly value gain estimate post-training",
            "Estimated_ROI_USD": "Estimated ROI from training per record",
            "Performance_Score": "Current performance rating/score",
            "Predicted_Performance_Score": "Predicted future performance score",
            "Data_Generated_On": "Date when this record was generated"
        }
        dd["Description"] = dd["Column"].map(desc_map).fillna("")
        st.dataframe(dd, use_container_width=True)
        download_df(dd, "skill_gap_data_dictionary.csv", "Download data dictionary")

        
        st.markdown("#### Independent & Dependent Variables")
        
        # Var groups
        independent_vars = [
            ("Employee_ID"),
            ("Department"),
            ("Role"),
            ("Tenure_Years"),
            ("Annual_Salary"),
            ("Skill"),
            ("Required_Level"),
            ("Current_Level"),
            ("Skill_Gap"),
            ("Training_Count"),
            ("Last_Training_Type"),
        ]
        
        dependent_vars = [
            ("Recommend_Training_Flag"),
            ("Predicted_Improvement_pts"),
            ("Predicted_Level"),
            ("Estimated_ROI_USD"),
            ("Predicted_Performance_Score"),
        ]
        
        left, right = st.columns(2)
        
        with left:
            st.markdown("### Independent Variables")
            for name in independent_vars:
                st.markdown(
                    f"""
                    <div class='card'>
                        <b>{name}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        
        with right:
            st.markdown("### Dependent Variables")
            for name in dependent_vars:
                st.markdown(
                    f"""
                    <div class='card'>
                        <b>{name}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
