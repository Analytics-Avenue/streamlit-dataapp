# app_skill_gap_streamlit.py
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

# -------------------------

hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# -------------------------
# Company Logo + Name
# -------------------------
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


# App header & page config
# -------------------------
st.set_page_config(page_title="Skill Gap & Training Needs — Analytics", layout="wide")
# ----------------------------------------------------------
st.markdown("<div class='big-header'>Skill Gap & Training Needs — Analytics</div>", unsafe_allow_html=True)


# -------------------------
# Card Glow CSS (hover effects)
# -------------------------
st.markdown("""
<style>
.card {
    padding: 18px;
    border-radius: 12px;
    background: #ffffff;
    border: 1px solid #ececec;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    transition: transform .18s ease, box-shadow .18s ease;
    text-align: left;
}
.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 10px 30px rgba(6,75,134,0.12);
    border-color: #064b86;
}
.kpi {
    padding: 24px;
    border-radius: 12px;
    background: #ffffff;
    border: 1px solid #e6eef7;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    transition: transform .18s ease, box-shadow .18s ease;
    text-align: center;
    font-weight:700;
    color:#064b86;
    font-size:20px;
}
.kpi:hover {
    transform: translateY(-6px);
    box-shadow: 0 10px 30px rgba(6,75,134,0.12);
}
.small { font-size:13px; color:#666; line-height:1.3; }
.left-align { text-align:left; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Utilities
# -------------------------
DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/hr/skill_gap_dataset.csv"
# Replace above with your raw GitHub CSV URL for default dataset mode.

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

def download_df(df, filename, button_label="Download CSV"):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(button_label, b, file_name=filename, mime="text/csv")

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

# -------------------------
# Tabs: Overview & Application
# -------------------------
tabs = st.tabs(["Overview", "Application"])

# -------------------------
# Overview content (left-aligned)
# -------------------------
with tabs[0]:
    st.markdown("### About this Application")
    st.markdown("""
    <div class='card left-align'>
        <b>Purpose</b>: Identify current vs required skill levels across the workforce, recommend targeted training, estimate ROI, and prioritise interventions by impact.
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Capabilities")
        st.markdown("""
        <div class='card left-align'>
        • Employee × Skill matrix analysis<br>
        • Skill gap detection & prioritized training recommendations<br>
        • Predictive models: who needs training, expected performance lift<br>
        • Clustering and dimensionality reduction to spot cohorts<br>
        • Downloadable reports and ML outputs for operational use
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("#### Business Impact")
        st.markdown("""
        <div class='card left-align'>
        • Higher training ROI via targeted interventions<br>
        • Faster upskilling and improved productivity<br>
        • Reduced time-to-competency for critical roles<br>
        • Data-driven budget allocation & measurable outcomes
        </div>
        """, unsafe_allow_html=True)

    # KPI cards: 5 in a row label-only
    st.markdown("### Key Performance Indicators")
    kp1,kp2,kp3,kp4,kp5 = st.columns(5)
    kp1.markdown("<div class='kpi'>Employees Tracked</div>", unsafe_allow_html=True)
    kp2.markdown("<div class='kpi'>Skills Covered</div>", unsafe_allow_html=True)
    kp3.markdown("<div class='kpi'>Avg Skill Gap</div>", unsafe_allow_html=True)
    kp4.markdown("<div class='kpi'>% Recommend Training</div>", unsafe_allow_html=True)
    kp5.markdown("<div class='kpi'>Estimated ROI</div>", unsafe_allow_html=True)

    st.markdown("### Who should use this app & How")
    st.markdown("""
    <div class='card left-align'>
    <b>Who</b>: L&D leads, HR analysts, Plant managers, Ops leads.<br><br>
    <b>How</b>: 1) Use default dataset from GitHub or upload your CSV. 2) Filter by department/role. 3) Inspect clusters, top skill gaps and automated insights. 4) Export prioritized training lists and ML predictions to integrate with LMS procurement.
    </div>
    """, unsafe_allow_html=True)


# -------------------------
# Application tab
# -------------------------
with tabs[1]:
    st.header("Application")

    st.markdown("**Step 1 — Load dataset**")
    load_mode = st.radio("Data source:", ["Default dataset (GitHub URL)", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)

    df = None
    # expected canonical columns for this skill-gap dataset long-format
    EXPECTED_COLS = [
        "Employee_ID","Department","Role","Tenure_Years","Annual_Salary",
        "Skill","Required_Level","Current_Level","Skill_Gap",
        "Recommend_Training_Flag","Recommended_Training","Training_Count","Last_Training_Date","Last_Training_Type",
        "Training_Cost_USD","Predicted_Improvement_pts","Predicted_Level",
        "Estimated_Yearly_Productivity_Value_Gain_USD","Estimated_ROI_USD",
        "Performance_Score","Predicted_Performance_Score","Data_Generated_On"
    ]

    if load_mode == "Default dataset (GitHub URL)":
        try:
            df = read_csv_safe(DEFAULT_URL)
            st.success("Loaded default dataset from DEFAULT_URL (edit DEFAULT_URL in the app to point to your file).")
            st.dataframe(df.head())
        except Exception as e:
            st.error("Failed to load default dataset. Replace DEFAULT_URL at top of file with your raw GitHub CSV URL. Error: " + str(e))
            st.stop()

    elif load_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            try:
                df = read_csv_safe(uploaded)
                st.success("Uploaded CSV read.")
                st.dataframe(df.head())
            except Exception as e:
                st.error("Failed to read uploaded CSV: " + str(e))
                st.stop()
        else:
            st.stop()

    else:  # upload + mapping
        uploaded = st.file_uploader("Upload CSV to map columns", type=["csv"], key="map_upload")
        if uploaded:
            try:
                raw = read_csv_safe(uploaded)
            except Exception as e:
                st.error("Failed to read CSV: " + str(e))
                st.stop()
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head())
            st.markdown("Map the columns from your file to the expected schema (map at least Employee_ID, Skill, Current_Level, Required_Level).")
            mapping = {}
            cols_list = list(raw.columns)
            for col in EXPECTED_COLS:
                mapping[col] = st.selectbox(f"Map → {col}", options=["-- Select --"] + cols_list, key=f"map_{col}")
            if st.button("Apply mapping"):
                missing = [k for k,v in mapping.items() if v=="-- Select --" and k in ["Employee_ID","Skill","Current_Level","Required_Level"]]
                if missing:
                    st.error("Please map at minimum: " + ", ".join(missing))
                else:
                    # rename mapped ones (only those the user selected)
                    rename_map = {mapping[k]: k for k in mapping if mapping[k]!="-- Select --"}
                    df = raw.rename(columns=rename_map)
                    st.success("Mapping applied.")
                    st.dataframe(df.head())
        else:
            st.stop()

    if df is None:
        st.stop()

    # -------------------------
    # Basic cleaning & canonicalization
    # -------------------------
    df.columns = [str(c).strip() for c in df.columns]

    # if there are duplicate logical columns like 'Current_Level__dup1', prefer first occurrence
    canonical_map = {}
    for base in ["Employee_ID","Skill","Required_Level","Current_Level","Recommend_Training_Flag","Performance_Score"]:
        col_found = prefer_column(df, base)
        if col_found and col_found != base:
            canonical_map[col_found] = base
    if canonical_map:
        df = df.rename(columns=canonical_map)

    # Ensure numeric conversions
    num_cols = ["Required_Level","Current_Level","Skill_Gap","Training_Cost_USD","Predicted_Improvement_pts",
                "Predicted_Level","Estimated_Yearly_Productivity_Value_Gain_USD","Estimated_ROI_USD",
                "Performance_Score","Predicted_Performance_Score","Tenure_Years","Annual_Salary"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Coerce Recommend_Training_Flag into int 0/1 if exists
    if "Recommend_Training_Flag" in df.columns:
        df["Recommend_Training_Flag"] = pd.to_numeric(df["Recommend_Training_Flag"], errors="coerce").fillna(0).astype(int)
    else:
        # if missing, create from gap threshold
        if "Skill_Gap" in df.columns:
            df["Recommend_Training_Flag"] = (df["Skill_Gap"] >= 10).astype(int)
        else:
            df["Recommend_Training_Flag"] = 0

    # Fill NaNs sensibly
    if "Skill_Gap" not in df.columns and "Required_Level" in df.columns and "Current_Level" in df.columns:
        df["Skill_Gap"] = df["Required_Level"] - df["Current_Level"]
    for c in ["Skill_Gap"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Convert date-like columns if present
    if "Last_Training_Date" in df.columns:
        try:
            df["Last_Training_Date"] = pd.to_datetime(df["Last_Training_Date"], errors="coerce")
        except:
            df["Last_Training_Date"] = pd.to_datetime(df["Last_Training_Date"].astype(str), errors="coerce")

    # Derive some helpful columns if not present
    if "Department" not in df.columns:
        df["Department"] = "Unknown"
    if "Role" not in df.columns:
        df["Role"] = "Unknown"

    # Basic preview
    st.markdown("### Dataset preview (first 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)
    download_df(df.head(200), "skill_gap_preview.csv", "Download sample preview")

    # -------------------------
    # Filters & Slicer (date range slicer not required, but if Last_Training_Date exists allow date filter)
    # -------------------------
    st.markdown("### Filters")
    f1, f2, f3, f4 = st.columns([2,2,2,1])

    departments = sorted(df["Department"].dropna().unique().tolist())
    roles = sorted(df["Role"].dropna().unique().tolist())
    skills = sorted(df["Skill"].dropna().unique().tolist())
    employees = sorted(df["Employee_ID"].dropna().unique().tolist())

    sel_depts = f1.multiselect("Department", options=departments, default=departments if departments else [])
    sel_roles = f2.multiselect("Role", options=roles, default=roles if roles else [])
    sel_skills = f3.multiselect("Skill", options=skills, default=skills if skills else [])
    sel_emp = f4.multiselect("Employee", options=employees, default=[])

    filt = df.copy()
    if sel_depts:
        filt = filt[filt["Department"].isin(sel_depts)]
    if sel_roles:
        filt = filt[filt["Role"].isin(sel_roles)]
    if sel_skills:
        filt = filt[filt["Skill"].isin(sel_skills)]
    if sel_emp:
        filt = filt[filt["Employee_ID"].isin(sel_emp)]

    st.markdown(f"Filtered rows: **{len(filt):,}**")

    # -------------------------
    # Aggregate KPIs (dynamic)
    # -------------------------
    st.markdown("### KPIs (dynamic)")
    kk1,kk2,kk3,kk4,kk5 = st.columns(5)
    total_employees = filt["Employee_ID"].nunique()
    total_skills = filt["Skill"].nunique()
    avg_gap = round(filt["Skill_Gap"].mean(),2) if "Skill_Gap" in filt.columns else "N/A"
    pct_recommend = (filt["Recommend_Training_Flag"].mean()*100) if "Recommend_Training_Flag" in filt.columns else 0
    est_roi = round(filt["Estimated_ROI_USD"].sum(),2) if "Estimated_ROI_USD" in filt.columns else 0

    kk1.markdown(f"<div class='kpi'>{total_employees}<div style='font-size:12px; font-weight:600; color:#333;'>Employees Tracked</div></div>", unsafe_allow_html=True)
    kk2.markdown(f"<div class='kpi'>{total_skills}<div style='font-size:12px; font-weight:600; color:#333;'>Skills Covered</div></div>", unsafe_allow_html=True)
    kk3.markdown(f"<div class='kpi'>{avg_gap}<div style='font-size:12px; font-weight:600; color:#333;'>Avg Skill Gap</div></div>", unsafe_allow_html=True)
    kk4.markdown(f"<div class='kpi'>{pct_recommend:.1f}%<div style='font-size:12px; font-weight:600; color:#333;'>% Recommend Training</div></div>", unsafe_allow_html=True)
    kk5.markdown(f"<div class='kpi'>${est_roi:,.0f}<div style='font-size:12px; font-weight:600; color:#333;'>Estimated ROI (sum)</div></div>", unsafe_allow_html=True)

    # -------------------------
    # Charts (multiple different ones)
    # -------------------------
    st.markdown("### Exploratory Charts")

    # 1) Skill gap distribution (histogram)
    c1, c2 = st.columns(2)
    with c1:
        if "Skill_Gap" in filt.columns:
            fig_gap = px.histogram(filt, x="Skill_Gap", nbins=30, title="Skill Gap Distribution", marginal="box")
            st.plotly_chart(fig_gap, use_container_width=True)
        else:
            st.info("Skill_Gap column not available for histogram.")

    # 2) Top skills by avg gap (bar)
    with c2:
        if "Skill_Gap" in filt.columns:
            grp_skill = filt.groupby("Skill")["Skill_Gap"].mean().reset_index().sort_values("Skill_Gap", ascending=False).head(20)
            fig_bar = px.bar(grp_skill, x="Skill", y="Skill_Gap", title="Top 20 Skills by Avg Gap", labels={"Skill_Gap":"Avg Gap"})
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Skill_Gap column not available for bar chart.")

    # 3) Heatmap: employee vs skill pivot (showing current level)
    st.markdown("#### Skill matrix heatmap (sample)")
    try:
        # pivot takes memory — sample if too large
        piv_sample = filt.sample(min(len(filt), 2000), random_state=42) if len(filt)>2000 else filt
        heat = piv_sample.pivot_table(index="Employee_ID", columns="Skill", values="Current_Level", aggfunc="mean").fillna(0)
        if heat.shape[0] > 0 and heat.shape[1] > 0:
            fig_heat = px.imshow(heat, aspect="auto", color_continuous_scale="Viridis", title="Employee x Skill (Current Level)")
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Not enough data for heatmap.")
    except Exception as e:
        st.info("Heatmap could not be generated: " + str(e))

    # 4) Box plot: current level by role
    st.markdown("#### Skill level distributions by Role")
    try:
        if "Current_Level" in filt.columns and "Role" in filt.columns:
            fig_box = px.box(filt, x="Role", y="Current_Level", title="Current Skill Level by Role")
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Current_Level or Role column missing.")
    except Exception as e:
        st.info("Box plot error: " + str(e))

    # 5) Scatter: Predicted improvement vs cost (if present)
    st.markdown("#### Predicted improvement vs training cost")
    try:
        if "Predicted_Improvement_pts" in filt.columns and "Training_Cost_USD" in filt.columns:
            fig_sc = px.scatter(filt, x="Training_Cost_USD", y="Predicted_Improvement_pts", color="Department",
                                hover_data=["Employee_ID","Skill","Recommended_Training"], title="Cost vs Predicted Improvement")
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("Predicted_Improvement_pts or Training_Cost_USD missing.")
    except Exception as e:
        st.info("Scatter plot error: " + str(e))

    # -------------------------
    # ML Concepts (4): PCA, KMeans, Classifier, Regressor
    # -------------------------
    st.markdown("### ML: 4 Concepts")
    st.markdown("We'll run PCA, clustering, a classifier (Recommend_Training_Flag), and a regressor (Performance_Score).")

    # Prepare modeling dataframe (one row per employee-skill). We'll create a feature matrix of skills per employee using pivot.
    # For clustering and PCA we prefer employee-level features: average current levels per skill.
    employee_matrix = None
    try:
        pivot_emp = filt.pivot_table(index="Employee_ID", columns="Skill", values="Current_Level", aggfunc="mean").fillna(0)
        pivot_emp.columns = [str(c) for c in pivot_emp.columns]
        employee_matrix = pivot_emp.copy()
        st.markdown("#### PCA (employee-level skill profile)")
        if employee_matrix.shape[0] >= 2 and employee_matrix.shape[1] >= 2:
            # Standardize
            scaler_emp = StandardScaler()
            X_emp = scaler_emp.fit_transform(employee_matrix.values)
            pca = PCA(n_components=min(5, X_emp.shape[1]))
            pcs = pca.fit_transform(X_emp)
            pc_df = pd.DataFrame(pcs[:, :2], columns=["PC1","PC2"], index=employee_matrix.index).reset_index()
            fig_pca = px.scatter(pc_df, x="PC1", y="PC2", hover_name="Employee_ID", title="PCA of Employee Skill Profiles")
            st.plotly_chart(fig_pca, use_container_width=True)
            st.markdown(f"PCA explained variance (first 2): {pca.explained_variance_ratio_[:2].sum():.2f}")
        else:
            st.info("Not enough employee-skill columns for PCA.")
    except Exception as e:
        st.info("PCA error: " + str(e))

    # KMeans clustering on employee skill matrix
    st.markdown("#### KMeans clustering (cohorting employees by skills)")
    try:
        if employee_matrix is not None and employee_matrix.shape[0] >= 3:
            n_clusters = st.slider("Select number of clusters for KMeans", min_value=2, max_value=10, value=3)
            scaler_emp = StandardScaler()
            X_emp = scaler_emp.fit_transform(employee_matrix.values)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_emp)
            cl_df = pd.DataFrame({"Employee_ID": employee_matrix.index, "Cluster": clusters})
            # show cluster sizes
            st.write(cl_df["Cluster"].value_counts().sort_index().to_frame("Size"))
            # scatter of first two PCA components colored by cluster
            pca2 = PCA(n_components=2)
            pcs2 = pca2.fit_transform(X_emp)
            pc_cl = pd.DataFrame(pcs2, columns=["PC1","PC2"])
            pc_cl["Cluster"] = clusters.astype(str)
            st.plotly_chart(px.scatter(pc_cl, x="PC1", y="PC2", color="Cluster", title="KMeans clusters on PCA projection"), use_container_width=True)
        else:
            st.info("Not enough employees for clustering.")
    except Exception as e:
        st.info("KMeans error: " + str(e))

    # Classification: predict Recommend_Training_Flag at skill-row level
    st.markdown("#### Classification — Predict Recommend_Training_Flag (RandomForestClassifier)")
    try:
        if "Recommend_Training_Flag" in filt.columns and filt["Recommend_Training_Flag"].nunique() > 1:
            # Use simple features: Current_Level, Required_Level, Skill_Gap, Tenure_Years, Annual_Salary
            clf_features = [c for c in ["Current_Level","Required_Level","Skill_Gap","Tenure_Years","Annual_Salary"] if c in filt.columns]
            if len(clf_features) >= 1:
                data_clf = filt[clf_features + ["Recommend_Training_Flag"]].dropna()
                # If dataset large, sample to speed up
                sample_frac = st.slider("Sample fraction for classifier (reduce if slow)", min_value=0.05, max_value=1.0, value=1.0)
                if sample_frac < 1.0:
                    data_clf = data_clf.sample(frac=sample_frac, random_state=42)
                X = data_clf[clf_features].fillna(0)
                y = data_clf["Recommend_Training_Flag"].astype(int)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()>1 else None)
                clf = RandomForestClassifier(n_estimators=150, random_state=42)
                clf.fit(X_train, y_train)
                prob = clf.predict_proba(X_test)[:,1] if hasattr(clf, "predict_proba") else clf.predict(X_test)
                pred_label = (prob >= 0.5).astype(int)
                acc = accuracy_score(y_test, pred_label)
                try:
                    auc = roc_auc_score(y_test, prob)
                    st.write(f"Classifier Accuracy: {acc:.3f}  | ROC AUC: {auc:.3f}")
                except:
                    st.write(f"Classifier Accuracy: {acc:.3f}")
                # Feature importances
                fi = pd.DataFrame({"feature":clf_features, "importance":clf.feature_importances_}).sort_values("importance", ascending=False)
                st.dataframe(fi)
                # Prepare downloadable predictions
                preds_df = X_test.reset_index(drop=True).copy()
                preds_df["Actual"] = y_test.reset_index(drop=True)
                preds_df["Pred_Prob"] = prob
                preds_df["Pred_Label"] = pred_label
                st.markdown("Sample classifier predictions (test set)")
                st.dataframe(preds_df.head(10))
                download_df(preds_df, "classifier_predictions.csv", "Download classifier predictions")
            else:
                st.info("Not enough features for classifier.")
        else:
            st.info("Recommend_Training_Flag missing or constant — cannot run classifier.")
    except Exception as e:
        st.info("Classifier error: " + str(e))

    # Regression: predict Performance_Score at employee level
    st.markdown("#### Regression — Predict Performance_Score (RandomForestRegressor)")
    try:
        if "Performance_Score" in filt.columns:
            # build employee-level features: avg current level, avg gap, tenure, salary
            emp_feats = filt.groupby("Employee_ID").agg(
                Avg_Current_Level=("Current_Level","mean"),
                Avg_Required_Level=("Required_Level","mean"),
                Avg_Gap=("Skill_Gap","mean"),
                Training_Count=("Training_Count","sum"),
                Annual_Salary=("Annual_Salary","mean"),
                Tenure_Years=("Tenure_Years","mean"),
                Performance_Score=("Performance_Score","mean")
            ).reset_index()
            emp_feats = emp_feats.fillna(0)
            # remove rows without target
            emp_feats = emp_feats[~emp_feats["Performance_Score"].isna()]
            if len(emp_feats) >= 10:
                reg_features = ["Avg_Current_Level","Avg_Required_Level","Avg_Gap","Training_Count","Annual_Salary","Tenure_Years"]
                Xr = emp_feats[reg_features]
                yr = emp_feats["Performance_Score"]
                Xtr,Xte,ytr,yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)
                rfr = RandomForestRegressor(n_estimators=150, random_state=42)
                rfr.fit(Xtr, ytr)
                preds_r = rfr.predict(Xte)
                st.write(f"Regression RMSE: {mean_squared_error(yte,preds_r,squared=False):.3f}, R2: {r2_score(yte,preds_r):.3f}")
                out_reg = Xte.reset_index(drop=True).copy()
                out_reg["Actual_Perf"] = yte.reset_index(drop=True)
                out_reg["Pred_Perf"] = preds_r
                st.dataframe(out_reg.head(10))
                download_df(out_reg, "regression_predictions.csv", "Download regression predictions")
                # feature importances
                fi_r = pd.DataFrame({"feature": reg_features, "importance": rfr.feature_importances_}).sort_values("importance", ascending=False)
                st.dataframe(fi_r)
            else:
                st.info("Not enough employee records with Performance_Score to train regression (need >=10).")
        else:
            st.info("Performance_Score not found — cannot run regression.")
    except Exception as e:
        st.info("Regression error: " + str(e))

    # -------------------------
    # Automated Insights generation
    # -------------------------
    st.markdown("### Automated Insights")
    insights = []
    try:
        # Top skills with largest average gap
        if "Skill_Gap" in filt.columns:
            top_skill_gaps = filt.groupby("Skill")["Skill_Gap"].mean().sort_values(ascending=False).head(10)
            for skill, gap in top_skill_gaps.items():
                insights.append({"Insight_Type":"Top Skill Gap", "Skill":skill, "Avg_Gap":round(float(gap),2)})
        # Employees with highest average gap
        emp_gap = filt.groupby("Employee_ID")["Skill_Gap"].mean().sort_values(ascending=False).head(10)
        for emp, gap in emp_gap.items():
            insights.append({"Insight_Type":"High Gap Employee", "Employee_ID": emp, "Avg_Gap": round(float(gap),2)})
        # Top recommended trainings by count
        if "Recommended_Training" in filt.columns:
            top_recs = filt[filt["Recommend_Training_Flag"]==1]["Recommended_Training"].value_counts().head(10)
            for rec, cnt in top_recs.items():
                insights.append({"Insight_Type":"Recommended Training", "Training": rec, "Count": int(cnt)})
        # ROI summary
        if "Estimated_ROI_USD" in filt.columns:
            tot_roi = round(filt["Estimated_ROI_USD"].sum(),2)
            insights.append({"Insight_Type":"ROI Summary", "Estimated_Total_ROI_USD": tot_roi})
    except Exception as e:
        insights.append({"Insight_Type":"Insight Error", "Message": str(e)})

    insights_df = pd.DataFrame(insights).fillna("")
    if insights_df.empty:
        st.info("No automated insights available for this filter.")
    else:
        st.dataframe(insights_df, use_container_width=True)
        download_df(insights_df, "automated_insights_skillgap.csv", "Download automated insights")

    # -------------------------
    # Final exports & notes
    # -------------------------
    st.markdown("### Exports & Next steps")
    col_a, col_b = st.columns(2)
    col_a.markdown("Download full filtered dataset:")
    download_df(filt.reset_index(drop=True), "skill_gap_filtered.csv", "Download filtered dataset")
    col_b.markdown("Save aggregated employee matrix (useful for clustering/PCA):")
    if 'employee_matrix' in locals() and employee_matrix is not None:
        download_df(employee_matrix.reset_index(), "employee_skill_matrix.csv", "Download employee skill matrix")

    st.markdown("""
    <div style='margin-top:14px; font-size:13px; color:#555'>
    Notes: This app runs small ML models in-browser. For production, train and persist models in a proper ML pipeline (feature store, model registry). Treat the ROI numbers as indicative — adjust monetary params and improvement multipliers to reflect your business.
    </div>
    """, unsafe_allow_html=True)

# End of app
