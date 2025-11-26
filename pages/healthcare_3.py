import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
import shap
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings("ignore")

# ==========================================================
# PAGE CONFIG & HIDE SIDEBAR
# ==========================================================
st.set_page_config(page_title="PatientFlow Navigator", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebarNav"] {display:none;}
section[data-testid="stSidebar"] {display:none;}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# GLOBAL CSS – MARKETING LAB STYLE
# ==========================================================
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
@keyframes fadeIn {
  from {opacity:0; transform:translateY(10px);}
  to   {opacity:1; transform:translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# COMPANY HEADER
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

st.markdown("<div class='big-header'>PatientFlow Navigator</div>", unsafe_allow_html=True)
st.write("Enterprise-capable healthcare analytics: patient trends, treatment efficiency, readmission prediction, and cost forecasting.")

# ==========================================================
# REQUIRED COLUMNS
# ==========================================================
REQUIRED_HEALTH_COLS = [
    "Patient_ID", "Age", "Gender", "Disease", "Symptoms", "Treatment",
    "Admission_Date", "Discharge_Date", "Treatment_Cost", "Readmission",
    "Department", "Visit_Date", "Outcome", "Risk_Level", "Risk_Score"
]

def auto_map_health_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    cols = [c.strip() for c in df.columns]
    mapping_candidates = {
        "Patient_ID":["patient_id","id","PatientID"],
        "Age":["age","Age"],
        "Gender":["gender","Gender","sex"],
        "Disease":["disease","Diagnosis","Condition"],
        "Symptoms":["symptoms","Symptom","Symptom_List"],
        "Treatment":["treatment","Treatment_Type"],
        "Admission_Date":["admission","admission_date","Admit_Date"],
        "Discharge_Date":["discharge","discharge_date","Discharge_Date"],
        "Treatment_Cost":["cost","Treatment_Cost","Bill"],
        "Readmission":["readmission","Readmit","Rehospitalization"],
        "Department":["department","Dept"],
        "Visit_Date":["visit_date","VisitDate"],
        "Outcome":["outcome","Treatment_Outcome"],
        "Risk_Level":["risk_level","RiskLevel"],
        "Risk_Score":["risk_score","RiskScore"]
    }
    for req, candidates in mapping_candidates.items():
        for c in cols:
            low = c.lower().strip()
            for cand in candidates:
                if cand.lower() in low or low in cand.lower():
                    rename[c] = req
                    break
            if c in rename:
                break
    if rename:
        df = df.rename(columns=rename)
    return df

def download_df(df, filename, label="Download CSV"):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")

def to_currency(val):
    try:
        return f"₹ {float(val):,.2f}"
    except:
        return "₹ 0.00"

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
    <div class='card'>
        Track patient journeys, treatment patterns, readmission risk, and cost dynamics.
        PatientFlow Navigator lets hospitals move from reactive firefighting to proactive care planning.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">What This Lab Does</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class='card'>
        • Profiles patient cohorts by <b>disease, age, risk & department</b><br>
        • Measures <b>length of stay</b>, treatment cost, and readmission patterns<br>
        • Flags <b>high-risk patients</b> using ML classification<br>
        • Predicts <b>treatment cost</b> for upcoming admissions<br>
        • Supports <b>capacity & resource planning</b> for hospitals
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class='card'>
        • Reduced <b>avoidable readmissions</b><br>
        • Optimized <b>bed & staff allocation</b><br>
        • Better <b>cost control & patient billing transparency</b><br>
        • Improved <b>quality of care & patient outcomes</b><br>
        • Strong analytic layer for <b>hospital administrators & clinical leads</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Key KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Total Patients</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Avg Age</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Avg Length of Stay</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Readmission Rate</div>", unsafe_allow_html=True)

# ==========================================================
# TAB 2 – IMPORTANT ATTRIBUTES
# ==========================================================
with tab2:
    st.markdown('<div class="section-title">Important Attributes</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        independents = [
            "Age",
            "Gender",
            "Disease",
            "Symptoms",
            "Treatment",
            "Department",
            "Visit_Date",
            "Risk_Level",
            "Risk_Score",
            "Admission_Date",
            "Discharge_Date",
            "Length_of_Stay (derived)"
        ]
        for v in independents:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        dependents = [
            "Readmission",
            "Treatment_Cost",
            "Outcome"
        ]
        for v in dependents:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# ==========================================================
# TAB 3 – APPLICATION
# ==========================================================
with tab3:
    st.markdown('<div class="section-title">Step 1: Load dataset</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Dataset option:",
        ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"],
        horizontal=True
    )

    df = None

    if mode == "Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/healthcare/healthcare_3.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            df = auto_map_health_columns(df)
            st.success("Default dataset loaded & auto-mapped.")
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load default dataset: {e}")
            st.stop()

    elif mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            df = auto_map_health_columns(df)
            st.success("File uploaded and auto-mapped.")
            st.dataframe(df.head(10), use_container_width=True)

    else:  # Upload + mapping
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded:
            raw = pd.read_csv(uploaded)
            st.write("Preview (first 10 rows):")
            st.dataframe(raw.head(10), use_container_width=True)
            mapping = {}
            for req in REQUIRED_HEALTH_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + list(raw.columns))
            if st.button("Apply mapping"):
                missing = [k for k, v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v: k for k, v in mapping.items()})
                    st.success("Mapping applied.")
                    st.dataframe(df.head(10), use_container_width=True)

    if df is None:
        st.stop()

    # ======================================================
    # CLEANUP & DERIVED COLUMNS
    # ======================================================
    # Ensure date fields
    for c in ["Admission_Date", "Discharge_Date", "Visit_Date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Length of Stay
    if "Admission_Date" in df.columns and "Discharge_Date" in df.columns:
        df["Length_of_Stay"] = (df["Discharge_Date"] - df["Admission_Date"]).dt.days
        df["Length_of_Stay"] = df["Length_of_Stay"].fillna(0).clip(lower=0)
    else:
        df["Length_of_Stay"] = 0

    # Treatment cost numeric
    if "Treatment_Cost" in df.columns:
        df["Treatment_Cost"] = pd.to_numeric(df["Treatment_Cost"], errors="coerce").fillna(0)
    else:
        df["Treatment_Cost"] = 0

    # Risk score numeric
    if "Risk_Score" in df.columns:
        df["Risk_Score"] = pd.to_numeric(df["Risk_Score"], errors="coerce").fillna(0)
    else:
        df["Risk_Score"] = 0

    # Readmission flag
    if "Readmission" in df.columns:
        df["Readmission_Flag"] = df["Readmission"].astype(str).str.lower().isin(["yes", "1", "true"]).astype(int)
    else:
        df["Readmission_Flag"] = 0

    # Age numeric
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(0)
    else:
        df["Age"] = 0

    # ======================================================
    # FILTERS
    # ======================================================
    st.markdown('<div class="section-title">Step 2: Filters & Preview</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    departments = df["Department"].dropna().unique().tolist() if "Department" in df.columns else []
    genders = df["Gender"].dropna().unique().tolist() if "Gender" in df.columns else []
    diseases = df["Disease"].dropna().unique().tolist() if "Disease" in df.columns else []

    with c1:
        sel_dept = st.multiselect("Department", options=departments, default=departments[:3] if departments else [])
    with c2:
        sel_gender = st.multiselect("Gender", options=genders, default=genders if genders else [])
    with c3:
        if "Visit_Date" in df.columns and not df["Visit_Date"].isna().all():
            min_d = df["Visit_Date"].min().date()
            max_d = df["Visit_Date"].max().date()
            date_range = st.date_input("Visit Date Range", value=(min_d, max_d))
        else:
            date_range = None

    filt = df.copy()
    if sel_dept:
        filt = filt[filt["Department"].isin(sel_dept)]
    if sel_gender:
        filt = filt[filt["Gender"].isin(sel_gender)]
    if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2 and "Visit_Date" in filt.columns:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filt = filt[(filt["Visit_Date"] >= start) & (filt["Visit_Date"] <= end)]

    # Display filter info
    st.markdown(f"""
    <div style='padding:8px; background:#eef4ff; border-radius:8px; font-weight:600; margin-bottom:8px;'>
    Filtered rows: {len(filt)} of {len(df)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Data Preview (first 10 rows)")
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(500), "patientflow_filtered_preview.csv", "Download filtered preview (first 500 rows)")

    if filt.empty:
        st.warning("Filtered dataset is empty. Relax your filters.")
        st.stop()

    # ======================================================
    # KPIs
    # ======================================================
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    total_patients = filt["Patient_ID"].nunique() if "Patient_ID" in filt.columns else len(filt)
    avg_age = filt["Age"].mean() if "Age" in filt.columns else np.nan
    avg_los = filt["Length_of_Stay"].mean()
    readm_rate = filt["Readmission_Flag"].mean() * 100 if "Readmission_Flag" in filt.columns else 0
    total_cost = filt["Treatment_Cost"].sum()

    k1.metric("Total Patients", f"{total_patients:,}")
    k2.metric("Avg Age", f"{avg_age:.1f}" if not math.isnan(avg_age) else "N/A")
    k3.metric("Avg Length of Stay (days)", f"{avg_los:.2f}")
    k4.metric("Readmission Rate", f"{readm_rate:.2f}%")

    # ======================================================
    # CHARTS
    # ======================================================
    st.markdown('<div class="section-title">Exploratory Charts</div>', unsafe_allow_html=True)

    # 1) Disease distribution
    if "Disease" in filt.columns:
        dis = filt["Disease"].value_counts().reset_index()
        dis.columns = ["Disease", "Count"]
        fig_dis = px.bar(dis, x="Disease", y="Count", title="Patient Count by Disease", text="Count")
        fig_dis.update_traces(textposition="outside")
        fig_dis.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_dis, use_container_width=True)

    # 2) Treatment cost distribution
    if "Treatment_Cost" in filt.columns:
        fig_cost = px.histogram(filt, x="Treatment_Cost", nbins=30, title="Treatment Cost Distribution")
        st.plotly_chart(fig_cost, use_container_width=True)

    # 3) Risk score vs Length of Stay
    if "Risk_Score" in filt.columns:
        fig_risk = px.scatter(
            filt,
            x="Risk_Score",
            y="Length_of_Stay",
            color="Department" if "Department" in filt.columns else None,
            title="Risk Score vs Length of Stay"
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    # ======================================================
    # PREDICTIVE ANALYTICS
    # ======================================================
    st.markdown('<div class="section-title">Predictive Analytics</div>', unsafe_allow_html=True)

    # ---------------------------
    # Readmission Prediction
    # ---------------------------
    st.markdown("#### Readmission Prediction (Classification)")

    req_clf_cols = ["Age", "Gender", "Department", "Risk_Score", "Length_of_Stay", "Readmission_Flag"]
    if all(c in filt.columns for c in req_clf_cols) and filt["Readmission_Flag"].nunique() > 1:

        clf_df = filt[req_clf_cols].dropna().copy()
        if len(clf_df) > 20:
            le_gender = LabelEncoder()
            le_dept = LabelEncoder()

            clf_df["Gender_enc"] = le_gender.fit_transform(clf_df["Gender"].astype(str))
            clf_df["Dept_enc"] = le_dept.fit_transform(clf_df["Department"].astype(str))

            X_clf = clf_df[["Age", "Gender_enc", "Dept_enc", "Risk_Score", "Length_of_Stay"]]
            y_clf = clf_df["Readmission_Flag"]

            X_train, X_test, y_train, y_test = train_test_split(
                X_clf, y_clf, test_size=0.2, random_state=42
            )

            clf_model = RandomForestClassifier(n_estimators=120, random_state=42)
            with st.spinner("Training readmission model..."):
                clf_model.fit(X_train, y_train)

            y_pred = clf_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write(f"Model accuracy: **{acc*100:.2f}%**")

            # New patient prediction
            st.markdown("##### Predict Readmission for a New Patient")
            age_in = st.number_input("Age", 0, 120, 40, key="clf_age")
            gender_in = st.selectbox("Gender", sorted(filt["Gender"].dropna().unique().tolist()), key="clf_gender")
            dept_in = st.selectbox("Department", sorted(filt["Department"].dropna().unique().tolist()), key="clf_dept")
            risk_in = st.number_input("Risk Score", 0.0, 10.0, 3.0, step=0.1, key="clf_risk")
            los_in = st.number_input("Length of Stay (days)", 0, 100, 5, key="clf_los")

            if st.button("Predict Readmission", key="btn_readmission"):
                x_new = pd.DataFrame([[
                    age_in,
                    le_gender.transform([gender_in])[0],
                    le_dept.transform([dept_in])[0],
                    risk_in,
                    los_in
                ]], columns=X_clf.columns)
                pred = clf_model.predict(x_new)[0]
                st.success(f"Predicted Readmission: {'Yes' if pred == 1 else 'No'}")

            # Feature importance
            fi_df = pd.DataFrame({
                "Feature": X_clf.columns,
                "Importance": clf_model.feature_importances_
            }).sort_values("Importance", ascending=False)
            fig_fi = px.bar(fi_df, x="Feature", y="Importance", title="Readmission Feature Importance")
            fig_fi.update_traces(text=fi_df["Importance"].round(3), textposition="outside")
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Not enough rows with complete data for classification (need >20).")

    else:
        st.warning("Required columns or label variation missing for readmission model.")

    # ---------------------------
    # Treatment Cost Prediction
    # ---------------------------
    st.markdown("#### Treatment Cost Prediction (Regression)")

    req_reg_cols = ["Age", "Gender", "Department", "Risk_Score", "Length_of_Stay", "Treatment_Cost"]
    if all(c in filt.columns for c in req_reg_cols) and filt["Treatment_Cost"].sum() > 0:

        reg_df = filt[req_reg_cols].dropna().copy()
        if len(reg_df) > 20:
            le_gender_r = LabelEncoder()
            le_dept_r = LabelEncoder()

            reg_df["Gender_enc"] = le_gender_r.fit_transform(reg_df["Gender"].astype(str))
            reg_df["Dept_enc"] = le_dept_r.fit_transform(reg_df["Department"].astype(str))

            X_reg = reg_df[["Age", "Gender_enc", "Dept_enc", "Risk_Score", "Length_of_Stay"]]
            y_reg = reg_df["Treatment_Cost"]

            X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
                X_reg, y_reg, test_size=0.2, random_state=42
            )

            reg_model = RandomForestRegressor(n_estimators=150, random_state=42)
            with st.spinner("Training treatment cost model..."):
                reg_model.fit(X_train_r, y_train_r)

            y_pred_r = reg_model.predict(X_test_r)
            rmse = math.sqrt(mean_squared_error(y_test_r, y_pred_r))
            st.write(f"RMSE: **{rmse:,.2f}**")

            # New prediction
            st.markdown("##### Predict Treatment Cost for New Patient")
            age_r = st.number_input("Age (Cost)", 0, 120, 40, key="reg_age")
            gender_r = st.selectbox("Gender (Cost)", sorted(filt["Gender"].dropna().unique().tolist()), key="reg_gender")
            dept_r = st.selectbox("Department (Cost)", sorted(filt["Department"].dropna().unique().tolist()), key="reg_dept")
            risk_r = st.number_input("Risk Score (Cost)", 0.0, 10.0, 3.0, step=0.1, key="reg_risk")
            los_r = st.number_input("Length of Stay (Cost)", 0, 100, 5, key="reg_los")

            if st.button("Predict Treatment Cost", key="btn_cost"):
                x_new_r = pd.DataFrame([[
                    age_r,
                    le_gender_r.transform([gender_r])[0],
                    le_dept_r.transform([dept_r])[0],
                    risk_r,
                    los_r
                ]], columns=X_reg.columns)
                pred_cost = reg_model.predict(x_new_r)[0]
                st.success(f"Predicted Treatment Cost: {to_currency(pred_cost)}")

            # Feature importance
            fi_reg = pd.DataFrame({
                "Feature": X_reg.columns,
                "Importance": reg_model.feature_importances_
            }).sort_values("Importance", ascending=False)
            fig_reg = px.bar(fi_reg, x="Feature", y="Importance", title="Treatment Cost Feature Importance")
            fig_reg.update_traces(text=fi_reg["Importance"].round(3), textposition="outside")
            st.plotly_chart(fig_reg, use_container_width=True)

            # SHAP summary plot
            try:
                explainer_r = shap.TreeExplainer(reg_model)
                shap_values_r = explainer_r.shap_values(X_test_r)
                fig_shap, ax = plt.subplots(figsize=(8, 5))
                shap.summary_plot(shap_values_r, X_test_r, plot_type="bar", show=False)
                st.pyplot(fig_shap)
            except Exception:
                st.info("SHAP visualization could not be generated for this environment.")
        else:
            st.info("Not enough rows with complete data for regression model (need >20).")
    else:
        st.warning("Required columns for treatment cost model not available or costs are all zero.")

    # ======================================================
    # AUTOMATED INSIGHTS
    # ======================================================
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)

    insights = []

    # 1) Disease with highest volume
    if "Disease" in filt.columns:
        dis_ct = filt["Disease"].value_counts()
        if not dis_ct.empty:
            insights.append(["Top Disease by Patient Count", dis_ct.index[0]])

    # 2) Department with highest average treatment cost
    if "Department" in filt.columns and "Treatment_Cost" in filt.columns:
        dept_cost = filt.groupby("Department")["Treatment_Cost"].mean().sort_values(ascending=False)
        if not dept_cost.empty:
            insights.append(["Highest Avg Treatment Cost Dept", dept_cost.index[0]])

    # 3) Department with highest readmission rate
    if "Department" in filt.columns and "Readmission_Flag" in filt.columns:
        dept_readm = filt.groupby("Department")["Readmission_Flag"].mean().sort_values(ascending=False)
        if not dept_readm.empty:
            insights.append(["Highest Readmission Rate Dept", dept_readm.index[0]])

    # 4) Risk bucket with longest length of stay
    if "Risk_Level" in filt.columns:
        risk_los = filt.groupby("Risk_Level")["Length_of_Stay"].mean().sort_values(ascending=False)
        if not risk_los.empty:
            insights.append(["Risk Level with Longest Stay", risk_los.index[0]])

    if insights:
        insights_df = pd.DataFrame(insights, columns=["Insight", "Value"])
        st.dataframe(insights_df, use_container_width=True)
        download_df(insights_df, "patientflow_automated_insights.csv", "Download insights CSV")
    else:
        st.info("No insights generated for the current selection.")
