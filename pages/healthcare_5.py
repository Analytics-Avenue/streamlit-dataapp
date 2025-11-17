# app_hospital.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
import math
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Config + CSS
# ---------------------------
st.set_page_config(page_title="HospitalOps — Capacity & Risk Lab", layout="wide")
st.markdown("<h1 style='margin-bottom:0.2rem'>HospitalOps — Capacity & Risk Lab</h1>", unsafe_allow_html=True)
st.markdown("Operational analytics for hospital chains: facility gaps, equipment shortages, risk scoring and ML suggestions.")

st.markdown("""
<style>
/* Card + metric styling */
div[data-testid="stMarkdownContainer"] .card {
    background: rgba(255,255,255,0.06);
    padding: 16px 18px;
    border-radius: 12px;
    margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 6px 22px rgba(0,0,0,0.12);
    backdrop-filter: blur(3px);
}
.metric-card {
    background: rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.16);
    font-weight: 700;
    transition: all 0.22s ease;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}
.metric-card:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 6px 28px rgba(0,0,0,0.18), 0 0 18px rgba(255,255,255,0.04) inset;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Utilities
# ---------------------------
RAW_DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/healthcare/healthcare_syn.csv"

REQUIRED_HEALTH_COLS = [
    "Patient_ID", "Visit_Date", "Age", "Gender", "Department", "Diagnosis",
    "Treatment", "Treatment_Cost", "Length_of_Stay_days", "Outcome", "Readmission",
    "Risk_Score", "Risk_Level", "Country", "City"
]

def to_currency(x):
    try:
        return "₹ " + f"{float(x):,.2f}"
    except:
        return x

def download_df(df, filename):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button("Download CSV", b, file_name=filename, mime="text/csv")

def auto_map_health_columns(df):
    """Try to map likely column names to canonical names defined above."""
    rename = {}
    cols = [c.strip() for c in df.columns]
    candidates = {
        "Patient_ID":["patient_id","id","patient id"],
        "Visit_Date":["visit_date","visit date","date","visit"],
        "Age":["age"],
        "Gender":["gender","sex"],
        "Department":["department","dept"],
        "Diagnosis":["diagnosis","disease","condition"],
        "Treatment":["treatment","treatment_type"],
        "Treatment_Cost":["treatment_cost","cost","bill","treatment cost"],
        "Length_of_Stay_days":["length_of_stay","length_of_stay_days","los","stay_days"],
        "Outcome":["outcome","discharge_status"],
        "Readmission":["readmission","readmit","re_admission"],
        "Risk_Score":["risk_score","score"],
        "Risk_Level":["risk_level","risk"],
        "Country":["country"],
        "City":["city"]
    }
    for req, cands in candidates.items():
        for c in cols:
            low = c.lower().strip()
            for cand in cands:
                if cand in low or low in cand:
                    rename[c] = req
                    break
            if c in rename:
                break
    if rename:
        df = df.rename(columns=rename)
    return df

# ---------------------------
# Session state cushions
# ---------------------------
if "hosp_pipeline" not in st.session_state:
    st.session_state.hosp_pipeline = None
if "hosp_preprocessor" not in st.session_state:
    st.session_state.hosp_preprocessor = None
if "loaded_df" not in st.session_state:
    st.session_state.loaded_df = None

# ---------------------------
# Tabs: Overview / Application
# ---------------------------
tabs = st.tabs(["Overview", "Application"])

# ---------------------------
# LOAD dataset — inside Application tab (but we show a small preview in Overview after load)
# ---------------------------
with tabs[1]:
    st.header("Step 1 — Load dataset")
    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True, key="dataset_mode_hosp_ops")

    df = None

    if mode == "Default dataset":
        try:
            df = pd.read_csv(RAW_DEFAULT_URL)
            df.columns = df.columns.str.strip()
            df = auto_map_health_columns(df)
            st.success("Default dataset loaded from repository (raw GitHub link).")
            st.dataframe(df.head())
            st.session_state.loaded_df = df
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    elif mode == "Upload CSV":
        st.markdown("#### Upload CSV")
        uploaded = st.file_uploader("Upload CSV file (healthcare)", type=["csv"], key="upload_hosp_ops")
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                df.columns = df.columns.str.strip()
                df = auto_map_health_columns(df)
                st.success("Uploaded dataset loaded and auto-mapped.")
                st.dataframe(df.head())
                st.session_state.loaded_df = df
            except Exception as e:
                st.error("Failed to read uploaded file: " + str(e))
                st.stop()

    else:  # Upload + mapping
        st.markdown("#### Upload CSV and map columns")
        uploaded_map = st.file_uploader("Upload CSV to map columns", type=["csv"], key="upload_map_hosp_ops")
        if uploaded_map is not None:
            raw = pd.read_csv(uploaded_map)
            raw.columns = raw.columns.str.strip()
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head())
            st.markdown("Map your columns to the required fields (only required fields shown).")
            mapping = {}
            for req in REQUIRED_HEALTH_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + list(raw.columns), key=f"map_{req}")
            if st.button("Apply mapping", key="apply_map_hosp_ops"):
                missing = [k for k,v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v:k for k,v in mapping.items()})
                    st.success("Mapping applied.")
                    st.dataframe(df.head())
                    st.session_state.loaded_df = df

# If nothing loaded yet, stop rest of app — we show Overview once loaded below
if st.session_state.loaded_df is None:
    with tabs[0]:
        st.markdown("### Overview")
        st.info("No dataset loaded yet. Go to Application tab and load a dataset (Default / Upload / Upload + mapping).")
    st.stop()

# Use the loaded dataset going forward
df = st.session_state.loaded_df.copy()

# ---------------------------
# Basic cleaning / parsing
# ---------------------------
# Ensure Visit_Date exists under canonical name
if "Visit_Date" in df.columns:
    try:
        df["Visit_Date"] = pd.to_datetime(df["Visit_Date"], errors="coerce")
    except:
        pass

# Fill likely numeric columns and standardize names
numeric_candidates = {
    "Age": 0,
    "Treatment_Cost": 0.0,
    "Length_of_Stay_days": 0.0,
    "Risk_Score": 0.0,
    "Prescription_Count": 0
}
for col, default in numeric_candidates.items():
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

# Normalize readmission field to 0/1 if possible
if "Readmission" in df.columns:
    df["Readmission_Flag"] = df["Readmission"].apply(lambda x: 1 if str(x).strip().lower() in ["yes","1","true","y","t"] else 0)
else:
    df["Readmission_Flag"] = 0

# Create or normalize length of stay
if "Length_of_Stay_days" not in df.columns and ("Admission_Date" in df.columns and "Discharge_Date" in df.columns):
    try:
        df["Admission_Date"] = pd.to_datetime(df["Admission_Date"], errors="coerce")
        df["Discharge_Date"] = pd.to_datetime(df["Discharge_Date"], errors="coerce")
        df["Length_of_Stay_days"] = (df["Discharge_Date"] - df["Admission_Date"]).dt.days.fillna(0)
    except:
        df["Length_of_Stay_days"] = 0

# Make sure some important columns exist; if not, add safe placeholders
for col in ["Patient_ID", "Visit_Date", "Age", "Gender", "Department", "Diagnosis", "Treatment", "Treatment_Cost", "Outcome", "Risk_Level", "Risk_Score"]:
    if col not in df.columns:
        df[col] = np.nan

# ---------------------------
# Overview tab (now that df loaded)
# ---------------------------
with tabs[0]:
    st.markdown("### Overview")
    st.markdown("""
    <div class='card'>
        HospitalOps gives operational visibility across patients & hospitals:
        patient trends, treatment cost, readmission insights, risk scoring and ML-powered suggestions.
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(f"<div class='metric-card'>Patients: {df['Patient_ID'].nunique()}</div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='metric-card'>Records: {len(df):,}</div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='metric-card'>Avg Age: {df['Age'].dropna().mean():.1f}</div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='metric-card'>Avg LOS (days): {df['Length_of_Stay_days'].dropna().mean():.1f}</div>", unsafe_allow_html=True)
    k5.markdown(f"<div class='metric-card'>Readmission Rate: {df['Readmission_Flag'].mean()*100:.2f}%</div>", unsafe_allow_html=True)

    st.markdown("### Capabilities & Impact")
    st.markdown("""
    <div class='card'>
      • Spot departments with high readmission and high cost.<br>
      • Identify top diagnoses, treatment cost drivers and seasonal patterns.<br>
      • Train readmission and cost prediction models and run single-row simulations.<br>
      • Export filtered results for procurement and operational decisions.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# Application tab content (continued)
# ---------------------------
with tabs[1]:
    st.markdown("### Step 2 — Filters & preview")
    c1, c2, c3 = st.columns([2,2,1])
    departments = sorted(df["Department"].dropna().unique().tolist())
    genders = sorted(df["Gender"].dropna().unique().tolist())
    date_min = df["Visit_Date"].min() if not df["Visit_Date"].isna().all() else None
    date_max = df["Visit_Date"].max() if not df["Visit_Date"].isna().all() else None

    with c1:
        sel_depts = st.multiselect("Department", options=departments, default=departments[:6])
    with c2:
        sel_genders = st.multiselect("Gender", options=genders, default=genders)
    with c3:
        date_range = st.date_input("Visit date range", value=(date_min, date_max))

    filt = df.copy()
    if sel_depts:
        filt = filt[filt["Department"].isin(sel_depts)]
    if sel_genders:
        filt = filt[filt["Gender"].isin(sel_genders)]
    if date_range and date_range[0] is not None:
        start, end = date_range
        if start and end:
            filt = filt[(filt["Visit_Date"] >= pd.to_datetime(start)) & (filt["Visit_Date"] <= pd.to_datetime(end))]

    st.markdown("Filtered preview (top 10 rows)")
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(200), "healthcare_filtered_preview.csv")

    # Key metrics for filtered
    st.markdown("### Key Metrics (filtered)")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Unique Patients", f"{filt['Patient_ID'].nunique():,}")
    m2.metric("Avg Treatment Cost", to_currency(filt['Treatment_Cost'].mean() if 'Treatment_Cost' in filt.columns else 0))
    m3.metric("Avg LOS (days)", f"{filt['Length_of_Stay_days'].mean():.2f}")
    m4.metric("Readmission Rate", f"{filt['Readmission_Flag'].mean()*100:.2f}%")

    # Charts
    st.markdown("### Charts")
    # Outcome pie
    if "Outcome" in filt.columns:
        fig_out = px.pie(filt, names="Outcome", title="Treatment Outcomes")
        st.plotly_chart(fig_out, use_container_width=True)

    # Age distribution
    if "Age" in filt.columns:
        fig_age = px.histogram(filt, x="Age", nbins=30, title="Age distribution")
        st.plotly_chart(fig_age, use_container_width=True)

    # Readmission by Department
    if "Department" in filt.columns and "Readmission_Flag" in filt.columns:
        dept_read = filt.groupby("Department")["Readmission_Flag"].mean().reset_index().sort_values("Readmission_Flag", ascending=False)
        fig_dept = px.bar(dept_read, x="Department", y="Readmission_Flag", title="Readmission rate by Department", labels={"Readmission_Flag":"Readmission Rate"})
        st.plotly_chart(fig_dept, use_container_width=True)

    # Top diagnoses by average cost
    if "Diagnosis" in filt.columns and "Treatment_Cost" in filt.columns:
        diag_cost = filt.groupby("Diagnosis").agg({"Treatment_Cost":"mean","Patient_ID":"count"}).rename(columns={"Patient_ID":"count"}).reset_index()
        diag_cost = diag_cost.sort_values("Treatment_Cost", ascending=False).head(15)
        fig_diag = px.bar(diag_cost, x="Diagnosis", y="Treatment_Cost", title="Top diagnoses by avg treatment cost")
        st.plotly_chart(fig_diag, use_container_width=True)

    # -------------------------
    # Step 3: ML — Readmission classification & Treatment cost regression
    st.markdown("### Step 3 — Predictive models (classification & regression)")

    # Choose which model(s) to run
    tasks = st.multiselect("Select tasks to run", options=["Readmission (classification)", "Treatment cost (regression)"], default=["Readmission (classification)"])

    # Provide feature choices (auto-suggest numeric + categorical)
    possible_feats = [c for c in filt.columns if c not in ["Patient_ID","Visit_Date","Doctor_Notes","Outcome","Readmission","Readmission_Flag","Treatment_Cost"]]
    numeric_feats = filt.select_dtypes(include=[np.number]).columns.tolist()
    suggested = [c for c in ["Age","Length_of_Stay_days","Risk_Score","Prescription_Count"] if c in possible_feats]

    feat_cols = st.multiselect("Features (choose at least 2)", options=possible_feats, default=[f for f in suggested if f in possible_feats][:4])

    if len(feat_cols) < 2 and len(tasks) > 0:
        st.info("Pick at least 2 features to train models.")
    else:
        # Build X, and train models separately as requested
        X = filt[feat_cols].copy().reset_index(drop=True)

        # Identify numeric and categorical in chosen features
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in feat_cols if c not in num_cols]

        # Preprocessor
        transformers = []
        if num_cols:
            transformers.append(("num", StandardScaler(), num_cols))
        if cat_cols:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

        if "Readmission (classification)" in tasks:
            st.markdown("#### Readmission (classification)")
            if "Readmission_Flag" not in filt.columns:
                st.warning("No Readmission field found in dataset to train classification.")
            else:
                y_clf = filt["Readmission_Flag"].reset_index(drop=True)
                # Align X,y lengths after dropna
                combined = pd.concat([X, y_clf], axis=1).dropna()
                X_clf = combined[feat_cols]
                y_clf = combined["Readmission_Flag"]
                if X_clf.shape[0] < 30:
                    st.warning("Not enough rows to reliably train classification (need ~30+).")
                else:
                    clf_pipe = Pipeline([("prep", preprocessor), ("model", RandomForestClassifier(n_estimators=150, random_state=42))])
                    test_size = st.slider("Test size for classification", 0.1, 0.4, 0.2, key="testsize_clf")
                    if st.button("Train classification model", key="train_clf"):
                        X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=test_size, random_state=42, stratify=y_clf if y_clf.nunique()>1 else None)
                        clf_pipe.fit(X_train, y_train)
                        st.session_state.hosp_pipeline = clf_pipe
                        st.session_state.hosp_preprocessor = preprocessor
                        y_pred = clf_pipe.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        auc = None
                        try:
                            if hasattr(clf_pipe, "predict_proba"):
                                auc = roc_auc_score(y_test, clf_pipe.predict_proba(X_test)[:,1])
                        except:
                            auc = None
                        st.success(f"Classification trained — accuracy: {acc:.3f}" + (f"  | ROC-AUC: {auc:.3f}" if auc else ""))

                        # Feature importance (approx) — map names
                        try:
                            importances = clf_pipe.named_steps["model"].feature_importances_
                            feat_names = []
                            if num_cols:
                                feat_names += num_cols
                            if cat_cols:
                                cat_names = clf_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
                                feat_names += cat_names
                            fi = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
                            st.markdown("Top feature importances (classification)")
                            st.dataframe(fi.head(10))
                            fig_fi = px.bar(fi.head(10), x="importance", y="feature", orientation="h", title="Feature importances (classification)")
                            st.plotly_chart(fig_fi, use_container_width=True)
                        except Exception:
                            st.info("Feature importance not available.")

                        # Optional SHAP
                        try:
                            import shap
                            import matplotlib.pyplot as plt
                            st.markdown("SHAP analysis (classification) — may take a few seconds")
                            explainer = shap.TreeExplainer(clf_pipe.named_steps["model"])
                            X_test_trans = clf_pipe.named_steps["prep"].transform(X_test)
                            shap_values = explainer.shap_values(X_test_trans)
                            fig_shap = plt.figure(figsize=(6,4))
                            # shap summary - for binary classification show class 1
                            if isinstance(shap_values, list) and len(shap_values) > 1:
                                shap.summary_plot(shap_values[1], features=X_test_trans, feature_names=fi["feature"].tolist(), show=False)
                            else:
                                shap.summary_plot(shap_values, features=X_test_trans, feature_names=fi["feature"].tolist(), show=False)
                            st.pyplot(fig_shap)
                        except Exception as e:
                            st.info("SHAP unavailable or failed: " + str(e))

        if "Treatment cost (regression)" in tasks:
            st.markdown("#### Treatment cost (regression)")
            if "Treatment_Cost" not in filt.columns:
                st.warning("No Treatment_Cost column found for regression target.")
            else:
                y_reg = filt["Treatment_Cost"].reset_index(drop=True)
                combined_r = pd.concat([X, y_reg], axis=1).dropna()
                X_reg = combined_r[feat_cols]
                y_reg = combined_r["Treatment_Cost"]
                if X_reg.shape[0] < 30:
                    st.warning("Not enough rows to reliably train regression (need ~30+).")
                else:
                    reg_pipe = Pipeline([("prep", preprocessor), ("model", RandomForestRegressor(n_estimators=150, random_state=42))])
                    test_size_r = st.slider("Test size for regression", 0.1, 0.4, 0.2, key="testsize_reg")
                    if st.button("Train regression model", key="train_reg"):
                        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=test_size_r, random_state=42)
                        reg_pipe.fit(X_train_r, y_train_r)
                        st.session_state.hosp_pipeline = reg_pipe
                        st.session_state.hosp_preprocessor = preprocessor
                        preds = reg_pipe.predict(X_test_r)
                        rmse = math.sqrt(mean_squared_error(y_test_r, preds))
                        r2 = r2_score(y_test_r, preds)
                        st.success(f"Regression trained — RMSE: {rmse:.2f} | R²: {r2:.3f}")

                        # Feature importance
                        try:
                            importances = reg_pipe.named_steps["model"].feature_importances_
                            feat_names = []
                            if num_cols:
                                feat_names += num_cols
                            if cat_cols:
                                feat_names += reg_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
                            fi = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
                            st.markdown("Top feature importances (regression)")
                            st.dataframe(fi.head(10))
                            fig_fi_r = px.bar(fi.head(10), x="importance", y="feature", orientation="h", title="Feature importances (regression)")
                            st.plotly_chart(fig_fi_r, use_container_width=True)
                        except Exception:
                            st.info("Feature importance not available for regression.")

                        # Optional SHAP
                        try:
                            import shap
                            import matplotlib.pyplot as plt
                            st.markdown("SHAP analysis (regression) — may take a few seconds")
                            explainer = shap.TreeExplainer(reg_pipe.named_steps["model"])
                            X_test_trans_r = reg_pipe.named_steps["prep"].transform(X_test_r)
                            shap_values_r = explainer.shap_values(X_test_trans_r)
                            fig_shap_r = plt.figure(figsize=(6,4))
                            shap.summary_plot(shap_values_r, features=X_test_trans_r, feature_names=fi["feature"].tolist(), show=False)
                            st.pyplot(fig_shap_r)
                        except Exception as e:
                            st.info("SHAP unavailable or failed: " + str(e))

    # -------------------------
    # Quick single-row prediction (if pipeline exists)
    st.markdown("### Quick single-row prediction & simulation")
    if st.session_state.hosp_pipeline is not None and st.session_state.hosp_preprocessor is not None:
        pipe = st.session_state.hosp_pipeline
        preproc = st.session_state.hosp_preprocessor
        # determine pipeline feature names based on what was used last (best effort)
        last_feat_cols = feat_cols
        input_cols_ui = st.columns(len(last_feat_cols))
        new_row = {}
        for idx, col in enumerate(last_feat_cols):
            if col in num_cols:
                val = input_cols_ui[idx].number_input(col, value=float(X[col].median()), key=f"pred_in_{col}")
            else:
                opts = sorted(X[col].dropna().unique().tolist())
                val = input_cols_ui[idx].selectbox(col, options=opts, index=0 if opts else None, key=f"pred_sel_{col}")
            new_row[col] = val
        if st.button("Predict (single-row)", key="predict_single_row"):
            xr = pd.DataFrame([new_row])
            try:
                pred = pipe.predict(xr)[0]
                if isinstance(pipe.named_steps["model"], RandomForestRegressor):
                    st.success(f"Predicted value: {pred:.2f}")
                else:
                    st.success(f"Predicted class/value: {pred}")
            except Exception as e:
                st.error("Prediction failed: " + str(e))
    else:
        st.info("Train a model (classification or regression) first to enable single-row prediction.")

    # -------------------------
    # Export filtered dataset
    st.markdown("### Export & Notes")
    if st.button("Download filtered dataset (CSV)", key="dl_filtered_hosp"):
        download_df(filt, "hospitalops_filtered.csv")
    st.markdown("Notes: Use this app to prioritize departments/hospitals for interventions based on readmission, cost and LOS.")

# End of app
