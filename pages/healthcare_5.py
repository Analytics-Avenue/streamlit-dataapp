# app_hospital_full.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Optional
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import math
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Page config + title
# -------------------------
st.set_page_config(page_title="HospitalOps — Full Dashboard", layout="wide")
st.title("🏥 HospitalOps — Capacity, Risk & Ops Lab")
st.caption("Overview → Dataset setup (default / upload / upload+mapping) → Charts → ML → Clustering → Profiles")

# -------------------------
# CSS / Glow styling for Overview
# -------------------------
OVERVIEW_CSS = """
<style>
/* page background */
section.main {
  background: linear-gradient(180deg, #0f1724 0%, #081020 100%);
  color: #e6eef8;
}

/* glow KPI cards */
.kpi-row { display:flex; gap:18px; flex-wrap:wrap; margin-bottom:18px; }
.kpi {
  background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
  border-radius: 12px;
  padding: 18px;
  width: 220px;
  box-shadow: 0 6px 30px rgba(0,120,255,0.08), inset 0 -4px 10px rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.06);
  transition: transform .16s ease, box-shadow .16s ease;
}
.kpi:hover { transform: translateY(-6px); box-shadow: 0 18px 50px rgba(0,120,255,0.18); }

/* title gradient */
.header-gradient {
  font-size: 20px;
  font-weight:700;
  background: linear-gradient(90deg,#9be7ff,#8affc1);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

/* card */
.card {
  background: rgba(255,255,255,0.03);
  padding: 14px;
  border-radius: 10px;
  border: 1px solid rgba(255,255,255,0.05);
  margin-bottom: 12px;
}

/* small muted */
.muted { color: rgba(230,238,248,0.7); font-size:14px; }
</style>
"""
st.markdown(OVERVIEW_CSS, unsafe_allow_html=True)

# -------------------------
# Utilities
# -------------------------
DEFAULT_DATA_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/healthcare/healthcare_syn.csv"

def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep first occurrence of duplicated column names and strip whitespace."""
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = df.columns.str.strip()
    return df

def download_df(df: pd.DataFrame, filename: str = "export.csv"):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button("Download CSV", b, file_name=filename, mime="text/csv")

def safe_onehot(sparse_output_arg=False):
    """Return kwargs for OneHotEncoder depending on sklearn version compatibility."""
    try:
        return {"handle_unknown":"ignore", "sparse_output": sparse_output_arg}
    except TypeError:
        # older sklearn uses 'sparse' param
        return {"handle_unknown":"ignore", "sparse": not sparse_output_arg}

# -------------------------
# Session state placeholders
# -------------------------
if "hospital_master" not in st.session_state:
    st.session_state["hospital_master"] = None
if "equipment_master" not in st.session_state:
    st.session_state["equipment_master"] = None
if "patient_risk" not in st.session_state:
    st.session_state["patient_risk"] = None
if "clf_pipe" not in st.session_state:
    st.session_state["clf_pipe"] = None
if "reg_pipe" not in st.session_state:
    st.session_state["reg_pipe"] = None
if "clustered_master" not in st.session_state:
    st.session_state["clustered_master"] = None

# -------------------------
# Sidebar navigation
# -------------------------
page = st.sidebar.radio("Navigate", ["Overview", "Application"])

# -------------------------
# Overview page
# -------------------------
if page == "Overview":
    st.markdown("<div class='header-gradient'>HospitalOps — Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'><b>About this application</b><p class='muted'>HospitalOps is a lightweight operational analytics app for hospital networks. It helps identify resource gaps, patient risk, and offers simple ML to predict readmission and risk score.</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><b>Purpose</b><p class='muted'>Improve resource allocation, reduce readmissions, prioritize procurement and support data-driven operational decisions.</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><b>Capabilities</b><p class='muted'>Upload or use default datasets, interactive KPIs & charts, clustering and ML models (classification & regression), single-row simulation, and export.</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><b>Business Impact</b><p class='muted'>Faster triage decisions, reduced bed shortages, better equipment utilization and targeted interventions for high-risk patients.</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='kpi-row'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi'><div style='font-size:12px;color:#bcdffb'>Total Hospitals</div><div style='font-size:20px;font-weight:700'>120</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='kpi'><div style='font-size:12px;color:#bcdffb'>High-Risk Hospitals</div><div style='font-size:20px;font-weight:700'>18</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='kpi'><div style='font-size:12px;color:#bcdffb'>Avg Bed Occupancy</div><div style='font-size:20px;font-weight:700'>72%</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='kpi'><div style='font-size:12px;color:#bcdffb'>Ventilators Available</div><div style='font-size:20px;font-weight:700'>1,240</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='kpi'><div style='font-size:12px;color:#bcdffb'>Avg Staff / Hospital</div><div style='font-size:20px;font-weight:700'>68</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # KPR chart sample (radar-like using polar)
    kpr = pd.DataFrame({
        "KPR": ["Response Time", "Bed Mgmt", "Equipment Readiness", "Staffing", "Triage Accuracy"],
        "Score": [72, 65, 60, 68, 70]
    })
    fig_kpr = px.line_polar(kpr, r="Score", theta="KPR", line_close=True, title="KPR - Sample Radar", range_r=[0,100])
    st.plotly_chart(fig_kpr, use_container_width=True)

# -------------------------
# Application page
# -------------------------
else:
    st.header("Application — Dataset Setup & Analytics")

    st.markdown("**Select dataset mode:** default dataset (fetched by URL), user upload, or user upload + manual mapping.")

    dataset_mode = st.radio("Dataset option:", ["Default dataset (URL)", "User upload (CSV)", "User upload + Manual mapping"], horizontal=True)

    # ---------- DEFAULT dataset ----------
    if dataset_mode == "Default dataset (URL)":
        st.subheader("Default dataset (fetched from URL)")
        try:
            df_default = pd.read_csv(DEFAULT_DATA_URL)
            df_default = remove_duplicate_columns(df_default)
            st.session_state["hospital_master"] = df_default  # treat default as hospital_master if possible
            st.success("Default dataset loaded.")
            st.dataframe(df_default.head())
        except Exception as e:
            st.error(f"Failed to load default dataset: {e}")

    # ---------- USER upload ----------
    elif dataset_mode == "User upload (CSV)":
        st.subheader("Upload CSV (interpreted as Hospital Master by default)")
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="upload_simple")
        if uploaded is not None:
            try:
                dfu = pd.read_csv(uploaded)
                dfu = remove_duplicate_columns(dfu)
                st.session_state["hospital_master"] = dfu
                st.success("Uploaded and cleaned (duplicates removed).")
                st.dataframe(dfu.head())
            except Exception as e:
                st.error(f"Failed to read uploaded CSV: {e}")

    # ---------- USER upload + manual mapping ----------
    else:
        st.subheader("Upload CSV + Manual mapping")
        uploaded_map = st.file_uploader("Upload CSV to map columns", type=["csv"], key="upload_map")
        mapped_df: Optional[pd.DataFrame] = None
        if uploaded_map is not None:
            try:
                dfu = pd.read_csv(uploaded_map)
                st.write("Preview of uploaded CSV")
                st.dataframe(dfu.head())
                # remove exact duplicate columns before mapping options
                dfu = dfu.loc[:, ~dfu.columns.duplicated()].copy()
                dfu.columns = dfu.columns.str.strip()
                st.markdown("Map uploaded columns to canonical names. Leave blank to create empty placeholder column.")
                canonical = [
                    "Hospital_Name", "Location", "Hospital_Type", "Monthly_Patients",
                    "Staff_Count", "Beds_Count", "Ventilators_Count", "Risk_Score", "Risk_Level"
                ]
                mapping = {}
                cols_list = ["-- None --"] + list(dfu.columns)
                for c in canonical:
                    mapping[c] = st.selectbox(f"Map to → {c}", options=cols_list, key=f"map_{c}")
                if st.button("Apply mapping"):
                    mapped = pd.DataFrame()
                    for c in canonical:
                        sel = mapping[c]
                        if sel != "-- None --":
                            mapped[c] = dfu[sel]
                        else:
                            mapped[c] = np.nan
                    mapped_df = mapped
                    st.session_state["hospital_master"] = mapped_df
                    st.success("Mapping applied and stored as hospital master.")
                    st.dataframe(mapped_df.head())
            except Exception as e:
                st.error(f"Failed to read file for mapping: {e}")

    # ---------- Show current datasets in session ----------
    st.markdown("---")
    st.subheader("Datasets in session (you can also generate sample datasets below)")

    col_h, col_e, col_p = st.columns(3)
    with col_h:
        st.write("Hospital Master")
        if st.session_state.get("hospital_master") is None:
            st.info("No hospital master loaded.")
            if st.button("Generate sample hospital master"):
                # small synthetic master
                def sample_hosp(n=20):
                    rng = np.random.RandomState(42)
                    locs = ["Chennai","Coimbatore","Madurai","Trichy","Tambaram"]
                    types = ["General","Multi","Clinic","Teaching"]
                    rows = []
                    for i in range(n):
                        rows.append({
                            "Hospital_Name": f"H_{i+1:03d}",
                            "Location": rng.choice(locs),
                            "Hospital_Type": rng.choice(types),
                            "Monthly_Patients": int(max(50, rng.normal(1200,400))),
                            "Staff_Count": int(max(4, rng.normal(60,20))),
                            "Beds_Count": int(max(10, rng.normal(120,50))),
                            "Ventilators_Count": int(max(0, rng.poisson(3)))
                        })
                    return pd.DataFrame(rows)
                st.session_state["hospital_master"] = sample_hosp()
                st.success("Sample hospital master generated.")
        else:
            st.dataframe(st.session_state["hospital_master"].head())

    with col_e:
        st.write("Equipment dataset")
        if st.session_state.get("equipment_master") is None:
            st.info("No equipment dataset loaded.")
            if st.button("Generate sample equipment dataset"):
                hm = st.session_state.get("hospital_master")
                if hm is None:
                    st.error("Please load or generate hospital master first.")
                else:
                    # simple equipment dataset
                    rows = []
                    eqs = ["ICU","MRI","CT","XRay","BloodBank","Pharmacy"]
                    rng = np.random.RandomState(24)
                    for _, r in hm.iterrows():
                        for eq in eqs:
                            rows.append({
                                "Hospital_Name": r["Hospital_Name"],
                                "Equipment": eq,
                                "Count": int(max(0, rng.poisson(3))),
                                "Functional_pct": round(min(100, max(25, rng.normal(88,10))),1)
                            })
                    st.session_state["equipment_master"] = pd.DataFrame(rows)
                    st.success("Equipment dataset generated.")
        else:
            st.dataframe(st.session_state["equipment_master"].head())

    with col_p:
        st.write("Patient & Risk dataset")
        if st.session_state.get("patient_risk") is None:
            st.info("No patient risk dataset.")
            if st.button("Generate sample patient dataset"):
                hm = st.session_state.get("hospital_master")
                if hm is None:
                    st.error("Please load hospital master first.")
                else:
                    rng = np.random.RandomState(7)
                    rows = []
                    depts = ["Emergency","Cardiology","Ortho","Pediatrics","General"]
                    hosp_names = hm["Hospital_Name"].tolist()
                    for i in range(500):
                        age = int(max(0, rng.normal(48, 19)))
                        los = int(max(0, rng.exponential(3)))
                        readmit = 1 if (rng.rand() < 0.06 + (los>5)*0.04) else 0
                        risk = round(min(30, age*0.05 + los*1.2 + readmit*5 + rng.normal(0,2)),2)
                        rows.append({
                            "Patient_ID": f"P_{i+1:06d}",
                            "Hospital_Name": rng.choice(hosp_names),
                            "Age": age,
                            "Department": rng.choice(depts),
                            "Treatment_Cost": round(abs(rng.normal(2500,1100)),2),
                            "Length_of_Stay_days": los,
                            "Readmission": "Yes" if readmit==1 else "No",
                            "Risk_Score": risk,
                            "Risk_Level": "High" if risk>=12 else ("Medium" if risk>=6 else "Low")
                        })
                    st.session_state["patient_risk"] = pd.DataFrame(rows)
                    st.success("Sample patient dataset generated.")
        else:
            st.dataframe(st.session_state["patient_risk"].head())

    st.markdown("---")

    # ---------- Data Overview & Filters ----------
    st.subheader("Data Overview & Filters")
    use_ds = st.selectbox("Choose dataset to preview", options=["hospital_master","equipment_master","patient_risk"])
    ds = st.session_state.get(use_ds)
    if ds is None:
        st.info(f"No {use_ds} dataset available. Upload or generate one above.")
    else:
        st.write(f"Showing first 8 rows of {use_ds}")
        st.dataframe(ds.head(8))
        st.write("Columns:", list(ds.columns))
        st.write("Missing values:")
        st.dataframe(ds.isna().sum().rename("missing_count"))

        # quick download
        download_df(ds, filename=f"{use_ds}.csv")

    st.markdown("---")

    # ---------- Charts ----------
    st.subheader("Charts")
    charts_ds = st.selectbox("Choose dataset for charts", options=["patient_risk","hospital_master","equipment_master"])
    if charts_ds == "patient_risk":
        pr = st.session_state.get("patient_risk")
        if pr is None:
            st.info("No patient_risk dataset.")
        else:
            if "Risk_Level" in pr.columns:
                fig = px.histogram(pr, x="Risk_Level", title="Risk Level Distribution", color="Risk_Level")
                st.plotly_chart(fig, use_container_width=True)
            if "Age" in pr.columns:
                st.plotly_chart(px.histogram(pr, x="Age", nbins=30, title="Age distribution"), use_container_width=True)
            if "Department" in pr.columns and "Readmission" in pr.columns:
                dept = pr.groupby("Department")["Readmission"].apply(lambda s: np.mean(s.astype(str).str.lower().isin(["yes","y","true","1"]))).reset_index(name="readmit_rate")
                st.plotly_chart(px.bar(dept, x="Department", y="readmit_rate", title="Readmission rate by Department"), use_container_width=True)
            if "Diagnosis" in pr.columns and "Treatment_Cost" in pr.columns:
                diag = pr.groupby("Diagnosis")["Treatment_Cost"].mean().reset_index().nlargest(10, "Treatment_Cost")
                st.plotly_chart(px.bar(diag, x="Diagnosis", y="Treatment_Cost", title="Top Diagnoses by Avg Treatment Cost"), use_container_width=True)
    elif charts_ds == "hospital_master":
        hm = st.session_state.get("hospital_master")
        if hm is None:
            st.info("No hospital_master dataset.")
        else:
            if "Location" in hm.columns:
                fig = px.bar(hm.groupby("Location").size().reset_index(name="count"), x="Location", y="count", title="Hospitals by Location")
                st.plotly_chart(fig, use_container_width=True)
            if "Beds_Count" in hm.columns:
                st.plotly_chart(px.histogram(hm, x="Beds_Count", nbins=25, title="Beds count distribution"), use_container_width=True)
            if "Monthly_Patients" in hm.columns and "Staff_Count" in hm.columns:
                fig = px.scatter(hm, x="Monthly_Patients", y="Staff_Count", hover_name="Hospital_Name", title="Patients vs Staff")
                st.plotly_chart(fig, use_container_width=True)
    else:
        em = st.session_state.get("equipment_master")
        if em is None:
            st.info("No equipment dataset.")
        else:
            if "Equipment" in em.columns and "Count" in em.columns:
                top = em.groupby("Equipment")["Count"].sum().reset_index().sort_values("Count", ascending=False)
                st.plotly_chart(px.bar(top, x="Equipment", y="Count", title="Total equipment counts"), use_container_width=True)

    st.markdown("---")

    # ---------- ML Section ----------
    st.subheader("ML: Readmission classification & Risk Score regression")

    pr = st.session_state.get("patient_risk")
    if pr is None:
        st.info("Upload or generate patient_risk dataset to train models.")
    else:
        # Prepare a copy and basic featurization
        data = pr.copy()
        # Create Readmission_Flag if possible
        if "Readmission" in data.columns:
            data["Readmission_Flag"] = data["Readmission"].astype(str).str.lower().isin(["yes","y","true","1"]).astype(int)
        else:
            data["Readmission_Flag"] = 0

        candidate_features = [c for c in data.columns if c not in ["Patient_ID","Hospital_Name","Risk_Level","Risk_Score","Readmission","Readmission_Flag"]]
        st.markdown("Select features for ML (both numeric & categorical supported).")
        selected_feats = st.multiselect("Features", options=candidate_features, default=[c for c in ["Age","Length_of_Stay_days","Treatment_Cost"] if c in candidate_features])

        if len(selected_feats) < 1:
            st.warning("Choose at least one feature to train models.")
        else:
            X = data[selected_feats].copy()
            # handle types
            num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = [c for c in X.columns if c not in num_cols]

            transformers = []
            if num_cols:
                transformers.append(("num", StandardScaler(), num_cols))
            if cat_cols:
                enc_kwargs = safe_onehot(sparse_output_arg=False)
                transformers.append(("cat", OneHotEncoder(**enc_kwargs), cat_cols))
            if not transformers:
                st.error("No usable feature columns detected for ML.")
            else:
                preproc = ColumnTransformer(transformers=transformers, remainder="drop")

                # Classification
                if st.checkbox("Enable Readmission classification (target = Readmission_Flag)"):
                    y = data["Readmission_Flag"]
                    df_comb = pd.concat([X, y], axis=1).dropna()
                    if df_comb.shape[0] < 40:
                        st.warning("Not enough rows to train classifier reliably (need ~40+).")
                    else:
                        test_size = st.slider("Test size (classification)", 0.1, 0.4, 0.2, key="clf_test_size")
                        if st.button("Train Readmission Classifier"):
                            X_tr, X_te, y_tr, y_te = train_test_split(df_comb[selected_feats], df_comb["Readmission_Flag"], test_size=test_size, random_state=42, stratify=df_comb["Readmission_Flag"] if df_comb["Readmission_Flag"].nunique()>1 else None)
                            clf_pipe = Pipeline([("pre", preproc), ("model", RandomForestClassifier(n_estimators=150, random_state=42))])
                            clf_pipe.fit(X_tr, y_tr)
                            st.session_state["clf_pipe"] = clf_pipe
                            y_pred = clf_pipe.predict(X_te)
                            acc = accuracy_score(y_te, y_pred)
                            st.success(f"Classifier trained — accuracy: {acc:.3f}")

                # Regression
                if st.checkbox("Enable Risk Score regression (target = Risk_Score)"):
                    if "Risk_Score" not in data.columns:
                        st.warning("No Risk_Score column found in patient dataset.")
                    else:
                        df_r = pd.concat([X, data["Risk_Score"]], axis=1).dropna()
                        if df_r.shape[0] < 40:
                            st.warning("Not enough rows to train regression reliably (need ~40+).")
                        else:
                            test_size_r = st.slider("Test size (regression)", 0.1, 0.4, 0.2, key="reg_test_size")
                            if st.button("Train Risk Score Regression"):
                                X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(df_r[selected_feats], df_r["Risk_Score"], test_size=test_size_r, random_state=42)
                                reg_pipe = Pipeline([("pre", preproc), ("model", RandomForestRegressor(n_estimators=150, random_state=42))])
                                reg_pipe.fit(X_tr_r, y_tr_r)
                                st.session_state["reg_pipe"] = reg_pipe
                                preds = reg_pipe.predict(X_te_r)
                                rmse = math.sqrt(mean_squared_error(y_te_r, preds))
                                r2 = r2_score(y_te_r, preds)
                                st.success(f"Regression trained — RMSE: {rmse:.2f} | R²: {r2:.3f}")

        # Single-row predict UI (if pipelines exist)
        st.markdown("#### Quick single-row prediction")
        pipe_clf = st.session_state.get("clf_pipe")
        pipe_reg = st.session_state.get("reg_pipe")
        if pipe_clf is None and pipe_reg is None:
            st.info("Train at least one model to enable single-row prediction.")
        else:
            # Build UI for selected_feats
            sim_cols = selected_feats if selected_feats else []
            if sim_cols:
                cols_ui = st.columns(len(sim_cols))
                new_row = {}
                for i, c in enumerate(sim_cols):
                    if c in num_cols:
                        new_row[c] = cols_ui[i].number_input(c, value=float(X[c].median() if c in X.columns and X[c].notna().any() else 0.0), format="%.3f")
                    else:
                        opts = list(X[c].dropna().unique()) if c in X.columns else []
                        new_row[c] = cols_ui[i].selectbox(c, options=opts if opts else ["N/A"])
                if st.button("Predict single-row"):
                    xr = pd.DataFrame([new_row])
                    if pipe_clf is not None:
                        try:
                            pred_c = pipe_clf.predict(xr)[0]
                            st.success(f"Classifier predicts Readmission_Flag = {int(pred_c)}")
                        except Exception as e:
                            st.error(f"Classification prediction failed: {e}")
                    if pipe_reg is not None:
                        try:
                            pred_r = pipe_reg.predict(xr)[0]
                            st.success(f"Regression predicts Risk_Score = {pred_r:.2f}")
                        except Exception as e:
                            st.error(f"Regression prediction failed: {e}")
            else:
                st.info("No features available for single-row prediction. Train a model with selected features first.")

    st.markdown("---")

    # ---------- Clustering ----------
    st.subheader("Clustering (K-Means) for hospitals")
    hm = st.session_state.get("hospital_master")
    if hm is None:
        st.info("Upload or generate hospital master dataset to run clustering.")
    else:
        num_cols_hm = hm.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols_hm:
            st.warning("No numeric columns in hospital master to cluster on.")
        else:
            cluster_feats = st.multiselect("Numeric features for clustering", options=num_cols_hm, default=num_cols_hm[:3])
            if cluster_feats:
                n_clusters = st.slider("Number of clusters", 2, 8, 3)
                if st.button("Run clustering on hospital master"):
                    hm_clean = hm.dropna(subset=cluster_feats)
                    if hm_clean.shape[0] < n_clusters:
                        st.warning("Not enough rows to create that many clusters.")
                    else:
                        km = KMeans(n_clusters=n_clusters, random_state=42)
                        hm.loc[hm_clean.index, "Cluster"] = km.fit_predict(hm_clean[cluster_feats])
                        st.session_state["clustered_master"] = hm
                        st.success("Clustering complete.")
                        # plot first two features
                        fig = px.scatter(hm.dropna(subset=cluster_feats), x=cluster_feats[0], y=cluster_feats[1], color="Cluster", hover_name="Hospital_Name", title="Hospital clusters (projection)")
                        st.plotly_chart(fig, use_container_width=True)
                        st.write("Cluster counts:")
                        st.write(hm["Cluster"].value_counts().sort_index())
            else:
                st.info("Select numeric features to enable clustering.")

    st.markdown("---")

    # ---------- Hospital Profiles ----------
    st.subheader("Hospital Profiles Explorer")
    hm_final = st.session_state.get("clustered_master", st.session_state.get("hospital_master"))
    if hm_final is None:
        st.info("No hospital master dataset found.")
    else:
        hlist = hm_final["Hospital_Name"].unique().tolist()
        sel = st.selectbox("Choose hospital", options=hlist)
        profile = hm_final[hm_final["Hospital_Name"] == sel].iloc[0].to_dict()
        st.json(profile)

        # patient summary for selected hospital
        pr = st.session_state.get("patient_risk")
        if pr is not None:
            pr_h = pr[pr["Hospital_Name"] == sel]
            st.write("Patient summary (if present):")
            st.write(f"Records: {len(pr_h)}")
            if "Risk_Score" in pr_h.columns:
                st.write(f"Avg Risk Score: {pr_h['Risk_Score'].mean():.2f}")
            if "Readmission" in pr_h.columns:
                rr = pr_h["Readmission"].astype(str).str.lower().isin(["yes","y","true","1"]).mean()
                st.write(f"Readmission %: {rr*100:.2f}%")
        else:
            st.info("No patient dataset present to summarize.")

    st.markdown("---")
    st.info("App notes: This is an experimentation app — for production use add validation, logging, model monitoring, and secure storage for models/data.")
