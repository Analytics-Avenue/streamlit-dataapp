import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import math
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="HospitalOps — Full Dashboard", layout="wide")
st.title("🏥 HospitalOps — Full Dashboard")

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


# -------------------------
# CSS (black text + glow)
# -------------------------
st.markdown(
    """
    <style>
    /* Force black text */
    * { color: #000000 !important; }

    /* Glow card */
    .glow-card {
        background-color: #ffffff;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0,120,255,0.12), 0 0 12px rgba(0,120,255,0.06) inset;
        border: 1px solid rgba(0,120,255,0.18);
        margin-bottom: 12px;
    }

    .small-muted { color: rgba(0,0,0,0.65); font-size: 14px; }

    /* KPI row */
    .kpi-row { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:16px; }
    .kpi { background:#fff; padding:12px 16px; border-radius:10px; box-shadow:0 6px 18px rgba(0,0,0,0.04); border:1px solid rgba(0,0,0,0.04); width:220px; }

    /* tabs bigger text */
    .stTabs [role="tab"] { font-size:16px; padding:8px 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Utilities
# -------------------------
DEFAULT_DATA_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/healthcare/healthcare_syn.csv"

def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate column names, keep first occurrence, strip whitespace."""
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = df.columns.str.strip()
    return df

def download_df(df: pd.DataFrame, filename: str = "export.csv"):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button("Download CSV", b, file_name=filename, mime="text/csv")

def safe_onehot_kwargs():
    # adapt to sklearn versions
    try:
        return {"handle_unknown":"ignore", "sparse_output": False}
    except TypeError:
        return {"handle_unknown":"ignore", "sparse": False}

def generate_sample_hospital(n=30, seed=42):
    rng = np.random.RandomState(seed)
    locations = ["Chennai","Tambaram","Anna Nagar","Adyar","Velachery","Coimbatore","Madurai","Trichy"]
    types = ["General","Multispeciality","Clinic","Teaching","Cardiac"]
    rows = []
    for i in range(n):
        rows.append({
            "Hospital_Name": f"H_{i+1:03d}",
            "Location": rng.choice(locations),
            "Hospital_Type": rng.choice(types),
            "Monthly_Patients": int(max(50, rng.normal(1200,400))),
            "Staff_Count": int(max(4, rng.normal(60,20))),
            "Beds_Count": int(max(10, rng.normal(120,50))),
            "Ventilators_Count": int(max(0, rng.poisson(3)))
        })
    return pd.DataFrame(rows)

def generate_sample_equipment(hm: pd.DataFrame):
    eqs = ["ICU","MRI","CT","XRay","BloodBank","Pharmacy","Operation_Theatre"]
    rows = []
    rng = np.random.RandomState(24)
    for _, r in hm.iterrows():
        for eq in eqs:
            rows.append({
                "Hospital_Name": r["Hospital_Name"],
                "Equipment": eq,
                "Count": int(max(0, rng.poisson(3))),
                "Functional_pct": round(min(100, max(25, rng.normal(88,10))),1)
            })
    return pd.DataFrame(rows)

def generate_sample_patients(hm: pd.DataFrame, n=500, seed=7):
    rng = np.random.RandomState(seed)
    depts = ["Emergency","Cardiology","Ortho","Pediatrics","General"]
    hosp_names = hm["Hospital_Name"].tolist()
    rows = []
    for i in range(n):
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
    return pd.DataFrame(rows)

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
# Two main tabs: Overview, Application
# -------------------------
tab_overview, tab_app = st.tabs(["Overview", "Application"])

# -------------------------
# Overview Tab
# -------------------------
with tab_overview:
    st.markdown('<div class="glow-card"><h2>About this application</h2>'
                '<p class="small-muted">HospitalOps provides operational visibility across hospitals: patient trends, equipment readiness, risk scoring, and decision support.</p></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="glow-card"><h3>Purpose</h3>'
                '<p class="small-muted">Improve allocation of beds and equipment, identify high-risk patients and hospitals, and enable quick operational reporting.</p></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="glow-card"><h3>Capabilities</h3>'
                '<p class="small-muted">Dataset ingestion (default/upload/mapping), visual KPIs, clustering, ML (classification + regression), single-row simulation, model explainability.</p></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="glow-card"><h3>Business Impact</h3>'
                '<p class="small-muted">Faster triage, reduced bed shortages, targeted procurement, and data-driven expansion planning.</p></div>',
                unsafe_allow_html=True)

    # Static KPIs (no data import)
    st.markdown('<div class="kpi-row">', unsafe_allow_html=True)
    st.markdown('<div class="kpi"><div style="font-size:12px;color:#333">Total Hospitals</div><div style="font-size:20px;font-weight:700">120</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="kpi"><div style="font-size:12px;color:#333">High-Risk Hospitals</div><div style="font-size:20px;font-weight:700">18</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="kpi"><div style="font-size:12px;color:#333">Avg Bed Occupancy</div><div style="font-size:20px;font-weight:700">72%</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="kpi"><div style="font-size:12px;color:#333">Ventilators</div><div style="font-size:20px;font-weight:700">1,240</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="kpi"><div style="font-size:12px;color:#333">Avg Staff / Hospital</div><div style="font-size:20px;font-weight:700">68</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # KPR radar (sample)
    kpr = pd.DataFrame({
        "KPR": ["Response Time", "Bed Mgmt", "Equipment Readiness", "Staffing", "Triage Accuracy"],
        "Score": [72, 65, 60, 68, 70]
    })
    fig_kpr = px.line_polar(kpr, r="Score", theta="KPR", line_close=True, range_r=[0,100], title="KPR (sample)")
    st.plotly_chart(fig_kpr, use_container_width=True)

# -------------------------
# Application Tab - contains sub-tabs for everything
# -------------------------
with tab_app:
    st.header("Application")
    app_tab_dataset, app_tab_charts, app_tab_ml, app_tab_cluster, app_tab_profiles = st.tabs(
        ["Dataset Setup", "Charts", "ML Predictions", "Clustering", "Hospital Profiles"]
    )

    # -------------------------
    # Dataset Setup Tab
    # -------------------------
    with app_tab_dataset:
        st.subheader("Dataset Setup — choose one of three modes")
        st.markdown("1) Default dataset (fetched from URL)  —  2) User Upload CSV  —  3) Upload + Manual Mapping")

        mode = st.radio("Dataset mode:", ["Default (URL)", "Upload CSV", "Upload + Manual mapping"], horizontal=True)

        if mode == "Default (URL)":
            st.markdown("**Loading default dataset from repository**")
            try:
                df_default = pd.read_csv(DEFAULT_DATA_URL)
                df_default = remove_duplicate_columns(df_default)
                st.session_state["hospital_master"] = df_default  # treat default as hospital_master by default
                st.success("Default dataset loaded and stored as hospital_master (session).")
                st.dataframe(df_default.head())
            except Exception as e:
                st.error(f"Failed to load default dataset: {e}")

        elif mode == "Upload CSV":
            st.markdown("**Upload a CSV file — treated as Hospital Master by default**")
            uploaded = st.file_uploader("Upload hospital CSV", type=["csv"], key="upload_hm")
            if uploaded is not None:
                try:
                    dfu = pd.read_csv(uploaded)
                    dfu = remove_duplicate_columns(dfu)
                    st.session_state["hospital_master"] = dfu
                    st.success("Uploaded hospital master saved in session.")
                    st.dataframe(dfu.head())
                except Exception as e:
                    st.error(f"Failed to read uploaded CSV: {e}")

        else:  # Upload + mapping
            st.markdown("**Upload CSV and map columns manually to canonical hospital master fields**")
            uploaded_map = st.file_uploader("Upload CSV to map", type=["csv"], key="upload_map")
            if uploaded_map is not None:
                try:
                    raw = pd.read_csv(uploaded_map)
                    st.write("Preview of uploaded CSV")
                    st.dataframe(raw.head())
                    # Remove exact duplicate columns pre-mapping
                    raw = raw.loc[:, ~raw.columns.duplicated()].copy()
                    raw.columns = raw.columns.str.strip()
                    st.markdown("Map columns (leave as '-- None --' to create blank placeholder)")
                    canonical = [
                        "Hospital_Name", "Location", "Hospital_Type", "Monthly_Patients",
                        "Staff_Count", "Beds_Count", "Ventilators_Count", "Risk_Score", "Risk_Level"
                    ]
                    mapping = {}
                    cols_list = ["-- None --"] + list(raw.columns)
                    for c in canonical:
                        mapping[c] = st.selectbox(f"Map to → {c}", options=cols_list, key=f"map_{c}")
                    if st.button("Apply mapping"):
                        mapped = pd.DataFrame()
                        for c in canonical:
                            sel = mapping[c]
                            if sel != "-- None --":
                                mapped[c] = raw[sel]
                            else:
                                mapped[c] = np.nan
                        st.session_state["hospital_master"] = mapped
                        st.success("Mapping applied — stored as hospital_master in session.")
                        st.dataframe(mapped.head())
                except Exception as e:
                    st.error(f"Failed to read file for mapping: {e}")

        st.markdown("---")
        # equipment and patient upload/generate
        hcol1, hcol2, hcol3 = st.columns(3)
        with hcol1:
            st.write("Hospital Master in session")
            if st.session_state.get("hospital_master") is None:
                st.info("No hospital master loaded.")
                if st.button("Generate sample hospital master"):
                    st.session_state["hospital_master"] = generate_sample_hospital(n=30)
                    st.success("Sample hospital master generated.")
            else:
                st.dataframe(st.session_state["hospital_master"].head())

        with hcol2:
            st.write("Equipment dataset")
            if st.session_state.get("equipment_master") is None:
                st.info("No equipment dataset.")
                if st.button("Generate sample equipment dataset"):
                    hm = st.session_state.get("hospital_master")
                    if hm is None:
                        st.error("Please load or generate hospital master first.")
                    else:
                        st.session_state["equipment_master"] = generate_sample_equipment(hm)
                        st.success("Sample equipment dataset generated.")
            else:
                st.dataframe(st.session_state["equipment_master"].head())

        with hcol3:
            st.write("Patient risk dataset")
            if st.session_state.get("patient_risk") is None:
                st.info("No patient_risk dataset.")
                if st.button("Generate sample patient dataset"):
                    hm = st.session_state.get("hospital_master")
                    if hm is None:
                        st.error("Please load or generate hospital master first.")
                    else:
                        st.session_state["patient_risk"] = generate_sample_patients(hm, n=600)
                        st.success("Sample patient dataset generated.")
            else:
                st.dataframe(st.session_state["patient_risk"].head())

        st.markdown("---")
        st.info("Datasets stored in session_state: hospital_master, equipment_master, patient_risk. Use Charts / ML / Clustering tabs to analyze.")

    # -------------------------
    # Charts Tab
    # -------------------------
    with app_tab_charts:
        st.subheader("Charts & Visualizations")

        ds_choice = st.selectbox("Select dataset for charts", options=["hospital_master","equipment_master","patient_risk"])
        ds = st.session_state.get(ds_choice)

        if ds is None:
            st.info(f"No {ds_choice} dataset available. Upload or generate it in Dataset Setup.")
        else:
            st.write(f"Preview — {ds_choice} (first 10 rows)")
            st.dataframe(ds.head(10))

            # Numeric columns
            num_cols = ds.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = ds.select_dtypes(exclude=[np.number]).columns.tolist()

            # Basic charts
            st.markdown("### Basic Charts")
            if num_cols:
                col_x = st.selectbox("Numeric column for histogram", num_cols, index=0)
                bins = st.slider("Bins", 5, 60, 30)
                fig_hist = px.histogram(ds, x=col_x, nbins=bins, title=f"Distribution of {col_x}")
                st.plotly_chart(fig_hist, use_container_width=True)

                if len(num_cols) >= 2:
                    xcol = st.selectbox("X axis (numeric)", num_cols, index=0, key="scatter_x")
                    ycol = st.selectbox("Y axis (numeric)", num_cols, index=1, key="scatter_y")
                    color_col = st.selectbox("Color by (optional)", options=[None]+cat_cols, index=0)
                    fig_scatter = px.scatter(ds, x=xcol, y=ycol, color=color_col, hover_data=cat_cols, title=f"{ycol} vs {xcol}")
                    st.plotly_chart(fig_scatter, use_container_width=True)

                # correlation heatmap
                if len(num_cols) >= 2:
                    corr = ds[num_cols].corr()
                    fig_corr = px.imshow(corr, text_auto=True, title="Correlation matrix")
                    st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("No numeric columns available for charts in this dataset.")

            # Categorical charts
            if cat_cols:
                cat_choice = st.selectbox("Categorical column for bar chart", options=cat_cols)
                vc = ds[cat_choice].value_counts().reset_index()
                vc.columns = [cat_choice, "count"]
                fig_bar = px.bar(vc.head(20), x=cat_choice, y="count", title=f"Top values for {cat_choice}")
                st.plotly_chart(fig_bar, use_container_width=True)

            # Pie chart for risk level if present
            if "Risk_Level" in ds.columns:
                fig_pie = px.pie(ds, names="Risk_Level", title="Risk Level Distribution")
                st.plotly_chart(fig_pie, use_container_width=True)

            # If equipment dataset show equipment totals
            if ds_choice == "equipment_master" and "Equipment" in ds.columns and "Count" in ds.columns:
                em_sum = ds.groupby("Equipment")["Count"].sum().reset_index().sort_values("Count", ascending=False)
                st.plotly_chart(px.bar(em_sum, x="Equipment", y="Count", title="Total equipment counts"), use_container_width=True)

            # If hospital master show location counts
            if ds_choice == "hospital_master" and "Location" in ds.columns:
                loc_cnt = ds.groupby("Location").size().reset_index(name="Hospitals")
                st.plotly_chart(px.bar(loc_cnt, x="Location", y="Hospitals", title="Hospitals by Location"), use_container_width=True)

            # KPI cards derived from dataset if available
            st.markdown("### Derived KPIs")
            k1,k2,k3,k4 = st.columns(4)
            if ds_choice == "hospital_master":
                k1.metric("Hospitals", ds["Hospital_Name"].nunique() if "Hospital_Name" in ds.columns else "—")
                k2.metric("Avg Beds", f"{ds['Beds_Count'].mean():.1f}" if "Beds_Count" in ds.columns else "—")
                k3.metric("Avg Staff", f"{ds['Staff_Count'].mean():.1f}" if "Staff_Count" in ds.columns else "—")
                k4.metric("Avg Ventilators", f"{ds['Ventilators_Count'].mean():.1f}" if "Ventilators_Count" in ds.columns else "—")
            elif ds_choice == "patient_risk":
                k1.metric("Records", f"{len(ds):,}")
                k2.metric("Avg Risk Score", f"{ds['Risk_Score'].mean():.2f}" if "Risk_Score" in ds.columns else "—")
                if "Readmission" in ds.columns:
                    rr = ds["Readmission"].astype(str).str.lower().isin(["yes","y","true","1"]).mean()
                    k3.metric("Readmission %", f"{rr*100:.2f}%")
                else:
                    k3.metric("Readmission %", "—")
                k4.metric("Avg LOS", f"{ds['Length_of_Stay_days'].mean():.2f}" if "Length_of_Stay_days" in ds.columns else "—")
            else:
                k1.metric("Records", f"{len(ds):,}")
                k2.metric("Unique Hospitals", ds["Hospital_Name"].nunique() if "Hospital_Name" in ds.columns else "—")
                k3.metric("Equipment types", ds["Equipment"].nunique() if "Equipment" in ds.columns else "—")
                k4.metric("—", "—")

    # -------------------------
    # ML Predictions Tab
    # -------------------------
    with app_tab_ml:
        st.subheader("ML: Readmission Classification & Risk Score Regression")
        pr = st.session_state.get("patient_risk")
        if pr is None:
            st.info("No patient_risk dataset. Generate or upload one in Dataset Setup.")
        else:
            df = pr.copy()
            # normalize readmission
            if "Readmission" in df.columns:
                df["Readmission_Flag"] = df["Readmission"].astype(str).str.lower().isin(["yes","y","true","1"]).astype(int)
            else:
                df["Readmission_Flag"] = 0

            # Feature selection
            candidate_features = [c for c in df.columns if c not in ["Patient_ID","Hospital_Name","Risk_Level","Risk_Score","Readmission","Readmission_Flag"]]
            st.markdown("Select features (numeric and categorical supported)")
            selected_features = st.multiselect("Features", options=candidate_features, default=[c for c in ["Age","Length_of_Stay_days","Treatment_Cost"] if c in candidate_features])

            if len(selected_features) == 0:
                st.warning("Pick at least one feature to proceed.")
            else:
                X = df[selected_features].copy()
                num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                cat_cols = [c for c in X.columns if c not in num_cols]

                transformers = []
                if num_cols:
                    transformers.append(("num", StandardScaler(), num_cols))
                if cat_cols:
                    transformers.append(("cat", OneHotEncoder(**safe_onehot_kwargs()), cat_cols))

                if not transformers:
                    st.error("No usable numeric or categorical features detected.")
                else:
                    preproc = ColumnTransformer(transformers=transformers, remainder="drop")

                    # Train classifier
                    if st.checkbox("Train Readmission classifier"):
                        y_clf = df["Readmission_Flag"]
                        combined = pd.concat([X, y_clf], axis=1).dropna()
                        if combined.shape[0] < 40:
                            st.warning("Not enough rows to train classifier reliably (need ~40+).")
                        else:
                            test_size = st.slider("Test size (classifier)", 0.1, 0.4, 0.2, key="clf_test")
                            if st.button("Train classifier"):
                                X_tr, X_te, y_tr, y_te = train_test_split(combined[selected_features], combined["Readmission_Flag"], test_size=test_size, random_state=42, stratify=combined["Readmission_Flag"] if combined["Readmission_Flag"].nunique()>1 else None)
                                clf_pipe = Pipeline([("pre", preproc), ("model", RandomForestClassifier(n_estimators=150, random_state=42))])
                                clf_pipe.fit(X_tr, y_tr)
                                st.session_state["clf_pipe"] = clf_pipe
                                y_pred = clf_pipe.predict(X_te)
                                acc = accuracy_score(y_te, y_pred)
                                st.success(f"Classifier trained — accuracy: {acc:.3f}")
                                # feature importance (approx)
                                try:
                                    importances = clf_pipe.named_steps["model"].feature_importances_
                                    # get feature names
                                    feat_names = []
                                    if num_cols:
                                        feat_names += num_cols
                                    if cat_cols:
                                        feat_names += clf_pipe.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
                                    fi = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
                                    st.markdown("Top feature importances (classifier)")
                                    st.dataframe(fi.head(10))
                                    st.plotly_chart(px.bar(fi.head(10), x="importance", y="feature", orientation="h"), use_container_width=True)
                                except Exception:
                                    st.info("Feature importances not available.")

                    # Train regression
                    if st.checkbox("Train Risk Score regression"):
                        if "Risk_Score" not in df.columns:
                            st.warning("No Risk_Score column in patient dataset.")
                        else:
                            combined_r = pd.concat([X, df["Risk_Score"]], axis=1).dropna()
                            if combined_r.shape[0] < 40:
                                st.warning("Not enough rows to train regression reliably (need ~40+).")
                            else:
                                test_size_r = st.slider("Test size (regression)", 0.1, 0.4, 0.2, key="reg_test")
                                if st.button("Train regression"):
                                    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(combined_r[selected_features], combined_r["Risk_Score"], test_size=test_size_r, random_state=42)
                                    reg_pipe = Pipeline([("pre", preproc), ("model", RandomForestRegressor(n_estimators=150, random_state=42))])
                                    reg_pipe.fit(X_tr_r, y_tr_r)
                                    st.session_state["reg_pipe"] = reg_pipe
                                    preds = reg_pipe.predict(X_te_r)
                                    rmse = math.sqrt(mean_squared_error(y_te_r, preds))
                                    r2 = r2_score(y_te_r, preds)
                                    st.success(f"Regression trained — RMSE: {rmse:.2f} | R²: {r2:.3f}")
                                    # feature importance
                                    try:
                                        importances = reg_pipe.named_steps["model"].feature_importances_
                                        feat_names = []
                                        if num_cols:
                                            feat_names += num_cols
                                        if cat_cols:
                                            feat_names += reg_pipe.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
                                        fi = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
                                        st.markdown("Top feature importances (regression)")
                                        st.dataframe(fi.head(10))
                                        st.plotly_chart(px.bar(fi.head(10), x="importance", y="feature", orientation="h"), use_container_width=True)
                                    except Exception:
                                        st.info("Feature importances not available for regression.")

                    # Single-row prediction UI
                    st.markdown("### Quick single-row prediction (use trained models)")
                    pipe_clf = st.session_state.get("clf_pipe")
                    pipe_reg = st.session_state.get("reg_pipe")
                    if pipe_clf is None and pipe_reg is None:
                        st.info("Train at least one model to enable single-row prediction.")
                    else:
                        if len(selected_features) == 0:
                            st.info("Select features used for training to show inputs.")
                        else:
                            cols_ui = st.columns(len(selected_features))
                            new_row = {}
                            for i, f in enumerate(selected_features):
                                if f in num_cols:
                                    new_row[f] = cols_ui[i].number_input(f, value=float(X[f].median() if f in X.columns and X[f].notna().any() else 0.0))
                                else:
                                    opts = X[f].dropna().unique().tolist() if f in X.columns else []
                                    new_row[f] = cols_ui[i].selectbox(f, options=opts if opts else ["N/A"])
                            if st.button("Predict single-row"):
                                xr = pd.DataFrame([new_row])
                                if pipe_clf is not None:
                                    try:
                                        p = pipe_clf.predict(xr)[0]
                                        st.success(f"Classifier predicts Readmission_Flag = {int(p)}")
                                    except Exception as e:
                                        st.error(f"Classifier prediction failed: {e}")
                                if pipe_reg is not None:
                                    try:
                                        p = pipe_reg.predict(xr)[0]
                                        st.success(f"Regression predicts Risk_Score = {p:.2f}")
                                    except Exception as e:
                                        st.error(f"Regression prediction failed: {e}")

    # -------------------------
    # Clustering Tab
    # -------------------------
    with app_tab_cluster:
        st.subheader("Clustering — Segment hospitals using KMeans")
        hm = st.session_state.get("hospital_master")
        if hm is None:
            st.info("No hospital_master dataset found. Load or generate one in Dataset Setup.")
        else:
            numeric_cols = hm.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.warning("No numeric columns in hospital_master for clustering.")
            else:
                feat_sel = st.multiselect("Numeric features to cluster on", options=numeric_cols, default=numeric_cols[:3])
                n_clusters = st.slider("Number of clusters", 2, 8, 3)
                if st.button("Run KMeans clustering"):
                    if len(feat_sel) == 0:
                        st.warning("Select at least one numeric feature.")
                    else:
                        hm_clean = hm.dropna(subset=feat_sel)
                        if hm_clean.shape[0] < n_clusters:
                            st.warning("Not enough rows to form requested clusters.")
                        else:
                            km = KMeans(n_clusters=n_clusters, random_state=42)
                            labels = km.fit_predict(hm_clean[feat_sel])
                            hm.loc[hm_clean.index, "Cluster"] = labels
                            st.session_state["clustered_master"] = hm
                            st.success("Clustering complete and stored in session.")
                            # 2D scatter of first two features
                            if len(feat_sel) >= 2:
                                fig = px.scatter(hm.dropna(subset=feat_sel), x=feat_sel[0], y=feat_sel[1], color="Cluster", hover_name="Hospital_Name", title="Clusters (projection)")
                                st.plotly_chart(fig, use_container_width=True)
                            st.write("Cluster counts:")
                            st.write(hm["Cluster"].value_counts().sort_index())

    # -------------------------
    # Profiles Tab
    # -------------------------
    with app_tab_profiles:
        st.subheader("Hospital Profiles Explorer")
        hm_final = st.session_state.get("clustered_master", st.session_state.get("hospital_master"))
        if hm_final is None:
            st.info("No hospital_master dataset. Load or generate in Dataset Setup.")
        else:
            if "Hospital_Name" not in hm_final.columns:
                # try alternative column name tries
                alt = [c for c in hm_final.columns if "hospital" in c.lower()]
                if alt:
                    hm_final = hm_final.rename(columns={alt[0]: "Hospital_Name"})
            hospital_list = hm_final["Hospital_Name"].dropna().unique().tolist()
            if not hospital_list:
                st.info("No hospital names found in hospital_master dataset.")
            else:
                sel = st.selectbox("Choose hospital", hospital_list)
                prof = hm_final[hm_final["Hospital_Name"] == sel].iloc[0].to_dict()
                st.markdown("**Profile (hospital master fields)**")
                st.json(prof)

                # patient summary
                pr = st.session_state.get("patient_risk")
                if pr is not None:
                    pr_h = pr[pr["Hospital_Name"] == sel]
                    st.markdown("**Patient summary (if patient dataset present)**")
                    st.write(f"Records: {len(pr_h)}")
                    if "Risk_Score" in pr_h.columns:
                        st.write(f"Avg Risk Score: {pr_h['Risk_Score'].mean():.2f}")
                    if "Readmission" in pr_h.columns:
                        rr = pr_h["Readmission"].astype(str).str.lower().isin(["yes","y","true","1"]).mean()
                        st.write(f"Readmission %: {rr*100:.2f}%")
                else:
                    st.info("No patient dataset present to summarize.")

                # equipment summary
                em = st.session_state.get("equipment_master")
                if em is not None:
                    em_h = em[em["Hospital_Name"] == sel]
                    st.markdown("**Equipment summary (if equipment dataset present)**")
                    if not em_h.empty:
                        st.dataframe(em_h.groupby("Equipment").agg({"Count":"sum","Functional_pct":"mean"}).reset_index())
                    else:
                        st.info("No equipment records for this hospital.")

    # End of Application tab

# End of app
