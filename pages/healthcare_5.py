# ============================
# app_hospital.py  (FULL FIXED VERSION)
# ============================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="HospitalOps — Capacity & Risk Lab", layout="wide")

st.markdown("<h1 style='margin-bottom:0.2rem'>HospitalOps — Capacity & Risk Lab</h1>", unsafe_allow_html=True)
st.markdown("Operational analytics for hospital chains: facility gaps, equipment shortages, risk scoring and ML suggestions.")

# ---------------------------------------------------
# STYLING
# ---------------------------------------------------
st.markdown("""
<style>
.card {
    background: rgba(255,255,255,0.06);
    padding: 16px 18px;
    border-radius: 12px;
    margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 6px 22px rgba(0,0,0,0.12);
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
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# UTILITIES
# ---------------------------------------------------
RAW_DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/healthcare/healthcare_syn.csv"

REQUIRED_COLS = [
    "Patient_ID","Visit_Date","Age","Gender","Department","Diagnosis",
    "Treatment","Treatment_Cost","Length_of_Stay_days","Outcome",
    "Readmission","Risk_Score","Risk_Level","Country","City"
]

def to_currency(x):
    try:
        return "₹ " + f"{float(x):,.2f}"
    except:
        return x

def download_df(df, filename):
    buffer = BytesIO()
    buffer.write(df.to_csv(index=False).encode("utf-8"))
    buffer.seek(0)
    st.download_button("Download CSV", buffer, file_name=filename, mime="text/csv")

def auto_map(df):
    rename = {}
    candidates = {
        "Patient_ID":["patient"],
        "Visit_Date":["date","visit"],
        "Age":["age"],
        "Gender":["gender","sex"],
        "Department":["department","dept"],
        "Diagnosis":["diagnosis"],
        "Treatment":["treat"],
        "Treatment_Cost":["cost","bill","charge"],
        "Length_of_Stay_days":["los","stay"],
        "Outcome":["outcome"],
        "Readmission":["readmission","readmit"],
        "Risk_Score":["risk_score","score"],
        "Risk_Level":["risk"],
        "Country":["country"],
        "City":["city"]
    }
    for req, cands in candidates.items():
        for col in df.columns:
            lc = col.lower().strip()
            if any(c in lc for c in cands):
                rename[col] = req
    return df.rename(columns=rename)


# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
if "pipeline_clf" not in st.session_state: st.session_state.pipeline_clf = None
if "pipeline_reg" not in st.session_state: st.session_state.pipeline_reg = None
if "loaded_df" not in st.session_state: st.session_state.loaded_df = None

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tabs = st.tabs(["Overview", "Application"])

# ===================================================
# TAB 2 - APPLICATION
# ===================================================
with tabs[1]:

    st.header("Step 1 — Load dataset")

    mode = st.radio("Select:", ["Default dataset","Upload CSV","Upload CSV + Mapping"], horizontal=True)

    df = None

    # DEFAULT DATASET
    if mode == "Default dataset":
        try:
            df = pd.read_csv(RAW_DEFAULT_URL)
            df = auto_map(df)
            st.success("Default dataset loaded.")
            st.dataframe(df.head())
            st.session_state.loaded_df = df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()

    # UPLOAD
    elif mode == "Upload CSV":
        file = st.file_uploader("Upload healthcare CSV", type=["csv"])
        if file:
            try:
                df = pd.read_csv(file)
                df = auto_map(df)
                st.success("File uploaded.")
                st.dataframe(df.head())
                st.session_state.loaded_df = df
            except Exception as e:
                st.error(f"Upload failed: {e}")
                st.stop()

    # MAPPING MODE
    else:
        file = st.file_uploader("Upload CSV to map", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.write("Preview:")
            st.dataframe(raw.head())

            mapping = {}
            for req in REQUIRED_COLS:
                mapping[req] = st.selectbox(
                    f"Map column for {req}",
                    ["--Select--"] + list(raw.columns)
                )
            if st.button("Apply mapping"):
                missing = [k for k,v in mapping.items() if v == "--Select--"]
                if missing:
                    st.error("Missing mappings: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v:k for k,v in mapping.items()})
                    st.success("Mapping applied.")
                    st.dataframe(df.head())
                    st.session_state.loaded_df = df

# Nothing loaded - stop further UI
if st.session_state.loaded_df is None:
    with tabs[0]:
        st.info("Load a dataset first in Application tab.")
    st.stop()

df = st.session_state.loaded_df.copy()

# CLEANING
if "Visit_Date" in df.columns:
    df["Visit_Date"] = pd.to_datetime(df["Visit_Date"], errors="coerce")

if "Readmission" in df.columns:
    df["Readmission_Flag"] = df["Readmission"].astype(str).str.lower().isin(["yes","1","true","y","t"]).astype(int)
else:
    df["Readmission_Flag"] = 0

# Ensure numeric
for col in ["Age","Treatment_Cost","Length_of_Stay_days","Risk_Score"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)


# ===================================================
# TAB 1 — OVERVIEW
# ===================================================
with tabs[0]:
    st.markdown("### High-level Overview")

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.markdown(f"<div class='metric-card'>Patients: {df['Patient_ID'].nunique()}</div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='metric-card'>Records: {len(df):,}</div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='metric-card'>Avg Age: {df['Age'].mean():.1f}</div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='metric-card'>Avg LOS: {df['Length_of_Stay_days'].mean():.1f}</div>", unsafe_allow_html=True)
    k5.markdown(f"<div class='metric-card'>Readmission Rate: {df['Readmission_Flag'].mean()*100:.2f}%</div>", unsafe_allow_html=True)


# ===================================================
# TAB 2 CONTINUES — FILTERS & ANALYSIS
# ===================================================
with tabs[1]:

    st.subheader("Step 2 — Filters")

    departments = sorted(df["Department"].dropna().unique())
    genders = sorted(df["Gender"].dropna().unique())
    date_min = df["Visit_Date"].min()
    date_max = df["Visit_Date"].max()

    c1,c2,c3 = st.columns([2,2,1])
    with c1:
        f_dept = st.multiselect("Department", departments, default=departments[:5])
    with c2:
        f_gender = st.multiselect("Gender", genders, default=genders)
    with c3:
        f_range = st.date_input("Visit Range", (date_min, date_max))

    filt = df.copy()
    if f_dept:
        filt = filt[filt["Department"].isin(f_dept)]
    if f_gender:
        filt = filt[filt["Gender"].isin(f_gender)]
    if len(f_range)==2:
        s,e = f_range
        filt = filt[(filt["Visit_Date"]>=pd.to_datetime(s)) & (filt["Visit_Date"]<=pd.to_datetime(e))]

    st.write("Filtered preview:")
    st.dataframe(filt.head(10))

    download_df(filt, "filtered_health_data.csv")

    # METRICS
    st.subheader("Key Metrics")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Unique Patients", filt["Patient_ID"].nunique())
    c2.metric("Avg Cost", to_currency(filt["Treatment_Cost"].mean()))
    c3.metric("Avg LOS", f"{filt['Length_of_Stay_days'].mean():.2f}")
    c4.metric("Readmission %", f"{filt['Readmission_Flag'].mean()*100:.2f}%")

    # CHARTS
    st.subheader("Charts")

    if "Outcome" in filt.columns:
        st.plotly_chart(px.pie(filt, names="Outcome", title="Outcomes"), use_container_width=True)

    if "Age" in filt.columns:
        st.plotly_chart(px.histogram(filt, x="Age", nbins=30, title="Age Distribution"), use_container_width=True)

    if "Department" in filt.columns:
        dep = filt.groupby("Department")["Readmission_Flag"].mean().reset_index()
        st.plotly_chart(px.bar(dep, x="Department", y="Readmission_Flag", title="Readmission by Department"), use_container_width=True)

    if "Diagnosis" in filt.columns:
        diag = filt.groupby("Diagnosis")["Treatment_Cost"].mean().reset_index().sort_values("Treatment_Cost", ascending=False).head(15)
        st.plotly_chart(px.bar(diag, x="Diagnosis", y="Treatment_Cost", title="Top Diagnosis by Avg Cost"), use_container_width=True)


    # ===================================================
    # ML MODELS
    # ===================================================
    st.subheader("Step 3 — ML Models")

    task = st.multiselect("Choose Models:", ["Readmission (Classification)", "Treatment Cost (Regression)"])

    # Feature selection
    excl = ["Patient_ID","Visit_Date","Outcome","Readmission","Readmission_Flag","Treatment_Cost"]
    feat = [c for c in filt.columns if c not in excl]
    feat_selected = st.multiselect("Features", feat, default=feat[:4])

    if len(feat_selected) < 2 and task:
        st.warning("Choose at least 2 features.")
        st.stop()

    X = filt[feat_selected].copy().dropna()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feat_selected if c not in num_cols]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    # -------- Classification --------
    if "Readmission (Classification)" in task:

        st.markdown("#### Readmission Prediction")

        y = filt.loc[X.index, "Readmission_Flag"]

        if y.nunique() < 2:
            st.warning("Not enough classes to train classifier.")
        else:
            test_size = st.slider("Test size (classification)", 0.1, 0.4, 0.2)
            if st.button("Train Classification Model"):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )

                pipe = Pipeline([
                    ("prep", preprocessor),
                    ("model", RandomForestClassifier(n_estimators=150, random_state=42))
                ])
                pipe.fit(X_train, y_train)

                st.session_state.pipeline_clf = pipe

                pred = pipe.predict(X_test)
                acc = accuracy_score(y_test, pred)
                auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1])

                st.success(f"Model Trained — Accuracy {acc:.3f} | AUC {auc:.3f}")

    # -------- Regression --------
    if "Treatment Cost (Regression)" in task:

        st.markdown("#### Treatment Cost Prediction")

        if "Treatment_Cost" not in filt.columns:
            st.warning("No Treatment_Cost column available.")
        else:
            y = filt.loc[X.index, "Treatment_Cost"]

            test_size = st.slider("Test size (regression)", 0.1, 0.4, 0.2)
            if st.button("Train Regression Model"):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                pipe = Pipeline([
                    ("prep", preprocessor),
                    ("model", RandomForestRegressor(n_estimators=150, random_state=42))
                ])
                pipe.fit(X_train, y_train)

                st.session_state.pipeline_reg = pipe

                pred = pipe.predict(X_test)
                r2 = r2_score(y_test, pred)

                st.success(f"Regression Model Trained — R2 Score {r2:.3f}")

