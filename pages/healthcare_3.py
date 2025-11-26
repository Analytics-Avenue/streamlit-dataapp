import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import shap
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PAGE CONFIG + HIDE SIDEBAR
# ---------------------------------------------------------
st.set_page_config(page_title="PatientFlow Navigator", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] {display:none;}
[data-testid="stSidebarNav"] {display:none;}

.big-header {
    font-size:42px;
    font-weight:900;
    color:black;
    margin-bottom:10px;
}

/* Card (Glassmorphism) */
.card {
    padding:22px;
    border-radius:16px;
    background:rgba(255,255,255,0.55);
    border:1px solid rgba(255,255,255,0.4);
    box-shadow:0 4px 18px rgba(0,0,0,0.15);
    backdrop-filter: blur(6px);
    margin-bottom:18px;
    transition: all 0.25s ease;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.22);
}

/* KPI Cards */
.metric-card {
    padding:20px;
    border-radius:14px;
    background:rgba(255,255,255,0.5);
    border:1px solid rgba(0,0,0,0.15);
    text-align:center;
    color:#064b86;
    font-weight:700;
    font-size:18px;
    transition:0.25s ease;
    box-shadow:0 3px 15px rgba(0,0,0,0.15);
    backdrop-filter: blur(4px);
}
.metric-card:hover {
    box-shadow:0 0 25px rgba(0,123,255,0.55);
    transform:scale(1.05);
}

/* Section Titles */
.section-title {
    font-size:28px;
    font-weight:700;
    margin-top:25px;
    margin-bottom:12px;
    position:relative;
}
.section-title::after {
    content:"";
    position:absolute;
    bottom:-6px;
    left:0;
    width:45%;
    height:3px;
    background:#064b86;
    border-radius:4px;
}

/* Variable Boxes */
.variable-box {
    padding:15px;
    border-radius:10px;
    background:white;
    border:1px solid #ddd;
    margin-bottom:10px;
    font-size:17px;
    font-weight:600;
    color:#064b86;
    text-align:center;
    box-shadow:0 2px 10px rgba(0,0,0,0.1);
    transition:0.25s ease;
}
.variable-box:hover {
    transform:translateY(-5px);
    box-shadow:0 10px 18px rgba(0,0,0,0.2);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HEADER + BRAND BLOCK
# ---------------------------------------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

st.markdown(f"""
<div style="display:flex; align-items:center;">
    <img src="{logo_url}" width="65" style="margin-right:12px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:700;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:700;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='big-header'>PatientFlow Navigator</div>", unsafe_allow_html=True)
st.markdown("Next-gen healthcare intelligence: patient journey analytics, readmission prediction, cost forecasting, and risk scoring.")

# ---------------------------------------------------------
# REQUIRED COLUMNS
# ---------------------------------------------------------
REQUIRED_COLS = [
    "Patient_ID","Age","Gender","Disease","Symptoms","Treatment",
    "Admission_Date","Discharge_Date","Treatment_Cost","Readmission",
    "Department","Visit_Date","Outcome","Risk_Level","Risk_Score"
]

def auto_map(df):
    rename = {}
    for col in df.columns:
        low = col.lower().strip()
        if "patient" in low: rename[col] = "Patient_ID"
        elif "age" in low: rename[col] = "Age"
        elif "gender" in low or "sex" in low: rename[col] = "Gender"
        elif "disease" in low: rename[col] = "Disease"
        elif "symptom" in low: rename[col] = "Symptoms"
        elif "treat" in low: rename[col] = "Treatment"
        elif "admis" in low: rename[col] = "Admission_Date"
        elif "discha" in low: rename[col] = "Discharge_Date"
        elif "cost" in low or "bill" in low: rename[col] = "Treatment_Cost"
        elif "read" in low: rename[col] = "Readmission"
        elif "dept" in low: rename[col] = "Department"
        elif "visit" in low: rename[col] = "Visit_Date"
        elif "out" in low: rename[col] = "Outcome"
        elif "risk_s" in low: rename[col] = "Risk_Score"
        elif "risk" in low: rename[col] = "Risk_Level"
    return df.rename(columns=rename)

def download(df, name):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode())
    b.seek(0)
    st.download_button("Download CSV", b, file_name=name, mime="text/csv")

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# ---------------------------------------------------------
# TAB 1 – OVERVIEW
# ---------------------------------------------------------
with tab1:
    st.markdown("<div class='section-title'>Overview</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        PatientFlow Navigator is a predictive, operational, and clinical intelligence engine.
        It helps hospitals reduce readmissions, forecast treatment cost, analyze symptoms, and classify patient risk.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>KPIs</div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Total Patients</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Avg Length of Stay</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Readmission Rate</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Avg Treatment Cost</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2 – ATTRIBUTES
# ---------------------------------------------------------
with tab2:
    st.markdown("<div class='section-title'>Important Attributes</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-title'>Independent Variables</div>", unsafe_allow_html=True)
        indep = [
            "Age","Gender","Symptoms","Disease","Treatment",
            "Admission_Date","Discharge_Date","Department",
            "Risk_Score","Risk_Level"
        ]
        for v in indep:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='section-title'>Dependent Variables</div>", unsafe_allow_html=True)
        dep = ["Length_of_Stay","Readmission","Treatment_Cost","Outcome"]
        for v in dep:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 3 – APPLICATION
# ---------------------------------------------------------
with tab3:

    # ---------------------------------------------------------
    # LOAD DATASET
    # ---------------------------------------------------------
    st.markdown("<div class='section-title'>Step 1 – Load Dataset</div>", unsafe_allow_html=True)

    mode = st.radio("Load data:", ["Default dataset", "Upload CSV", "Upload CSV + Mapping"], horizontal=True)
    df = None

    if mode == "Default dataset":
        url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/healthcare/healthcare_3.csv"
        df = pd.read_csv(url)
        df = auto_map(df)
        st.success("Loaded default dataset")
        st.dataframe(df.head())

    elif mode == "Upload CSV":
        f = st.file_uploader("Upload CSV", type=["csv"])
        if f:
            df = pd.read_csv(f)
            df = auto_map(df)
            st.success("Uploaded and auto-mapped")
            st.dataframe(df.head())

    else:
        f = st.file_uploader("Upload CSV to map", type=["csv"])
        if f:
            raw = pd.read_csv(f)
            st.dataframe(raw.head())
            mapping = {}
            for col in REQUIRED_COLS:
                mapping[col] = st.selectbox(f"Map → {col}", ["-- Select --"] + list(raw.columns))
            if st.button("Apply mapping"):
                missing = [c for c, v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Map all columns")
                else:
                    df = raw.rename(columns={v: c for c, v in mapping.items()})
                    st.success("Mapping applied")
                    st.dataframe(df.head())

    if df is None:
        st.stop()

    # ---------------------------------------------------------
    # CLEANING + FEATURE ENGINEERING
    # ---------------------------------------------------------
    df["Admission_Date"] = pd.to_datetime(df["Admission_Date"], errors="coerce")
    df["Discharge_Date"] = pd.to_datetime(df["Discharge_Date"], errors="coerce")
    df["Visit_Date"] = pd.to_datetime(df["Visit_Date"], errors="coerce")

    df["Length_of_Stay"] = (df["Discharge_Date"] - df["Admission_Date"]).dt.days.fillna(0)
    df["Readmission_Flag"] = df["Readmission"].astype(str).str.lower().isin(["1","yes","true"])

    # ---------------------------------------------------------
    # FILTERS
    # ---------------------------------------------------------
    st.markdown("<div class='section-title'>Step 2 – Filters</div>", unsafe_allow_html=True)

    dep = st.multiselect("Department", df["Department"].unique())
    gen = st.multiselect("Gender", df["Gender"].unique())
    disease = st.multiselect("Disease", df["Disease"].unique())

    fdf = df.copy()
    if dep: fdf = fdf[fdf["Department"].isin(dep)]
    if gen: fdf = fdf[fdf["Gender"].isin(gen)]
    if disease: fdf = fdf[fdf["Disease"].isin(disease)]

    st.dataframe(fdf.head(10))
    download(fdf, "filtered_data.csv")

    # ---------------------------------------------------------
    # READMISSION MODEL
    # ---------------------------------------------------------
    st.markdown("<div class='section-title'>Predictive Analytics – Readmission</div>", unsafe_allow_html=True)

    clf_cols = ["Age","Gender","Department","Risk_Score","Length_of_Stay","Readmission_Flag"]
    if all(c in fdf.columns for c in clf_cols):

        clf_df = fdf[clf_cols].dropna()
        le1 = LabelEncoder()
        le2 = LabelEncoder()
        clf_df["Gender"] = le1.fit_transform(clf_df["Gender"])
        clf_df["Department"] = le2.fit_transform(clf_df["Department"])

        X = clf_df.drop("Readmission_Flag", axis=1)
        y = clf_df["Readmission_Flag"]

        if len(X) > 25:
            Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, random_state=42)
            mdl = RandomForestClassifier(n_estimators=120)
            mdl.fit(Xtr, ytr)

            pred = mdl.predict(Xts)
            st.success(f"Readmission Model Accuracy: {accuracy_score(yts, pred)*100:.2f}%")

            # Feature importance
            fi = pd.DataFrame({"Feature": X.columns, "Importance": mdl.feature_importances_}).sort_values("Importance", ascending=False)
            st.bar_chart(fi.set_index("Feature"))

    # ---------------------------------------------------------
    # COST MODEL
    # ---------------------------------------------------------
    st.markdown("<div class='section-title'>Predictive Analytics – Treatment Cost</div>", unsafe_allow_html=True)

    reg_cols = ["Age","Gender","Department","Risk_Score","Length_of_Stay","Treatment_Cost"]

    if all(c in fdf.columns for c in reg_cols):

        reg_df = fdf[reg_cols].dropna()
        reg_df["Gender"] = le1.fit_transform(reg_df["Gender"])
        reg_df["Department"] = le2.fit_transform(reg_df["Department"])

        X = reg_df.drop("Treatment_Cost", axis=1)
        y = reg_df["Treatment_Cost"]

        if len(X) > 25:
            Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, random_state=42)
            rmdl = RandomForestRegressor(n_estimators=120)
            rmdl.fit(Xtr, ytr)

            pred = rmdl.predict(Xts)
            rmse = np.sqrt(mean_squared_error(yts, pred))
            st.success(f"Treatment Cost RMSE: {rmse:,.2f}")

            # Feature importance
            fi = pd.DataFrame({"Feature": X.columns, "Importance": rmdl.feature_importances_}).sort_values("Importance", ascending=False)
            st.bar_chart(fi.set_index("Feature"))

            # SHAP
            expl = shap.TreeExplainer(rmdl)
            shap_vals = expl.shap_values(Xts)

            fig, ax = plt.subplots(figsize=(8,5))
            shap.summary_plot(shap_vals, Xts, plot_type="bar", show=False)
            st.pyplot(fig)
