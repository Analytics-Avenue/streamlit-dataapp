import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import math
import warnings
warnings.filterwarnings("ignore")

# --- Page Setup ---
st.set_page_config(page_title="Healthcare Analytics & Insights", layout="wide")
st.markdown("<h1 style='margin-bottom:0.2rem'>Healthcare Analytics & Insights</h1>", unsafe_allow_html=True)
st.markdown("Enterprise-capable healthcare analytics: patient trends, treatment efficiency, readmission prediction, and cost forecasting.")

# -------------------------
# Dataset handling
# -------------------------
REQUIRED_HEALTH_COLS = [
    "Patient_ID", "Age", "Gender", "Disease", "Symptoms", "Treatment", 
    "Admission_Date", "Discharge_Date", "Treatment_Cost", "Readmission"
]

def auto_map_health_columns(df):
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
        "Readmission":["readmission","Readmit","Rehospitalization"]
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

# --- Dataset selection ---
st.markdown("### Step 1 — Load dataset")
mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)
df = None

if mode == "Default dataset":
    DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/healthcare/healthcare.csv"
    try:
        df = pd.read_csv(DEFAULT_URL)
        df = auto_map_health_columns(df)
        st.success("Default dataset loaded")
        st.dataframe(df.head())
    except Exception as e:
        st.error("Failed to load default dataset: " + str(e))
        st.stop()

elif mode == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        df = auto_map_health_columns(df)
        st.success("File uploaded and mapped")
        st.dataframe(df.head())

else:  # Upload + mapping
    uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
    if uploaded:
        raw = pd.read_csv(uploaded)
        st.write("Preview (first 5 rows):")
        st.dataframe(raw.head())
        mapping = {}
        for req in REQUIRED_HEALTH_COLS:
            mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + list(raw.columns))
        if st.button("Apply mapping"):
            missing = [k for k,v in mapping.items() if v == "-- Select --"]
            if missing:
                st.error("Please map all required columns: " + ", ".join(missing))
            else:
                df = raw.rename(columns={v:k for k,v in mapping.items()})
                st.success("Mapping applied")
                st.dataframe(df.head())

if df is None:
    st.stop()

# Validate required columns
missing_cols = [c for c in REQUIRED_HEALTH_COLS if c not in df.columns]
if missing_cols:
    st.error("Missing required columns: " + ", ".join(missing_cols))
    st.stop()

# -------------------------
# Type conversions
# -------------------------
df = df.copy()
df["Admission_Date"] = pd.to_datetime(df["Admission_Date"], errors="coerce")
df["Discharge_Date"] = pd.to_datetime(df["Discharge_Date"], errors="coerce")
numeric_cols = ["Age", "Treatment_Cost"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Derived metrics
df["Length_of_Stay"] = (df["Discharge_Date"] - df["Admission_Date"]).dt.days.fillna(0)
df["Readmission_Flag"] = df["Readmission"].apply(lambda x: 1 if str(x).lower() in ["yes","1","true"] else 0)

# -------------------------
# Tabs: Overview & Application
# -------------------------
tabs = st.tabs(["Overview", "Application"])

with tabs[0]:
    st.markdown("### Overview")
    st.markdown("""
    <div style='background:#f0f0f0;padding:20px;border-radius:12px;box-shadow:0 4px 15px rgba(0,0,0,0.15)'>
    Healthcare analytics insights app: Track patient trends, treatment efficiency, costs, readmissions, and predict outcomes.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Total Patients", f"{df['Patient_ID'].nunique():,}")
    k2.metric("Average Age", f"{df['Age'].mean():.1f}")
    k3.metric("Avg Length of Stay (days)", f"{df['Length_of_Stay'].mean():.1f}")
    k4.metric("Total Treatment Cost", f"₹ {df['Treatment_Cost'].sum():,.0f}")
    k5.metric("Readmission Rate", f"{df['Readmission_Flag'].mean()*100:.2f}%")


# --------------------------
# Application Tab
# --------------------------
with tabs[1]:
    st.header("Application / Filters")
    
    sel_dept = st.multiselect("Department", df["Department"].unique(), default=df["Department"].unique())
    sel_gender = st.multiselect("Gender", df["Gender"].unique(), default=df["Gender"].unique())
    date_range = st.date_input("Visit Date Range", value=(df["Visit_Date"].min(), df["Visit_Date"].max()))
    
    filt = df.copy()
    filt = filt[filt["Department"].isin(sel_dept)]
    filt = filt[filt["Gender"].isin(sel_gender)]
    filt = filt[(filt["Visit_Date"] >= pd.to_datetime(date_range[0])) & 
                (filt["Visit_Date"] <= pd.to_datetime(date_range[1]))]
    
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(10), "filtered_healthcare_preview.csv")
    
    # Metrics
    st.markdown("### Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", filt['Patient_ID'].nunique())
    c2.metric("Avg Age", f"{filt['Age'].mean():.1f}")
    c3.metric("Readmission Rate", f"{(filt['Readmission'].sum()/len(filt)*100):.2f}%")
    c4.metric("Total Treatment Cost", to_currency(filt['Treatment_Cost'].sum()))
    
    # Charts
    fig_outcome = px.pie(filt, names="Outcome", title="Treatment Outcomes")
    st.plotly_chart(fig_outcome, use_container_width=True)
    
    fig_risk = px.bar(filt['Risk_Level'].value_counts().reset_index(), x='index', y='Risk_Level',
                      title="Patient Risk Levels")
    st.plotly_chart(fig_risk, use_container_width=True)

# --------------------------
# Predictions Tab
# --------------------------
with tabs[2]:
    st.header("Predictive Analytics")

    st.markdown("### Readmission Prediction (Classification)")
    # Prepare data
    clf_df = df[['Age','Gender','Department','Risk_Score','Length_of_Stay_days','Readmission']].copy()
    # Encode categorical
    le_gender = LabelEncoder()
    le_dept = LabelEncoder()
    clf_df['Gender'] = le_gender.fit_transform(clf_df['Gender'])
    clf_df['Department'] = le_dept.fit_transform(clf_df['Department'])
    
    X_clf = clf_df.drop('Readmission', axis=1)
    y_clf = clf_df['Readmission']
    
    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_model.fit(X_train, y_train)
    
    y_pred = clf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.markdown(f"Model Accuracy: **{acc*100:.2f}%**")
    
    st.markdown("#### Predict for new patient")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", df['Gender'].unique())
    dept = st.selectbox("Department", df['Department'].unique())
    risk_score = st.number_input("Risk Score", min_value=0.0, max_value=10.0, value=3.0)
    los = st.number_input("Length of Stay (days)", min_value=0, max_value=100, value=5)
    
    if st.button("Predict Readmission"):
        x_new = pd.DataFrame([[age, le_gender.transform([gender])[0], le_dept.transform([dept])[0], risk_score, los]],
                             columns=X_clf.columns)
        pred = clf_model.predict(x_new)[0]
        st.success(f"Predicted Readmission: {'Yes' if pred==1 else 'No'}")
    
    # --------------------------
    st.markdown("### Treatment Cost Prediction (Regression)")
    reg_df = df[['Age','Gender','Department','Risk_Score','Length_of_Stay_days','Treatment_Cost']].copy()
    # Encode categorical
    reg_df['Gender'] = le_gender.fit_transform(reg_df['Gender'])
    reg_df['Department'] = le_dept.fit_transform(reg_df['Department'])
    
    X_reg = reg_df.drop('Treatment_Cost', axis=1)
    y_reg = reg_df['Treatment_Cost']
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train_r, y_train_r)
    
    y_pred_r = reg_model.predict(X_test_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    st.markdown(f"Model RMSE: **{rmse:.2f}**")
    
    st.markdown("#### Predict Treatment Cost for new patient")
    age_r = st.number_input("Age (Cost)", min_value=0, max_value=120, value=30, key='age_r')
    gender_r = st.selectbox("Gender (Cost)", df['Gender'].unique(), key='gender_r')
    dept_r = st.selectbox("Department (Cost)", df['Department'].unique(), key='dept_r')
    risk_score_r = st.number_input("Risk Score (Cost)", min_value=0.0, max_value=10.0, value=3.0, key='risk_r')
    los_r = st.number_input("Length of Stay (Cost)", min_value=0, max_value=100, value=5, key='los_r')
    
    if st.button("Predict Treatment Cost", key="predict_cost"):
        x_new_r = pd.DataFrame([[age_r, le_gender.transform([gender_r])[0], le_dept.transform([dept_r])[0], risk_score_r, los_r]],
                               columns=X_reg.columns)
        pred_cost = reg_model.predict(x_new_r)[0]
        st.success(f"Predicted Treatment Cost: {to_currency(pred_cost)}")

        # Add at the end of Predictions tab after the previous code
        
        import shap
        import matplotlib.pyplot as plt
        
        st.markdown("### Feature Importance & SHAP Analysis")
        
        # --- Readmission Classification Feature Importance ---
        st.markdown("#### Readmission Prediction Feature Importance")
        clf_importances = pd.DataFrame({
            'Feature': X_clf.columns,
            'Importance': clf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        fig_imp = px.bar(clf_importances, x='Feature', y='Importance', title="Feature Importance (Readmission)")
        st.plotly_chart(fig_imp, use_container_width=True)
        
        # --- SHAP Analysis for Classification ---
        st.markdown("#### SHAP Summary Plot (Readmission)")
        explainer_clf = shap.TreeExplainer(clf_model)
        shap_values_clf = explainer_clf.shap_values(X_test)
        
        fig_shap, ax = plt.subplots()
        shap.summary_plot(shap_values_clf[1], X_test, plot_type="bar", show=False)  # for class 1 (Readmission=Yes)
        st.pyplot(fig_shap)
        
        # --- Treatment Cost Regression Feature Importance ---
        st.markdown("#### Treatment Cost Feature Importance")
        reg_importances = pd.DataFrame({
            'Feature': X_reg.columns,
            'Importance': reg_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        fig_reg_imp = px.bar(reg_importances, x='Feature', y='Importance', title="Feature Importance (Treatment Cost)")
        st.plotly_chart(fig_reg_imp, use_container_width=True)
        
        # --- SHAP Analysis for Regression ---
        st.markdown("#### SHAP Summary Plot (Treatment Cost)")
        explainer_reg = shap.TreeExplainer(reg_model)
        shap_values_reg = explainer_reg.shap_values(X_test_r)
        
        fig_shap_r, ax2 = plt.subplots()
        shap.summary_plot(shap_values_reg, X_test_r, plot_type="bar", show=False)
        st.pyplot(fig_shap_r)


