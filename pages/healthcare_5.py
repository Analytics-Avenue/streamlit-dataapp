import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from datetime import datetime

# =========================================
# Streamlit Page Config
# =========================================
st.set_page_config(page_title="Hospital Risk & Resource Dashboard", layout="wide")
st.title("🏥 Hospital Risk & Resource Analytics Dashboard")
st.write("Upload hospital dataset to run analytics, ML predictions & clustering.")

# =========================================
# File Upload
# =========================================
uploaded_file = st.file_uploader("Upload Hospital Dataset (CSV)", type="csv")

df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # -------- Duplicate Column Fix --------
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = df.columns.str.strip()
    # --------------------------------------

    st.success("Dataset loaded successfully!")
    st.write("### Preview")
    st.dataframe(df.head())

# Exit if no file
if df is None:
    st.stop()

# =========================================
# Column Detection
# =========================================
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# Remove Patient_ID if numeric
if "Patient_ID" in numeric_cols:
    numeric_cols.remove("Patient_ID")

# Risk labels
risk_score_col = "Risk_Score" if "Risk_Score" in df.columns else None
risk_level_col = "Risk_Level" if "Risk_Level" in df.columns else None

# =========================================
# Sidebar Navigation
# =========================================
menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "ML Prediction", "Clustering", "Hospital Profiles"]
)

# =========================================
# 1. OVERVIEW
# =========================================
if menu == "Overview":
    st.header("📊 Data Overview")

    st.write("#### Summary Stats")
    st.dataframe(df.describe())

    # Risk distribution
    if risk_level_col:
        fig = px.histogram(df, x=risk_level_col, title="Risk Level Distribution",
                           color=risk_level_col)
        st.plotly_chart(fig, use_container_width=True)

    # Numeric exploration
    st.write("#### Explore Numeric Columns")
    col_x = st.selectbox("Select numeric column for histogram", numeric_cols)
    fig = px.histogram(df, x=col_x, title=f"Distribution of {col_x}")
    st.plotly_chart(fig, use_container_width=True)

# =========================================
# 2. MACHINE LEARNING
# =========================================
if menu == "ML Prediction":
    st.header("🤖 ML Model Training & Prediction")

    # Target Selection
    target = st.selectbox("Select Target Column", [risk_score_col, risk_level_col])

    if target is None:
        st.error("No suitable target column found.")
        st.stop()

    # Type of ML task
    if target == risk_score_col:
        model_task = "Regression"
    else:
        model_task = "Classification"

    st.info(f"Detected ML Task: **{model_task}**")

    # Prepare Features
    feature_cols = [c for c in df.columns if c != target]

    X = df[feature_cols]
    y = df[target]

    # Preprocessor
    pre = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), [c for c in numeric_cols if c in X.columns]),
            ('cat', OneHotEncoder(handle_unknown='ignore'), [c for c in cat_cols if c in X.columns])
        ]
    )

    # Model Selection
    if model_task == "Regression":
        model = RandomForestRegressor()
    else:
        model = RandomForestClassifier()

    pipe = Pipeline([("preprocessor", pre), ("model", model)])

    if st.button("Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe.fit(X_train, y_train)

        st.session_state["ml_pipeline"] = pipe
        st.success("Model trained successfully!")

        # Metrics
        preds = pipe.predict(X_test)

        if model_task == "Regression":
            st.write("### Performance")
            st.write("MSE:", mean_squared_error(y_test, preds))
            st.write("R2 Score:", r2_score(y_test, preds))
        else:
            st.write("### Accuracy")
            st.write("Accuracy:", accuracy_score(y_test, preds))

    # Single-row Predict
    st.write("### 🔮 Single Hospital Prediction")
    if "ml_pipeline" in st.session_state:
        pipe = st.session_state["ml_pipeline"]

        sim_vals = {}
        st.write("Enter hospital values to simulate prediction:")
        for col in feature_cols:
            if col in numeric_cols:
                sim_vals[col] = st.number_input(col, value=float(df[col].mean()))
            else:
                sim_vals[col] = st.selectbox(col, df[col].unique())

        if st.button("Predict Now"):
            row = pd.DataFrame([sim_vals])
            pred = pipe.predict(row)[0]
            st.success(f"Predicted Output: **{pred}**")

    else:
        st.info("Train a model first.")

# =========================================
# 3. CLUSTERING
# =========================================
if menu == "Clustering":
    st.header("🧩 Hospital Clustering")

    cluster_features = st.multiselect("Select clustering features", numeric_cols, default=numeric_cols[:4])
    k = st.slider("Number of Clusters", 2, 10, 3)

    if st.button("Run Clustering"):
        km = KMeans(n_clusters=k, random_state=42)
        df["Cluster"] = km.fit_predict(df[cluster_features])

        st.success("Clustering complete!")

        fig = px.scatter(df, x=cluster_features[0], y=cluster_features[1],
                         color="Cluster", title="Cluster Visualization")
        st.plotly_chart(fig, use_container_width=True)

        st.write("### Cluster Counts")
        st.write(df["Cluster"].value_counts())

# =========================================
# 4. HOSPITAL PROFILES
# =========================================
if menu == "Hospital Profiles":
    st.header("🏥 Hospital Explorer")

    if "Hospital Name" in df.columns:
        selected = st.selectbox("Select Hospital", df["Hospital Name"].unique())

        profile = df[df["Hospital Name"] == selected].iloc[0]
        st.write("### Details")
        st.write(profile)

        # Radar Chart (if numeric cols exist)
        if len(numeric_cols) >= 3:
            fig = px.line_polar(
                r=[profile[col] for col in numeric_cols[:6]],
                theta=numeric_cols[:6],
                line_close=True,
                title="Hospital Capability Radar"
            )
            st.plotly_chart(fig, use_container_width=True)
