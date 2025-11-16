import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ----------------------------------------------------
# APP CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="Marketing ML Dashboard", layout="wide")
st.markdown("""
<style>

.card {
    background: rgba(255,255,255,0.08);
    padding: 18px 20px;
    border-radius: 14px;
    margin-bottom: 15px;
    border: 1px solid rgba(255,255,255,0.25);
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    backdrop-filter: blur(6px);
    transition: all .25s ease;
}

.card:hover {
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.55);
    box-shadow: 0 0 22px rgba(255,255,255,0.5);
    transform: scale(1.03);
}

/* Metric card */
.metric-card {
    background: rgba(255,255,255,0.12);
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.30);
    font-weight: 600;
    font-size: 16px;
    transition: all 0.25s ease;
    box-shadow: 0 2px 10px rgba(0,0,0,0.18);
    backdrop-filter: blur(5px);
}

.metric-card:hover {
    background: rgba(255,255,255,0.20);
    box-shadow: 0 0 16px rgba(255,255,255,0.45);
    transform: scale(1.04);
}
</style>
""", unsafe_allow_html=True)

st.title("Marketing Analytics App — ML Dashboard")
st.markdown("Build insights from your marketing data using classical ML. No AI API required.")

# ----------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Data", "EDA", "ML Model"])

# ----------------------------------------------------
# SESSION STATE
# ----------------------------------------------------
if "marketing_df" not in st.session_state:
    st.session_state.marketing_df = None

# ----------------------------------------------------
# UPLOAD DATA
# ----------------------------------------------------
if page == "Upload Data":
    st.header("Step 1: Upload Marketing Dataset")

    tab1, tab2 = st.tabs(["Overview", "Application"])

    # ----------------- TAB 1 -----------------
    with tab1:
        st.markdown("### Welcome to the Marketing ML Dashboard")
        st.markdown("""
        <div class='card'>
            A lightweight machine-learning workspace built for marketers.
            Upload your dataset, explore patterns, and run classical ML models
            without requiring any external AI API.
        </div>
        """, unsafe_allow_html=True)
    
        st.markdown("### What This App Does")
        st.markdown("""
        <div class='card'>
            • Understand campaign-level trends<br>
            • Create automated EDA visualizations<br>
            • Train ML classification/regression models<br>
            • Predict outcomes from new input values<br>
            • Explore correlations, feature importance, and KPIs<br>
        </div>
        """, unsafe_allow_html=True)
    
        k1, k2, k3 = st.columns(3)
    
        k1.markdown("<div class='metric-card'>EDA</div>", unsafe_allow_html=True)
        k2.markdown("<div class='metric-card'>ML Models</div>", unsafe_allow_html=True)
        k3.markdown("<div class='metric-card'>Predictions</div>", unsafe_allow_html=True)

    # ----------------- TAB 2 -----------------
    with tab2:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded:
            df = pd.read_csv(uploaded)
            st.session_state.marketing_df = df
            st.success("Dataset uploaded successfully")

        st.subheader("Preview")
        if st.session_state.marketing_df is not None:
            st.dataframe(st.session_state.marketing_df)

# ----------------------------------------------------
# EDA SECTION
# ----------------------------------------------------
elif page == "EDA":
    st.header("Step 2: Explore the Data")

    df = st.session_state.marketing_df
    if df is None:
        st.warning("Upload a dataset first.")
        st.stop()

    tab1, tab2 = st.tabs(["Overview", "Application"])

    # ----------------- TAB 1 -----------------
    with tab1:
        st.subheader("Overview")
        st.write("""
        This section shows:
        - Summary statistics  
        - Correlation heatmap  
        - Campaign performance charts  
        """)

    # ----------------- TAB 2 -----------------
    with tab2:
        st.subheader("Dataset Overview")
        st.write(df.describe(include="all"))

        st.subheader("Correlation Heatmap")
        num_df = df.select_dtypes(include=[np.number])
        if not num_df.empty:
            fig = px.imshow(num_df.corr(), text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Campaign Metrics")
        if "Campaign" in df.columns and "Clicks" in df.columns:
            fig = px.bar(df, x="Campaign", y="Clicks", color="Campaign")
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# ML SECTION
# ----------------------------------------------------
elif page == "ML Model":
    st.header("Step 3: Build ML Model")

    df = st.session_state.marketing_df
    if df is None:
        st.warning("Upload a dataset first.")
        st.stop()

    tab1, tab2 = st.tabs(["Overview", "Application"])

    # ----------------- TAB 1 -----------------
    with tab1:
        st.subheader("Overview")
        st.write("""
        Build classical ML models on your marketing dataset.  
        Steps:
        1. Select target column  
        2. Choose classification or regression  
        3. Train/test split  
        4. Train model  
        5. View feature importance  
        6. Predict on new values  
        """)

    # ----------------- TAB 2 -----------------
    with tab2:
        st.subheader("Target Column")
        target = st.selectbox("Select target variable", df.columns)

        if target:
            X = df.drop(columns=[target])
            y = df[target]

            numeric_cols = X.select_dtypes(include=np.number).columns
            cat_cols = X.select_dtypes(exclude=np.number).columns

            st.markdown("### ML Pipeline Setup")

            preprocessor = ColumnTransformer([
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
            ])

            st.subheader("Choose Model Type")
            model_type = st.radio("Model Type", ["Classification", "Regression"])

            if model_type == "Classification":
                model = RandomForestClassifier()
            else:
                model = RandomForestRegressor()

            pipeline = Pipeline([
                ("prep", preprocessor),
                ("model", model)
            ])

            test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

            if st.button("Train Model"):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                pipeline.fit(X_train, y_train)
                score = pipeline.score(X_test, y_test)

                st.success(f"Model trained successfully! Accuracy/Score: {score:.3f}")

                try:
                    importances = pipeline.named_steps["model"].feature_importances_
                    st.subheader("Feature Importance")

                    feature_names = (
                        pipeline.named_steps["prep"]
                        .named_transformers_["cat"]
                        .get_feature_names_out(cat_cols).tolist()
                        + numeric_cols.tolist()
                    )

                    fig = px.bar(
                        x=importances,
                        y=feature_names,
                        orientation="h"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Feature importance not available for this model.")

            st.subheader("Predict on New Data")
            st.write("Enter values for prediction:")

            user_input = {}
            for col in X.columns:
                if col in numeric_cols:
                    user_input[col] = st.number_input(col, value=0.0)
                else:
                    user_input[col] = st.text_input(col, "")

            if st.button("Predict"):
                input_df = pd.DataFrame([user_input])
                pred = pipeline.predict(input_df)[0]
                st.success(f"Prediction: {pred}")
