import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ---------------------------
# App Config & CSS
# ---------------------------
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
.card:hover { background: rgba(255,255,255,0.18); transform: scale(1.03);}
.metric-card { background: rgba(255,255,255,0.12); padding: 20px; border-radius: 14px; text-align: center; font-weight: 600; font-size: 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.18); }
.metric-card:hover { background: rgba(255,255,255,0.20); transform: scale(1.04);}
</style>
""", unsafe_allow_html=True)

st.title("Marketing Analytics ML Dashboard")
st.markdown("Upload, explore, train ML, and predict in one place.")

# ---------------------------
# Session State
# ---------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

# ---------------------------
# Dataset Selection
# -------------------------
# Dataset input: default, upload, mapping
# -------------------------
st.markdown("### Step 1 — Load dataset")
mode = st.radio(
    "Dataset option:",
    ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"],
    horizontal=True
)

df = None

# REQUIRED FIELDS (for mapping)
REQUIRED_MARKETING_COLS = ["Campaign", "Clicks", "Impressions", "Budget"]

# Placeholder: replace with actual auto-mapping logic
def auto_map_columns(df):
    # Example: just ensure required columns exist; real mapping logic can go here
    for col in REQUIRED_MARKETING_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df

# -------------------------
# Dataset input: default, upload, mapping
# -------------------------
st.markdown("### Step 1 — Load dataset")
mode = st.radio(
    "Dataset option:",
    ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"],
    horizontal=True
)

df = None

# REQUIRED FIELDS (for mapping)
REQUIRED_MARKETING_COLS = ["Campaign", "Clicks", "Impressions", "Budget"]

# Placeholder auto-map function
def auto_map_columns(df):
    for col in REQUIRED_MARKETING_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df

# -------------------------
# Mode: Default dataset
# -------------------------
if mode == "Default dataset":
    DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/marketing.csv"
    try:
        df = pd.read_csv(DEFAULT_URL)
        df.columns = df.columns.str.strip()
        df = auto_map_columns(df)
        st.session_state.df = df  # Always assign here
        st.success("Default dataset loaded")
        st.dataframe(df.head())
    except Exception as e:
        st.error("Failed to load default dataset: " + str(e))
        st.stop()

# -------------------------
# Mode: Upload CSV
# -------------------------
elif mode == "Upload CSV":
    st.markdown("#### Download Sample CSV for Reference")
    SAMPLE_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/Marketing_Analytics.csv"
    try:
        sample_df = pd.read_csv(SAMPLE_URL).head(5)
        sample_csv = sample_df.to_csv(index=False)
        st.download_button("Download Sample CSV", sample_csv, "sample_dataset.csv", "text/csv")
    except Exception as e:
        st.info(f"Sample CSV unavailable: {e}")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        df.columns = df.columns.str.strip()
        df = auto_map_columns(df)
        st.session_state.df = df  # Always assign here
        st.success("File uploaded.")
        st.dataframe(df.head())
        sample_small = df.head(5).to_csv(index=False)
        st.download_button("Download sample (first 5 rows)", sample_small, "sample_uploaded_5rows.csv", "text/csv")

# -------------------------
# Mode: Upload + Column Mapping
# -------------------------
else:
    uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        raw.columns = raw.columns.str.strip()
        st.write("Preview (first 5 rows):")
        st.dataframe(raw.head())

        st.markdown("Map your columns to required fields (only required fields shown).")
        mapping = {}
        for req in REQUIRED_MARKETING_COLS:
            mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + list(raw.columns))

        if st.button("Apply mapping"):
            missing = [k for k, v in mapping.items() if v == "-- Select --"]
            if missing:
                st.error("Please map all required columns: " + ", ".join(missing))
            else:
                df = raw.rename(columns={v: k for k, v in mapping.items()})
                st.session_state.df = df  # Always assign here
                st.success("Mapping applied.")
                st.dataframe(df.head())
                sample_small = df.head(5).to_csv(index=False)
                st.download_button(
                    "Download mapped sample (5 rows)",
                    sample_small,
                    "mapped_sample_5rows.csv",
                    "text/csv"
                )

# -------------------------
# Preview Dataset
# -------------------------
if st.session_state.df is not None:
    st.subheader("Dataset Preview")
    st.dataframe(st.session_state.df)



# ---------------------------
# EDA Section
# ---------------------------
if st.session_state.df is not None:
    st.header("Step 2: EDA & Visualization")
    df = st.session_state.df
    st.subheader("Summary Statistics")
    st.write(df.describe(include="all"))

    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        st.subheader("Correlation Heatmap")
        fig = px.imshow(df[num_cols].corr(), text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

    if "Campaign" in df.columns and "Clicks" in df.columns:
        st.subheader("Clicks per Campaign")
        fig = px.bar(df, x="Campaign", y="Clicks", color="Campaign")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# ML Section
# ---------------------------
if st.session_state.df is not None:
    st.header("Step 3: ML Model & Prediction")

    target = st.selectbox("Select Target Column", df.columns)
    model_type = st.radio("Select Model Type", ["Regression", "Classification"])

    if target:
        X = df.drop(columns=[target]).copy()
        y = df[target].copy()

        # Numeric / categorical split
        numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

        # Convert all numeric columns safely
        for col in numeric_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # Regression target numeric conversion
        if model_type == "Regression":
            y = pd.to_numeric(y, errors='coerce')

        # Drop rows with any NaNs in features or target
        combined = X.copy()
        combined[target] = y
        combined = combined.dropna()
        X = combined.drop(columns=[target])
        y = combined[target]

        if len(X) < 2:
            st.error("Not enough valid rows after cleaning for training.")
            st.stop()

        # Recalculate numeric/categorical columns after dropping NaNs
        numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

        # Pipeline
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ])
        model = RandomForestRegressor() if model_type == "Regression" else RandomForestClassifier()
        pipeline = Pipeline([("prep", preprocessor), ("model", model)])

        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

        if st.button("Train Model"):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            # Fit model
            pipeline.fit(X_train, y_train)
            st.session_state.pipeline = pipeline
            score = pipeline.score(X_test, y_test)
            st.success(f"Trained — Score: {score:.3f}")

            # Feature importance
            try:
                importances = pipeline.named_steps["model"].feature_importances_
                feature_names = numeric_cols.copy()
                if cat_cols:
                    cat_names = pipeline.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols)
                    feature_names += cat_names.tolist()
                fig = px.bar(x=importances, y=feature_names, orientation="h", title="Feature Importances")
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Feature importance not available for this model.")


# ---------------------------
# Quick Predict Single Row
# ---------------------------
if st.session_state.pipeline is not None:
    st.header("Step 4: Quick Predict (Single Row)")

    X = st.session_state.df.drop(columns=[target])
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    feat_cols = X.columns

    pcols = st.columns(len(feat_cols))
    inputs = {}
    for i, col in enumerate(feat_cols):
        if col in numeric_cols:
            inputs[col] = pcols[i].number_input(col, value=float(X[col].median()))
        else:
            options = sorted(X[col].dropna().unique().tolist())[:200]
            default_index = 0 if options else None
            inputs[col] = pcols[i].selectbox(col, options=options, index=default_index if default_index is not None else 0)

    if st.button("Predict Single Row"):
        row = pd.DataFrame([inputs])
        try:
            pred_val = st.session_state.pipeline.predict(row)[0]
            st.success(f"Prediction: {pred_val}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("Train a model first to enable quick prediction.")
