import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# ----------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Price Prediction",
    layout="wide"
)

st.title("🏠 Real Estate Price Analytics & Prediction Dashboard")

st.markdown("""
This workspace brings data analytics + ML prediction together  
so you can decode real estate pricing like a seasoned analyst.
""")

# ----------------------------------------------------------------
# UPLOAD DATA
# ----------------------------------------------------------------
st.sidebar.header("Upload Your Real Estate Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("📌 Raw Dataset")
    st.dataframe(df, use_container_width=True)

    # -------------------------------------------------------------
    # CHOOSE FEATURES
    # -------------------------------------------------------------
    st.sidebar.subheader("Model Configuration")

    target_col = st.sidebar.selectbox(
        "Select Target Variable (Price Column)",
        df.columns
    )

    feature_cols = st.sidebar.multiselect(
        "Select Feature Columns",
        [col for col in df.columns if col != target_col],
        default=[col for col in df.columns if col != target_col]
    )

    # Basic Validation
    if not feature_cols:
        st.warning("Choose at least one feature column.")
        st.stop()

    X = df[feature_cols]
    y = df[target_col]

    # -------------------------------------------------------------
    # SPLIT DATA
    # -------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Separate numerical and categorical features
    num_cols = X.select_dtypes(include=["int", "float"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # -------------------------------------------------------------
    # COLUMN TRANSFORMER
    # -------------------------------------------------------------
    transformer = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ],
        remainder="passthrough"
    )

    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)

    # -------------------------------------------------------------
    # RANDOM FOREST MODEL
    # -------------------------------------------------------------
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42
    )
    model.fit(X_train_transformed, y_train)

    y_pred = model.predict(X_test_transformed)

    # FIX: RMSE WITHOUT unexpected keyword args
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # -------------------------------------------------------------
    # KPI ROW
    # -------------------------------------------------------------
    st.subheader("📊 Model Performance")

    kpi1, kpi2 = st.columns(2)
    kpi1.metric("RMSE", f"{rmse:,.2f}")
    kpi2.metric("R² Score", f"{r2:.3f}")

    st.markdown("---")

    # -------------------------------------------------------------
    # FEATURE IMPORTANCE
    # -------------------------------------------------------------
    st.subheader("🔥 Feature Importance")

    # Get feature names after OneHotEncoding
    encoded_cat = list(transformer.named_transformers_["cat"].get_feature_names_out(cat_cols))
    final_features = encoded_cat + list(num_cols)

    importance_df = pd.DataFrame({
        "Feature": final_features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig_imp = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance Chart",
        text="Importance"
    )
    fig_imp.update_traces(texttemplate='%{text:.3f}')
    fig_imp.update_layout(height=600)

    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")

    # -------------------------------------------------------------
    # PRICE DISTRIBUTION
    # -------------------------------------------------------------
    st.subheader("💰 Price Distribution")
    fig_price = px.histogram(
        df,
        x=target_col,
        nbins=40,
        title="Distribution of Property Prices",
        text_auto=True
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # -------------------------------------------------------------
    # SCATTERPLOT
    # -------------------------------------------------------------
    st.subheader("📌 Relationship Analysis")

    numeric_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()

    scatter_x = st.selectbox("Select X-axis Feature", numeric_cols, index=0)

    fig_scatter = px.scatter(
        df,
        x=scatter_x,
        y=target_col,
        trendline="ols",
        title=f"{scatter_x} vs {target_col}",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # -------------------------------------------------------------
    # REAL ESTATE ANALYTICS NOTES (YOUR REQUEST)
    # -------------------------------------------------------------
    st.markdown("## 🧠 Real Estate Data Analytics Overview")

    st.markdown("""
### What Analysts Usually Track:
• Price trends across locations  
• Sqft vs price correlation  
• Amenities impact on pricing  
• Demand-supply differences  
• Seasonal buying patterns  
• Builder reputation scoring  
• Rental yield vs purchase value  

### Why It Matters:
Real estate pricing is part logic, part chaos.  
Good analytics reduces the chaos.
""")

    # -------------------------------------------------------------
    # CHART PURPOSES (YOUR REQUEST)
    # -------------------------------------------------------------
    st.markdown("## 🎯 Purpose of Each Chart")

    st.markdown("""
• **Feature Importance:** Shows which variables actually influence the price.  
• **Price Distribution:** Helps spot skewness, outliers, unrealistic values.  
• **Scatterplot:** Shows whether sqft, bathrooms, bedrooms actually relate to price.  
• **KPIs (RMSE, R²):** Tell if your model is decent or trash.  
""")

    # -------------------------------------------------------------
    # QUICK TIPS (Your request)
    # -------------------------------------------------------------
    st.markdown("## ⚡ Quick Tips")

    st.markdown("""
• Remove extreme outliers before training.  
• Convert sqft to price per sqft metrics.  
• Use latitude/longitude instead of “location name” when possible.  
• Avoid dumping 50+ categorical features into ML models.  
• More clean data beats more complex models.  
""")

else:
    st.info("Upload a CSV file to begin.")
