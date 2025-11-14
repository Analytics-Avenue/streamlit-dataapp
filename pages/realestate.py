import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import io

# ----------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Price Prediction",
    layout="wide"
)

st.title("🏠 Real Estate Price Analytics & ML Prediction Dashboard")
st.markdown("Upload your dataset or use the provided sample dataset below.")

# ----------------------------------------------------------------
# CREATE SAMPLE DATASET
# ----------------------------------------------------------------
sample_df = pd.DataFrame({
    "location": ["Anna Nagar", "Velachery", "Adyar", "OMR", "Tambaram"] * 40,
    "sqft": np.random.randint(600, 2500, 200),
    "bhk": np.random.randint(1, 5, 200),
    "bathrooms": np.random.randint(1, 4, 200),
    "age": np.random.randint(1, 15, 200),
    "price": np.random.randint(25, 250, 200) * 100000
})

# Download button
csv_buffer = io.StringIO()
sample_df.to_csv(csv_buffer, index=False)
st.download_button(
    label="📥 Download Sample CSV",
    data=csv_buffer.getvalue(),
    file_name="sample_real_estate.csv",
    mime="text/csv"
)

# ----------------------------------------------------------------
# UPLOAD DATA
# ----------------------------------------------------------------
st.sidebar.header("Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
else:
    st.info("Using sample dataset (you can upload your own).")
    df = sample_df.copy()

# Display dataset
st.subheader("📌 Dataset Preview")
st.dataframe(df, use_container_width=True)

# ----------------------------------------------------------------
# COLUMN MAPPING
# ----------------------------------------------------------------
st.sidebar.header("Column Mapping")

all_cols = df.columns.tolist()

target_col = st.sidebar.selectbox("Price Column", all_cols, index=all_cols.index("price") if "price" in all_cols else 0)
sqft_col = st.sidebar.selectbox("Sqft Column", all_cols, index=all_cols.index("sqft") if "sqft" in all_cols else 0)
location_col = st.sidebar.selectbox("Location Column", all_cols, index=all_cols.index("location") if "location" in all_cols else 0)

numeric_opts = [c for c in all_cols if c not in [target_col, sqft_col, location_col]]
extra_numeric = st.sidebar.multiselect("Extra Numeric Columns", numeric_opts)

feature_cols = [sqft_col, location_col] + extra_numeric

# Build ML dataset
X = df[feature_cols]
y = df[target_col]

num_cols = X.select_dtypes(include=["int", "float"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# ----------------------------------------------------------------
# TRAIN TEST SPLIT
# ----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------------------
# TRANSFORMER
# ----------------------------------------------------------------
transformer = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ],
    remainder="passthrough"
)

X_train_t = transformer.fit_transform(X_train)
X_test_t = transformer.transform(X_test)

# ----------------------------------------------------------------
# MODEL TRAINING
# ----------------------------------------------------------------
model = RandomForestRegressor(
    n_estimators=350,
    random_state=42
)

model.fit(X_train_t, y_train)
y_pred = model.predict(X_test_t)

# ----------------------------------------------------------------
# KPIs
# ----------------------------------------------------------------
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model KPIs")
k1, k2 = st.columns(2)
k1.metric("RMSE", f"{rmse:,.2f}")
k2.metric("R² Score", f"{r2:.3f}")

st.markdown("---")

# ----------------------------------------------------------------
# FEATURE IMPORTANCE
# ----------------------------------------------------------------
st.subheader("🔥 Feature Importance")

encoded_cat = list(transformer.named_transformers_["cat"].get_feature_names_out(cat_cols))
final_cols = encoded_cat + list(num_cols)

importance_df = pd.DataFrame({
    "Feature": final_cols,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

fig_imp = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Importance",
    text="Importance"
)
fig_imp.update_traces(texttemplate='%{text:.3f}')
fig_imp.update_layout(height=650)

st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------------------
# PRICE DISTRIBUTION
# ----------------------------------------------------------------
st.subheader("💰 Price Distribution")

fig_price = px.histogram(
    df,
    x=target_col,
    nbins=35,
    text_auto=True,
    title="Distribution of Property Prices"
)
st.plotly_chart(fig_price, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------------------
# SQFT VS PRICE SCATTER
# ----------------------------------------------------------------
st.subheader("📌 Relationship: Sqft vs Price")

fig_scatter = px.scatter(
    df,
    x=sqft_col,
    y=target_col,
    trendline="ols",
    title=f"{sqft_col} vs {target_col}"
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------------------
# REAL ESTATE ANALYTICS NOTES
# ----------------------------------------------------------------
st.subheader("🧠 Real Estate Analytics Overview")

st.markdown("""
### Key Metrics Analysts Monitor
• Price per sqft  
• Market supply vs demand  
• Locality premium score  
• Construction age value impact  
• Amenities impact  
• Buyer sentiment index  
• Seasonality trends (Festive months spike)  
""")

# ----------------------------------------------------------------
# PURPOSE OF CHARTS
# ----------------------------------------------------------------
st.subheader("🎯 Purpose of the Charts")

st.markdown("""
• **Feature Importance**: Shows which inputs influence price the most.  
• **Price Distribution**: Identifies skewness, outliers, unusual price spikes.  
• **Scatter Plot**: Checks how sqft affects pricing.  
• **KPIs**: Measures accuracy and reliability of your ML model.  
""")

# ----------------------------------------------------------------
# QUICK TIPS
# ----------------------------------------------------------------
st.subheader("⚡ Quick Tips for Better Predictions")

st.markdown("""
• Remove outliers above 3 standard deviations.  
• Use latitude and longitude instead of locality name.  
• More data improves accuracy more than hyperparameters.  
• Convert bathroom/BHK/amenities into numeric ratios.  
""")
