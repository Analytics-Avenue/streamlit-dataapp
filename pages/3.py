import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


st.set_page_config(layout="wide")

hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(page_title="Real Estate Analytics", layout="wide")
st.title("🏙️ Real Estate Analytics Dashboard")

# ---------------------------------------------------------------
# REQUIRED COLUMNS
# ---------------------------------------------------------------
REQUIRED_COLS = [
    "Price",
    "City",
    "Area_sqft",
    "Bedrooms",
    "Bathrooms",
    "Property_Type",
    "Parking",
    "Furnishing"
]

# ---------------------------------------------------------------
# 1. DATASET SELECTION
# ---------------------------------------------------------------
mode = st.sidebar.radio(
    "Select Data Source",
    ["Default Dataset", "Upload CSV", "Upload CSV + Column Mapping"]
)

df = None

# ---------------------------------------------------------------
# DEFAULT DATASET
# ---------------------------------------------------------------
if mode == "Default Dataset":
    URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
    
    try:
        df = pd.read_csv(URL)
        st.success("Default dataset loaded successfully.")
    except Exception as e:
        st.error(f"Could not load dataset: {e}")

# ---------------------------------------------------------------
# UPLOAD CSV
# ---------------------------------------------------------------
if mode == "Upload CSV":
    file = st.file_uploader("Upload your dataset", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.success("Dataset uploaded successfully.")

# ---------------------------------------------------------------
# UPLOAD CSV + MAPPING
# ---------------------------------------------------------------
if mode == "Upload CSV + Column Mapping":
    file = st.file_uploader("Upload dataset", type=["csv"])
    if file:
        raw = pd.read_csv(file)
        st.write("Uploaded Data Preview", raw.head())

        st.subheader("Map Required Columns")
        mapping = {}
        for col in REQUIRED_COLS:
            mapping[col] = st.selectbox(f"Select column for {col}", raw.columns)

        df = raw.rename(columns=mapping)
        st.success("Column mapping applied.")

# ---------------------------------------------------------------
# STOP IF NO DATA
# ---------------------------------------------------------------
if df is None:
    st.warning("Dataset not loaded.")
    st.stop()

# ---------------------------------------------------------------
# FILTERS
# ---------------------------------------------------------------
st.sidebar.header("Filters")

cities = st.sidebar.multiselect("City", df["City"].unique(), default=df["City"].unique())
ptype = st.sidebar.multiselect("Property Type", df["Property_Type"].unique(), default=df["Property_Type"].unique())

filtered_df = df[df["City"].isin(cities) & df["Property_Type"].isin(ptype)]

# ---------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Avg Price", f"₹{filtered_df['Price'].mean():,.0f}")
col2.metric("Avg Area (sqft)", f"{filtered_df['Area_sqft'].mean():,.0f}")
col3.metric("Listings", f"{len(filtered_df)}")

# ---------------------------------------------------------------
# DOWNLOAD FILTERED DATA
# ---------------------------------------------------------------
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Filtered Data", csv, "filtered_realestate.csv", "text/csv")

# ---------------------------------------------------------------
# CHART SECTION
# ---------------------------------------------------------------
st.subheader("Price by City")

# Purpose + Quick Tips
with st.expander("📘 Purpose of Chart"):
    st.write("""
    This chart shows the average property prices across selected cities.  
    Helps in understanding market variation and demand-supply gaps.
    """)

with st.expander("💡 Quick Tips"):
    st.write("""
    • Compare metro vs tier-2 markets  
    • Spot overpriced cities  
    • Great for investor portfolio planning  
    """)

# Chart
city_data = filtered_df.groupby("City")["Price"].mean().reset_index()

fig = px.bar(
    city_data,
    x="City",
    y="Price",
    text="Price",
)

fig.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")

fig.update_layout(
    title="Average Price by City",
    xaxis_title="City",
    yaxis_title="Avg Price",
    xaxis=dict(title_font=dict(size=16, color="black")),
    yaxis=dict(title_font=dict(size=16, color="black")),
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# INSIGHTS
# ---------------------------------------------------------------
st.subheader("🔎 Automated Insights")

try:
    max_city = city_data.loc[city_data["Price"].idxmax()]["City"]
    min_city = city_data.loc[city_data["Price"].idxmin()]["City"]

    st.write(f"• Highest prices in **{max_city}**")  
    st.write(f"• Lowest prices in **{min_city}**")  
    st.write("• Price variation suggests investment opportunities in low-cost high-growth cities")
except:
    st.write("Not enough data for insights.")

# ---------------------------------------------------------------
# ML SECTION
# ---------------------------------------------------------------
st.subheader("ML Prediction Model")

try:
    df_ml = df.dropna()

    X = df_ml.drop("Price", axis=1)
    y = df_ml["Price"]

    cat_cols = X.select_dtypes(include=["object"]).columns
    num_cols = X.select_dtypes(include=["number"]).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("rf", RandomForestRegressor())
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = ((y_test - preds) ** 2).mean() ** 0.5

    st.metric("Model RMSE", f"₹{rmse:,.0f}")

except Exception as e:
    st.error(f"ML failed: {e}")
