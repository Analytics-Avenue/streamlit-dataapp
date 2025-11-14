import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="Real Estate Analytics Dashboard",
    layout="wide"
)

# -----------------------------------------------------
# TITLE
# -----------------------------------------------------
st.title("🏡 Real Estate Analytics Dashboard")
st.markdown("""
Welcome to your **modern real estate intelligence suite**.  
Track prices, understand demand, evaluate regions, and predict future value.
""")

# -----------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------
st.sidebar.header("📁 Upload Data")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Synthetic dataset fallback
def load_sample_data():
    n = 250
    np.random.seed(42)
    df = pd.DataFrame({
        "Date": pd.date_range(start="2022-01-01", periods=n),
        "Property_Type": np.random.choice(["Apartment", "Condo", "House", "Villa", "Townhouse"], n),
        "Location": np.random.choice(["Downtown", "Suburban", "Rural"], n),
        "Price": np.random.randint(250000, 950000, n),
        "Bedrooms": np.random.randint(1, 5, n),
        "Bathrooms": np.random.randint(1, 4, n),
        "Square_Footage": np.random.randint(600, 3000, n),
        "Days_On_Market": np.random.randint(10, 200, n),
        "School_Rating": np.random.randint(1, 10, n),
        "Demand_Score": np.random.randint(1, 10, n),
        "Agent": np.random.choice(["Amit", "Divya", "Karan", "Pooja", "Rahul"], n),
        "Region_Code": np.random.choice(["N1", "N2", "E1", "S1", "W1"], n)
    })
    df["Price_per_SqFt"] = (df["Price"] / df["Square_Footage"]).round()
    return df


if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully")
else:
    df = load_sample_data()
    st.sidebar.info("Using sample dataset")

# Sidebar filters
st.sidebar.header("🔍 Filters")
loc_filter = st.sidebar.multiselect("Location", df["Location"].unique(), default=df["Location"].unique())
ptype_filter = st.sidebar.multiselect("Property Type", df["Property_Type"].unique(), default=df["Property_Type"].unique())

df = df[df["Location"].isin(loc_filter)]
df = df[df["Property_Type"].isin(ptype_filter)]

# -----------------------------------------------------
# KPIs
# -----------------------------------------------------
st.markdown("## 📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Avg Price", f"₹{df['Price'].mean():,.0f}")
col2.metric("Avg Price/SqFt", f"₹{df['Price_per_SqFt'].mean():,.0f}")
col3.metric("Avg Days on Market", f"{df['Days_On_Market'].mean():.0f} days")
col4.metric("Total Listings", len(df))

# -----------------------------------------------------
# TABS
# -----------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Overview",
    "🗺 Region Heatmap",
    "🏆 Agent Leaderboard",
    "🤖 ML Price Prediction"
])

# -----------------------------------------------------
# TAB 1 – OVERVIEW
# -----------------------------------------------------
with tab1:
    st.subheader("📈 Market Overview Charts")
    st.caption("Understand how prices, demand, and listings behave across different segments.")

    # Price trend
    st.markdown("### 💹 Price Trend Over Time")
    st.caption("Purpose: Helps understand macro pricing movement and volatility")
    fig1 = px.line(df, x="Date", y="Price", title="Price Trend", markers=True)
    fig1.update_traces(text=df["Price"], textposition="top center")
    st.plotly_chart(fig1, use_container_width=True)

    # Distribution
    colA, colB = st.columns(2)

    with colA:
        st.markdown("### 🏘 Price Distribution")
        st.caption("Purpose: Identify affordability range and pricing concentration")
        fig2 = px.histogram(df, x="Price", nbins=20)
        st.plotly_chart(fig2, use_container_width=True)
    
    with colB:
        st.markdown("### 📦 Property Type Breakdown")
        st.caption("Purpose: Understand supply composition across property categories")
        fig3 = px.pie(df, names="Property_Type")
        st.plotly_chart(fig3, use_container_width=True)

# -----------------------------------------------------
# TAB 2 – REGION HEATMAP
# -----------------------------------------------------
with tab2:
    st.subheader("🗺 Region Heatmap")
    st.caption("Purpose: Identify hot zones, high-price pockets, and undervalued regions.")

    region_map = df.groupby("Region_Code").agg({
        "Price": "mean",
        "Price_per_SqFt": "mean",
        "Demand_Score": "mean"
    }).reset_index()

    fig_hm = px.density_heatmap(
        region_map,
        x="Region_Code",
        y="Demand_Score",
        z="Price",
        text_auto=True,
        title="Region Heatmap: Price vs Demand",
        color_continuous_scale="RdBu"
    )
    st.plotly_chart(fig_hm, use_container_width=True)

# -----------------------------------------------------
# TAB 3 – AGENT LEADERBOARD
# -----------------------------------------------------
with tab3:
    st.subheader("🏆 Agent Leaderboard")
    st.caption("Purpose: Identify high-performing sales agents and conversion strengths.")

    agent_df = df.groupby("Agent").agg({
        "Price": "sum",
        "Days_On_Market": "mean",
        "Bedrooms": "count"
    }).rename(columns={"Bedrooms": "Listings"}).reset_index()

    agent_df["Rank"] = agent_df["Price"].rank(ascending=False).astype(int)

    agent_df = agent_df.sort_values("Rank")

    st.dataframe(agent_df)

# -----------------------------------------------------
# TAB 4 – ML PRICE PREDICTION
# -----------------------------------------------------
with tab4:
    st.subheader("🤖 ML Price Prediction")
    st.caption("Purpose: Predict optimal pricing for new listings using machine learning.")

    ml_df = df.dropna()

    target = "Price"
    features = ["Bedrooms", "Bathrooms", "Square_Footage", "Property_Type", "Location", "School_Rating", "Demand_Score"]

    X = ml_df[features]
    y = ml_df[target]

    numeric_cols = ["Bedrooms", "Bathrooms", "Square_Footage", "School_Rating", "Demand_Score"]
    cat_cols = ["Property_Type", "Location"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ], remainder="passthrough")

    X_processed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    st.write(f"📉 **Model RMSE:** {rmse:,.2f}")

    st.markdown("### 🔮 Predict Price for New Property")

    # User inputs
    colA, colB, colC = st.columns(3)

    with colA:
        p_type = st.selectbox("Property Type", df["Property_Type"].unique())
        bed = st.number_input("Bedrooms", 1, 6, 3)
        school = st.slider("School Rating", 1, 10, 7)

    with colB:
        loc = st.selectbox("Location", df["Location"].unique())
        bath = st.number_input("Bathrooms", 1, 4, 2)
        demand = st.slider("Demand Score", 1, 10, 5)

    with colC:
        sqft = st.number_input("Square Footage", 500, 4000, 1200)

    input_data = pd.DataFrame([{
        "Property_Type": p_type,
        "Location": loc,
        "Bedrooms": bed,
        "Bathrooms": bath,
        "Square_Footage": sqft,
        "School_Rating": school,
        "Demand_Score": demand
    }])

    input_processed = preprocessor.transform(input_data)

    pred_price = model.predict(input_processed)[0]

    st.success(f"🏷 Estimated Price: ₹{pred_price:,.0f}")

# END
