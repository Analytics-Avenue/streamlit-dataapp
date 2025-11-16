import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ---------------------------- Page Config ----------------------------
st.set_page_config(page_title="Tenant Risk & Market Trend Analyzer", layout="wide")

# ---------------------------- CSS ----------------------------
hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
.big-header {
    font-size: 40px; font-weight: 900;
    background: linear-gradient(90deg,#1E90FF,#00CED1);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.card {background:#fff;border-radius:15px;padding:20px;margin-bottom:15px;
box-shadow:0 4px 20px rgba(0,0,0,0.08);}
.metric-card {background:#eef4ff;padding:15px;border-radius:8px;text-align:center;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# ---------------------------- Header ----------------------------
st.markdown("<div class='big-header'>Tenant Risk & Market Trend Analyzer</div>", unsafe_allow_html=True)

# ---------------------------- Required Columns ----------------------------
REQUIRED_COLS = [
    "City", "Property_Type", "Price", "Area_sqft",
    "Age_Years", "Conversion_Probability", "Latitude", "Longitude"
]

# ==========================================================
# TABS
# ==========================================================
tab1, tab2 = st.tabs(["Overview", "Application"])

# ==========================================================
# TAB 1 - OVERVIEW
# ==========================================================
with tab1:
    st.markdown("### Overview")
    st.markdown("""
    <div class='card'>
    This app identifies occupancy risk and emerging market trends. Investors and property managers can proactively manage portfolios
    by analyzing property age, location, and conversion probability. The system highlights properties with high vacancy risk and areas
    showing growth potential.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Purpose")
    st.markdown("""
    <div class='card'>
    • Evaluate properties for potential rental income risk<br>
    • Identify neighborhoods with growing demand<br>
    • Compare property types and ages for occupancy trends<br>
    • Assist investors in maximizing returns while minimizing risk
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Capabilities")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='card'>
        <b>Technical</b><br>
        • Occupancy risk scoring<br>
        • Interactive charts and maps<br>
        • City and property segmentation<br>
        • ML prediction for new properties
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='card'>
        <b>Business</b><br>
        • Investment risk management<br>
        • Market opportunity identification<br>
        • Data-driven portfolio planning<br>
        • Optimized property selection
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>High Risk Properties</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Low Risk Properties</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>City with Highest Growth</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Average Occupancy Probability</div>", unsafe_allow_html=True)

# ==========================================================
# TAB 2 - APPLICATION
# ==========================================================
with tab2:
    st.markdown("### Step 1: Load Dataset")
    df = None

    mode = st.radio(
        "Select Option:",
        ["Default Dataset", "Upload CSV", "Upload CSV + Column Mapping"],
        horizontal=True
    )

    if mode == "Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        try:
            df = pd.read_csv(URL)
            st.success("Default dataset loaded successfully.")
        except Exception as e:
            st.error(f"Could not load dataset: {e}")

    elif mode == "Upload CSV":
        st.markdown("#### Download Sample CSV for Reference")
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        try:
            # Load default dataset
            sample_df = pd.read_csv(URL).head(5)  # Take first 5 rows
            sample_csv = sample_df.to_csv(index=False)
            st.download_button("Download Sample CSV", sample_csv, "sample_dataset.csv", "text/csv")
        except Exception as e:
            st.info(f"Sample CSV unavailable: {e}")
    
        # Upload actual CSV
        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)
    
    elif mode == "Upload CSV + Column Mapping":
        file = st.file_uploader("Upload dataset", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.write("Uploaded Data", raw.head())
            st.markdown("### Map Required Columns Only")
            mapping = {}
            for col in REQUIRED_COLS:
                mapping[col] = st.selectbox(
                    f"Map your column to: {col}",
                    options=["-- Select --"] + list(raw.columns)
                )
            if st.button("Apply Mapping"):
                missing = [col for col, mapped in mapping.items() if mapped == "-- Select --"]
                if missing:
                    st.error(f"Please map all required columns: {missing}")
                else:
                    df = raw.rename(columns=mapping)
                    st.success("Mapping applied successfully.")
                    st.dataframe(df.head())

    if df is None:
        st.stop()

    if not all(col in df.columns for col in REQUIRED_COLS):
        st.error("Dataset missing required columns.")
        st.write("Required columns:", REQUIRED_COLS)
        st.stop()

    df = df.dropna()

    # ---------------- Filters ----------------
    st.markdown("### Step 2: Filters")
    f1, f2, f3 = st.columns(3)
    with f1:
        city = st.multiselect("City", df["City"].unique())
    with f2:
        ptype = st.multiselect("Property Type", df["Property_Type"].unique())
    with f3:
        age_filter = st.slider("Property Age (Years)", int(df["Age_Years"].min()), int(df["Age_Years"].max()), (int(df["Age_Years"].min()), int(df["Age_Years"].max())))

    filt = df.copy()
    if city:
        filt = filt[filt["City"].isin(city)]
    if ptype:
        filt = filt[filt["Property_Type"].isin(ptype)]
    filt = filt[(filt["Age_Years"] >= age_filter[0]) & (filt["Age_Years"] <= age_filter[1])]

    st.markdown("### Data Preview")
    st.dataframe(filt.head(), use_container_width=True)

    # ---------------- KPIs ----------------
    st.markdown("### Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("High Risk Properties", len(filt[filt["Conversion_Probability"] < 0.4]))
    k2.metric("Low Risk Properties", len(filt[filt["Conversion_Probability"] > 0.7]))
    k3.metric("City with Highest Growth", filt.groupby("City")["Conversion_Probability"].mean().sort_values(ascending=False).head(1).index[0])
    k4.metric("Average Occupancy Probability", f"{filt['Conversion_Probability'].mean():.2f}")

    # ---------------- Charts with Purpose & Quick Tip ----------------
    # Chart 1 - Occupancy Risk by Property Type
    st.markdown("### Occupancy Risk by Property Type")
    with st.expander("Purpose & Quick Tip"):
        st.markdown("""
        **Purpose:** Compare average occupancy probability across property types to spot high-risk categories.<br>
        **Quick Tip:** Focus on property types with low occupancy probability for risk mitigation.
        """, unsafe_allow_html=True)
    ptype_risk = filt.groupby("Property_Type")["Conversion_Probability"].mean().reset_index()
    fig1 = px.bar(ptype_risk, x="Property_Type", y="Conversion_Probability",
                  color="Conversion_Probability", color_continuous_scale=px.colors.sequential.Plasma,
                  text="Conversion_Probability")
    fig1.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2 - Age vs Price per Sqft
    st.markdown("### Age vs Price per Sqft")
    with st.expander("Purpose & Quick Tip"):
        st.markdown("""
        **Purpose:** Examine how property age affects price per square foot and occupancy.<br>
        **Quick Tip:** Newer properties usually have higher occupancy and better ROI.
        """, unsafe_allow_html=True)
    filt["Price_per_sqft"] = filt["Price"] / filt["Area_sqft"]
    fig2 = px.scatter(filt, x="Age_Years", y="Price_per_sqft",
                      color="Conversion_Probability", size="Price_per_sqft",
                      hover_data=["City", "Property_Type", "Price"],
                      color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3 - City-wise Occupancy Trend
    st.markdown("### City-wise Occupancy Trend")
    with st.expander("Purpose & Quick Tip"):
        st.markdown("""
        **Purpose:** Identify cities with high or low occupancy probability.<br>
        **Quick Tip:** Target investment in cities with high occupancy probability.
        """, unsafe_allow_html=True)
    city_trend = filt.groupby("City")["Conversion_Probability"].mean().reset_index()
    fig3 = px.bar(city_trend, x="City", y="Conversion_Probability",
                  color="City", text="Conversion_Probability",
                  color_discrete_sequence=px.colors.qualitative.Bold)
    fig3.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig3, use_container_width=True)

    # Chart 4 - High Risk Properties Map
    st.markdown("### High Risk Properties Map")
    with st.expander("Purpose & Quick Tip"):
        st.markdown("""
        **Purpose:** Visualize high-risk properties on map to identify potential hotspots.<br>
        **Quick Tip:** Hover over points to see city, price, age, and occupancy probability.
        """, unsafe_allow_html=True)
    filt["Conversion_Normalized"] = (filt["Conversion_Probability"] - filt["Conversion_Probability"].min()) / (
        filt["Conversion_Probability"].max() - filt["Conversion_Probability"].min()
    )
    fig4 = px.scatter_mapbox(filt, lat="Latitude", lon="Longitude",
                             size="Price_per_sqft", color="Conversion_Normalized",
                             hover_name="Property_Type",
                             hover_data=["City", "Price", "Age_Years", "Conversion_Probability"],
                             color_continuous_scale=px.colors.sequential.Viridis,
                             size_max=20, zoom=10)
    fig4.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig4, use_container_width=True)

    # ---------------- ML Prediction ----------------
    st.markdown("### Step 3: Predict Occupancy Risk for New Property")
    c1, c2, c3 = st.columns(3)
    with c1:
        p_city = st.selectbox("City", df["City"].unique())
    with c2:
        p_ptype = st.selectbox("Property Type", df["Property_Type"].unique())
    with c3:
        p_age = st.number_input("Property Age (Years)", min_value=0, max_value=100, value=10)
    p_area = st.number_input("Area (sqft)", min_value=100, max_value=10000, value=1200)

    # Model
    X = df[["City", "Property_Type", "Age_Years", "Area_sqft"]]
    y = df["Conversion_Probability"]
    transformer = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["City", "Property_Type"]),
        ("num", StandardScaler(), ["Age_Years", "Area_sqft"])
    ])
    X_trans = transformer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    new_input = transformer.transform(pd.DataFrame([[p_city, p_ptype, p_age, p_area]],
                                                   columns=["City","Property_Type","Age_Years","Area_sqft"]))
    pred_risk = model.predict(new_input)[0]
    st.metric("Predicted Occupancy Probability", f"{pred_risk:.2f}")
