import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Neighborhood Lifestyle & Risk Aware Analyzer", layout="wide")

# -------------------------------
# HIDE SIDEBAR
# -------------------------------
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
.big-header {font-size: 40px; font-weight: 900;
background: linear-gradient(90deg,#FF6B6B,#FFD93D);
-webkit-background-clip:text; -webkit-text-fill-color:transparent;}
.card {background:#fff;border-radius:15px;padding:20px;margin-bottom:15px;
box-shadow:0 4px 20px rgba(0,0,0,0.08);}
.metric-card {background:#eef4ff;padding:15px;border-radius:8px;text-align:center;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# REQUIRED COLUMNS
# -------------------------------
REQUIRED_COLS = [
    "City", "Neighborhood", "Property_Type", "Price", "Latitude", "Longitude",
    "Lifestyle_Score", "Climate_Risk_Score"
]

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<div class='big-header'>Neighborhood Lifestyle & Risk Aware Analyzer</div>", unsafe_allow_html=True)

# -------------------------------
# TABS
# -------------------------------
tab1, tab2 = st.tabs(["Overview", "Application"])

# -------------------------------
# TAB 1: OVERVIEW
# -------------------------------
with tab1:
    st.markdown("### Overview")
    st.markdown("<div class='card'>This app evaluates properties based on neighborhood lifestyle amenities and climate/risk exposure, enabling investors to make holistic, risk-aware decisions.</div>", unsafe_allow_html=True)
    
    st.markdown("### Purpose")
    st.markdown("<div class='card'>• Identify top neighborhoods by lifestyle score<br>• Assess climate and risk exposure<br>• Combine lifestyle and risk insights for investment decisions<br>• Map neighborhoods with interactive scoring</div>", unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Top Lifestyle Neighborhoods</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Safest Neighborhoods</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Top Risk-Adjusted ROI Areas</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Average Lifestyle Score</div>", unsafe_allow_html=True)

# -------------------------------
# TAB 2: APPLICATION
# -------------------------------
with tab2:
    st.markdown("### Step 1: Load Dataset")
    df = None

    mode = st.radio(
        "Select Dataset Option:",
        ["Default Dataset", "Upload CSV", "Upload CSV + Column Mapping"],
        horizontal=True
    )

    # -------------------------------
    # DEFAULT DATASET
    # -------------------------------
    if mode == "Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/realestate_neighborhood.csv"
        try:
            df = pd.read_csv(URL)
        except Exception as e:
            st.error(f"Failed to load default dataset: {e}")

    # -------------------------------
    # UPLOAD CSV
    # -------------------------------
    elif mode == "Upload CSV":
        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)

    # -------------------------------
    # UPLOAD CSV + COLUMN MAPPING
    # -------------------------------
    elif mode == "Upload CSV + Column Mapping":
        file = st.file_uploader("Upload dataset", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.write("Uploaded Data Preview", raw.head())
            mapping = {}
            for col in REQUIRED_COLS:
                mapping[col] = st.selectbox(f"Map your column to: {col}", options=["-- Select --"] + list(raw.columns))
            if st.button("Apply Mapping"):
                missing = [col for col, mapped in mapping.items() if mapped == "-- Select --"]
                if missing:
                    st.error(f"Please map all required columns: {missing}")
                else:
                    df = raw.rename(columns=mapping)
                    st.success("Mapping applied successfully.")

    # -------------------------------
    # VALIDATION
    # -------------------------------
    if df is None:
        st.warning("Please select or upload a dataset to continue.")
        st.stop()

    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        st.error(f"The following required columns are missing: {missing_cols}")
        st.stop()

    df = df.dropna(subset=REQUIRED_COLS)

    # -------------------------------
    # FILTERS
    # -------------------------------
    city = st.multiselect("City", df["City"].unique())
    neighborhood = st.multiselect("Neighborhood", df["Neighborhood"].unique())
    ptype = st.multiselect("Property Type", df["Property_Type"].unique())

    filt = df.copy()
    if city: filt = filt[filt["City"].isin(city)]
    if neighborhood: filt = filt[filt["Neighborhood"].isin(neighborhood)]
    if ptype: filt = filt[filt["Property_Type"].isin(ptype)]

    st.markdown("### Data Preview")
    st.dataframe(filt.head(), use_container_width=True)

    # -------------------------------
    # KPIs
    # -------------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Top Lifestyle Neighborhoods", filt.groupby("Neighborhood")["Lifestyle_Score"].mean().sort_values(ascending=False).head(1).index[0])
    k2.metric("Safest Neighborhoods", filt.groupby("Neighborhood")["Climate_Risk_Score"].mean().sort_values().head(1).index[0])
    k3.metric("Best Risk-Adjusted Areas", (filt["Lifestyle_Score"] / (filt["Climate_Risk_Score"]+0.01)).sort_values(ascending=False).head(1).index[0])
    k4.metric("Average Lifestyle Score", f"{filt['Lifestyle_Score'].mean():.2f}")

    # -------------------------------
    # PURPOSE & QUICK TIP
    # -------------------------------
    with st.expander("Purpose & Quick Tip"):
        st.markdown("**Purpose:** Combine neighborhood lifestyle and climate risk insights to guide investment choices.")
        st.markdown("**Quick Tip:** Prioritize neighborhoods with high lifestyle scores and low climate risk.")

    # -------------------------------
    # CHARTS
    # -------------------------------
    st.markdown("### Lifestyle Score Distribution by Property Type")
    fig1 = px.histogram(filt, x="Lifestyle_Score", nbins=20, color="Property_Type", marginal="box", color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### City-wise Average Lifestyle Score")
    city_avg = filt.groupby("City")["Lifestyle_Score"].mean().reset_index()
    fig2 = px.bar(city_avg, x="City", y="Lifestyle_Score", color="City", text="Lifestyle_Score", color_discrete_sequence=px.colors.qualitative.Bold)
    fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Lifestyle & Risk Hotspot Map")
    filt["Lifestyle_Norm"] = (filt["Lifestyle_Score"] - filt["Lifestyle_Score"].min()) / (filt["Lifestyle_Score"].max() - filt["Lifestyle_Score"].min())
    filt["Risk_Norm"] = (filt["Climate_Risk_Score"] - filt["Climate_Risk_Score"].min()) / (filt["Climate_Risk_Score"].max() - filt["Climate_Risk_Score"].min())
    filt["Combined_Color"] = filt["Lifestyle_Norm"] * (1 - filt["Risk_Norm"])
    fig3 = px.scatter_mapbox(
        filt,
        lat="Latitude",
        lon="Longitude",
        size="Price",
        color="Combined_Color",
        hover_name="Neighborhood",
        hover_data=["City","Property_Type","Lifestyle_Score","Climate_Risk_Score"],
        color_continuous_scale=px.colors.diverging.RdYlGn,
        size_max=15,
        zoom=10
    )
    fig3.update_layout(mapbox_style="open-street-map", coloraxis_colorbar=dict(title="Lifestyle-Risk Score"), margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig3, use_container_width=True)

    # -------------------------------
    # NEW CHARTS
    # -------------------------------
    st.markdown("### City-Neighborhood Heatmap (Lifestyle / Risk)")
    heatmap_data = filt.groupby(["City","Neighborhood"])[["Lifestyle_Score","Climate_Risk_Score"]].mean().reset_index()
    heatmap_data["Score"] = heatmap_data["Lifestyle_Score"] / (heatmap_data["Climate_Risk_Score"]+0.01)
    fig4 = px.density_heatmap(
        heatmap_data,
        x="City",
        y="Neighborhood",
        z="Score",
        color_continuous_scale="Viridis",
        text_auto=True
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Lifestyle vs Climate Risk Scatter")
    fig5 = px.scatter(
        filt,
        x="Lifestyle_Score",
        y="Climate_Risk_Score",
        color="City",
        size="Price",
        hover_name="Neighborhood",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig5, use_container_width=True)

    # -------------------------------
    # ML Suggestions
    # -------------------------------
    st.markdown("### ML Suggested Top Neighborhoods")
    filt["Risk_Adjusted_Score"] = filt["Lifestyle_Score"] / (filt["Climate_Risk_Score"] + 0.01)
    X = filt[["Lifestyle_Score","Climate_Risk_Score","Price","Property_Type"]]
    y = filt["Risk_Adjusted_Score"]
    pipeline = Pipeline([
        ("preprocess", ColumnTransformer([("ptype", OneHotEncoder(), ["Property_Type"])], remainder="passthrough")),
        ("model", RandomForestRegressor(n_estimators=50, random_state=42))
    ])
    pipeline.fit(X, y)
    filt["Predicted_Score"] = pipeline.predict(X)
    top_suggestions = filt.sort_values("Predicted_Score", ascending=False).head(10)
    st.dataframe(top_suggestions[["City","Neighborhood","Property_Type","Predicted_Score"]], use_container_width=True)

    # -------------------------------
    # Download
    # -------------------------------
    csv = filt.to_csv(index=False)
    st.download_button("Download Filtered Dataset", csv, "neighborhood_lifestyle_risk.csv", "text/csv")
