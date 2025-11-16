import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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
    # Normalize scores
    if filt["Lifestyle_Score"].max() - filt["Lifestyle_Score"].min() > 0:
        filt["Lifestyle_Norm"] = (filt["Lifestyle_Score"] - filt["Lifestyle_Score"].min()) / (filt["Lifestyle_Score"].max() - filt["Lifestyle_Score"].min())
    else:
        filt["Lifestyle_Norm"] = 0.5
    if filt["Climate_Risk_Score"].max() - filt["Climate_Risk_Score"].min() > 0:
        filt["Risk_Norm"] = (filt["Climate_Risk_Score"] - filt["Climate_Risk_Score"].min()) / (filt["Climate_Risk_Score"].max() - filt["Climate_Risk_Score"].min())
    else:
        filt["Risk_Norm"] = 0.5

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

    # --- Previous code above stays the same ---

    # -------------------------------
    # ADDITIONAL CHARTS
    # -------------------------------
    st.markdown("### Neighborhood Lifestyle vs. Climate Risk")
    fig4 = px.scatter(
        filt,
        x="Climate_Risk_Score",
        y="Lifestyle_Score",
        color="City",
        size="Price",
        hover_name="Neighborhood",
        hover_data=["Property_Type", "Price"],
        color_discrete_sequence=px.colors.qualitative.Set2,
        size_max=15
    )
    fig4.update_layout(xaxis_title="Climate Risk Score", yaxis_title="Lifestyle Score")
    st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown("### Top 10 Neighborhoods by Risk-Adjusted Score")
    filt["Risk_Adjusted_Score"] = filt["Lifestyle_Score"] / (filt["Climate_Risk_Score"] + 0.01)
    top10 = filt.groupby("Neighborhood")["Risk_Adjusted_Score"].mean().sort_values(ascending=False).head(10).reset_index()
    fig5 = px.bar(top10, x="Neighborhood", y="Risk_Adjusted_Score", text="Risk_Adjusted_Score", color="Risk_Adjusted_Score", color_continuous_scale=px.colors.sequential.Teal)
    fig5.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig5, use_container_width=True)
    
    st.markdown("### Property Type vs. Average Lifestyle Score")
    ptype_avg = filt.groupby("Property_Type")["Lifestyle_Score"].mean().reset_index()
    fig6 = px.pie(ptype_avg, names="Property_Type", values="Lifestyle_Score", color="Property_Type", color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig6, use_container_width=True)


    # -------------------------------
    # DOWNLOAD
    # -------------------------------
    csv = filt.to_csv(index=False)
    st.download_button("Download Filtered Dataset", csv, "neighborhood_lifestyle_risk.csv", "text/csv")
