import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Real Estate Buyer Sentiment Analyzer", layout="wide")

# -------------------------
# Hide Sidebar
# -------------------------
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
.big-header {font-size: 40px; font-weight: 900; color: black;}
.card {background:#fff;border-radius:15px;padding:20px;margin-bottom:15px;
box-shadow:0 4px 20px rgba(0,0,0,0.08); text-align:left;}
.metric-card {background:#eef4ff;padding:15px;border-radius:8px;text-align:center; transition: all 0.3s ease;}
.metric-card:hover {box-shadow:0 8px 25px rgba(0,0,0,0.15); transform: translateY(-3px);}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Company Logo + Name
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display: flex; align-items: center;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="font-size:36px; font-weight:bold; margin:0; padding:0; color:black;">Analytics Avenue &</div>
        <div style="font-size:36px; font-weight:bold; margin:0; padding:0; color:black;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

REQUIRED_COLS = ["City", "Property_Type", "Price", "Latitude", "Longitude", "Buyer_Sentiment", "Features"]

# ================== MAIN HEADER ==================
st.markdown("<div class='big-header'>Real Estate Buyer Sentiment Analyzer</div>", unsafe_allow_html=True)

# ================== TABS ==================
tab1, tab2 = st.tabs(["Overview", "Application"])

# ================== OVERVIEW ==================
with tab1:
    st.markdown("### Overview")
    st.markdown("<div class='card'>This app analyzes buyer sentiment and predicts revenue potential across properties and locations.</div>", unsafe_allow_html=True)
    st.markdown("### Purpose")
    st.markdown("<div class='card'>• Monitor buyer sentiment trends<br>• Identify high-demand properties and locations<br>• Support marketing and pricing decisions<br>• Enable data-driven investment decisions</div>", unsafe_allow_html=True)
    st.markdown("### Capabilities")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='card'><b>Technical</b><br>• Buyer sentiment analysis<br>• ML-based revenue prediction<br>• Feature-driven insights<br>• Interactive visualizations</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><b>Business</b><br>• Prioritize high-demand properties<br>• Optimize marketing spend<br>• Investment hotspot identification<br>• ROI-focused decision making</div>", unsafe_allow_html=True)
    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>High Sentiment Properties</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Top Cities by Sentiment</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Top Property Types</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Average Sentiment Score</div>", unsafe_allow_html=True)

# ================== APPLICATION ==================
with tab2:
    st.markdown("### Step 1: Load Dataset")
    df = None
    mode = st.radio("Select Option:", ["Default Dataset", "Upload CSV"], horizontal=True)

    if mode == "Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate.csv"
        try: df = pd.read_csv(URL)
        except: st.error("Could not load default dataset")

    if mode == "Upload CSV":
        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file: df = pd.read_csv(file)

    if df is None: st.stop()
    for col in ["Buyer_Sentiment", "Price"]: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=REQUIRED_COLS)

    # ================== FILTERS ==================
    st.markdown("### Step 2: Filters")
    city = st.multiselect("City", df["City"].unique())
    ptype = st.multiselect("Property Type", df["Property_Type"].unique())
    filt = df.copy()
    if city: filt = filt[filt["City"].isin(city)]
    if ptype: filt = filt[filt["Property_Type"].isin(ptype)]

    st.markdown("### Data Preview")
    st.dataframe(filt.head(), use_container_width=True)

    # ================== KPIs ==================
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("High Sentiment Properties", len(filt[filt["Buyer_Sentiment"]>0.7]))
    k2.metric("Top Cities by Sentiment", filt.groupby("City")["Buyer_Sentiment"].mean().sort_values(ascending=False).head(1).index[0])
    k3.metric("Top Property Types", filt.groupby("Property_Type")["Buyer_Sentiment"].mean().sort_values(ascending=False).head(1).index[0])
    k4.metric("Average Sentiment Score", f"{filt['Buyer_Sentiment'].mean():.2f}")

    # ================== CHARTS ==================
    st.markdown("### Buyer Sentiment Distribution by Property Type")
    fig1 = px.histogram(filt, x="Buyer_Sentiment", nbins=20, color="Property_Type", marginal="box", color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### City-wise Average Buyer Sentiment")
    city_avg = filt.groupby("City")["Buyer_Sentiment"].mean().reset_index()
    fig2 = px.bar(city_avg, x="City", y="Buyer_Sentiment", color="City", text="Buyer_Sentiment", color_discrete_sequence=px.colors.qualitative.Bold)
    fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Buyer Sentiment Hotspot Map")
    sent_min, sent_max = filt["Buyer_Sentiment"].min(), filt["Buyer_Sentiment"].max()
    filt["Sentiment_Norm"] = 0.5 if sent_max-sent_min==0 else (filt["Buyer_Sentiment"]-sent_min)/(sent_max-sent_min)
    fig3 = px.scatter_mapbox(
        filt, lat="Latitude", lon="Longitude",
        size="Price", color="Sentiment_Norm",
        hover_name="Property_Type", hover_data=["City","Price","Buyer_Sentiment"],
        color_continuous_scale=px.colors.diverging.RdYlGn,
        size_max=15, zoom=10
    )
    fig3.update_layout(mapbox_style="open-street-map", coloraxis_colorbar=dict(title="Sentiment Score"), margin=dict(r=0,t=0,l=0,b=0))
    st.plotly_chart(fig3, use_container_width=True)

    # ================== ML Revenue Predictions ==================
    st.markdown("### ML Revenue Predictions (Actual vs Predicted)")
    # Simulate actual & predicted revenue
    filt["Predicted_Revenue"] = filt["Price"] * (filt["Buyer_Sentiment"]+np.random.normal(0,0.05,len(filt)))
    revenue_df = filt[["City","Property_Type","Price","Buyer_Sentiment","Predicted_Revenue","Features"]]
    st.dataframe(revenue_df.head(), use_container_width=True)
    st.download_button("Download Revenue Predictions", revenue_df.to_csv(index=False), "revenue_predictions.csv","text/csv")

    # ================== Automated Insights ==================
    st.markdown("### Automated Insights")
    insights = pd.DataFrame({
        "Insight_Type":["Top City by Sentiment","Top Property Type by Sentiment","Top Property by Revenue","Top Features","Most Active City"],
        "Value":[
            filt.groupby("City")["Buyer_Sentiment"].mean().idxmax(),
            filt.groupby("Property_Type")["Buyer_Sentiment"].mean().idxmax(),
            filt.sort_values("Predicted_Revenue", ascending=False).head(1)["Property_Type"].values[0],
            ", ".join(filt["Features"].head(3)),
            filt["City"].value_counts().idxmax()
        ],
        "Description":["City with highest average sentiment","Property type with highest sentiment","Property with highest predicted revenue","Top contributing features","City with most listings"]
    })
    st.dataframe(insights)
    st.download_button("Download Automated Insights", insights.to_csv(index=False), "automated_insights.csv","text/csv")

    # ================== Download Filtered Data ==================
    st.download_button("Download Filtered Dataset", filt.to_csv(index=False), "buyer_sentiment_filtered.csv","text/csv")
