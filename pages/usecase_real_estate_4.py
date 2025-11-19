import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from io import BytesIO

st.set_page_config(page_title="Real Estate Agent & Market Insights", layout="wide")

# --------------------------
# Hide default sidebar
# --------------------------
hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# -------------------------
# Company Logo + Name
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display: flex; align-items: center;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0; padding:0;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0; padding:0;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------
# Required Columns
# --------------------------
REQUIRED_COLS = [
    "City",
    "Property_Type",
    "Agent_Name",
    "Price",
    "Lead_Score",
    "Conversion_Probability",
    "Days_On_Market"
]

# --------------------------
# CSS for cards & hover effects
# --------------------------
st.markdown("""
<style>
.big-header {font-size:40px; font-weight:900; color:black;}
.card, .metric-card, .hover-card {background:#fff;border-radius:15px;padding:20px;margin-bottom:15px;box-shadow:0 4px 20px rgba(0,0,0,0.08);transition: all 0.25s ease;text-align:left;}
.card:hover, .metric-card:hover, .hover-card:hover {box-shadow:0 0 18px rgba(0,0,0,0.4); transform:scale(1.03);}
.metric-card {font-weight:600;}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Main header
# --------------------------
st.markdown("<div class='big-header'>Real Estate Agent & Market Insights</div>", unsafe_allow_html=True)

# ==========================================================
# Tabs
# ==========================================================
tab1, tab2 = st.tabs(["Overview", "Application"])

# ==========================================================
# TAB 1 - OVERVIEW
# ==========================================================
with tab1:
    st.markdown("### Overview")
    st.markdown("<div class='hover-card'>This platform provides insights into agent performance, lead conversion, and market segmentation. It enables analytics-driven decision making for real estate teams and investors.</div>", unsafe_allow_html=True)

    st.markdown("### Purpose")
    st.markdown("<div class='hover-card'>• Evaluate agent performance<br>• Track lead-to-sale conversion<br>• Understand market segments and pricing trends<br>• Identify high-demand property areas</div>", unsafe_allow_html=True)

    st.markdown("### Capabilities & Business Impact")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='hover-card'><b>Analytics</b><br>• Agent performance dashboard<br>• Lead conversion metrics<br>• Market segmentation with clustering<br>• Pricing trends visualization</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='hover-card'><b>Business</b><br>• Incentive planning for agents<br>• Target high-conversion segments<br>• Optimize marketing spend<br>• Strategic portfolio allocation</div>", unsafe_allow_html=True)

    st.markdown("### KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Avg Lead Score</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Avg Conversion %</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Avg Days on Market</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Top Performing Agent</div>", unsafe_allow_html=True)

    st.markdown("### Use of App")
    st.markdown("<div class='hover-card'>• Monitor agent & property performance<br>• Evaluate lead conversion efficiency<br>• Segment markets for strategic decisions<br>• Forecast potential revenue streams</div>", unsafe_allow_html=True)

    st.markdown("### Who Should Use")
    st.markdown("<div class='hover-card'>• Real estate agents & managers<br>• Investors & portfolio managers<br>• Market analysts<br>• Property marketing teams</div>", unsafe_allow_html=True)

# ==========================================================
# TAB 2 - APPLICATION
# ==========================================================
with tab2:
    st.markdown("### Step 1: Load Dataset")
    df = None

    mode = st.radio("Select Option:", ["Default Dataset", "Upload CSV", "Upload CSV + Column Mapping"], horizontal=True)

    if mode=="Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        try:
            df = pd.read_csv(URL)
            st.success("Default dataset loaded successfully.")
            st.dataframe(df.head(), use_container_width=True)
        except:
            st.error("Could not load default dataset.")

    if mode=="Upload CSV":
        st.markdown("#### Download Sample CSV")
        sample_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
        sample_df = pd.read_csv(sample_url).head(5)
        st.download_button("Download Sample CSV", sample_df.to_csv(index=False), "sample.csv","text/csv")
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file: df=pd.read_csv(file)

    if mode=="Upload CSV + Column Mapping":
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.write(raw.head())
            mapping={}
            for col in REQUIRED_COLS:
                mapping[col]=st.selectbox(f"Map column to {col}", options=["-- Select --"]+list(raw.columns))
            if st.button("Apply Mapping"):
                missing=[c for c,v in mapping.items() if v=="-- Select --"]
                if missing: st.error(f"Map all columns: {missing}")
                else:
                    df=raw.rename(columns=mapping)
                    st.success("Mapping applied")
                    st.dataframe(df.head())

    if df is None: st.stop()
    if not all(col in df.columns for col in REQUIRED_COLS):
        st.error("Missing required columns"); st.stop()

    df=df.dropna(subset=REQUIRED_COLS)

    # --------------------------
    # Filters
    # --------------------------
    st.markdown("### Step 2: Filters")
    f1,f2,f3=st.columns(3)
    with f1: city_filter=st.multiselect("City", df["City"].unique())
    with f2: ptype_filter=st.multiselect("Property Type", df["Property_Type"].unique())
    with f3: agent_filter=st.multiselect("Agent Name", df["Agent_Name"].unique())

    filt=df.copy()
    if city_filter: filt=filt[filt["City"].isin(city_filter)]
    if ptype_filter: filt=filt[filt["Property_Type"].isin(ptype_filter)]
    if agent_filter: filt=filt[filt["Agent_Name"].isin(agent_filter)]

    st.markdown("### Filtered Data Preview")
    st.dataframe(filt.head(), use_container_width=True)

    # --------------------------
    # KPIs
    # --------------------------
    st.markdown("### Key Metrics")
    k1,k2,k3,k4=st.columns(4)
    k1.metric("Avg Lead Score", f"{filt['Lead_Score'].mean():.2f}")
    k2.metric("Avg Conversion %", f"{filt['Conversion_Probability'].mean()*100:.2f}%")
    k3.metric("Avg Days on Market", f"{filt['Days_On_Market'].mean():.1f}")
    top_agent=filt.groupby("Agent_Name")["Conversion_Probability"].mean().idxmax()
    k4.metric("Top Performing Agent", top_agent)

    # --------------------------
    # Charts
    # --------------------------
    st.markdown("### Agent-wise Conversion")
    agent_conv=filt.groupby("Agent_Name")["Conversion_Probability"].mean().reset_index()
    fig1=px.bar(agent_conv, x="Agent_Name", y="Conversion_Probability", text="Conversion_Probability",
                color="Conversion_Probability", color_continuous_scale=px.colors.sequential.Viridis)
    fig1.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### City-wise Average Price")
    city_price=filt.groupby("City")["Price"].mean().reset_index()
    fig2=px.bar(city_price,x="City",y="Price",color="City",text="Price",color_discrete_sequence=px.colors.qualitative.Bold)
    fig2.update_traces(texttemplate="₹ %{text:,.0f}",textposition="outside")
    st.plotly_chart(fig2,use_container_width=True)

    st.markdown("### Market Segmentation (KMeans)")
    seg_features=filt[["Price","Days_On_Market","Conversion_Probability"]]
    scaler=StandardScaler()
    seg_scaled=scaler.fit_transform(seg_features)
    kmeans=KMeans(n_clusters=4,random_state=42)
    filt["Segment"]=kmeans.fit_predict(seg_scaled)
    fig3=px.scatter(filt,x="Price",y="Conversion_Probability",color="Segment",
                    hover_data=["Agent_Name","City","Property_Type"],
                    color_discrete_sequence=px.colors.qualitative.D3)
    st.plotly_chart(fig3,use_container_width=True)

    # --------------------------
    # ML Revenue Prediction
    # --------------------------
    st.markdown("### Step 3: ML Revenue Predictions")
    X=filt[["City","Property_Type","Lead_Score","Days_On_Market"]]
    y=filt["Price"]

    transformer=ColumnTransformer([
        ("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),["City","Property_Type"]),
        ("num","passthrough",["Lead_Score","Days_On_Market"])
    ])
    X_trans=transformer.fit_transform(X)
    X_train,X_test,y_train,y_test=train_test_split(X_trans,y,test_size=0.2,random_state=42)
    model=RandomForestRegressor(n_estimators=300,random_state=42)
    model.fit(X_train,y_train)

    y_pred=model.predict(X_trans)
    results=X.copy()
    results["Actual_Revenue"]=y
    results["Predicted_Revenue"]=y_pred
    st.dataframe(results,use_container_width=True)

    buf=BytesIO()
    results.to_csv(buf,index=False)
    st.download_button("Download ML Revenue Predictions", buf.getvalue(),"ml_revenue_predictions.csv","text/csv")

    # --------------------------
    # Automated Insights
    # --------------------------
    st.markdown("### Automated Insights")
    insights=filt.groupby(["City","Property_Type"]).agg({"Price":["mean","max","min"]}).reset_index()
    insights.columns=["City","Property_Type","Avg_Revenue","Max_Revenue","Min_Revenue"]
    st.dataframe(insights,use_container_width=True)
    buf2=BytesIO()
    insights.to_csv(buf2,index=False)
    st.download_button("Download Automated Insights",buf2.getvalue(),"automated_insights.csv","text/csv")
