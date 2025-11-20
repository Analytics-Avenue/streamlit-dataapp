import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="Order-to-Delivery Analytics", layout="wide")
st.markdown("""<style>[data-testid="stSidebarNav"]{display:none;}</style>""", unsafe_allow_html=True)

# ------------------------------
# Card & KPI Glow CSS
# ------------------------------
st.markdown("""
<style>
.card {
    padding: 20px;
    border-radius: 12px;
    background: white;
    border: 1px solid #d9d9d9;
    transition: 0.3s;
    box-shadow: 0px 2px 4px rgba(0,0,0,0.1);
    text-align:left;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0px 4px 18px rgba(6,75,134,0.35);
    border-color: #064b86;
}
.kpi {
    padding: 25px;
    border-radius: 14px;
    background: white;
    border: 1px solid #ccc;
    font-size: 22px;
    font-weight: bold;
    color: #064b86;
    text-align: center;
    transition: 0.3s;
}
.kpi:hover {
    transform: translateY(-4px);
    box-shadow: 0px 4px 15px rgba(6,75,134,0.30);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Header
# ------------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display:flex; align-items:center;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:34px; font-weight:bold;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:34px; font-weight:bold;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# CSV Download Helper
# ------------------------------
def download_df(df, filename):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode())
    b.seek(0)
    st.download_button("Download CSV", b, file_name=filename, mime="text/csv")

# ------------------------------
# Tabs
# ------------------------------
tabs = st.tabs(["Overview", "Application"])

# ------------------------------
# Overview Tab
# ------------------------------
with tabs[0]:
    st.markdown("## Overview")
    
    st.markdown("<div class='card'>This application analyzes order-to-delivery lead times, predicts delivery delays, and provides actionable insights to optimize production scheduling and dispatch.</div>", unsafe_allow_html=True)

    # Grid: Capabilities | Business Impact
    c1, c2 = st.columns(2)
    c1.markdown("### Capabilities")
    c1.markdown("""
    <div class='card'>
    • Predictive lead-time analytics<br>
    • End-to-end production visibility dashboards<br>
    • Dispatch simulation & optimization<br>
    • Order delay root-cause analysis<br>
    • Multi-filter exploration (Product, Machine, Dates)
    </div>
    """, unsafe_allow_html=True)

    c2.markdown("### Business Impact")
    c2.markdown("""
    <div class='card'>
    • Faster delivery<br>
    • Higher customer satisfaction<br>
    • Reduced production & dispatch delays<br>
    • Optimized resource allocation<br>
    • Minimized late deliveries
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards
    st.markdown("## KPIs")
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.markdown("<div class='kpi'>Orders Tracked</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Avg Lead Time</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Avg Scheduling Delay</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Avg Production Time</div>", unsafe_allow_html=True)
    k5.markdown("<div class='kpi'>Avg Dispatch Delay</div>", unsafe_allow_html=True)

    # Who should use
    st.markdown("### Who Should Use This App?")
    st.markdown("<div class='card'>• Plant Managers<br>• Production Planners<br>• Dispatch Coordinators<br>• Operations Analysts</div>", unsafe_allow_html=True)

    st.markdown("### How to Use")
    st.markdown("<div class='card'>• Load dataset (default GitHub or CSV)<br>• Filter by product, machine, or date<br>• View KPIs and charts<br>• Use ML predictions and automated insights to optimize scheduling</div>", unsafe_allow_html=True)

# ------------------------------
# Application Tab
# ------------------------------
with tabs[1]:
    st.header("Application")
    st.markdown("### Choose Dataset Option")

    mode = st.radio("Select:", [
        "Default dataset (GitHub URL)",
        "Upload CSV",
        "Upload CSV + Manual Column Mapping"
    ])

    df = None
    REQUIRED_COLS = [
        "Order_ID","Order_Date","Product_Type","Machine_Type","Scheduling_Delay_Hrs",
        "Production_Time_Hrs","Machine_Delay_Hrs","Dispatch_Delay_Hrs",
        "Total_Lead_Time_Hrs","Estimated_Delivery_Date","Predicted_Lead_Time_Hrs","Customer_Satisfaction_Score"
    ]

    # ---------------------- DEFAULT DATASET ----------------------
    if mode=="Default dataset (GitHub URL)":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/manufacturing/order_to_delivery_dataset.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            st.success("Default dataset loaded successfully.")
            st.dataframe(df.head())
        except Exception as e:
            st.error("Failed to load default dataset. Error: " + str(e))
            st.stop()

    # ---------------------- UPLOAD CSV ----------------------
    elif mode=="Upload CSV":
        file = st.file_uploader("Upload CSV", type=["csv"], key="upload_simple")
        if file:
            df = pd.read_csv(file)
            st.success("File uploaded.")
            st.dataframe(df.head())
        else:
            st.stop()

    # ---------------------- UPLOAD + MAP ----------------------
    else:
        file = st.file_uploader("Upload CSV for mapping", type=["csv"], key="upload_map")
        if file:
            raw = pd.read_csv(file)
            st.write("Preview:")
            st.dataframe(raw.head())
            mapping = {}
            for col in REQUIRED_COLS:
                mapping[col] = st.selectbox(f"Map → {col}", ["-- Select --"] + list(raw.columns), key=f"map_{col}")
            if st.button("Apply Mapping", key="apply_map"):
                miss = [m for m in mapping if mapping[m]=="-- Select --"]
                if miss:
                    st.error("Map all columns: " + ", ".join(miss))
                else:
                    df = raw.rename(columns={mapping[k]:k for k in mapping})
                    st.success("Mapping applied.")
                    st.dataframe(df.head())
        else:
            st.stop()

    if df is None: st.stop()

    # Convert dates
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors='coerce')
    if "Estimated_Delivery_Date" in df.columns:
        df["Estimated_Delivery_Date"] = pd.to_datetime(df["Estimated_Delivery_Date"], errors='coerce')

    # Filters
    st.markdown("### Filters")
    m1,m2 = st.columns(2)
    products = df["Product_Type"].unique() if "Product_Type" in df.columns else []
    machines = df["Machine_Type"].unique() if "Machine_Type" in df.columns else []
    sel_prod = m1.multiselect("Product Type", products, default=products)
    sel_mach = m2.multiselect("Machine Type", machines, default=machines)
    df_f = df.copy()
    if sel_prod: df_f = df_f[df_f["Product_Type"].isin(sel_prod)]
    if sel_mach: df_f = df_f[df_f["Machine_Type"].isin(sel_mach)]
    st.dataframe(df_f.head(5), use_container_width=True)
    download_df(df_f,"filtered_orders.csv")

    # KPIs Dynamic
    st.markdown("### Key Metrics")
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Orders", df_f.shape[0])
    k2.metric("Avg Lead Time", f"{df_f['Total_Lead_Time_Hrs'].mean():.2f}" if "Total_Lead_Time_Hrs" in df_f else "N/A")
    k3.metric("Avg Scheduling Delay", f"{df_f['Scheduling_Delay_Hrs'].mean():.2f}" if "Scheduling_Delay_Hrs" in df_f else "N/A")
    k4.metric("Avg Production Time", f"{df_f['Production_Time_Hrs'].mean():.2f}" if "Production_Time_Hrs" in df_f else "N/A")
    k5.metric("Avg Dispatch Delay", f"{df_f['Dispatch_Delay_Hrs'].mean():.2f}" if "Dispatch_Delay_Hrs" in df_f else "N/A")

    # ------------------------------
    # Charts
    # ------------------------------
    st.markdown("### Charts")
    if "Order_Date" in df_f.columns and "Total_Lead_Time_Hrs" in df_f.columns:
        fig = px.line(df_f.sort_values("Order_Date"), x="Order_Date", y="Total_Lead_Time_Hrs", color="Product_Type", title="Lead Time Trend")
        st.plotly_chart(fig, use_container_width=True)
    if "Machine_Type" in df_f.columns and "Total_Lead_Time_Hrs" in df_f.columns:
        fig2 = px.box(df_f, x="Machine_Type", y="Total_Lead_Time_Hrs", title="Lead Time by Machine Type")
        st.plotly_chart(fig2, use_container_width=True)

    # ------------------------------
    # ML: Lead Time Prediction
    # ------------------------------
    st.markdown("### Machine Learning: Lead Time Prediction")
    features = ["Scheduling_Delay_Hrs","Production_Time_Hrs","Machine_Delay_Hrs","Dispatch_Delay_Hrs"]
    features = [f for f in features if f in df_f.columns]
    if len(df_f)>=30 and len(features)>=2:
        X = df_f[features].fillna(0)
        y = df_f["Total_Lead_Time_Hrs"].fillna(0)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        Xtr,Xte,ytr,yte = train_test_split(Xs,y,test_size=0.2,random_state=42)
        model = RandomForestRegressor(n_estimators=150, random_state=42)
        model.fit(Xtr,ytr)
        preds = model.predict(Xte)
        results = pd.DataFrame({**{"Actual":yte.values,"Predicted":preds}, **{f:Xte[:,i] for i,f in enumerate(features)}})
        st.dataframe(results.head(10))
        download_df(results,"ml_leadtime_predictions.csv")
    else:
        st.info("Not enough data or features for ML model.")

    # ------------------------------
    # Automated Insights
    # ------------------------------
    st.markdown("### Automated Insights")
    insights=[]
    if "Product_Type" in df_f.columns and "Total_Lead_Time_Hrs" in df_f.columns:
        avg_lead = df_f.groupby("Product_Type")["Total_Lead_Time_Hrs"].mean().sort_values(ascending=False)
        insights.append({"Insight":"Product with Highest Avg Lead Time","Value":f"{avg_lead.index[0]} ({avg_lead.iloc[0]:.2f} hrs)"})
    if "Machine_Type" in df_f.columns and "Total_Lead_Time_Hrs" in df_f.columns:
        avg_mach = df_f.groupby("Machine_Type")["Total_Lead_Time_Hrs"].mean().sort_values(ascending=False)
        insights.append({"Insight":"Machine causing highest lead time","Value":f"{avg_mach.index[0]} ({avg_mach.iloc[0]:.2f} hrs)"})
    if insights:
        st.dataframe(pd.DataFrame(insights))
        download_df(pd.DataFrame(insights),"automated_insights.csv")
    else:
        st.info("No insights could be generated for this filter.")
