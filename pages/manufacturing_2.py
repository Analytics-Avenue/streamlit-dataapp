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

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(page_title="Order-to-Delivery Analytics", layout="wide")

# Hide sidebar
st.markdown("""<style>[data-testid="stSidebarNav"]{display:none;}</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Card Hover Glow CSS
# ---------------------------------------------------------
st.markdown("""
<style>
.card {
    padding: 20px;
    border-radius: 12px;
    background: white;
    border: 1px solid #d9d9d9;
    transition: 0.3s;
    box-shadow: 0px 2px 4px rgba(0,0,0,0.1);
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0px 4px 18px rgba(6,75,134,0.35);
    border-color: #064b86;
    text-align: left;
}
.kpi {
    padding: 30px;
    border-radius: 14px;
    background: white;
    border: 1px solid #ccc;
    font-size: 24px;
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

# ---------------------------------------------------------
# Header + Logo
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Helper for CSV download
# ---------------------------------------------------------
def download_df(df, filename):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode())
    b.seek(0)
    st.download_button("Download CSV", b, file_name=filename, mime="text/csv")

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tabs = st.tabs(["Overview", "Application"])

# ---------------------------------------------------------
# Overview Tab
# ---------------------------------------------------------
with tabs[0]:
    st.markdown("## Overview")

    st.markdown("""
    <div class='card'>
        This application analyzes order-to-delivery processes, predicts lead times, simulates dispatch, and provides actionable insights to reduce delivery delays.
    </div>
    """, unsafe_allow_html=True)

    # Capabilities / Impact
    c1, c2 = st.columns(2)
    c1.markdown("### Capabilities")
    c1.markdown("""
    <div class='card'>
        • Predictive lead-time analytics<br>
        • End-to-end production visibility dashboards<br>
        • Dispatch simulation<br>
        • Multi-filter exploration by Product, Machine Type, Date Range<br>
        • Order delay analysis
    </div>
    """, unsafe_allow_html=True)

    c2.markdown("### Business Impact")
    c2.markdown("""
    <div class='card'>
        • Faster deliveries<br>
        • Improved customer satisfaction<br>
        • Reduced scheduling and machine delays<br>
        • Optimized dispatch planning<br>
        • Data-driven production decisions
    </div>
    """, unsafe_allow_html=True)

    # KPI CARDS (5 in a row)
    st.markdown("## KPIs")
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.markdown("<div class='kpi'>Total Orders</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Avg Lead Time (Hrs)</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Avg Scheduling Delay</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Avg Production Time</div>", unsafe_allow_html=True)
    k5.markdown("<div class='kpi'>Avg Dispatch Delay</div>", unsafe_allow_html=True)

    # Who should use
    st.markdown("### Who Should Use This App?")
    st.markdown("""
    <div class='card'>
        • Production/Plant Managers<br>
        • Operations Teams<br>
        • Dispatch/Logistics Planners<br>
        • Data Analysts
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### How to Use")
    st.markdown("""
    <div class='card'>
        • Upload dataset or use default<br>
        • Apply filters for Product, Machine, Date<br>
        • Explore KPIs and trend charts<br>
        • Review ML predicted lead times<br>
        • Download insights for reporting
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Application Tab
# ---------------------------------------------------------
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
        "Production_Time_Hrs","Machine_Delay_Hrs","Dispatch_Delay_Hrs","Total_Lead_Time_Hrs",
        "Estimated_Delivery_Date","Predicted_Lead_Time_Hrs","Customer_Satisfaction_Score"
    ]

    # ---------------------- DEFAULT DATASET ----------------------
    if mode == "Default dataset (GitHub URL)":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/manufacturing/machine_failure_data.csv"
        # Replace with realistic default dataset URL for orders if available
        df = pd.DataFrame([{
            "Order_ID":"ORD-1000",
            "Order_Date":"2023-09-13 15:04:06",
            "Product_Type":"Component-A",
            "Machine_Type":"Assembly",
            "Scheduling_Delay_Hrs":5.99,
            "Production_Time_Hrs":11.45,
            "Machine_Delay_Hrs":3.97,
            "Dispatch_Delay_Hrs":7.05,
            "Total_Lead_Time_Hrs":28.46,
            "Estimated_Delivery_Date":"2023-09-14 19:31:34",
            "Predicted_Lead_Time_Hrs":27.99,
            "Customer_Satisfaction_Score":7.15
        }]*10)

    # ---------------------- UPLOAD CSV ----------------------
    elif mode == "Upload CSV":
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)
        else:
            st.stop()

    # ---------------------- UPLOAD + MAP ----------------------
    else:
        file = st.file_uploader("Upload CSV for mapping", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.write("Preview:")
            st.dataframe(raw.head())
            mapping = {}
            for col in REQUIRED_COLS:
                mapping[col] = st.selectbox(f"Map → {col}", ["-- Select --"] + list(raw.columns))
            if st.button("Apply Mapping"):
                miss = [m for m in mapping if mapping[m] == "-- Select --"]
                if miss:
                    st.error("Map all columns: " + ", ".join(miss))
                else:
                    df = raw.rename(columns={mapping[k]:k for k in mapping})
                    st.success("Mapping applied.")
        else:
            st.stop()

    if df is None:
        st.stop()

    # Convert dates
    if "Order_Date" in df.columns:
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    if "Estimated_Delivery_Date" in df.columns:
        df["Estimated_Delivery_Date"] = pd.to_datetime(df["Estimated_Delivery_Date"], errors="coerce")

    # ---------------------- Filters ----------------------
    st.markdown("### Filters")
    products = df["Product_Type"].unique().tolist() if "Product_Type" in df.columns else []
    machines = df["Machine_Type"].unique().tolist() if "Machine_Type" in df.columns else []

    p1,p2 = st.columns(2)
    sel_prod = p1.multiselect("Product Type", products, default=products)
    sel_mach = p2.multiselect("Machine Type", machines, default=machines)

    df_f = df.copy()
    if sel_prod:
        df_f = df_f[df_f["Product_Type"].isin(sel_prod)]
    if sel_mach:
        df_f = df_f[df_f["Machine_Type"].isin(sel_mach)]

    st.dataframe(df_f.head(5), use_container_width=True)
    download_df(df_f, "filtered_orders.csv")

    # ---------------------- Dynamic KPIs ----------------------
    st.markdown("### Key Metrics")
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Total Orders", df_f.shape[0])
    k2.metric("Avg Lead Time", round(df_f["Total_Lead_Time_Hrs"].mean(),2) if "Total_Lead_Time_Hrs" in df_f.columns else 0)
    k3.metric("Avg Scheduling Delay", round(df_f["Scheduling_Delay_Hrs"].mean(),2) if "Scheduling_Delay_Hrs" in df_f.columns else 0)
    k4.metric("Avg Production Time", round(df_f["Production_Time_Hrs"].mean(),2) if "Production_Time_Hrs" in df_f.columns else 0)
    k5.metric("Avg Dispatch Delay", round(df_f["Dispatch_Delay_Hrs"].mean(),2) if "Dispatch_Delay_Hrs" in df_f.columns else 0)

    # ---------------------- Charts ----------------------
    st.markdown("### Charts")
    if "Order_Date" in df_f.columns and "Total_Lead_Time_Hrs" in df_f.columns:
        fig1 = px.line(df_f.sort_values("Order_Date"), x="Order_Date", y="Total_Lead_Time_Hrs",
                       color="Product_Type", title="Lead Time Trend")
        st.plotly_chart(fig1, use_container_width=True)

    # ---------------------- ML: Lead Time Prediction ----------------------
    st.markdown("### Machine Learning: Lead Time Prediction")
    features = ["Scheduling_Delay_Hrs","Production_Time_Hrs","Machine_Delay_Hrs","Dispatch_Delay_Hrs"]
    features = [f for f in features if f in df_f.columns]
    target = "Total_Lead_Time_Hrs"

    if len(df_f) >= 10 and len(features) >= 2 and target in df_f.columns:
        X = df_f[features].fillna(0)
        y = df_f[target].fillna(0)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        Xtr,Xte,ytr,yte = train_test_split(Xs,y,test_size=0.2,random_state=42)

        model = RandomForestRegressor(n_estimators=150, random_state=42)
        model.fit(Xtr,ytr)

        preds = model.predict(Xte)

        results = pd.DataFrame({
            "Actual_Lead_Time": yte,
            "Predicted_Lead_Time": preds
        })
        st.dataframe(results.head(10))
        download_df(results,"ml_predictions.csv")
    else:
        st.info("Not enough data or features for ML prediction.")

    # ---------------------- Automated Insights ----------------------
    st.markdown("### Automated Insights")
    insights = []
    if "Total_Lead_Time_Hrs" in df_f.columns:
        insights.append({"Insight":"Max Lead Time", "Value": round(df_f["Total_Lead_Time_Hrs"].max(),2)})
        insights.append({"Insight":"Min Lead Time", "Value": round(df_f["Total_Lead_Time_Hrs"].min(),2)})
        insights.append({"Insight":"Avg Lead Time", "Value": round(df_f["Total_Lead_Time_Hrs"].mean(),2)})
    ins_df = pd.DataFrame(insights)
    st.dataframe(ins_df)
    download_df(ins_df,"automated_insights.csv")
