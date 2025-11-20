# app_inventory_analytics.py
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
st.set_page_config(page_title="Inventory Analytics: Pileup & Shortages", layout="wide")

# Hide sidebar
st.markdown("""<style>[data-testid="stSidebarNav"]{display:none;}</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Card Glow CSS
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
.small { font-size:13px; color:#666; }
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
# Safe CSV loader (handles duplicate column names)
# ---------------------------------------------------------
def read_csv_safe(url_or_file):
    df = pd.read_csv(url_or_file)
    cols = list(df.columns)
    if len(cols) != len(set(cols)):
        new_cols = []
        seen = {}
        for c in cols:
            base = str(c).strip()
            if base not in seen:
                seen[base] = 0
                new_cols.append(base)
            else:
                seen[base] += 1
                new_cols.append(f"{base}__dup{seen[base]}")
        df.columns = new_cols
    df.columns = [str(c).strip() for c in df.columns]
    return df

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
        This application monitors inventory levels, detects pileups or shortages,
        forecasts demand, and provides actionable insights to optimize stock and production flow.
    </div>
    """, unsafe_allow_html=True)

    # Capabilities / Impact
    c1, c2 = st.columns(2)
    c1.markdown("### Capabilities")
    c1.markdown("""
    <div class='card'>
        • SKU-level demand forecasting<br>
        • Inventory pileup/shortage detection<br>
        • Production & dispatch delay analytics<br>
        • Multi-filter analysis by SKU, Date, Machine Type<br>
        • Predictive lead time & delivery insights
    </div>
    """, unsafe_allow_html=True)

    c2.markdown("### Business Impact")
    c2.markdown("""
    <div class='card'>
        • Reduce cash stuck in inventory<br>
        • Stabilize production & delivery<br>
        • Lower inventory wastage<br>
        • Improve customer satisfaction<br>
        • Optimize procurement & scheduling
    </div>
    """, unsafe_allow_html=True)

    # KPI CARDS (5 in a row)
    st.markdown("## KPIs")
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.markdown("<div class='kpi'>SKUs Tracked</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Avg Inventory</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Avg Lead Time</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Shortages</div>", unsafe_allow_html=True)
    k5.markdown("<div class='kpi'>Pileups</div>", unsafe_allow_html=True)

    st.markdown("### Who Should Use This App?")
    st.markdown("""
    <div class='card'>
        • Inventory Managers<br>
        • Supply Chain Analysts<br>
        • Production Planners<br>
        • Operations Managers
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### How To Use This App?")
    st.markdown("""
    <div class='card'>
        • Upload your inventory dataset or use the default dataset.<br>
        • Filter by SKU, Date Range, or Machine Type.<br>
        • Review KPIs, trends, and automated insights.<br>
        • Use ML predictions to forecast shortages or pileups.<br>
        • Export filtered data or insights for reporting.
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
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/manufacturing/inventory_data.csv"
        try:
            df = read_csv_safe(DEFAULT_URL)
            st.success("Default dataset loaded successfully.")
            st.dataframe(df.head())
        except Exception as e:
            st.error("Failed to load default dataset. Error: " + str(e))
            st.stop()

    # ---------------------- UPLOAD CSV ----------------------
    elif mode == "Upload CSV":
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            df = read_csv_safe(file)
            st.success("File uploaded.")
            st.dataframe(df.head())
        else:
            st.stop()

    # ---------------------- UPLOAD + MAP ----------------------
    else:
        file = st.file_uploader("Upload CSV for mapping", type=["csv"])
        if file:
            raw = read_csv_safe(file)
            st.write("Preview:")
            st.dataframe(raw.head())
            mapping = {}
            for col in REQUIRED_COLS:
                mapping[col] = st.selectbox(f"Map → {col}", ["-- Select --"] + list(raw.columns))
            if st.button("Apply Mapping"):
                miss = [m for m in mapping if mapping[m]=="-- Select --"]
                if miss:
                    st.error("Map all columns: " + ", ".join(miss))
                else:
                    rename_map = {mapping[k]: k for k in mapping}
                    df = raw.rename(columns=rename_map)
                    st.success("Mapping applied.")
                    st.dataframe(df.head())
        else:
            st.stop()

    if df is None:
        st.stop()

    # ---------- Filters ----------
    st.markdown("### Filters")
    skus = df["Product_Type"].unique().tolist() if "Product_Type" in df.columns else []
    machine_types = df["Machine_Type"].unique().tolist() if "Machine_Type" in df.columns else []

    m1,m2 = st.columns(2)
    sel_sku = m1.multiselect("Product Type", skus, default=skus)
    sel_machine = m2.multiselect("Machine Type", machine_types, default=machine_types)

    df_f = df.copy()
    if sel_sku:
        df_f = df_f[df_f["Product_Type"].isin(sel_sku)]
    if sel_machine:
        df_f = df_f[df_f["Machine_Type"].isin(sel_machine)]

    st.dataframe(df_f.head(), use_container_width=True)
    download_df(df_f, "filtered_inventory.csv")

    # ---------------------- Dynamic KPIs ----------------------
    st.markdown("### Key Metrics (Dynamic)")
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("SKUs", df_f["Product_Type"].nunique())
    k2.metric("Avg Inventory", round(df_f["Total_Lead_Time_Hrs"].mean(),2))
    k3.metric("Avg Lead Time", round(df_f["Predicted_Lead_Time_Hrs"].mean(),2) if "Predicted_Lead_Time_Hrs" in df_f.columns else 0)
    k4.metric("Shortages", int((df_f["Total_Lead_Time_Hrs"]>50).sum()))
    k5.metric("Pileups", int((df_f["Total_Lead_Time_Hrs"]<10).sum()))

    # ---------------------- Charts ----------
    if "Order_Date" in df_f.columns and "Total_Lead_Time_Hrs" in df_f.columns:
        df_f["Order_Date"] = pd.to_datetime(df_f["Order_Date"])
        fig = px.line(df_f.sort_values("Order_Date"), x="Order_Date", y="Total_Lead_Time_Hrs", color="Product_Type", title="Lead Time Trend")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------- ML Prediction ----------
    st.markdown("### ML: Lead Time Prediction")
    numeric_cols = ["Scheduling_Delay_Hrs","Production_Time_Hrs","Machine_Delay_Hrs","Dispatch_Delay_Hrs"]
    if len(df_f)>=20 and all(col in df_f.columns for col in numeric_cols):
        X = df_f[numeric_cols].fillna(0)
        y = df_f["Total_Lead_Time_Hrs"].fillna(0)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        Xtr,Xte,ytr,yte = train_test_split(Xs,y,test_size=0.2,random_state=42)
        model = RandomForestRegressor(n_estimators=100,random_state=42)
        model.fit(Xtr,ytr)
        preds = model.predict(Xte)
        results = pd.DataFrame({**{"Actual":yte},**{"Predicted":preds}})
        st.dataframe(results.head(10))
        download_df(results,"ml_leadtime_predictions.csv")
    else:
        st.info("Not enough data for ML prediction.")

    # ---------------------- Automated Insights ----------
    st.markdown("### Automated Insights")
    insights = []
    if "Product_Type" in df_f.columns and "Total_Lead_Time_Hrs" in df_f.columns:
        lead_time_avg = df_f.groupby("Product_Type")["Total_Lead_Time_Hrs"].mean().sort_values(ascending=False)
        if not lead_time_avg.empty:
            insights.append({"Insight":"Product with highest lead time","Product":lead_time_avg.index[0],"Avg_Lead_Time":round(lead_time_avg.iloc[0],2)})
    ins_df = pd.DataFrame(insights)
    if not ins_df.empty:
        st.dataframe(ins_df)
        download_df(ins_df,"automated_insights.csv")
    else:
        st.info("No automated insights available.")
