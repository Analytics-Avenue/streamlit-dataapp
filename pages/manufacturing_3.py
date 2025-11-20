import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(page_title="Inventory Optimization Analytics", layout="wide")

# Hide sidebar
st.markdown("""<style>[data-testid="stSidebarNav"]{display:none;}</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Card Glow + KPI Hover CSS
# ---------------------------------------------------------
st.markdown("""
<style>

.card {
    padding: 20px;
    border-radius: 12px;
    background: white;
    border: 1px solid #d9d9d9;
    transition: 0.25s;
    box-shadow: 0px 2px 4px rgba(0,0,0,0.10);
    text-align: left;
}
.card:hover {
    transform: translateY(-6px);
    box-shadow: 0px 5px 20px rgba(6,75,134,0.35);
    border-color: #064b86;
}

.kpi {
    padding: 25px;
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
    transform: translateY(-5px);
    box-shadow: 0px 5px 18px rgba(6,75,134,0.30);
}

.title-bar {
    color:#064b86;
    font-size:36px; 
    font-weight:bold; 
    margin-bottom:20px;
    text-align:left;
}

.small { font-size:13px; color:#777; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

st.markdown(f"""
<div style="display:flex; align-items:center; margin-bottom:20px;">
    <img src="{logo_url}" width="60" style="margin-right:12px;">
    <div class="title-bar">Analytics Avenue & Advanced Analytics</div>
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
# Safe CSV loader
# ---------------------------------------------------------
def read_csv_safe(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tabs = st.tabs(["Overview", "Application"])

# ---------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------
with tabs[0]:

    st.markdown("## Overview")

    st.markdown("""
    <div class='card'>
        This application provides SKU-level inventory insights, predicts shortages and pileups,
        analyzes lead time patterns, and optimizes stock decisions using advanced analytics.
    </div>
    """, unsafe_allow_html=True)

    # Capabilities and Impact
    c1, c2 = st.columns(2)

    c1.markdown("### Capabilities")
    c1.markdown("""
    <div class='card'>
        • Demand forecasting and error analysis<br>
        • Real-time inventory monitoring dashboards<br>
        • Procurement delay & production delay analytics<br>
        • Lead-time prediction using ML<br>
        • SKU-level optimization metrics & alerts
    </div>
    """, unsafe_allow_html=True)

    c2.markdown("### Business Impact")
    c2.markdown("""
    <div class='card'>
        • Reduce stock-outs and production halts<br>
        • Lower excess inventory & carrying costs<br>
        • Improve demand planning accuracy<br>
        • Boost customer service levels<br>
        • Optimize supply chain decision-making
    </div>
    """, unsafe_allow_html=True)

    # KPI 5 CARDS
    st.markdown("## KPIs")

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.markdown("<div class='kpi'>Total SKUs</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Avg Inventory</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Avg Lead Time</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Shortages</div>", unsafe_allow_html=True)
    k5.markdown("<div class='kpi'>Pileups</div>", unsafe_allow_html=True)

    st.markdown("### Who Should Use This App?")
    st.markdown("""
    <div class='card'>
        • Supply Chain Managers<br>
        • Production Planners<br>
        • Inventory Controllers<br>
        • Operations Managers<br>
        • Data Analysts working on demand forecasting
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# APPLICATION TAB
# ---------------------------------------------------------
with tabs[1]:

    st.header("Application")
    st.markdown("### Choose Dataset Option")

    mode = st.radio("Select:", [
        "Default dataset (GitHub URL)",
        "Upload CSV",
        "Upload CSV + Manual Column Mapping"
    ])

    REQUIRED_COLS = [
        "SKU","Date","Daily_Demand","Predicted_Demand","Forecast_Error",
        "Production_Qty","Production_Delay_Hrs","Procurement_Qty",
        "Procurement_Delay_Hrs","Inventory_Level","Safety_Stock",
        "Stock_Turnover","Lead_Time_Days","Backorder_Qty",
        "Wastage_Qty","Shortage_Flag","Pileup_Flag"
    ]

    df = None

    # ----------- DEFAULT DATASET -----------
    if mode == "Default dataset (GitHub URL)":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/manufacturing/inventory_dataset.csv"

        try:
            df = read_csv_safe(DEFAULT_URL)
            st.success("Default dataset loaded.")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error("Failed to load dataset: " + str(e))
            st.stop()

    # ----------- UPLOAD CSV -----------
    elif mode == "Upload CSV":
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            df = read_csv_safe(file)
            st.success("File uploaded.")
            st.dataframe(df.head(), use_container_width=True)
        else:
            st.stop()

    # ----------- UPLOAD & MAP -----------
    else:
        up = st.file_uploader("Upload for mapping", type=["csv"])
        if up:
            raw = read_csv_safe(up)
            st.dataframe(raw.head())

            mapping = {}
            cols = list(raw.columns)
            for col in REQUIRED_COLS:
                mapping[col] = st.selectbox(f"Map → {col}", ["-- Select --"] + cols)

            if st.button("Apply Mapping"):
                miss = [m for m in mapping if mapping[m] == "-- Select --"]
                if miss:
                    st.error("Map all columns.")
                else:
                    df = raw.rename(columns={mapping[k]: k for k in mapping})
                    st.success("Mapping applied.")
                    st.dataframe(df.head())
        else:
            st.stop()

    if df is None:
        st.stop()

    # convert dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # ---------------------------------------------------------
    # FILTERS
    # ---------------------------------------------------------
    st.markdown("### Filters")

    skus = df["SKU"].unique().tolist()
    dates = df["Date"].unique().tolist()

    f1, f2 = st.columns(2)

    sel_sku = f1.multiselect("SKU", skus, default=skus)
    sel_date = f2.multiselect("Date", dates, default=dates)

    df_f = df[df["SKU"].isin(sel_sku) & df["Date"].isin(sel_date)]

    st.dataframe(df_f.head(), use_container_width=True)
    download_df(df_f, "filtered_inventory.csv")

    # ---------------------------------------------------------
    # DYNAMIC KPIs
    # ---------------------------------------------------------
    st.markdown("### Key Metrics (Dynamic)")
    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric("SKUs", df_f["SKU"].nunique())
    k2.metric("Avg Inventory", round(df_f["Inventory_Level"].mean(), 2))
    k3.metric("Avg Lead Time", round(df_f["Lead_Time_Days"].mean(), 2))
    k4.metric("Shortages", int(df_f["Shortage_Flag"].sum()))
    k5.metric("Pileups", int(df_f["Pileup_Flag"].sum()))

    # ---------------------------------------------------------
    # CHARTS (ALL INCLUDED)
    # ---------------------------------------------------------
    st.markdown("## Charts")

    if "Inventory_Level" in df_f.columns:
        fig1 = px.line(df_f.sort_values("Date"), x="Date", y="Inventory_Level",
                       color="SKU", title="Inventory Level Trend")
        st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(df_f, x="SKU", y="Backorder_Qty", title="Backorders by SKU")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(df_f, x="Daily_Demand", y="Forecast_Error",
                      color="SKU", title="Forecast Error vs Demand")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.box(df_f, x="SKU", y="Lead_Time_Days", title="Lead Time Distribution")
    st.plotly_chart(fig4, use_container_width=True)

    # ---------------------------------------------------------
    # MACHINE LEARNING (Lead Time Prediction)
    # ---------------------------------------------------------
    st.markdown("## Machine Learning: Lead Time Prediction")

    ML_COLS = [
        "Daily_Demand","Predicted_Demand","Forecast_Error",
        "Production_Qty","Production_Delay_Hrs","Procurement_Qty",
        "Procurement_Delay_Hrs","Inventory_Level"
    ]

    ML_COLS = [c for c in ML_COLS if c in df_f.columns]

    if len(df_f) > 50 and len(ML_COLS) >= 3:

        X = df_f[ML_COLS].fillna(0)
        y = df_f["Lead_Time_Days"].fillna(0)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        Xtr, Xte, ytr, yte = train_test_split(Xs, y.values, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=180, random_state=42)
        model.fit(Xtr, ytr)

        preds = model.predict(Xte)

        results = pd.DataFrame({
            "Actual Lead Time": yte,
            "Predicted Lead Time": preds
        })

        st.dataframe(results.head(15))
        download_df(results, "leadtime_predictions.csv")

    else:
        st.warning("Not enough rows or features for ML model.")

    # ---------------------------------------------------------
    # AUTOMATED INSIGHTS
    # ---------------------------------------------------------
    st.markdown("## Automated Insights")

    insights = []

    # Highest stock
    inv = df_f.groupby("SKU")["Inventory_Level"].mean().sort_values(ascending=False)
    insights.append({
        "Insight": "Highest Avg Inventory SKU",
        "SKU": inv.index[0],
        "Value": round(inv.iloc[0], 2)
    })

    # Stockouts
    insights.append({
        "Insight": "Total Shortages",
        "Value": int(df_f["Shortage_Flag"].sum())
    })

    # Pileups
    insights.append({
        "Insight": "Total Pileups",
        "Value": int(df_f["Pileup_Flag"].sum())
    })

    # Worst forecast error
    fe = df_f.groupby("SKU")["Forecast_Error"].mean().sort_values(ascending=False)
    insights.append({
        "Insight": "Worst Forecast Accuracy SKU",
        "SKU": fe.index[0],
        "Error": round(fe.iloc[0], 2)
    })

    ins_df = pd.DataFrame(insights)
    st.dataframe(ins_df, use_container_width=True)
    download_df(ins_df, "automated_insights.csv")
