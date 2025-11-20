# app_inventory_pileup_shortage.py
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
st.set_page_config(
    page_title="Inventory Pileup & Shortage Analytics",
    layout="wide"
)

# Hide sidebar
st.markdown(
    """<style>[data-testid="stSidebarNav"]{display:none;}</style>""",
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Card Glow CSS
# ---------------------------------------------------------
st.markdown(
    """
<style>
.card {
    padding: 20px;
    border-radius: 12px;
    background: white;
    border: 1px solid #d9d9d9;
    transition: 0.3s;
    box-shadow: 0px 2px 4px rgba(0,0,0,0.1);
    text-align: left;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0px 4px 18px rgba(6,75,134,0.35);
    border-color: #064b86;
}
.kpi {
    padding: 26px;
    border-radius: 14px;
    background: white;
    border: 1px solid #ccc;
    font-size: 24px;
    font-weight: 700;
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
""",
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Header + Logo
# ---------------------------------------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

st.markdown(
    f"""
<div style="display:flex; align-items:center; margin-bottom:10px;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:34px; font-weight:bold;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:34px; font-weight:bold;">Advanced Analytics</div>
    </div>
</div>
<h1 style="text-align:left; margin-top:4px; margin-bottom:2px;">
    Inventory Pileup & Shortage Analytics
</h1>
<p style="margin-top:0; color:#444;">
    Track demand vs supply, predict stock risks, and balance working capital with service levels.
</p>
""",
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Helper for CSV download
# ---------------------------------------------------------
def download_df(df, filename, label="Download CSV", key=None):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode())
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv", key=key)

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

    st.markdown(
        """
    <div class='card'>
        This application monitors SKU-level demand, supply, and inventory to prevent both stockouts and excess pileups.
        It combines demand, production, procurement, and stock data to help you maintain lean but reliable inventory.
    </div>
    """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Capabilities")
        st.markdown(
            """
        <div class='card'>
            • Track daily demand vs production and procurement across SKUs<br>
            • Monitor inventory levels vs safety stock in near real-time<br>
            • Detect shortage & pileup risk via rule-based flags and analytics<br>
            • Run SKU-level lead-time & service-level diagnostics<br>
            • Export-ready EDA & ML outputs for planning / ERP / finance
        </div>
        """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown("### Business Impact")
        st.markdown(
            """
        <div class='card'>
            • Reduce cash stuck in slow-moving inventory<br>
            • Avoid production stoppages due to stockouts<br>
            • Improve service levels and customer OTIF<br>
            • Align production, procurement, and demand more tightly<br>
            • Support data-driven safety stock & reorder policies
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("## KPIs")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown("<div class='kpi'>Active SKUs</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Avg Inventory Level</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Shortage Incidents</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Pileup Incidents</div>", unsafe_allow_html=True)
    k5.markdown("<div class='kpi'>Service Level</div>", unsafe_allow_html=True)

    st.markdown("### Who Should Use This App & How")
    st.markdown(
        """
    <div class='card'>
        <b>Who</b>: Supply chain managers, production planners, inventory controllers, plant heads, and finance partners.<br><br>
        <b>How</b>:<br>
        1) Load the latest inventory & demand file (or use default demo data).<br>
        2) Filter by SKU & date to review demand, stock, lead time, and risk patterns.<br>
        3) Use the charts to identify chronic shortages, pileups, and unstable SKUs.<br>
        4) Download ML predictions & insights to feed planning rules / ERP / S&OP decks.
    </div>
    """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Application Tab
# ---------------------------------------------------------
with tabs[1]:
    st.header("Application")
    st.markdown("### Choose Dataset Option")

    mode = st.radio(
        "Select:",
        ["Default dataset (GitHub URL)", "Upload CSV", "Upload CSV + Manual Column Mapping"],
        horizontal=True,
    )

    df = None

    REQUIRED_COLS = [
        "SKU",
        "Date",
        "Daily_Demand",
        "Predicted_Demand",
        "Forecast_Error",
        "Production_Qty",
        "Production_Delay_Hrs",
        "Procurement_Qty",
        "Procurement_Delay_Hrs",
        "Inventory_Level",
        "Safety_Stock",
        "Stock_Turnover",
        "Lead_Time_Days",
        "Backorder_Qty",
        "Wastage_Qty",
        "Shortage_Flag",
        "Pileup_Flag",
    ]

    # Adjust this to your actual file path
    DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/manufacturing/inventory_data.csv"

    # ---------------------- DEFAULT DATASET ----------------------
    if mode == "Default dataset (GitHub URL)":
        try:
            df = read_csv_safe(DEFAULT_URL)
            st.success("Default dataset loaded successfully from GitHub URL.")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(
                "Failed to load default dataset. Check DEFAULT_URL or network. Error: "
                + str(e)
            )
            st.stop()

    # ---------------------- UPLOAD CSV ----------------------
    elif mode == "Upload CSV":
        file = st.file_uploader("Upload CSV", type=["csv"], key="upload_simple")
        if file:
            try:
                df = read_csv_safe(file)
                st.success("File uploaded successfully.")
                st.dataframe(df.head(), use_container_width=True)
            except Exception as e:
                st.error("Failed to read uploaded CSV: " + str(e))
                st.stop()
        else:
            st.stop()

    # ---------------------- UPLOAD + MAP ----------------------
    else:
        file = st.file_uploader("Upload CSV for mapping", type=["csv"], key="upload_map")
        if file:
            try:
                raw = read_csv_safe(file)
            except Exception as e:
                st.error("Failed to read uploaded CSV: " + str(e))
                st.stop()

            st.write("Preview:")
            st.dataframe(raw.head(), use_container_width=True)

            mapping = {}
            cols_list = list(raw.columns)
            for col in REQUIRED_COLS:
                mapping[col] = st.selectbox(
                    f"Map → {col}", ["-- Select --"] + cols_list, key=f"map_{col}"
                )

            if st.button("Apply Mapping", key="apply_map"):
                miss = [m for m in mapping if mapping[m] == "-- Select --"]
                if miss:
                    st.error("Map all required columns: " + ", ".join(miss))
                else:
                    rename_map = {mapping[k]: k for k in mapping}
                    df = raw.rename(columns=rename_map)
                    st.success("Mapping applied.")
                    st.dataframe(df.head(), use_container_width=True)
        else:
            st.stop()

    if df is None:
        st.stop()

    # ---------- Basic cleanup ----------
    df.columns = [str(c).strip() for c in df.columns]

    missing_req = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_req:
        st.warning("Some expected columns are missing: " + ", ".join(missing_req))

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    num_cols = [
        "Daily_Demand",
        "Predicted_Demand",
        "Forecast_Error",
        "Production_Qty",
        "Production_Delay_Hrs",
        "Procurement_Qty",
        "Procurement_Delay_Hrs",
        "Inventory_Level",
        "Safety_Stock",
        "Stock_Turnover",
        "Lead_Time_Days",
        "Backorder_Qty",
        "Wastage_Qty",
        "Shortage_Flag",
        "Pileup_Flag",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for flag in ["Shortage_Flag", "Pileup_Flag"]:
        if flag in df.columns:
            df[flag] = df[flag].fillna(0).astype(int)

    if "Date" in df.columns:
        df = df[df["Date"].notna()]

    # ---------------------------------------------------------
    # Filters (SKU + Date slicer)
    # ---------------------------------------------------------
    st.markdown("### Filters")

    f1, f2, _ = st.columns([2, 2, 2])

    sku_list = (
        sorted(df["SKU"].dropna().unique()) if "SKU" in df.columns else []
    )
    with f1:
        sel_skus = st.multiselect(
            "SKU", options=sku_list, default=sku_list[:10] if sku_list else []
        )

    if "Date" in df.columns and not df["Date"].empty:
        min_date = df["Date"].min().date()
        max_date = df["Date"].max().date()
        with f2:
            date_range = st.date_input("Date range", value=(min_date, max_date))
    else:
        date_range = None

    df_f = df.copy()
    if sel_skus:
        df_f = df_f[df_f["SKU"].isin(sel_skus)]
    if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        if "Date" in df_f.columns:
            df_f = df_f[(df_f["Date"] >= start) & (df_f["Date"] <= end)]

    st.markdown("#### Filtered data (first 10 rows)")
    st.dataframe(df_f.head(10), use_container_width=True)
    download_df(df_f, "inventory_filtered_data.csv", key="dl_filtered")

    # ---------------------------------------------------------
    # Dynamic KPIs
    # ---------------------------------------------------------
    st.markdown("### Key Metrics (Dynamic)")
    k1, k2, k3, k4, k5 = st.columns(5)

    active_skus = df_f["SKU"].nunique() if "SKU" in df_f.columns else 0
    avg_inv = df_f["Inventory_Level"].mean() if "Inventory_Level" in df_f.columns else np.nan
    shortage_events = df_f["Shortage_Flag"].sum() if "Shortage_Flag" in df_f.columns else 0
    pileup_events = df_f["Pileup_Flag"].sum() if "Pileup_Flag" in df_f.columns else 0
    service_level = 1 - (
        df_f["Backorder_Qty"].gt(0).mean() if "Backorder_Qty" in df_f.columns else 0
    )

    k1.metric("Active SKUs", int(active_skus))
    k2.metric("Avg Inventory", f"{avg_inv:.1f}" if not np.isnan(avg_inv) else "N/A")
    k3.metric("Shortage Events", int(shortage_events))
    k4.metric("Pileup Events", int(pileup_events))
    k5.metric("Service Level", f"{service_level*100:.1f}%" if service_level is not None else "N/A")

    # ---------------------------------------------------------
    # CHARTS 1–10 (Core EDA)
    # ---------------------------------------------------------
    st.markdown("## Exploratory Analysis — Core Views")

    # 1. Inventory level over time by SKU
    if "Date" in df_f.columns and "Inventory_Level" in df_f.columns:
        st.markdown("#### 1. Inventory Level over Time (by SKU)")
        fig1 = px.line(
            df_f.sort_values("Date"),
            x="Date",
            y="Inventory_Level",
            color="SKU",
            height=350,
        )
        st.plotly_chart(fig1, use_container_width=True)

    # 2. Daily demand vs production (aggregated)
    if (
        "Date" in df_f.columns
        and "Daily_Demand" in df_f.columns
        and "Production_Qty" in df_f.columns
    ):
        st.markdown("#### 2. Daily Demand vs Production")
        agg_dp = (
            df_f.groupby("Date")
            .agg(
                Total_Demand=("Daily_Demand", "sum"),
                Total_Production=("Production_Qty", "sum"),
            )
            .reset_index()
        )
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=agg_dp["Date"], y=agg_dp["Total_Demand"], name="Total Demand"
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=agg_dp["Date"],
                y=agg_dp["Total_Production"],
                name="Total Production",
            )
        )
        fig2.update_layout(yaxis_title="Qty", xaxis_title="Date")
        st.plotly_chart(fig2, use_container_width=True)

    # 3. Safety stock breach heatmap
    if (
        "Date" in df_f.columns
        and "Inventory_Level" in df_f.columns
        and "Safety_Stock" in df_f.columns
    ):
        st.markdown("#### 3. Safety Stock Breach Heatmap")
        tmp = df_f.copy()
        tmp["Breach"] = np.where(
            tmp["Inventory_Level"] < tmp["Safety_Stock"], 1, 0
        )
        tmp["Date_only"] = tmp["Date"].dt.date
        heat = (
            tmp.groupby(["SKU", "Date_only"])["Breach"]
            .max()
            .reset_index()
        )
        pivot = heat.pivot(index="SKU", columns="Date_only", values="Breach").fillna(0)
        fig3 = px.imshow(
            pivot,
            aspect="auto",
            color_continuous_scale="Reds",
        )
        fig3.update_layout(coloraxis_colorbar_title="Breach")
        st.plotly_chart(fig3, use_container_width=True)

    # 4. Shortage vs Pileup counts
    if "Shortage_Flag" in df_f.columns and "Pileup_Flag" in df_f.columns:
        st.markdown("#### 4. Shortage vs Pileup Events")
        counts = pd.DataFrame(
            {
                "Type": ["Shortage", "Pileup"],
                "Events": [
                    df_f["Shortage_Flag"].sum(),
                    df_f["Pileup_Flag"].sum(),
                ],
            }
        )
        fig4 = px.bar(counts, x="Type", y="Events")
        st.plotly_chart(fig4, use_container_width=True)

    # 5. Lead time distribution
    if "Lead_Time_Days" in df_f.columns:
        st.markdown("#### 5. Lead Time Distribution")
        fig5 = px.histogram(df_f, x="Lead_Time_Days", nbins=30)
        st.plotly_chart(fig5, use_container_width=True)

    # 6. Lead time vs Forecast Error
    if "Lead_Time_Days" in df_f.columns and "Forecast_Error" in df_f.columns:
        st.markdown("#### 6. Lead Time vs Forecast Error")
        fig6 = px.scatter(
            df_f,
            x="Forecast_Error",
            y="Lead_Time_Days",
            color="SKU",
        )
        st.plotly_chart(fig6, use_container_width=True)

    # 7. Stock turnover by SKU
    if "Stock_Turnover" in df_f.columns and "SKU" in df_f.columns:
        st.markdown("#### 7. Stock Turnover by SKU")
        st_turn = (
            df_f.groupby("SKU")["Stock_Turnover"].mean().reset_index()
        )
        fig7 = px.bar(st_turn, x="SKU", y="Stock_Turnover")
        st.plotly_chart(fig7, use_container_width=True)

    # 8. Backorder & Wastage trend
    if (
        "Date" in df_f.columns
        and "Backorder_Qty" in df_f.columns
        and "Wastage_Qty" in df_f.columns
    ):
        st.markdown("#### 8. Backorder & Wastage Trend")
        bw = (
            df_f.groupby("Date")
            .agg(
                Backorder=("Backorder_Qty", "sum"),
                Wastage=("Wastage_Qty", "sum"),
            )
            .reset_index()
        )
        fig8 = go.Figure()
        fig8.add_trace(
            go.Bar(x=bw["Date"], y=bw["Backorder"], name="Backorder")
        )
        fig8.add_trace(
            go.Bar(x=bw["Date"], y=bw["Wastage"], name="Wastage")
        )
        fig8.update_layout(
            barmode="group", xaxis_title="Date", yaxis_title="Qty"
        )
        st.plotly_chart(fig8, use_container_width=True)

    # 9. Top SKUs by average inventory
    if "Inventory_Level" in df_f.columns and "SKU" in df_f.columns:
        st.markdown("#### 9. Top 10 SKUs by Average Inventory")
        top_inv = (
            df_f.groupby("SKU")["Inventory_Level"]
            .mean()
            .reset_index()
            .sort_values("Inventory_Level", ascending=False)
            .head(10)
        )
        fig9 = px.bar(top_inv, x="SKU", y="Inventory_Level")
        st.plotly_chart(fig9, use_container_width=True)

    # 10. SKU performance matrix
    if (
        "Daily_Demand" in df_f.columns
        and "Inventory_Level" in df_f.columns
        and "Backorder_Qty" in df_f.columns
    ):
        st.markdown("#### 10. SKU Performance Matrix")
        perf = (
            df_f.groupby("SKU")
            .agg(
                Avg_Demand=("Daily_Demand", "mean"),
                Avg_Inventory=("Inventory_Level", "mean"),
                Backorder=("Backorder_Qty", "sum"),
            )
            .reset_index()
        )
        fig10 = px.scatter(
            perf,
            x="Avg_Demand",
            y="Avg_Inventory",
            size="Backorder",
            color="Backorder",
            hover_name="SKU",
        )
        st.plotly_chart(fig10, use_container_width=True)

    # ---------------------------------------------------------
    # CHARTS 11–15 (Advanced)
    # ---------------------------------------------------------
    st.markdown("## Advanced Analytics — Additional Views")

    # 11. Production vs Demand gap
    if (
        "Date" in df_f.columns
        and "Daily_Demand" in df_f.columns
        and "Production_Qty" in df_f.columns
    ):
        st.markdown("#### 11. Production vs Demand Gap")
        agg_gap = (
            df_f.groupby("Date")
            .agg(
                Total_Demand=("Daily_Demand", "sum"),
                Total_Production=("Production_Qty", "sum"),
            )
            .reset_index()
        )
        agg_gap["Gap"] = agg_gap["Total_Production"] - agg_gap["Total_Demand"]
        fig11 = go.Figure()
        fig11.add_trace(
            go.Scatter(
                x=agg_gap["Date"],
                y=agg_gap["Total_Demand"],
                name="Demand",
                fill="tozeroy",
            )
        )
        fig11.add_trace(
            go.Scatter(
                x=agg_gap["Date"],
                y=agg_gap["Total_Production"],
                name="Production",
                fill="tonexty",
            )
        )
        fig11.update_layout(xaxis_title="Date", yaxis_title="Qty")
        st.plotly_chart(fig11, use_container_width=True)

    # 12. Inventory risk matrix: shortage vs pileup rate
    if (
        "Shortage_Flag" in df_f.columns
        and "Pileup_Flag" in df_f.columns
        and "SKU" in df_f.columns
    ):
        st.markdown("#### 12. Inventory Risk Matrix (Shortage vs Pileup Rate)")
        risk = (
            df_f.groupby("SKU")
            .agg(
                Shortage_Rate=("Shortage_Flag", "mean"),
                Pileup_Rate=("Pileup_Flag", "mean"),
            )
            .reset_index()
        )
        fig12 = px.scatter(
            risk, x="Shortage_Rate", y="Pileup_Rate", hover_name="SKU"
        )
        st.plotly_chart(fig12, use_container_width=True)

    # 13. Inventory level moving average + naive forecast
    if "Date" in df_f.columns and "Inventory_Level" in df_f.columns:
        st.markdown("#### 13. Inventory Level Moving Average & Naive Forecast")
        inv_ts = (
            df_f.groupby("Date")["Inventory_Level"]
            .mean()
            .reset_index()
            .sort_values("Date")
        )
        inv_ts["MA_7"] = inv_ts["Inventory_Level"].rolling(
            window=7, min_periods=1
        ).mean()
        if not inv_ts.empty:
            last_val = inv_ts["Inventory_Level"].iloc[-1]
            future_dates = pd.date_range(
                inv_ts["Date"].iloc[-1] + pd.Timedelta(days=1), periods=7
            )
            future_df = pd.DataFrame(
                {"Date": future_dates, "Inventory_Level": [last_val] * len(future_dates)}
            )
        else:
            future_df = pd.DataFrame(columns=["Date", "Inventory_Level"])

        fig13 = go.Figure()
        fig13.add_trace(
            go.Scatter(
                x=inv_ts["Date"], y=inv_ts["Inventory_Level"], name="Actual"
            )
        )
        fig13.add_trace(
            go.Scatter(x=inv_ts["Date"], y=inv_ts["MA_7"], name="7-day MA")
        )
        if not future_df.empty:
            fig13.add_trace(
                go.Scatter(
                    x=future_df["Date"],
                    y=future_df["Inventory_Level"],
                    name="Naive Forecast",
                    line=dict(dash="dash"),
                )
            )
        fig13.update_layout(xaxis_title="Date", yaxis_title="Avg Inventory")
        st.plotly_chart(fig13, use_container_width=True)

    # 14. Procurement delay vs backorders
    if "Procurement_Delay_Hrs" in df_f.columns and "Backorder_Qty" in df_f.columns:
        st.markdown("#### 14. Procurement Delay vs Backorder Quantity")
        fig14 = px.scatter(
            df_f,
            x="Procurement_Delay_Hrs",
            y="Backorder_Qty",
            color="SKU",
        )
        st.plotly_chart(fig14, use_container_width=True)

    # 15. Lead time component breakdown
    if (
        "Lead_Time_Days" in df_f.columns
        and "Production_Delay_Hrs" in df_f.columns
        and "Procurement_Delay_Hrs" in df_f.columns
    ):
        st.markdown("#### 15. Lead Time Component Breakdown (Approximate)")
        comp = df_f.copy()
        comp["Prod_Days"] = comp["Production_Delay_Hrs"] / 24.0
        comp["Proc_Days"] = comp["Procurement_Delay_Hrs"] / 24.0
        lt_break = (
            comp.groupby("SKU")
            .agg(
                Lead_Time=("Lead_Time_Days", "mean"),
                Prod_Days=("Prod_Days", "mean"),
                Proc_Days=("Proc_Days", "mean"),
            )
            .reset_index()
            .head(15)
        )
        lt_break["Other_Days"] = lt_break["Lead_Time"] - (
            lt_break["Prod_Days"] + lt_break["Proc_Days"]
        )
        fig15 = go.Figure()
        fig15.add_trace(
            go.Bar(
                x=lt_break["SKU"],
                y=lt_break["Prod_Days"],
                name="Production Delay (days)",
            )
        )
        fig15.add_trace(
            go.Bar(
                x=lt_break["SKU"],
                y=lt_break["Proc_Days"],
                name="Procurement Delay (days)",
            )
        )
        fig15.add_trace(
            go.Bar(
                x=lt_break["SKU"],
                y=lt_break["Other_Days"],
                name="Other Lead Time (days)",
            )
        )
        fig15.update_layout(
            barmode="stack", xaxis_title="SKU", yaxis_title="Days"
        )
        st.plotly_chart(fig15, use_container_width=True)

    # ---------------------------------------------------------
    # ML: Lead Time Prediction (Regression) + Download
    # ---------------------------------------------------------
    st.markdown("## ML: Lead Time Prediction (Regression)")

    target_col = "Lead_Time_Days"
    feature_cols = [
        "Daily_Demand",
        "Predicted_Demand",
        "Forecast_Error",
        "Production_Qty",
        "Production_Delay_Hrs",
        "Procurement_Qty",
        "Procurement_Delay_Hrs",
        "Inventory_Level",
        "Safety_Stock",
        "Stock_Turnover",
        "Backorder_Qty",
        "Wastage_Qty",
        "Shortage_Flag",
        "Pileup_Flag",
    ]
    usable_features = [c for c in feature_cols if c in df_f.columns]

    if (
        target_col in df_f.columns
        and len(df_f) >= 80
        and len(usable_features) >= 3
    ):
        ml_df = df_f.dropna(subset=[target_col]).copy()
        X = ml_df[usable_features].fillna(0)
        y = ml_df[target_col].values

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X.values)

        X_train, X_test, y_train, y_test = train_test_split(
            Xs, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        from sklearn.metrics import mean_squared_error, r2_score

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        st.write(f"Model performance — RMSE: {rmse:.2f}, R²: {r2:.3f}")

        # Re-split using indices so we can show original feature values
        idx = np.arange(X.shape[0])
        _, test_idx, _, y_test_idx = train_test_split(
            idx, y, test_size=0.2, random_state=42
        )
        X_test_orig = X.iloc[test_idx].reset_index(drop=True)

        res = X_test_orig.copy()
        res["Actual_Lead_Time_Days"] = y_test
        res["Predicted_Lead_Time_Days"] = preds

        st.markdown("#### Sample ML Predictions (Lead Time)")
        st.dataframe(res.head(20), use_container_width=True)
        download_df(res, "lead_time_predictions_with_features.csv", key="dl_ml")
    else:
        st.info(
            "Not enough data or required columns for ML lead-time prediction "
            "(need Lead_Time_Days + ≥3 feature columns and ~80+ rows)."
        )

    # ---------------------------------------------------------
    # Automated Insights
    # ---------------------------------------------------------
    st.markdown("## Automated Insights")

    insights = []

    if "SKU" in df_f.columns and "Shortage_Flag" in df_f.columns:
        srt = (
            df_f.groupby("SKU")["Shortage_Flag"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        for sku, val in srt.items():
            insights.append(
                {
                    "Insight_Type": "High Shortage SKU",
                    "SKU": sku,
                    "Metric": "Shortage Rate",
                    "Value": round(float(val), 3),
                }
            )

    if "SKU" in df_f.columns and "Pileup_Flag" in df_f.columns:
        prt = (
            df_f.groupby("SKU")["Pileup_Flag"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        for sku, val in prt.items():
            insights.append(
                {
                    "Insight_Type": "High Pileup SKU",
                    "SKU": sku,
                    "Metric": "Pileup Rate",
                    "Value": round(float(val), 3),
                }
            )

    if "SKU" in df_f.columns and "Lead_Time_Days" in df_f.columns:
        lt = (
            df_f.groupby("SKU")["Lead_Time_Days"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        for sku, val in lt.items():
            insights.append(
                {
                    "Insight_Type": "Long Lead Time SKU",
                    "SKU": sku,
                    "Metric": "Avg Lead Time (days)",
                    "Value": round(float(val), 2),
                }
            )

    if "SKU" in df_f.columns and "Forecast_Error" in df_f.columns:
        fe = (
            df_f.groupby("SKU")["Forecast_Error"]
            .mean()
            .abs()
            .sort_values(ascending=False)
            .head(10)
        )
        for sku, val in fe.items():
            insights.append(
                {
                    "Insight_Type": "Unstable Forecast SKU",
                    "SKU": sku,
                    "Metric": "Avg |Forecast Error|",
                    "Value": round(float(val), 2),
                }
            )

    insights_df = pd.DataFrame(insights)
    if insights_df.empty:
        st.info(
            "No automated insights could be generated for the current filter."
        )
    else:
        st.dataframe(insights_df, use_container_width=True)
        download_df(
            insights_df,
            "inventory_automated_insights.csv",
            key="dl_insights",
        )
