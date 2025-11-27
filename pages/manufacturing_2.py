import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# HIDE SIDEBAR
# ---------------------------------------------------------
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Order-to-Delivery Analytics Lab", layout="wide")

# ---------------------------------------------------------
# LOGO + HEADER
# ---------------------------------------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

st.markdown(f"""
<div style="display:flex; align-items:center; margin-bottom:15px;">
    <img src="{logo_url}" width="62" style="margin-right:14px;">
    <div style="color:#064b86; font-size:40px; font-weight:800; line-height:1;">
        Analytics Avenue <span style="font-weight:500;">&</span><br>Advanced Analytics
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='font-size:44px; font-weight:900; margin-top:-10px;'>
Order-to-Delivery Analytics Lab
</h1>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# GLOBAL CSS (CARDS / KPI / VAR BOXES)
# ---------------------------------------------------------
st.markdown("""
<style>
* { color:#000; font-family: 'Inter', sans-serif; }

.glow-card {
    background:#fff;
    padding:18px 20px;
    border-radius:14px;
    border:1px solid rgba(0,120,255,0.25);
    box-shadow:0 8px 28px rgba(0,120,255,0.15);
    transition:0.25s;
    margin-bottom:12px;
}
.glow-card:hover {
    transform:translateY(-6px);
    box-shadow:0 15px 40px rgba(0,120,255,0.25);
}

.metric-card {
    background:white;
    padding:20px;
    border-radius:12px;
    text-align:center;
    font-weight:600;
    border:1px solid rgba(0,0,0,0.07);
    box-shadow: 0 5px 15px rgba(0,0,0,0.10);
    transition:0.25s ease;
}
.metric-card:hover {
    transform:translateY(-7px) scale(1.03);
    box-shadow:0 0 22px rgba(0,120,255,0.40);
}

.var-box {
    padding:14px 18px;
    border-radius:10px;
    border:1px solid rgba(0,120,255,0.25);
    background:#f7fbff;
    margin-bottom:8px;
    font-size:15px;
}
.section-title {
    font-size:22px;
    font-weight:700;
    margin-top:18px;
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# REQUIRED COLUMNS + DICTIONARY
# ---------------------------------------------------------
REQUIRED_COLS = {
    "Order_ID": "Unique order reference number.",
    "Order_Date": "Date when the order was created.",
    "Product_Type": "Category / family of the ordered product.",
    "Machine_Type": "Machine or line used for production.",
    "Scheduling_Delay_Hrs": "Delay between order creation and production scheduling (in hours).",
    "Production_Time_Hrs": "Pure production / processing time on machine (in hours).",
    "Machine_Delay_Hrs": "Additional delay due to breakdowns or availability (in hours).",
    "Dispatch_Delay_Hrs": "Delay between production completion and dispatch (in hours).",
    "Total_Lead_Time_Hrs": "End-to-end order-to-delivery lead time (in hours).",
    "Estimated_Delivery_Date": "Estimated date when order is expected to be delivered.",
    "Predicted_Lead_Time_Hrs": "Pre-computed / legacy model predicted lead time (if available).",
    "Customer_Satisfaction_Score": "Customer rating or satisfaction score (e.g. 1–5)."
}

DEPENDENT_VAR = "Total_Lead_Time_Hrs"
INDEPENDENT_VARS = [
    "Scheduling_Delay_Hrs",
    "Production_Time_Hrs",
    "Machine_Delay_Hrs",
    "Dispatch_Delay_Hrs"
]

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def download_df(df, filename, label="Download CSV"):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode())
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")

# ---------------------------------------------------------
# 3 TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# ---------------------------------------------------------
# TAB 1 — OVERVIEW
# ---------------------------------------------------------
with tab1:
    st.markdown("## Overview")
    st.markdown("""
    <div class='glow-card'>
        This lab provides full visibility into customer orders, production flow, machine delays,
        and dispatch performance. It helps you understand where time is getting burned in the
        order-to-delivery journey and how much you can recover with smarter scheduling.
    </div>
    """, unsafe_allow_html=True)

    colA, colB = st.columns(2)

    colA.markdown("### Capabilities")
    colA.markdown("""
    <div class='glow-card'>
        • End-to-end order-to-dispatch analytics<br>
        • Lead-time breakdown and bottleneck diagnosis<br>
        • ML & AutoML for delivery-time prediction<br>
        • Real-time KPI monitoring across products & machines<br>
        • Scenario-based scheduling simulator
    </div>
    """, unsafe_allow_html=True)

    colB.markdown("### Business Impact")
    colB.markdown("""
    <div class='glow-card'>
        • Faster and more predictable delivery<br>
        • Reduced production & dispatch delays<br>
        • Better capacity planning for machines<br>
        • Increased customer satisfaction & retention<br>
        • Evidence-backed scheduling & dispatch policies
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## KPI Framework")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown("<div class='metric-card'>Orders Tracked</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Avg Lead Time</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Avg Production Time</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Avg Machine Delay</div>", unsafe_allow_html=True)
    k5.markdown("<div class='metric-card'>Avg Dispatch Delay</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2 — DATA DICTIONARY
# ---------------------------------------------------------
with tab2:
    st.markdown("## Required Columns – Data Dictionary")

    # Create tabular data dictionary
    dd_data = []
    for col, desc in REQUIRED_COLS.items():
        dd_data.append({"Column": col, "Description": desc})

    dd_df = pd.DataFrame(dd_data)

    st.dataframe(dd_df, use_container_width=True)

    st.markdown("---")
    st.markdown("## Model Variables")

    left, right = st.columns(2)

    # ---------------- LEFT: Independent Variables ----------------
    with left:
        st.markdown("<div class='section-title'>Independent Variables</div>", unsafe_allow_html=True)
        for v in INDEPENDENT_VARS:
            desc = REQUIRED_COLS.get(v, "")
            st.markdown(
                f"""
                <div class='var-box'>
                    <b>{v}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ---------------- RIGHT: Dependent Variable -------------------
    with right:
        st.markdown("<div class='section-title'>Dependent Variable</div>", unsafe_allow_html=True)

        desc = REQUIRED_COLS.get(DEPENDENT_VAR, "")

        st.markdown(
            f"""
            <div class='var-box'>
                <b>{DEPENDENT_VAR}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------------------------------------------------
# TAB 3 — APPLICATION
# ---------------------------------------------------------
with tab3:
    st.header("Application")

    # -------------------------------------
    # DATASET MODE
    # -------------------------------------
    st.subheader("Step 1 — Load Dataset")
    mode = st.radio(
        "Choose mode:",
        ["Default dataset", "Upload CSV", "Upload CSV + Manual Mapping"],
        horizontal=True
    )

    df = None

    if mode == "Default dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/manufacturing/order_to_delivery_dataset.csv"
        try:
            df = pd.read_csv(URL)
            st.success("Default dataset loaded from GitHub.")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()

    elif mode == "Upload CSV":
        file = st.file_uploader("Upload order-to-delivery CSV", type=["csv"])
        if file:
            try:
                df = pd.read_csv(file)
                st.success("File uploaded.")
                st.dataframe(df.head(), use_container_width=True)
            except Exception as e:
                st.error("Failed to read uploaded CSV: " + str(e))
                st.stop()
        else:
            st.stop()

    else:  # Upload + Manual Mapping
        file = st.file_uploader("Upload CSV for manual mapping", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            st.markdown("Preview of uploaded data:")
            st.dataframe(raw.head(), use_container_width=True)

            mapping = {}
            for col in REQUIRED_COLS.keys():
                mapping[col] = st.selectbox(
                    f"Map → {col}",
                    ["-- Select --"] + list(raw.columns),
                    key=f"map_{col}"
                )

            if st.button("Apply Mapping"):
                miss = [k for k,v in mapping.items() if v == "-- Select --"]
                if miss:
                    st.error("Please map all required columns: " + ", ".join(miss))
                    st.stop()
                df = raw.rename(columns={v:k for k,v in mapping.items()})
                st.success("Mapping applied successfully.")
                st.dataframe(df.head(), use_container_width=True)
        else:
            st.stop()

    if df is None:
        st.stop()

    # -------------------------------------
    # VALIDATE COLUMNS
    # -------------------------------------
    missing_cols = [c for c in REQUIRED_COLS.keys() if c not in df.columns]
    if missing_cols:
        st.error("Missing required columns: " + ", ".join(missing_cols))
        st.stop()

    # -------------------------------------
    # CLEANUP
    # -------------------------------------
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    if "Estimated_Delivery_Date" in df.columns:
        df["Estimated_Delivery_Date"] = pd.to_datetime(df["Estimated_Delivery_Date"], errors="coerce")

    # Safe numeric conversion
    for col in ["Scheduling_Delay_Hrs", "Production_Time_Hrs",
                "Machine_Delay_Hrs", "Dispatch_Delay_Hrs",
                "Total_Lead_Time_Hrs", "Customer_Satisfaction_Score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -------------------------------------
    # FILTERS
    # -------------------------------------
    st.subheader("Step 2 — Filters & Preview")

    f1, f2 = st.columns(2)
    products = sorted(df["Product_Type"].dropna().unique().tolist())
    machines = sorted(df["Machine_Type"].dropna().unique().tolist())

    sel_p = f1.multiselect("Product Type", products, default=products)
    sel_m = f2.multiselect("Machine Type", machines, default=machines)

    df_f = df.copy()
    if sel_p:
        df_f = df_f[df_f["Product_Type"].isin(sel_p)]
    if sel_m:
        df_f = df_f[df_f["Machine_Type"].isin(sel_m)]

    if df_f.empty:
        st.warning("Filtered dataset is empty. Relax filters.")
        st.stop()

    st.markdown(f"Filtered rows: **{len(df_f)}**")
    st.dataframe(df_f.head(10), use_container_width=True)
    download_df(df_f.head(500), "filtered_orders.csv", "Download filtered preview (first 500 rows)")

    # -------------------------------------
    # KPIs
    # -------------------------------------
    st.subheader("Step 3 — Key Metrics")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Orders", df_f.shape[0])

    def safe_mean(col):
        return df_f[col].mean() if col in df_f.columns and df_f[col].notna().any() else None

    avg_lead = safe_mean("Total_Lead_Time_Hrs")
    avg_prod = safe_mean("Production_Time_Hrs")
    avg_mach = safe_mean("Machine_Delay_Hrs")
    avg_disp = safe_mean("Dispatch_Delay_Hrs")

    k2.metric("Avg Lead Time (hrs)", f"{avg_lead:.2f}" if avg_lead is not None else "N/A")
    k3.metric("Avg Production Time (hrs)", f"{avg_prod:.2f}" if avg_prod is not None else "N/A")
    k4.metric("Avg Machine Delay (hrs)", f"{avg_mach:.2f}" if avg_mach is not None else "N/A")
    k5.metric("Avg Dispatch Delay (hrs)", f"{avg_disp:.2f}" if avg_disp is not None else "N/A")

    # -------------------------------------
    # CHARTS
    # -------------------------------------
    st.subheader("Step 4 — Charts")

    if "Order_Date" in df_f.columns and df_f["Order_Date"].notna().any():
        fig = px.line(
            df_f.sort_values("Order_Date"),
            x="Order_Date",
            y="Total_Lead_Time_Hrs",
            color="Product_Type",
            title="Lead Time Trend by Product Type"
        )
        st.plotly_chart(fig, use_container_width=True)

    if "Machine_Type" in df_f.columns:
        fig2 = px.box(
            df_f,
            x="Machine_Type",
            y="Total_Lead_Time_Hrs",
            title="Lead Time Distribution by Machine Type"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # -------------------------------------
    # AUTO ML — LEAD TIME PREDICTION
    # -------------------------------------
    st.subheader("Step 5 — AutoML: Lead Time Prediction")

    features = [f for f in INDEPENDENT_VARS if f in df_f.columns]

    if len(df_f) >= 40 and len(features) >= 2:
        X_raw = df_f[features].fillna(0)
        y = df_f[DEPENDENT_VAR].fillna(0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        models = {
            "RandomForest": RandomForestRegressor(n_estimators=150, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(random_state=42),
            "LinearRegression": LinearRegression()
        }

        metrics_rows = []
        best_name = None
        best_r2 = -999
        best_preds = None

        for name, mdl in models.items():
            mdl.fit(X_train, y_train)
            preds = mdl.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            rmse = mse ** 0.5
            r2 = r2_score(y_test, preds)
            metrics_rows.append({"Model": name, "RMSE": rmse, "R2": r2})
            if r2 > best_r2:
                best_r2 = r2
                best_name = name
                best_preds = preds

        metrics_df = pd.DataFrame(metrics_rows).sort_values("R2", ascending=False)
        st.markdown("**Model Comparison (AutoML)**")
        st.dataframe(metrics_df, use_container_width=True)

        st.markdown(f"**Selected Best Model:** `{best_name}` (highest R²)")

        results = pd.DataFrame({
            "Actual_Lead_Time_Hrs": y_test.values,
            "Predicted_Lead_Time_Hrs": best_preds
        })
        st.dataframe(results.head(15), use_container_width=True)
        download_df(results, "automl_leadtime_predictions.csv", "Download AutoML prediction sample")

    else:
        st.info("Need at least 40 rows and ≥2 numeric delay features for AutoML training.")

    # -------------------------------------
    # SCHEDULING SIMULATOR
    # -------------------------------------
    st.subheader("Step 6 — Scheduling Simulator (What-if Analysis)")

    if not all(c in df_f.columns for c in INDEPENDENT_VARS):
        st.info("Simulator requires all delay columns: " + ", ".join(INDEPENDENT_VARS))
    else:
        sim_col1, sim_col2 = st.columns(2)

        with sim_col1:
            sched_delta = st.slider("Change in Scheduling Delay (%)", -80, 80, -20)
            prod_delta = st.slider("Change in Production Time (%)", -80, 80, 0)
        with sim_col2:
            mach_delta = st.slider("Change in Machine Delay (%)", -80, 80, -10)
            disp_delta = st.slider("Change in Dispatch Delay (%)", -80, 80, -10)

        sim_df = df_f.copy()

        sim_df["Scheduling_Delay_sim"] = sim_df["Scheduling_Delay_Hrs"] * (1 + sched_delta / 100)
        sim_df["Production_Time_sim"] = sim_df["Production_Time_Hrs"] * (1 + prod_delta / 100)
        sim_df["Machine_Delay_sim"] = sim_df["Machine_Delay_Hrs"] * (1 + mach_delta / 100)
        sim_df["Dispatch_Delay_sim"] = sim_df["Dispatch_Delay_Hrs"] * (1 + disp_delta / 100)

        sim_df["Sim_Total_Lead_Time_Hrs"] = (
            sim_df["Scheduling_Delay_sim"] +
            sim_df["Production_Time_sim"] +
            sim_df["Machine_Delay_sim"] +
            sim_df["Dispatch_Delay_sim"]
        )

        base_avg = df_f["Total_Lead_Time_Hrs"].mean()
        sim_avg = sim_df["Sim_Total_Lead_Time_Hrs"].mean()
        delta_hours = base_avg - sim_avg
        delta_pct = (delta_hours / base_avg * 100) if base_avg and not np.isnan(base_avg) else 0

        ksa, ksb, ksc = st.columns(3)
        ksa.metric("Baseline Avg Lead Time (hrs)", f"{base_avg:.2f}")
        ksb.metric("Simulated Avg Lead Time (hrs)", f"{sim_avg:.2f}")
        ksc.metric("Improvement", f"{delta_hours:.2f} hrs ({delta_pct:.1f}%)")

        st.markdown("Top 10 orders with highest lead time reduction:")
        sim_df["LeadTime_Reduction_Hrs"] = df_f["Total_Lead_Time_Hrs"] - sim_df["Sim_Total_Lead_Time_Hrs"]
        top_sim = sim_df.sort_values("LeadTime_Reduction_Hrs", ascending=False).head(10)[
            ["Order_ID", "Product_Type", "Machine_Type",
             "Total_Lead_Time_Hrs", "Sim_Total_Lead_Time_Hrs", "LeadTime_Reduction_Hrs"]
        ]
        st.dataframe(top_sim, use_container_width=True)
        download_df(top_sim, "scheduling_simulator_top_orders.csv", "Download simulator result")

    # -------------------------------------
    # AUTOMATED INSIGHTS
    # -------------------------------------
    st.subheader("Step 7 — Automated Insights")

    insights_rows = []

    if "Product_Type" in df_f.columns:
        prod_lead = df_f.groupby("Product_Type")[DEPENDENT_VAR].mean().sort_values(ascending=False)
        if len(prod_lead) > 0:
            insights_rows.append({
                "Insight": "Product with highest average lead time",
                "Detail": f"{prod_lead.index[0]} ({prod_lead.iloc[0]:.2f} hrs)"
            })

    if "Machine_Type" in df_f.columns:
        mach_lead = df_f.groupby("Machine_Type")[DEPENDENT_VAR].mean().sort_values(ascending=False)
        if len(mach_lead) > 0:
            insights_rows.append({
                "Insight": "Machine with highest average lead time",
                "Detail": f"{mach_lead.index[0]} ({mach_lead.iloc[0]:.2f} hrs)"
            })

    if avg_disp is not None and avg_disp > 0:
        insights_rows.append({
            "Insight": "Dispatch delay contribution",
            "Detail": f"Average dispatch delay is {avg_disp:.2f} hrs. Tighten dispatch readiness & loading slots."
        })

    if avg_mach is not None and avg_mach > 0:
        insights_rows.append({
            "Insight": "Machine-related delay",
            "Detail": f"Machines add {avg_mach:.2f} hrs on average. Review breakdowns, changeover, and maintenance windows."
        })

    if insights_rows:
        ins_df = pd.DataFrame(insights_rows)
        st.dataframe(ins_df, use_container_width=True)
        download_df(ins_df, "order_to_delivery_insights.csv", "Download insights")
    else:
        st.info("No insights generated for current filter selection.")
