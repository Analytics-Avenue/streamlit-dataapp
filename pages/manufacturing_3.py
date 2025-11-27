# inventory_pileup_shortage_analytics_lab.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# -------------------------
# BASIC PAGE & SIDEBAR HIDING
# -------------------------
hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

st.set_page_config(
    page_title="Inventory Pileup & Shortage Analytics Lab",
    layout="wide"
)

# -------------------------
# LOGO + HEADER
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

st.markdown(
    f"""
<div style="display:flex; align-items:center; margin-bottom:10px;">
    <img src="{logo_url}" width="60" style="margin-right:12px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:34px; font-weight:700;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:34px; font-weight:700;">Advanced Analytics</div>
    </div>
</div>
""",
    unsafe_allow_html=True
)

# -------------------------
# GLOBAL CSS – STANDARD LAB STYLE
# -------------------------
st.markdown(
    """
<style>
* { font-family: "Inter", sans-serif; }

/* Main header */
.big-header {
    font-size: 34px !important;
    font-weight: 700 !important;
    color:#000 !important;
    margin-bottom:8px;
}

/* Section title with underline hover */
.section-title {
    font-size: 22px !important;
    font-weight: 600 !important;
    margin-top:24px;
    margin-bottom:10px;
    color:#000 !important;
    position:relative;
}
.section-title:after {
    content:"";
    position:absolute;
    bottom:-5px;
    left:0;
    height:2px;
    width:0%;
    background:#064b86;
    transition: width 0.35s ease;
}
.section-title:hover:after { width:40%; }

/* Card */
.card {
    background:#ffffff;
    padding:18px 20px;
    border-radius:14px;
    border:1px solid #e3e3e3;
    font-size:15.5px;
    color:#000 !important;
    font-weight:500;
    box-shadow:0 3px 14px rgba(0,0,0,0.08);
    transition: all 0.25s ease;
}
.card:hover {
    transform:translateY(-3px);
    box-shadow:0 10px 24px rgba(6,75,134,0.18);
    border-color:#064b86;
}

/* KPI cards – blue text */
.kpi {
    background:#ffffff;
    padding:18px;
    border-radius:14px;
    border:1px solid #e2e2e2;
    font-size:18px !important;
    font-weight:600 !important;
    text-align:center;
    color:#064b86 !important;
    box-shadow:0 3px 12px rgba(0,0,0,0.07);
    transition:0.25s ease;
}
.kpi:hover {
    transform:translateY(-4px);
    box-shadow:0 13px 26px rgba(6,75,134,0.22);
    border-color:#064b86;
}
.kpi-value {
    display:block;
    font-size:20px;
    font-weight:700;
    margin-top:6px;
}

/* Variable boxes – blue text */
.variable-box {
    padding:12px 14px;
    border-radius:10px;
    background:#ffffff;
    border:1px solid #e5e5e5;
    box-shadow:0 2px 10px rgba(0,0,0,0.08);
    transition:0.25s ease;
    font-size:15px !important;
    font-weight:500 !important;
    color:#064b86 !important;
    margin-bottom:8px;
}
.variable-box:hover {
    transform:translateY(-3px);
    box-shadow:0 10px 20px rgba(6,75,134,0.18);
    border-color:#064b86;
}

/* Dataframe table (required fields) */
.required-table th {
    background:#ffffff !important;
    color:#000 !important;
    font-size:16px !important;
    border-bottom:2px solid #000 !important;
}
.required-table td {
    color:#000 !important;
    font-size:14.5px !important;
    padding:7px !important;
    border-bottom:1px solid #dcdcdc !important;
}
.required-table tr:hover td {
    background:#f8f8f8 !important;
}

/* Buttons */
.stButton>button,
.stDownloadButton>button {
    background:#064b86 !important;
    color:white !important;
    border:none;
    padding:8px 18px;
    border-radius:8px !important;
    font-size:14.5px !important;
    font-weight:600 !important;
    transition:0.25s ease;
}
.stButton>button:hover,
.stDownloadButton>button:hover {
    transform:translateY(-2px);
    background:#0a6eb3 !important;
}

/* Page fade */
.block-container { animation: fadeIn 0.4s ease; }
@keyframes fadeIn { from {opacity:0; transform:translateY(8px);} to {opacity:1; transform:translateY(0);} }
</style>
""",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='big-header'>Inventory Pileup & Shortage Analytics Lab</div>",
    unsafe_allow_html=True
)
st.markdown(
    "Track demand vs supply, predict stock risks, and simulate inventory strategies to balance working capital with service levels.",
    unsafe_allow_html=True
)

# -------------------------
# REQUIRED COLUMNS & DATA DICTIONARY
# -------------------------
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

REQUIRED_DICT = {
    "SKU": "Item code / material ID / product identifier.",
    "Date": "Transaction / snapshot date for the inventory and demand record.",
    "Daily_Demand": "Actual demand (orders or consumption) on the given Date.",
    "Predicted_Demand": "Forecasted demand for that SKU on the given Date.",
    "Forecast_Error": "Predicted_Demand − Daily_Demand or similar forecast error metric.",
    "Production_Qty": "Quantity produced / manufactured for the SKU on the Date.",
    "Production_Delay_Hrs": "Delay in production vs plan, expressed in hours.",
    "Procurement_Qty": "Quantity received from procurement / purchase orders.",
    "Procurement_Delay_Hrs": "Delay in inbound supply vs expected time, in hours.",
    "Inventory_Level": "On-hand inventory quantity at end of day / snapshot.",
    "Safety_Stock": "Planned safety stock threshold for the SKU.",
    "Stock_Turnover": "Inventory turnover ratio (e.g., Annual COGS / Avg inventory).",
    "Lead_Time_Days": "Supply lead time from replenishment trigger to stock available.",
    "Backorder_Qty": "Unfulfilled demand (backorders) for the SKU on that Date.",
    "Wastage_Qty": "Discarded / expired / damaged quantity on the Date.",
    "Shortage_Flag": "1 if inventory fell below safety stock (or stockout) on the Date; else 0.",
    "Pileup_Flag": "1 if inventory exceeded upper threshold (high excess); else 0.",
}

INDEPENDENT_VARS = [
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
]

DEPENDENT_VARS = [
    "Lead_Time_Days",
    "Shortage_Flag",
    "Pileup_Flag",
]

DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/manufacturing/inventory_data.csv"

# -------------------------
# HELPERS
# -------------------------
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


def download_df(df, filename, label="Download CSV", key=None):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode())
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv", key=key)


def to_numeric_safe(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# -------------------------
# SESSION STATE
# -------------------------
if "automl_model" not in st.session_state:
    st.session_state.automl_model = None
if "automl_scaler" not in st.session_state:
    st.session_state.automl_scaler = None
if "automl_features" not in st.session_state:
    st.session_state.automl_features = None
if "base_df" not in st.session_state:
    st.session_state.base_df = None

# -------------------------
# TABS
# -------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Data Dictionary", "Application"])

# =======================================================
# TAB 1: OVERVIEW
# =======================================================
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
        This lab monitors SKU-level demand, production, procurement, and inventory to prevent both
        <b>stockouts</b> and <b>excess pileups</b>. It brings together operational and planning signals
        into one decision workspace.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="card">
            • Track daily <b>demand vs production vs procurement</b><br>
            • Monitor <b>inventory vs safety stock</b> and breach patterns<br>
            • Analyze <b>forecast error, lead time, stock turnover</b><br>
            • Run <b>AutoML</b> on lead-time drivers<br>
            • Simulate inventory strategies via parameter tweaks
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="card">
            • Reduce <b>cash locked</b> in slow-moving inventory<br>
            • Avoid <b>production stoppages</b> due to shortages<br>
            • Improve <b>service levels & OTIF</b><br>
            • Support <b>S&OP, safety stock</b> and <b>reorder logic</b><br>
            • Give finance & supply chain a shared view of risk
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">High-level KPIs (conceptual)</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown("<div class='kpi'>Active SKUs</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Avg Inventory Level</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Shortage Incidents</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Pileup Incidents</div>", unsafe_allow_html=True)
    k5.markdown("<div class='kpi'>Service Level</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Who Should Use This</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
        Supply chain managers, production planners, inventory controllers, plant heads, and finance partners
        who want a <b>SKU-level, time-series view</b> of inventory risk and working capital.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =======================================================
# TAB 2: DATA DICTIONARY & VARIABLES
# =======================================================
with tab2:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)

    req_df = pd.DataFrame(
        [{"Attribute": k, "Description": v} for k, v in REQUIRED_DICT.items()]
    )

    st.markdown(
        """
        <style>
        .required-wrap table { width:100% !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="required-wrap">', unsafe_allow_html=True)
    st.dataframe(
        req_df.style.set_table_attributes('class="required-table"'),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        for v in INDEPENDENT_VARS:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        for v in DEPENDENT_VARS:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# =======================================================
# TAB 3: APPLICATION
# =======================================================
with tab3:
    st.markdown('<div class="section-title">Step 1: Load dataset</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Dataset option:",
        ["Default dataset (GitHub URL)", "Upload CSV", "Upload CSV + Manual Column Mapping"],
        horizontal=True
    )

    df = None

    if mode == "Default dataset (GitHub URL)":
        try:
            df = read_csv_safe(DEFAULT_URL)
            st.success("Default dataset loaded successfully from GitHub URL.")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load default dataset. Error: {e}")
            st.stop()

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

    # Keep in session for reuse
    st.session_state.base_df = df.copy()

    # ---------- Cleanup ----------
    df = df.copy()
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
    df = to_numeric_safe(df, num_cols)

    for flag in ["Shortage_Flag", "Pileup_Flag"]:
        if flag in df.columns:
            df[flag] = df[flag].fillna(0).astype(int)

    if "Date" in df.columns:
        df = df[df["Date"].notna()]

    # ---------------------------------------------------------
    # FILTERS
    # ---------------------------------------------------------
    st.markdown('<div class="section-title">Step 2: Filters & Preview</div>', unsafe_allow_html=True)

    f1, f2 = st.columns([2, 2])

    sku_list = sorted(df["SKU"].dropna().unique()) if "SKU" in df.columns else []
    with f1:
        sel_skus = st.multiselect(
            "SKU",
            options=sku_list,
            default=sku_list[:10] if sku_list else []
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

    st.markdown(
        f"""
        <div class="card" style="margin-top:10px; margin-bottom:8px;">
        <b>Filtered Rows:</b> {len(df_f)} of {len(df)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Filtered data (first 10 rows)")
    st.dataframe(df_f.head(10), use_container_width=True)
    download_df(df_f, "inventory_filtered_data.csv", label="Download filtered data", key="dl_filtered")

    if df_f.empty:
        st.warning("Filtered dataset is empty. Adjust filters above.")
        st.stop()

    # ---------------------------------------------------------
    # KPIs (Dynamic)
    # ---------------------------------------------------------
    st.markdown('<div class="section-title">Dynamic Key Metrics</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)

    active_skus = df_f["SKU"].nunique() if "SKU" in df_f.columns else 0
    avg_inv = df_f["Inventory_Level"].mean() if "Inventory_Level" in df_f.columns else np.nan
    shortage_events = df_f["Shortage_Flag"].sum() if "Shortage_Flag" in df_f.columns else 0
    pileup_events = df_f["Pileup_Flag"].sum() if "Pileup_Flag" in df_f.columns else 0
    service_level = 1 - (
        df_f["Backorder_Qty"].gt(0).mean() if "Backorder_Qty" in df_f.columns else 0
    )
    # Prepare safe display values
    avg_inv_disp = f"{avg_inv:.1f}" if avg_inv is not None and not np.isnan(avg_inv) else "N/A"
    service_level_disp = f"{service_level*100:.1f}%" if service_level is not None else "N/A"
    
    k1.markdown(f"<div class='kpi'>Active SKUs<span class='kpi-value'>{int(active_skus)}</span></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'>Avg Inventory<span class='kpi-value'>{avg_inv_disp}</span></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'>Shortage Events<span class='kpi-value'>{int(shortage_events)}</span></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'>Pileup Events<span class='kpi-value'>{int(pileup_events)}</span></div>", unsafe_allow_html=True)
    k5.markdown(f"<div class='kpi'>Service Level<span class='kpi-value'>{service_level_disp}</span></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # CORE EDA CHARTS
    # ---------------------------------------------------------
    st.markdown('<div class="section-title">Exploratory Analysis — Core Views</div>', unsafe_allow_html=True)

    # 1. Inventory level over time by SKU
    if "Date" in df_f.columns and "Inventory_Level" in df_f.columns:
        st.markdown("##### 1. Inventory Level over Time (by SKU)")
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
        st.markdown("##### 2. Daily Demand vs Production")
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
        st.markdown("##### 3. Safety Stock Breach Heatmap")
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
        if not heat.empty:
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
        st.markdown("##### 4. Shortage vs Pileup Events")
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
        st.markdown("##### 5. Lead Time Distribution")
        fig5 = px.histogram(df_f, x="Lead_Time_Days", nbins=30)
        st.plotly_chart(fig5, use_container_width=True)

    # ---------------------------------------------------------
    # ADVANCED: Inventory Moving Average & Naive Forecast
    # ---------------------------------------------------------
    st.markdown('<div class="section-title">Advanced View — Inventory Forecast</div>', unsafe_allow_html=True)

    if "Date" in df_f.columns and "Inventory_Level" in df_f.columns:
        inv_ts = (
            df_f.groupby("Date")["Inventory_Level"]
            .mean()
            .reset_index()
            .sort_values("Date")
        )
        inv_ts["MA_7"] = inv_ts["Inventory_Level"].rolling(
            window=7, min_periods=1
        ).mean()
        inv_ts["MA_30"] = inv_ts["Inventory_Level"].rolling(
            window=30, min_periods=1
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

        fig_inv = go.Figure()
        fig_inv.add_trace(
            go.Scatter(
                x=inv_ts["Date"], y=inv_ts["Inventory_Level"], name="Actual"
            )
        )
        fig_inv.add_trace(
            go.Scatter(
                x=inv_ts["Date"], y=inv_ts["MA_7"], name="7-day MA"
            )
        )
        fig_inv.add_trace(
            go.Scatter(
                x=inv_ts["Date"], y=inv_ts["MA_30"], name="30-day MA"
            )
        )
        if not future_df.empty:
            fig_inv.add_trace(
                go.Scatter(
                    x=future_df["Date"],
                    y=future_df["Inventory_Level"],
                    name="Naive Forecast",
                    line=dict(dash="dash"),
                )
            )
        fig_inv.update_layout(xaxis_title="Date", yaxis_title="Avg Inventory")
        st.plotly_chart(fig_inv, use_container_width=True)

    # ---------------------------------------------------------
    # AutoML: Lead Time Regression
    # ---------------------------------------------------------
    st.markdown('<div class="section-title">AutoML — Lead Time Prediction</div>', unsafe_allow_html=True)

    target_col = "Lead_Time_Days"
    candidate_features = [
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
    usable_features = [c for c in candidate_features if c in df.columns]

    if (target_col in df.columns) and (len(df) >= 80) and (len(usable_features) >= 3):
        with st.expander("Run AutoML for Lead_Time_Days", expanded=False):
            st.write(f"Using features: {', '.join(usable_features)}")

            ml_df = df.dropna(subset=[target_col]).copy()
            X = ml_df[usable_features].fillna(0)
            y = ml_df[target_col].astype(float).values

            scaler = StandardScaler()
            Xs = scaler.fit_transform(X.values)

            X_train, X_test, y_train, y_test = train_test_split(
                Xs, y, test_size=0.2, random_state=42
            )

            models = {
                "LinearRegression": LinearRegression(),
                "RandomForestRegressor": RandomForestRegressor(
                    n_estimators=200, random_state=42
                ),
                "GradientBoostingRegressor": GradientBoostingRegressor(
                    random_state=42
                ),
            }

            results = []
            best_model_name = None
            best_rmse = np.inf
            best_model = None

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                mae = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                results.append(
                    {
                        "Model": name,
                        "RMSE": rmse,
                        "MAE": mae,
                        "R2": r2,
                    }
                )
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model_name = name
                    best_model = model

            res_df = pd.DataFrame(results).sort_values("RMSE")
            st.write("AutoML model comparison:")
            st.dataframe(res_df, use_container_width=True)

            if best_model is not None:
                st.success(f"Best model: {best_model_name} (RMSE={best_rmse:.2f})")
                st.session_state.automl_model = best_model
                st.session_state.automl_scaler = scaler
                st.session_state.automl_features = usable_features

                # Show sample predictions on filtered data if same structure
                try:
                    df_f_ml = df_f.dropna(subset=[target_col]).copy()
                    if not df_f_ml.empty:
                        X_f = df_f_ml[usable_features].fillna(0)
                        X_f_scaled = scaler.transform(X_f.values)
                        preds_f = best_model.predict(X_f_scaled)
                        sample = df_f_ml.copy()
                        sample["Predicted_Lead_Time_Days"] = preds_f
                        st.write("Sample predictions on filtered dataset:")
                        st.dataframe(sample[[ "SKU", "Date", "Lead_Time_Days", "Predicted_Lead_Time_Days"]].head(20), use_container_width=True)
                        download_df(sample, "automl_lead_time_predictions.csv", label="Download AutoML predictions", key="dl_automl")
                except Exception as e:
                    st.warning(f"Could not score filtered dataset: {e}")
    else:
        st.info(
            "AutoML requires Lead_Time_Days, at least 3 numeric feature columns and ~80+ rows on full dataset."
        )

    # ---------------------------------------------------------
    # Inventory Strategy Simulator (What-if)
    # ---------------------------------------------------------
    st.markdown('<div class="section-title">Inventory Strategy Simulator (What-if)</div>', unsafe_allow_html=True)

    if "Inventory_Level" in df_f.columns and "Daily_Demand" in df_f.columns and "Production_Qty" in df_f.columns and "Procurement_Qty" in df_f.columns and "Safety_Stock" in df_f.columns:
        sim_c1, sim_c2, sim_c3, sim_c4, sim_c5 = st.columns(5)
        with sim_c1:
            dem_change = st.slider("Demand change %", -50, 50, 0)
        with sim_c2:
            prod_change = st.slider("Production change %", -50, 50, 0)
        with sim_c3:
            proc_change = st.slider("Procurement change %", -50, 50, 0)
        with sim_c4:
            ss_change = st.slider("Safety Stock change %", -50, 50, 0)
        with sim_c5:
            lt_change = st.slider("Lead Time change % (for info)", -50, 50, 0)

        if st.button("Run simulation"):
            sim = df_f.copy()

            dem_factor = 1 + dem_change / 100.0
            prod_factor = 1 + prod_change / 100.0
            proc_factor = 1 + proc_change / 100.0
            ss_factor = 1 + ss_change / 100.0

            sim["Sim_Demand"] = sim["Daily_Demand"] * dem_factor
            sim["Sim_Production"] = sim["Production_Qty"] * prod_factor
            sim["Sim_Procurement"] = sim["Procurement_Qty"] * proc_factor
            sim["Sim_Safety_Stock"] = sim["Safety_Stock"] * ss_factor

            # simple 1-period balance approximation
            sim["Sim_Inventory"] = (
                sim["Inventory_Level"]
                + sim["Sim_Production"]
                + sim["Sim_Procurement"]
                - sim["Sim_Demand"]
            )

            # new shortage / pileup flags
            sim["Sim_Shortage_Flag"] = np.where(
                sim["Sim_Inventory"] < sim["Sim_Safety_Stock"], 1, 0
            )
            sim["Sim_Pileup_Flag"] = np.where(
                sim["Sim_Inventory"] > sim["Sim_Safety_Stock"] * 2, 1, 0
            )

            # summary metrics
            shortage_rate_new = sim["Sim_Shortage_Flag"].mean()
            pileup_rate_new = sim["Sim_Pileup_Flag"].mean()
            avg_inv_new = sim["Sim_Inventory"].mean()

            # baseline for comparison
            base_short_rate = (
                df_f["Shortage_Flag"].mean() if "Shortage_Flag" in df_f.columns else np.nan
            )
            base_pile_rate = (
                df_f["Pileup_Flag"].mean() if "Pileup_Flag" in df_f.columns else np.nan
            )
            base_inv = df_f["Inventory_Level"].mean() if "Inventory_Level" in df_f.columns else np.nan

            st.markdown("#### Simulation summary")
            s1, s2, s3 = st.columns(3)
            s1.metric(
                "Shortage rate (new)",
                f"{shortage_rate_new*100:.1f}%",
                f"{(shortage_rate_new - (base_short_rate or 0))*100:.1f}%" if not np.isnan(base_short_rate) else None,
            )
            s2.metric(
                "Pileup rate (new)",
                f"{pileup_rate_new*100:.1f}%",
                f"{(pileup_rate_new - (base_pile_rate or 0))*100:.1f}%" if not np.isnan(base_pile_rate) else None,
            )
            s3.metric(
                "Avg inventory (new)",
                f"{avg_inv_new:.1f}",
                f"{avg_inv_new - (base_inv or 0):.1f}" if not np.isnan(base_inv) else None,
            )

            sku_view = (
                sim.groupby("SKU")
                .agg(
                    Sim_Inventory=("Sim_Inventory", "mean"),
                    Sim_Shortage_Rate=("Sim_Shortage_Flag", "mean"),
                    Sim_Pileup_Rate=("Sim_Pileup_Flag", "mean"),
                )
                .reset_index()
            )

            st.markdown("#### SKU-level simulation result")
            st.dataframe(sku_view.head(30), use_container_width=True)
            download_df(sku_view, "inventory_simulation_sku_view.csv", label="Download simulation results", key="dl_sim")

    else:
        st.info("Simulator requires Inventory_Level, Daily_Demand, Production_Qty, Procurement_Qty, Safety_Stock columns in filtered data.")

    # ---------------------------------------------------------
    # Automated Insights
    # ---------------------------------------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)

    insights = []

    # High shortage SKUs
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

    # High pileup SKUs
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

    # Long lead-time SKUs
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

    # Unstable forecast SKUs
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

    # AutoML-based insight if model exists
    if (
        st.session_state.automl_model is not None
        and st.session_state.automl_scaler is not None
        and st.session_state.automl_features is not None
        and "Lead_Time_Days" in df_f.columns
    ):
        try:
            ml_df_local = df_f.dropna(subset=["Lead_Time_Days"]).copy()
            if not ml_df_local.empty:
                X_local = ml_df_local[st.session_state.automl_features].fillna(0)
                X_local_scaled = st.session_state.automl_scaler.transform(X_local.values)
                preds_local = st.session_state.automl_model.predict(X_local_scaled)
                ml_df_local["Predicted_Lead_Time_Days"] = preds_local
                top_risk = (
                    ml_df_local.groupby("SKU")["Predicted_Lead_Time_Days"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(5)
                )
                for sku, val in top_risk.items():
                    insights.append(
                        {
                            "Insight_Type": "ML: High Predicted Lead Time",
                            "SKU": sku,
                            "Metric": "Predicted Lead Time (days)",
                            "Value": round(float(val), 2),
                        }
                    )
        except Exception:
            # If something breaks, just skip the ML insight silently
            pass

    insights_df = pd.DataFrame(insights)
    if insights_df.empty:
        st.info("No automated insights could be generated for the current filter.")
    else:
        st.dataframe(insights_df, use_container_width=True)
        download_df(
            insights_df,
            "inventory_automated_insights.csv",
            label="Download insights",
            key="dl_insights",
        )
