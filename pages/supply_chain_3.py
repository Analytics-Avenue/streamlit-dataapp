import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Warehouse Operations Analytics", layout="wide")

# ---------------------------------------------------------
# HIDE SIDEBAR
# ---------------------------------------------------------
hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# ---------------------------------------------------------
# GLOBAL CSS (cards, KPIs, layout)
# ---------------------------------------------------------
st.markdown("""
<style>
.card {
    padding: 18px;
    border-radius: 12px;
    background: #ffffff;
    border: 1px solid #e4e8f0;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    box-shadow: 0 1px 4px rgba(2,6,23,0.04);
    text-align: left;
}
.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 10px 30px rgba(6,75,134,0.18);
    border-color: #064b86;
}
.kpi {
    padding: 24px;
    border-radius: 12px;
    background: #ffffff;
    border: 1px solid #e2e8f5;
    font-size: 20px;
    font-weight: 700;
    text-align: center;
    color: #064b86;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.kpi:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 26px rgba(6,75,134,0.22);
}
.small {
    font-size: 13px;
    color: #666;
    line-height: 1.3;
}
.left-text { text-align:left !important; }
.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #064b86;
    margin-top: 12px;
    margin-bottom: 4px;
}
.subtle {
    font-size: 13px;
    color: #8a8f9b;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def download_csv(df, name, label="Download CSV"):
    buffer = BytesIO()
    buffer.write(df.to_csv(index=False).encode("utf-8"))
    buffer.seek(0)
    st.download_button(label, buffer, file_name=name, mime="text/csv")


def read_csv_safe(src):
    df = pd.read_csv(src)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------
# HEADER: LOGO + TITLE
# ---------------------------------------------------------
logo = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display:flex; align-items:center; margin-bottom:6px;">
    <img src="{logo}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:32px; font-weight:bold;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:32px; font-weight:bold;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:left; margin-bottom:12px;">
  <h1 style="margin:0; padding:0; font-size:26px;">Warehouse Operations Analytics</h1>
  <p style="margin:4px 0 0 0; color:#555;">
    Diagnose picking, slotting, congestion and picker productivity to squeeze every ounce of efficiency out of your warehouse.
  </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# REQUIRED COLUMNS & DATA DICTIONARY
# ---------------------------------------------------------
REQUIRED_COLS = [
    "Order_ID", "Warehouse", "Zone", "Aisle", "Bin", "Order_Timestamp",
    "Shift", "Picker_ID", "SKU", "SKU_Weight_KG", "SKU_Cube_M3",
    "SKU_Class", "Pick_Qty", "Travel_Distance_M", "Travel_Time_Sec",
    "Pick_Time_Sec", "Pack_Time_Sec", "Heatmap_Level", "Slotting_Score",
    "Equipment_Type", "Picker_Productivity_Items_Hour", "Congestion_Factor",
    "Delay_Reason", "Delay_Minutes"
]

DATA_DICTIONARY = [
    ("Order_ID", "Unique identifier of the customer order", "Meta"),
    ("Warehouse", "Warehouse / FC identifier", "Independent"),
    ("Zone", "Warehouse zone / segment", "Independent"),
    ("Aisle", "Aisle where items are stored", "Independent"),
    ("Bin", "Bin / location code for SKU", "Independent"),
    ("Order_Timestamp", "Timestamp when order was created / released", "Independent"),
    ("Shift", "Shift (e.g. Morning, Afternoon, Night)", "Independent"),
    ("Picker_ID", "ID of the picker who handled the order", "Independent"),
    ("SKU", "Stock-keeping unit code", "Independent"),
    ("SKU_Weight_KG", "Weight of the SKU in kg", "Independent"),
    ("SKU_Cube_M3", "Cubic volume of SKU in cubic meters", "Independent"),
    ("SKU_Class", "ABC / velocity class of SKU", "Independent"),
    ("Pick_Qty", "Quantity of the SKU picked", "Independent"),
    ("Travel_Distance_M", "Distance traveled by picker (meters)", "Independent"),
    ("Travel_Time_Sec", "Time spent travelling (seconds)", "Independent"),
    ("Pick_Time_Sec", "Time taken to pick items (seconds)", "Dependent"),
    ("Pack_Time_Sec", "Time taken to pack items (seconds)", "Dependent / Derived"),
    ("Heatmap_Level", "Zone congestion / heat level (0–5)", "Independent / Feature"),
    ("Slotting_Score", "Score for how optimal SKU placement is", "Independent / Feature"),
    ("Equipment_Type", "Type of material handling equipment used", "Independent"),
    ("Picker_Productivity_Items_Hour", "Items per hour handled by picker", "Dependent"),
    ("Congestion_Factor", "Overall congestion factor derived from movements", "Independent / Feature"),
    ("Delay_Reason", "Categorical reason for delay", "Independent"),
    ("Delay_Minutes", "Total delay minutes attributed to the order", "Dependent")
]

INDEPENDENT_VARS = [
    "Warehouse", "Zone", "Aisle", "Bin", "Order_Timestamp", "Shift",
    "Picker_ID", "SKU", "SKU_Weight_KG", "SKU_Cube_M3", "SKU_Class",
    "Pick_Qty", "Travel_Distance_M", "Travel_Time_Sec", "Heatmap_Level",
    "Slotting_Score", "Equipment_Type", "Congestion_Factor", "Delay_Reason"
]

DEPENDENT_VARS = [
    "Pick_Time_Sec",
    "Delay_Minutes",
    "Picker_Productivity_Items_Hour",
    "Pack_Time_Sec"
]

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab_overview, tab_dict, tab_app, tab_play = st.tabs(
    ["Overview", "Data Dictionary", "Application", "Actionable Playbooks"]
)

# ---------------------------------------------------------
# OVERVIEW TAB
# ---------------------------------------------------------
with tab_overview:
    st.markdown("""
    <div class='card'>
      <b>Purpose</b>: Analyse picking efficiency, slotting, congestion, picker productivity and equipment usage. 
      Use this to redesign layout, rebalance workload, and cut cost per order without breaking your ops team.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-title'>Capabilities</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='card'>
          • Warehouse heatmaps & congestion analysis<br>
          • Pick-path & travel distance efficiency<br>
          • Slotting score & SKU-class optimisation<br>
          • Picker productivity benchmarking<br>
          • Delay root-cause analytics<br>
          • Equipment utilisation and mix analysis
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='section-title'>Business impact</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='card'>
          • Faster order fulfilment & higher throughput<br>
          • Lower labour + operational cost per order<br>
          • Reduced congestion & picker fatigue<br>
          • Better layout & slotting decisions<br>
          • More predictable performance during peaks
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Headline KPIs</div>", unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown("<div class='kpi'>Total Orders</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Avg Pick Time</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Avg Travel Time</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Heatmap Intensity</div>", unsafe_allow_html=True)
    k5.markdown("<div class='kpi'>Productivity Score</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Who should use this</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
      • Warehouse managers & DC heads<br>
      • Ops / logistics leaders<br>
      • Industrial engineering / IE teams<br>
      • Process excellence & BI teams
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# APPLICATION TAB
# ---------------------------------------------------------
with tab_app:
    st.header("Application")
    st.markdown("Choose how you want to load the dataset.")

    mode = st.radio(
        "Dataset option:",
        ["Default dataset (GitHub URL)", "Upload CSV", "Upload CSV + Column Mapping"],
        horizontal=True
    )

    df = None

    if mode == "Default dataset (GitHub URL)":
        try:
            DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/supply_chain/warehouse_operations_analytics.csv"
            df = read_csv_safe(DEFAULT_URL)
            st.success("Default dataset loaded from GitHub.")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load default dataset: {e}")

    elif mode == "Upload CSV":
        st.markdown("#### Sample CSV (structure reference)")
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/supply_chain/warehouse_operations_analytics.csv"
        try:
            sample_df = pd.read_csv(URL).head(5)
            sample_csv = sample_df.to_csv(index=False)
            st.download_button("Download Sample CSV", sample_csv, "sample_warehouse_dataset.csv", "text/csv")
        except Exception as e:
            st.info(f"Sample CSV unavailable: {e}")

        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = read_csv_safe(file)

    else:  # Upload + mapping
        map_file = st.file_uploader("Upload CSV for column mapping", type=["csv"])
        if map_file:
            raw = read_csv_safe(map_file)
            st.markdown("Preview (first 5 rows):")
            st.dataframe(raw.head(), use_container_width=True)

            st.markdown("Map your columns to the required schema:")
            mapping = {}
            cols = list(raw.columns)
            for c in REQUIRED_COLS:
                mapping[c] = st.selectbox(
                    f"Map → {c}",
                    ["-- Select --"] + cols,
                    key=f"map_{c}"
                )
            if st.button("Apply Mapping"):
                missing = [k for k, v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Missing mappings for: " + ", ".join(missing))
                    st.stop()
                rename = {mapping[k]: k for k in mapping}
                df = raw.rename(columns=rename)
                st.success("Mapping applied.")
                st.dataframe(df.head(), use_container_width=True)

    if df is None:
        st.stop()

    # ---------------------------------------------------------
    # BASIC CLEANING
    # ---------------------------------------------------------
    df.columns = [str(c).strip() for c in df.columns]

    if "Order_Timestamp" not in df.columns:
        st.error("Order_Timestamp column missing. Ensure dataset or mapping is correct.")
        st.stop()

    df["Order_Timestamp"] = pd.to_datetime(df["Order_Timestamp"], errors="coerce")
    if df["Order_Timestamp"].isna().all():
        st.error("Order_Timestamp has no valid datetime values.")
        st.stop()

    numeric_cols = [
        "SKU_Weight_KG", "SKU_Cube_M3", "Pick_Qty", "Travel_Distance_M",
        "Travel_Time_Sec", "Pick_Time_Sec", "Pack_Time_Sec", "Heatmap_Level",
        "Slotting_Score", "Picker_Productivity_Items_Hour", "Congestion_Factor",
        "Delay_Minutes"
    ]
    df = safe_numeric(df, numeric_cols)

    # ---------------------------------------------------------
    # DATE FILTER
    # ---------------------------------------------------------
    min_d = df["Order_Timestamp"].min().to_pydatetime()
    max_d = df["Order_Timestamp"].max().to_pydatetime()

    st.markdown("### Date Filter")
    date_range = st.slider(
        "Select date range:",
        min_value=min_d,
        max_value=max_d,
        value=(min_d, max_d),
        format="YYYY-MM-DD"
    )

    df_f = df[(df["Order_Timestamp"] >= date_range[0]) &
              (df["Order_Timestamp"] <= date_range[1])]

    # ---------------------------------------------------------
    # EXTRA FILTERS
    # ---------------------------------------------------------
    st.markdown("### Filters")
    c1, c2, c3 = st.columns(3)

    wh_opts = sorted(df_f["Warehouse"].dropna().unique().tolist()) if "Warehouse" in df_f.columns else []
    sh_opts = sorted(df_f["Shift"].dropna().unique().tolist()) if "Shift" in df_f.columns else []
    pk_opts = sorted(df_f["Picker_ID"].dropna().unique().tolist()) if "Picker_ID" in df_f.columns else []

    wh = c1.multiselect("Warehouse", wh_opts, default=wh_opts)
    shift = c2.multiselect("Shift", sh_opts, default=sh_opts)
    picker = c3.multiselect("Picker ID", pk_opts, default=pk_opts)

    if wh and "Warehouse" in df_f.columns:
        df_f = df_f[df_f["Warehouse"].isin(wh)]
    if shift and "Shift" in df_f.columns:
        df_f = df_f[df_f["Shift"].isin(shift)]
    if picker and "Picker_ID" in df_f.columns:
        df_f = df_f[df_f["Picker_ID"].isin(picker)]

    st.markdown("### Filtered data preview")
    st.dataframe(df_f.head(10), use_container_width=True)
    download_csv(df_f, "filtered_warehouse_data.csv", label="Download filtered data")

    # ---------------------------------------------------------
    # DYNAMIC KPIs
    # ---------------------------------------------------------
    st.markdown("### KPIs (dynamic values)")
    k1, k2, k3, k4, k5 = st.columns(5)

    total_orders = int(df_f["Order_ID"].nunique()) if "Order_ID" in df_f.columns else len(df_f)
    avg_pick_time = float(df_f["Pick_Time_Sec"].mean()) if "Pick_Time_Sec" in df_f.columns else np.nan
    avg_travel_time = float(df_f["Travel_Time_Sec"].mean()) if "Travel_Time_Sec" in df_f.columns else np.nan
    avg_heatmap = float(df_f["Heatmap_Level"].mean()) if "Heatmap_Level" in df_f.columns else np.nan
    avg_prod = float(df_f["Picker_Productivity_Items_Hour"].mean()) if "Picker_Productivity_Items_Hour" in df_f.columns else np.nan

    k1.metric("Orders", total_orders)
    k2.metric("Avg Pick Time (sec)", f"{avg_pick_time:.1f}" if not np.isnan(avg_pick_time) else "N/A")
    k3.metric("Avg Travel Time (sec)", f"{avg_travel_time:.1f}" if not np.isnan(avg_travel_time) else "N/A")
    k4.metric("Avg Heatmap Level", f"{avg_heatmap:.2f}" if not np.isnan(avg_heatmap) else "N/A")
    k5.metric("Avg Productivity (items/hr)", f"{avg_prod:.1f}" if not np.isnan(avg_prod) else "N/A")

    # ---------------------------------------------------------
    # EDA SECTION
    # ---------------------------------------------------------
    st.markdown("## Exploratory Data Analysis")

    def chart(fig):
        try:
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        except Exception:
            pass
        st.plotly_chart(fig, use_container_width=True)

    # 1: Pick Qty Distribution
    if "Pick_Qty" in df_f.columns:
        with st.expander("Pick Qty Distribution — Purpose & Quick Tip", expanded=True):
            st.write("Purpose: See spread of order picking quantities and identify bulk orders.")
            st.write("Quick Tip: Long tail = irregular SKU demand, watch labour planning.")
        chart(px.histogram(df_f, x="Pick_Qty", nbins=40, title="Pick Quantity Distribution"))

    # 2: Travel Distance Over Time
    if "Travel_Distance_M" in df_f.columns:
        with st.expander("Travel Distance Over Time — Purpose & Quick Tip"):
            st.write("Purpose: Monitor how picker travel distance behaves over time / shifts.")
            st.write("Quick Tip: Spikes often link to bad slotting or batch picking rules.")
        chart(px.scatter(df_f, x="Order_Timestamp", y="Travel_Distance_M",
                         color="Warehouse" if "Warehouse" in df_f.columns else None,
                         title="Travel Distance Over Time"))

    # 3: Heatmap Activity by Zone
    if "Zone" in df_f.columns and "Heatmap_Level" in df_f.columns:
        with st.expander("Zone Heatmap Activity — Purpose & Quick Tip"):
            st.write("Purpose: Identify high-activity or congested warehouse zones.")
            st.write("Quick Tip: Zones with extremely wide spread are prime layout candidates.")
        chart(px.box(df_f, x="Zone", y="Heatmap_Level", title="Zone-Level Heatmap Activity"))

    # 4: Slotting Score by SKU Class
    if "SKU_Class" in df_f.columns and "Slotting_Score" in df_f.columns:
        with st.expander("Slotting Score by SKU Class — Purpose & Quick Tip"):
            st.write("Purpose: Check if high-velocity SKUs are actually well slotted.")
            st.write("Quick Tip: Class A with low Slotting_Score means wasted walking.")
        chart(px.violin(df_f, x="SKU_Class", y="Slotting_Score",
                        title="Slotting Score Distribution by SKU Class"))

    # 5: Delay Reasons %
    if "Delay_Reason" in df_f.columns:
        with st.expander("Delay Reasons — Purpose & Quick Tip"):
            st.write("Purpose: Understand what’s actually causing delays, not what people feel.")
            st.write("Quick Tip: Work on the biggest slice first, not the loudest complaint.")
        chart(px.pie(df_f, names="Delay_Reason", title="Delay Reasons Distribution"))

    # 6: Travel Time vs Distance
    if "Travel_Distance_M" in df_f.columns and "Travel_Time_Sec" in df_f.columns:
        with st.expander("Travel Time vs Distance — Purpose & Quick Tip"):
            st.write("Purpose: Validate sanity of travel rates and catch interruptions.")
            st.write("Quick Tip: Points far above the main cloud = wasted time.")
        chart(px.scatter(df_f, x="Travel_Distance_M", y="Travel_Time_Sec",
                         title="Travel Time vs Distance"))

    # 7: Equipment usage
    if "Equipment_Type" in df_f.columns:
        with st.expander("Equipment Usage — Purpose & Quick Tip"):
            st.write("Purpose: See distribution of equipment usage (trolleys, forklifts, AMRs, etc.).")
            st.write("Quick Tip: Over-reliance on one type usually bites you during breakdowns.")
        chart(px.bar(df_f, x="Equipment_Type", title="Equipment Usage"))

    # 8: Picker productivity
    if "Picker_Productivity_Items_Hour" in df_f.columns:
        with st.expander("Picker Productivity — Purpose & Quick Tip"):
            st.write("Purpose: Benchmark pickers objectively.")
            st.write("Quick Tip: Use bottom quartile for coaching, not punishment theatre.")
        chart(px.histogram(df_f, x="Picker_Productivity_Items_Hour",
                           title="Picker Productivity Distribution"))

    # 9: Congestion Factor
    if "Congestion_Factor" in df_f.columns and "Warehouse" in df_f.columns:
        with st.expander("Congestion Factor — Purpose & Quick Tip"):
            st.write("Purpose: Compare congestion across warehouses / buildings.")
            st.write("Quick Tip: Chronic high congestion needs layout or wave-planning changes.")
        chart(px.box(df_f, y="Congestion_Factor", color="Warehouse",
                     title="Congestion Factor by Warehouse"))

    # 10: SKU weight distribution
    if "SKU_Weight_KG" in df_f.columns:
        with st.expander("SKU Weight Distribution — Purpose & Quick Tip"):
            st.write("Purpose: Understand handling load profile for ergonomics.")
            st.write("Quick Tip: Very heavy clusters require different picking method or aids.")
        chart(px.histogram(df_f, x="SKU_Weight_KG", title="SKU Weight Distribution"))

    # 11: Pick Time vs Pack Time
    if "Pick_Time_Sec" in df_f.columns and "Pack_Time_Sec" in df_f.columns:
        with st.expander("Pick vs Pack Time — Purpose & Quick Tip"):
            st.write("Purpose: Check whether packing is becoming the real bottleneck.")
            st.write("Quick Tip: For similar pick times, wildly different pack times = station issues.")
        chart(px.scatter(df_f, x="Pick_Time_Sec", y="Pack_Time_Sec",
                         title="Pick vs Pack Time"))

    # 12: Heatmap Level Time-Series
    if "Heatmap_Level" in df_f.columns:
        with st.expander("Heatmap Level Over Time — Purpose & Quick Tip"):
            st.write("Purpose: Track congestion intensity through the day or week.")
            st.write("Quick Tip: Peaks often match cut-off times or inbound putaway clashes.")
        chart(px.line(df_f.sort_values("Order_Timestamp"),
                      x="Order_Timestamp", y="Heatmap_Level",
                      title="Heatmap Level Over Time"))

    # 13: Aisle Congestion Map
    if "Aisle" in df_f.columns and "Heatmap_Level" in df_f.columns:
        with st.expander("Aisle Congestion Map — Purpose & Quick Tip"):
            st.write("Purpose: See which aisles are doing all the suffering.")
            st.write("Quick Tip: Darker = more pain. Consider splitting demand to parallel aisles.")
        chart(px.density_heatmap(df_f, x="Aisle", y="Heatmap_Level",
                                 title="Aisle Congestion Heatmap"))

    # 14: Productivity by Shift
    if "Shift" in df_f.columns and "Picker_Productivity_Items_Hour" in df_f.columns:
        with st.expander("Productivity by Shift — Purpose & Quick Tip"):
            st.write("Purpose: Compare shift-level output controlling for volume.")
            st.write("Quick Tip: Night shift being lower is normal; the question is how much.")
        chart(px.box(df_f, x="Shift", y="Picker_Productivity_Items_Hour",
                     title="Productivity by Shift"))

    # 15: Delay Time Distribution
    if "Delay_Minutes" in df_f.columns:
        with st.expander("Delay Duration Distribution — Purpose & Quick Tip"):
            st.write("Purpose: Understand the tail of delays, not just the average.")
            st.write("Quick Tip: Long tails are where customer complaints come from.")
        chart(px.histogram(df_f, x="Delay_Minutes", nbins=40,
                           title="Delay Duration Distribution"))

    # ---------------------------------------------------------
    # MACHINE LEARNING MODELS
    # ---------------------------------------------------------
    st.markdown("## Machine Learning Models")

    # Model 1: Predict Pick Time (Regression)
    st.markdown("### 1. Predict Pick Time (RandomForestRegressor)")
    if "Pick_Time_Sec" in df_f.columns:
        features = [
            "Pick_Qty", "Travel_Distance_M", "Travel_Time_Sec",
            "SKU_Weight_KG", "SKU_Cube_M3", "Heatmap_Level",
            "Slotting_Score", "Picker_Productivity_Items_Hour"
        ]
        features = [f for f in features if f in df_f.columns]

        model_df = df_f.dropna(subset=["Pick_Time_Sec"])
        model_df = model_df.dropna(subset=features, how="any")

        if len(model_df) >= 50 and len(features) >= 2:
            X = model_df[features]
            y = model_df["Pick_Time_Sec"]

            numeric = features
            preprocessor = ColumnTransformer([
                ("num", StandardScaler(), numeric)
            ])

            model = Pipeline([
                ("prep", preprocessor),
                ("reg", RandomForestRegressor(n_estimators=120, random_state=42))
            ])

            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(Xtr, ytr)
            preds = model.predict(Xte)

            results = pd.DataFrame({
                "Actual_Pick_Time_Sec": yte.values,
                "Predicted_Pick_Time_Sec": preds
            })
            st.dataframe(results.head(20), use_container_width=True)
            download_csv(results, "ml_pick_time_predictions.csv", "Download Pick Time Predictions")
        else:
            st.info("Not enough clean rows / features to train pick-time model.")
    else:
        st.info("Pick_Time_Sec column missing; cannot train pick-time model.")

    # Model 2: Congestion Pattern Clustering
    st.markdown("### 2. Congestion Pattern Clustering (KMeans)")
    cluster_features = [c for c in ["Heatmap_Level", "Congestion_Factor", "Pick_Qty"] if c in df_f.columns]
    if len(cluster_features) >= 2 and len(df_f) >= 40:
        cl_df = df_f.dropna(subset=cluster_features).copy()
        if len(cl_df) >= 40:
            km = KMeans(n_clusters=4, random_state=42)
            cl_df["Cluster"] = km.fit_predict(cl_df[cluster_features])

            st.write("Cluster distribution:")
            st.dataframe(cl_df["Cluster"].value_counts().rename_axis("Cluster").reset_index(name="Count"))

            chart(px.scatter(cl_df, x="Pick_Qty", y="Congestion_Factor",
                             color="Cluster",
                             title="Congestion Clusters (Pick_Qty vs Congestion_Factor)"))
        else:
            st.info("Too few rows with non-null congestion fields for clustering.")
    else:
        st.info("Required fields for congestion clustering missing or insufficient rows.")

    # Model 3: Predict Delay Minutes
    st.markdown("### 3. Predict Delay Minutes (GradientBoostingRegressor)")
    if "Delay_Minutes" in df_f.columns:
        features2 = [c for c in ["Pick_Qty", "Travel_Time_Sec", "Congestion_Factor", "Heatmap_Level"] if c in df_f.columns]
        mdl2 = df_f.dropna(subset=["Delay_Minutes"] + features2)
        if len(mdl2) >= 50 and len(features2) >= 2:
            X2 = mdl2[features2]
            y2 = mdl2["Delay_Minutes"]
            model2 = GradientBoostingRegressor(random_state=42)
            model2.fit(X2, y2)
            preds2 = model2.predict(X2)
            df_delay = pd.DataFrame({"Actual_Delay_Min": y2.values, "Predicted_Delay_Min": preds2})
            st.dataframe(df_delay.head(20), use_container_width=True)
            download_csv(df_delay, "ml_delay_minutes_predictions.csv", "Download Delay Predictions")
        else:
            st.info("Not enough rows/features to train delay-minutes model.")
    else:
        st.info("Delay_Minutes column missing.")

    # Model 4: Predict Picker Productivity
    st.markdown("### 4. Predict Picker Productivity (RandomForestRegressor)")
    if "Picker_Productivity_Items_Hour" in df_f.columns:
        features3 = [c for c in ["Pick_Qty", "Travel_Distance_M", "Heatmap_Level", "SKU_Weight_KG"] if c in df_f.columns]
        mdl3 = df_f.dropna(subset=["Picker_Productivity_Items_Hour"] + features3)
        if len(mdl3) >= 50 and len(features3) >= 2:
            X3 = mdl3[features3]
            y3 = mdl3["Picker_Productivity_Items_Hour"]
            model3 = RandomForestRegressor(random_state=42)
            model3.fit(X3, y3)
            preds3 = model3.predict(X3)
            df_prod = pd.DataFrame({"Actual_Productivity": y3.values, "Predicted_Productivity": preds3})
            st.dataframe(df_prod.head(20), use_container_width=True)
            download_csv(df_prod, "ml_picker_productivity_predictions.csv",
                         "Download Productivity Predictions")
        else:
            st.info("Not enough rows/features to train productivity model.")
    else:
        st.info("Picker_Productivity_Items_Hour column missing.")

    # ---------------------------------------------------------
    # AUTOMATED INSIGHTS
    # ---------------------------------------------------------
    st.markdown("## Automated Insights")

    insights = []

    if "Zone" in df_f.columns and "Congestion_Factor" in df_f.columns:
        worst_zone = df_f.groupby("Zone")["Congestion_Factor"].mean().idxmax()
        insights.append(["Highest Congestion Zone", worst_zone])

    if "Equipment_Type" in df_f.columns and "Pick_Time_Sec" in df_f.columns:
        slow_eq = df_f.groupby("Equipment_Type")["Pick_Time_Sec"].mean().idxmax()
        insights.append(["Slowest Equipment Type (avg pick time)", slow_eq])

    if "Delay_Reason" in df_f.columns:
        top_delay = df_f["Delay_Reason"].value_counts().idxmax()
        insights.append(["Most Common Delay Reason", top_delay])

    if "Picker_ID" in df_f.columns and "Picker_Productivity_Items_Hour" in df_f.columns:
        low_picker = df_f.groupby("Picker_ID")["Picker_Productivity_Items_Hour"].mean().idxmin()
        insights.append(["Lowest Avg Productivity Picker", low_picker])

    ins_df = pd.DataFrame(insights, columns=["Insight", "Value"]) if insights else pd.DataFrame()
    if ins_df.empty:
        st.info("No automated insights could be generated for the current filters.")
    else:
        st.dataframe(ins_df, use_container_width=True)
        download_csv(ins_df, "automated_insights_warehouse.csv", "Download Automated Insights")


# ---------------------------------------------------------
# DATA DICTIONARY TAB
# ---------------------------------------------------------
with tab_dict:
    st.header("Data Dictionary & Variables")

    st.markdown("<div class='section-title'>Required Columns & Definitions</div>", unsafe_allow_html=True)
    dict_df = pd.DataFrame(DATA_DICTIONARY, columns=["Column Name", "Description", "Variable Role"])
    st.dataframe(dict_df, use_container_width=True)

    st.markdown("<div class='section-title'>Independent vs Dependent Variables</div>", unsafe_allow_html=True)
    c_ind, c_dep = st.columns(2)

    with c_ind:
        st.markdown("""
        <div class='card'>
          <b>Independent / Feature Variables</b>
          <div class='small'>Drivers, conditions, and attributes used to explain or predict performance.</div>
          <hr style="margin:6px 0;">
        """, unsafe_allow_html=True)
        for col in INDEPENDENT_VARS:
            st.markdown(f"<div class='small'>• {col}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_dep:
        st.markdown("""
        <div class='card'>
          <b>Dependent / Target Variables</b>
          <div class='small'>Outcomes you want to predict, optimise, or explain.</div>
          <hr style="margin:6px 0;">
        """, unsafe_allow_html=True)
        for col in DEPENDENT_VARS:
            st.markdown(f"<div class='small'>• {col}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='subtle'>Use this tab when mapping external datasets into the app or documenting the model schema.</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# ACTIONABLE PLAYBOOKS TAB
# ---------------------------------------------------------
with tab_play:
    st.header("Actionable Playbooks")

    play = [
        ["Reassign congested aisles",
         "Divert picks away from aisles with persistent high Heatmap_Level and Congestion_Factor."],
        ["Relocate slow-moving SKUs",
         "Move low-velocity SKUs to deeper locations; free golden zones for Class A SKUs."],
        ["Picker retraining",
         "Target pickers in the bottom productivity quartile for coaching and SOP refresh."],
        ["Forklift / AMR scheduling",
         "Assign AMRs or forklifts to long-distance routes and heavy SKUs to cut travel time."],
        ["Travel-time reduction",
         "Group orders and routes by Zone/Aisle to minimise zig-zag walk patterns."],
        ["Congestion hotfix",
         "Stagger high-volume picks across shifts or windows to flatten heatmap spikes."],
        ["Slotting optimisation",
         "Increase Slotting_Score for high-frequency SKUs by bringing them closer to docks."],
        ["Delay elimination",
         "Use Delay_Reason stats to run focused kaizen events on top-3 root causes."]
    ]

    play_df = pd.DataFrame(play, columns=["Action Recommendation", "Reason / Rationale"])
    st.dataframe(play_df, use_container_width=True)
    download_csv(play_df, "warehouse_actionable_playbooks.csv", "Download Playbooks")
