import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# -------------------------

hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Warehouse Operations Analytics", layout="wide")

# ---------------------------------------------------------
# CSS Glow Styles
# ---------------------------------------------------------
st.markdown("""
<style>
.card {
    padding: 20px;
    border-radius: 12px;
    background: white;
    border: 1px solid #ddd;
    transition: 0.3s;
}
.card:hover {
    transform: translateY(-4px);
    box-shadow: 0px 4px 18px rgba(6,75,134,0.28);
    border-color: #064b86;
}
.kpi {
    padding: 28px;
    border-radius: 14px;
    background: white;
    border: 1px solid #ccc;
    font-size: 24px;
    font-weight: bold;
    text-align:center;
    color:#064b86;
    transition: 0.3s;
}
.kpi:hover {
    transform: translateY(-5px);
    box-shadow: 0px 4px 16px rgba(6,75,134,0.35);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def download_csv(df, name):
    buffer = BytesIO()
    buffer.write(df.to_csv(index=False).encode())
    buffer.seek(0)
    st.download_button("Download CSV", buffer, file_name=name, mime="text/csv")


def read_csv_safe(src):
    df = pd.read_csv(src)
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
logo = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display:flex; align-items:center;">
    <img src="{logo}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:34px; font-weight:bold;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:34px; font-weight:bold;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("## Warehouse Operations Analytics")

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Application", "Actionable Playbooks"])

# ---------------------------------------------------------
# OVERVIEW TAB
# ---------------------------------------------------------
with tab1:
    st.markdown("""
    <div class='card'>
        Analyze picking efficiency, slotting patterns, congestion, picker productivity,
        equipment usage, and operational bottlenecks. Improve warehouse throughput with
        real-time analytics, forecasting, optimization, and automated insights.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    c1.markdown("### Capabilities")
    c1.markdown("""
    <div class='card'>
        • Warehouse heatmaps<br>
        • Pick-path optimization<br>
        • Slotting efficiency scoring<br>
        • Picker productivity analytics<br>
        • Congestion tracking<br>
        • Equipment utilization<br>
        • Order lifecycle metrics<br>
        • Real-time workforce performance
    </div>
    """, unsafe_allow_html=True)

    c2.markdown("### Business Impact")
    c2.markdown("""
    <div class='card'>
        • Faster order fulfillment<br>
        • Improved warehouse layout efficiency<br>
        • Lower labor + operational cost<br>
        • Balanced workforce utilization<br>
        • Reduced congestion bottlenecks<br>
        • Improved storage slotting accuracy
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    st.markdown("## KPIs")
    k1,k2,k3,k4,k5 = st.columns(5)

    k1.markdown("<div class='kpi'>Total Orders</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Avg Pick Time</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Avg Travel Time</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Heatmap Intensity</div>", unsafe_allow_html=True)
    k5.markdown("<div class='kpi'>Productivity Score</div>", unsafe_allow_html=True)

    st.markdown("### Who Should Use This App?")
    st.markdown("""
    <div class='card'>
        • Warehouse Managers<br>
        • Operations & Logistics Teams<br>
        • Industrial Engineers<br>
        • Process Improvement Analysts<br>
        • Data & BI Teams
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# APPLICATION TAB
# ---------------------------------------------------------
with tab2:

    st.header("Application")

    # Dataset Mode
    mode = st.radio("Choose dataset option:", [
        "Default dataset (GitHub URL)",
        "Upload CSV",
        "Upload CSV + Column Mapping"
    ])

    REQUIRED = [
        "Order_ID","Warehouse","Zone","Aisle","Bin","Order_Timestamp",
        "Shift","Picker_ID","SKU","SKU_Weight_KG","SKU_Cube_M3",
        "SKU_Class","Pick_Qty","Travel_Distance_M","Travel_Time_Sec",
        "Pick_Time_Sec","Pack_Time_Sec","Heatmap_Level","Slotting_Score",
        "Equipment_Type","Picker_Productivity_Items_Hour","Congestion_Factor",
        "Delay_Reason","Delay_Minutes"
    ]

    df = None

    if mode == "Default dataset (GitHub URL)":
        try:
            DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/supply_chain/warehouse_operations_analytics.csv"
            df = read_csv_safe(DEFAULT_URL)
            st.success("Default dataset loaded.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(str(e))

    elif mode == "Upload CSV":
        st.markdown("#### Download Sample CSV for Reference")
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/supply_chain/warehouse_operations_analytics.csv"
        try:
            # Load default dataset
            sample_df = pd.read_csv(URL).head(5)  # Take first 5 rows
            sample_csv = sample_df.to_csv(index=False)
            st.download_button("Download Sample CSV", sample_csv, "sample_dataset.csv", "text/csv")
        except Exception as e:
            st.info(f"Sample CSV unavailable: {e}")

        # Upload actual CSV
        file = st.file_uploader("Upload your dataset", type=["csv"])
        if file:
            df = pd.read_csv(file)
    
    elif mode == "Upload CSV + Column Mapping":
        if file:
            raw = read_csv_safe(file)
            st.write("Preview:")
            st.dataframe(raw.head())
    
            mapping = {}
            for c in REQUIRED:
                mapping[c] = st.selectbox(
                    f"Map column for → {c}",
                    ["-- Select --"] + list(raw.columns)
                )
    
            if st.button("Apply Mapping"):
                missing = [m for m in mapping if mapping[m] == "-- Select --"]
                if missing:
                    st.error("Missing mappings: " + ", ".join(missing))
                    st.stop()
                df = raw.rename(columns={mapping[k]: k for k in mapping})
                st.success("Mapping applied.")
                st.dataframe(df.head())
        
    if df is None:
        st.stop()

    # ----------------------------------
    # ----------------------------------
    # DATE FILTER (robust fix)
    # ----------------------------------
    if "Order_Timestamp" not in df.columns:
        st.error("Order_Timestamp column missing. Please upload a valid dataset or map columns properly.")
        st.stop()
    
    df["Order_Timestamp"] = pd.to_datetime(df["Order_Timestamp"], errors="coerce")
    
    if df["Order_Timestamp"].isna().all():
        st.error("Order_Timestamp column contains no valid datetime values. Fix your dataset.")
        st.stop()
    
    # Convert to native python datetime objects for Streamlit slider compatibility
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

    # ----------------------------------
    # Extra Filters
    # ----------------------------------
    st.markdown("### Filters")

    c1, c2, c3 = st.columns(3)

    wh = c1.multiselect("Warehouse", df["Warehouse"].unique(),
                        default=list(df["Warehouse"].unique()))
    shift = c2.multiselect("Shift", df["Shift"].unique(),
                           default=list(df["Shift"].unique()))
    picker = c3.multiselect("Picker ID", df["Picker_ID"].unique(),
                            default=list(df["Picker_ID"].unique()))

    df_f = df_f[df_f["Warehouse"].isin(wh)]
    df_f = df_f[df_f["Shift"].isin(shift)]
    df_f = df_f[df_f["Picker_ID"].isin(picker)]

    st.dataframe(df_f.head(), use_container_width=True)

    download_csv(df_f, "filtered_warehouse_data.csv")

    # ---------------------------------------------------------
    # EDA SECTION — 15 CHARTS
    # ---------------------------------------------------------
    st.markdown("## Exploratory Data Analysis")

    def chart(fig):
        try:
            fig.update_traces(texttemplate="%{value}", textposition="outside")
        except Exception:
            pass
        st.plotly_chart(fig, use_container_width=True)

    # 1: Pick Qty Distribution
    with st.expander("Pick Qty Distribution — Purpose & Quick Tip"):
        st.write("Purpose: Shows the distribution of order picking quantities to detect bulk orders.")
        st.write("Quick Tip: Watch for long tails indicating inconsistent SKU demand.")
    chart(px.histogram(df_f, x="Pick_Qty", nbins=40, title="Pick Quantity Distribution"))

    # 2: Travel Distance Over Time
    with st.expander("Travel Distance Over Time — Purpose & Quick Tip"):
        st.write("Purpose: Track how picker travel distance changes over time.")
        st.write("Quick Tip: Sudden spikes highlight layout or slotting inefficiencies.")
    chart(px.scatter(df_f, x="Order_Timestamp", y="Travel_Distance_M", color="Warehouse", title="Travel Distance Over Time"))

    # 3: Heatmap Activity by Zone
    with st.expander("Zone Heatmap Activity — Purpose & Quick Tip"):
        st.write("Purpose: Identify high-activity or congested warehouse zones.")
        st.write("Quick Tip: Zones with wide boxplots indicate unstable workload distribution.")
    chart(px.box(df_f, x="Zone", y="Heatmap_Level", title="Zone-Level Heatmap Activity"))

    # 4: Slotting Score by SKU Class
    with st.expander("Slotting Score by SKU Class — Purpose & Quick Tip"):
        st.write("Purpose: Compare slotting optimization across SKU classes.")
        st.write("Quick Tip: Low slotting scores for Class A items indicate layout problems.")
    chart(px.violin(df_f, x="SKU_Class", y="Slotting_Score", title="Slotting Score Distribution by SKU Class"))

    # 5: Delay Reasons %
    with st.expander("Delay Reasons — Purpose & Quick Tip"):
        st.write("Purpose: Understand root causes of operational delays.")
        st.write("Quick Tip: Focus improvement efforts on the dominant reason slice.")
    chart(px.pie(df_f, names="Delay_Reason", title="Delay Reasons Distribution"))

    # 6: Travel Time vs Distance
    with st.expander("Travel Time vs Distance — Purpose & Quick Tip"):
        st.write("Purpose: Measure relationship between distance walked and time consumed.")
        st.write("Quick Tip: Outliers may indicate interruptions or picker inefficiencies.")
    chart(px.scatter(df_f, x="Travel_Distance_M", y="Travel_Time_Sec", title="Travel Time vs Distance"))

    # 7: Equipment usage
    with st.expander("Equipment Usage — Purpose & Quick Tip"):
        st.write("Purpose: Understand which equipment types are used most.")
        st.write("Quick Tip: High reliance on a single type may signal imbalance.")
    chart(px.bar(df_f, x="Equipment_Type", title="Equipment Usage"))

    # 8: Picker productivity
    with st.expander("Picker Productivity — Purpose & Quick Tip"):
        st.write("Purpose: Evaluate distribution of picker productivity levels.")
        st.write("Quick Tip: Consider retraining bottom-performers.")
    chart(px.histogram(df_f, x="Picker_Productivity_Items_Hour", title="Picker Productivity Distribution"))

    # 9: Congestion Factor
    with st.expander("Congestion Factor — Purpose & Quick Tip"):
        st.write("Purpose: Analyze congestion differences across warehouses.")
        st.write("Quick Tip: Consistent high congestion requires layout revision.")
    chart(px.box(df_f, y="Congestion_Factor", color="Warehouse", title="Congestion Factor by Warehouse"))

    # 10: SKU weight distribution
    with st.expander("SKU Weight Distribution — Purpose & Quick Tip"):
        st.write("Purpose: Assess weight variation across SKUs.")
        st.write("Quick Tip: Heavy-SKU clusters help redesign ergonomic picking.")
    chart(px.histogram(df_f, x="SKU_Weight_KG", title="SKU Weight Distribution"))

    # 11: Pick Time vs Pack Time
    with st.expander("Pick vs Pack Time — Purpose & Quick Tip"):
        st.write("Purpose: Compare picking and packing time correlation.")
        st.write("Quick Tip: Large gaps may reveal packing station bottlenecks.")
    chart(px.scatter(df_f, x="Pick_Time_Sec", y="Pack_Time_Sec", title="Pick vs Pack Time"))

    # 12: Heatmap Level Time-Series
    with st.expander("Heatmap Level Over Time — Purpose & Quick Tip"):
        st.write("Purpose: Monitor congestion intensity across time.")
        st.write("Quick Tip: Peaks often align with shift overlaps or restocking.")
    chart(px.line(df_f.sort_values("Order_Timestamp"), x="Order_Timestamp", y="Heatmap_Level", title="Heatmap Level Over Time"))

    # 13: Aisle Congestion Map
    with st.expander("Aisle Congestion Map — Purpose & Quick Tip"):
        st.write("Purpose: Identify which aisles face maximum congestion.")
        st.write("Quick Tip: Darker zones suggest rerouting opportunities.")
    chart(px.density_heatmap(df_f, x="Aisle", y="Heatmap_Level", title="Aisle Congestion Heatmap"))

    # 14: Productivity by Shift
    with st.expander("Productivity by Shift — Purpose & Quick Tip"):
        st.write("Purpose: Compare picker performance across shifts.")
        st.write("Quick Tip: Low-performing shifts often align with staffing imbalance.")
    chart(px.box(df_f, x="Shift", y="Picker_Productivity_Items_Hour", title="Productivity by Shift"))

    # 15: Delay Time Distribution
    with st.expander("Delay Duration Distribution — Purpose & Quick Tip"):
        st.write("Purpose: Examine distribution of delay lengths.")
        st.write("Quick Tip: Long-tail delays require root-cause drilling.")
    chart(px.histogram(df_f, x="Delay_Minutes", nbins=40, title="Delay Duration Distribution"))


    # ---------------------------------------------------------
    # 4 ML MODELS
    # ---------------------------------------------------------
    st.markdown("## Machine Learning Models")

    # Target: Pick_Time_Sec Prediction
    st.markdown("### 1. Predict Pick Time (Regression)")

    features = ["Pick_Qty", "Travel_Distance_M", "Travel_Time_Sec",
                "SKU_Weight_KG", "SKU_Cube_M3", "Heatmap_Level",
                "Slotting_Score", "Picker_Productivity_Items_Hour"]

    X = df_f[features]
    y = df_f["Pick_Time_Sec"]

    numeric = features
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric)
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("reg", RandomForestRegressor(n_estimators=120))
    ])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(Xtr, ytr)

    preds = model.predict(Xte)

    results = pd.DataFrame({
        "Actual": yte,
        "Predicted": preds
    })
    st.dataframe(results.head())
    download_csv(results, "ml_pick_time_predictions.csv")

    # ---------------------------------------------------------
    # 2: Congestion Pattern Clustering
    st.markdown("### 2. Congestion Pattern Clustering (KMeans)")

    km = KMeans(n_clusters=4, random_state=42)
    df_f["Cluster"] = km.fit_predict(df_f[["Heatmap_Level",
                                           "Congestion_Factor",
                                           "Pick_Qty"]])

    chart(px.scatter(df_f, x="Pick_Qty", y="Congestion_Factor",
                     color="Cluster",
                     title="Congestion Clusters"))

    # ---------------------------------------------------------
    # 3: Predict Delay Minutes
    st.markdown("### 3. Predict Delay Minutes (Regression)")

    y2 = df_f["Delay_Minutes"]
    X2 = df_f[["Pick_Qty", "Travel_Time_Sec", "Congestion_Factor", "Heatmap_Level"]]

    model2 = GradientBoostingRegressor()
    model2.fit(X2, y2)
    preds2 = model2.predict(X2)

    df_delay = pd.DataFrame({"Actual": y2, "Predicted": preds2})
    st.dataframe(df_delay.head())

    # ---------------------------------------------------------
    # 4: Predict Productivity
    st.markdown("### 4. Predict Picker Productivity")

    y3 = df_f["Picker_Productivity_Items_Hour"]
    X3 = df_f[["Pick_Qty","Travel_Distance_M","Heatmap_Level","SKU_Weight_KG"]]

    model3 = RandomForestRegressor()
    model3.fit(X3, y3)
    preds3 = model3.predict(X3)

    df_prod = pd.DataFrame({"Actual": y3, "Predicted": preds3})
    st.dataframe(df_prod.head())

# ---------------------------------------------------------
# AUTOMATED INSIGHTS
# ---------------------------------------------------------
    st.markdown("## Automated Insights")

    insights = []

    # 1 Congestion zone
    worst_zone = df_f.groupby("Zone")["Congestion_Factor"].mean().idxmax()
    insights.append(["Highest Congestion Zone", worst_zone])

    # 2 Slowest Equipment
    slow_eq = df_f.groupby("Equipment_Type")["Pick_Time_Sec"].mean().idxmax()
    insights.append(["Slowest Equipment Type", slow_eq])

    # 3 Highest Delay Reason
    top_delay = df_f["Delay_Reason"].value_counts().idxmax()
    insights.append(["Top Delay Reason", top_delay])

    # 4 Lowest Productivity Picker
    low_picker = df_f.groupby("Picker_ID")["Picker_Productivity_Items_Hour"].mean().idxmin()
    insights.append(["Lowest Productivity Picker", low_picker])

    ins_df = pd.DataFrame(insights, columns=["Insight", "Value"])
    st.dataframe(ins_df)
    download_csv(ins_df, "automated_insights.csv")

# ---------------------------------------------------------
# PLAYBOOK TAB
# ---------------------------------------------------------
with tab3:

    st.header("Actionable Playbooks")

    play = [
        ["Reassign congested aisles", "Divert picks away from Aisles with Heatmap_Level > 4"],
        ["Relocate slow-moving SKUs", "Move low-velocity SKUs to back racks"],
        ["Picker retraining", "Train pickers with productivity < 25 items/hour"],
        ["Forklift/AMR scheduling", "Assign AMRs to long-distance pick routes"],
        ["Travel-time reduction", "Optimize zone routing for high-frequency SKUs"],
        ["Congestion hotfix", "Stagger Shift-1 pickers during peak hours"],
        ["Slotting optimization", "Boost Slotting_Score by relocating Class A SKUs"],
        ["Delay elimination", "Target root causes: Transport/Stockout/Equipment"]
    ]

    play_df = pd.DataFrame(play, columns=["Action Recommendation", "Reason"])
    st.dataframe(play_df, use_container_width=True)

    download_csv(play_df, "actionable_playbooks.csv")
