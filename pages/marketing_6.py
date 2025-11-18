import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import math
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Customer Retention & Churn Analysis", layout="wide")

# Hide default sidebar navigation (optional)
st.markdown("""<style>[data-testid="stSidebarNav"]{display:none;}</style>""", unsafe_allow_html=True)



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


# -------------------------
# Required Columns & Mapping
# -------------------------
REQUIRED_CUSTOMER_COLS = [
    "Customer_ID", "SignUp_Date", "Last_Active_Date", "Churn_Flag",
    "Total_Spend", "Total_Orders", "Avg_Order_Value", "Channel",
    "Country", "Device", "AgeGroup", "Gender"
]

AUTO_MAPS = {
    "Customer_ID": ["customer_id", "id", "Customer ID"],
    "SignUp_Date": ["signup_date", "registration_date", "SignUp_Date"],
    "Last_Active_Date": ["last_active_date", "last_seen", "Last_Active_Date"],
    "Churn_Flag": ["churn", "churn_flag", "Churn_Flag"],
    "Total_Spend": ["total_spend", "spend", "Total_Spend"],
    "Total_Orders": ["orders", "total_orders", "Total_Orders"],
    "Avg_Order_Value": ["avg_order_value", "AOV", "Avg_Order_Value"],
    "Channel": ["channel", "Source", "Channel"],
    "Country": ["country", "Country"],
    "Device": ["device", "Device"],
    "AgeGroup": ["agegroup", "Age_Group", "AgeGroup"],
    "Gender": ["gender", "Gender"]
}

def auto_map_columns(df):
    rename = {}
    cols = [c.strip() for c in df.columns]
    for req, candidates in AUTO_MAPS.items():
        for c in cols:
            low = c.lower().strip()
            for cand in candidates:
                if cand.lower() == low or cand.lower() in low or low in cand.lower():
                    rename[c] = req
                    break
            if c in rename:
                break
    if rename:
        df = df.rename(columns=rename)
    return df

def ensure_datetime(df, col):
    try:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    except:
        pass
    return df

def download_df(df, filename):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button("Download CSV", b, file_name=filename, mime="text/csv")

# -------------------------
# Page Header
# -------------------------
st.markdown("<h1 style='margin-bottom:0.2rem'>Customer Retention & Churn Analysis</h1>", unsafe_allow_html=True)
st.markdown("Understand which customers stay, which churn, and actionable insights to improve retention.")

# CSS for hover-glow metrics
st.markdown("""
<style>
div[data-testid="stMarkdownContainer"] .metric-card {
    background: rgba(255,255,255,0.10);
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.30);
    font-weight: 600;
    font-size: 16px;
    transition: all 0.25s ease;
    box-shadow: 0 2px 10px rgba(0,0,0,0.18);
}
div[data-testid="stMarkdownContainer"] .metric-card:hover {
    background: rgba(255,255,255,0.20);
    border: 1px solid rgba(255,255,255,0.55);
    box-shadow: 0 0 18px rgba(255,255,255,0.4);
    transform: scale(1.04);
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Application"])

# -------------------------
# Overview Tab
# -------------------------
with tabs[0]:
    st.markdown("### Overview")
    st.markdown("""
    <div class='metric-card'>
        This app provides <b>end-to-end customer retention & churn analysis</b>. 
        Track which customers stay, which churn, and identify at-risk segments for proactive engagement. 
        Aggregate historical behavior, spending, and engagement metrics to gain actionable insights and forecast churn probability.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Capabilities")
    st.markdown("""
    <div class='metric-card'>
        • Multi-dimensional retention analysis by Channel, Device, Country, AgeGroup, and Gender<br>
        • Churn probability prediction using <b>RandomForest Classifier</b><br>
        • Monthly retention trends and customer lifecycle insights<br>
        • Automated identification of at-risk segments<br>
        • Downloadable insights & ML predictions for reporting
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='metric-card' title='Total number of customers in the dataset'>Total Customers</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card' title='Number of churned customers'>Churned Customers</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card' title='Percentage of customers retained'>Retention Rate</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card' title='Average order value across customers'>Avg Order Value</div>", unsafe_allow_html=True)

    st.markdown("### Automated Insights")
    st.markdown("""
    <div class='metric-card'>
        • Identify top and bottom performing channels based on retention<br>
        • Highlight segments (AgeGroup, Gender, Device, Country) with high churn<br>
        • Downloadable insights table for executive reporting
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# Application Tab
# -------------------------
with tabs[1]:
    st.header("Application")
    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)
    df = None

    # --- Dataset loading ---
    if mode == "Default dataset":
        DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/customer_retention.csv"
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("Default dataset loaded")
            st.dataframe(df.head())
        except Exception as e:
            st.error("Failed to load default dataset: " + str(e))
            st.stop()
    elif mode == "Upload CSV":
        st.markdown("#### Download Sample CSV for Reference")
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/content_seo_dataset.csv"
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

            
    else:
        uploaded = st.file_uploader("Upload CSV to map", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head())
            st.markdown("Map your columns to required fields:")
            mapping = {}
            for req in REQUIRED_CUSTOMER_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + list(raw.columns))
            if st.button("Apply mapping"):
                missing = [k for k,v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    df = raw.rename(columns={v:k for k,v in mapping.items()})
                    st.success("Mapping applied.")
                    st.dataframe(df.head())

    if df is None: st.stop()

    # --- Type conversions ---
    df = ensure_datetime(df, "SignUp_Date")
    df = ensure_datetime(df, "Last_Active_Date")
    for col in ["Total_Spend","Total_Orders","Avg_Order_Value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if "Churn_Flag" in df.columns:
        df["Churn_Flag"] = df["Churn_Flag"].astype(int)

    # --- Filters ---
    channels = sorted(df["Channel"].dropna().unique()) if "Channel" in df.columns else []
    countries = sorted(df["Country"].dropna().unique()) if "Country" in df.columns else []
    devices = sorted(df["Device"].dropna().unique()) if "Device" in df.columns else []

    c1,c2,c3 = st.columns(3)
    sel_channel = c1.multiselect("Channel", options=channels, default=channels)
    sel_country = c2.multiselect("Country", options=countries, default=countries)
    sel_device = c3.multiselect("Device", options=devices, default=devices)

    filt = df.copy()
    if sel_channel: filt = filt[filt["Channel"].isin(sel_channel)]
    if sel_country: filt = filt[filt["Country"].isin(sel_country)]
    if sel_device: filt = filt[filt["Device"].isin(sel_device)]

    st.dataframe(filt.head(5))
    download_df(filt.head(5), "filtered_customers.csv")

    # --- KPIs ---
    st.markdown("### Key Metrics")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Customers", len(filt))
    k2.metric("Churned Customers", filt["Churn_Flag"].sum() if "Churn_Flag" in filt.columns else 0)
    k3.metric("Retention Rate", f"{100*(1-filt['Churn_Flag'].mean()):.2f}%" if "Churn_Flag" in filt.columns else "N/A")
    k4.metric("Avg Order Value", f"₹ {filt['Avg_Order_Value'].mean():,.2f}" if "Avg_Order_Value" in filt.columns else "N/A")

    # --- Retention Chart ---
    if "SignUp_Date" in filt.columns and "Churn_Flag" in filt.columns:
        retention = filt.groupby(filt["SignUp_Date"].dt.to_period("M")).agg(
            Total_Customers=("Customer_ID","count"),
            Churned=("Churn_Flag","sum")
        ).reset_index()
        retention["SignUp_Date"] = retention["SignUp_Date"].dt.to_timestamp()
        retention["Retention_Rate"] = 1 - retention["Churned"]/retention["Total_Customers"]
        fig = px.line(retention, x="SignUp_Date", y="Retention_Rate", title="Monthly Retention Rate", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    # --- ML: Churn Prediction + Downloadable ---
    st.markdown("### ML: Predict Churn Probability")
    if len(filt) > 40:
        feat_cols = ["Total_Spend","Total_Orders","Avg_Order_Value","Channel","Device","Country","AgeGroup","Gender"]
        feat_cols = [c for c in feat_cols if c in filt.columns]
        X = filt[feat_cols].copy()
        y = filt["Churn_Flag"]
        cat_cols = [c for c in X.columns if X[c].dtype == "object"]
        num_cols = [c for c in X.columns if c not in cat_cols]
        preprocessor = ColumnTransformer(transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols)
        ], remainder="drop")
        X_t = preprocessor.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        with st.spinner("Training RandomForest for churn prediction..."):
            rf.fit(X_train, y_train)
        preds = rf.predict_proba(X_test)[:,1]

        df_preds = pd.DataFrame({
            "Customer_ID": filt["Customer_ID"].iloc[:len(preds)],
            "Churn_Prob": preds
        })
        st.dataframe(df_preds.head(10))
        download_df(df_preds, "churn_predictions.csv")
    else:
        st.info("Not enough data to train ML model (min 40 rows).")

    # --- Automated Insights Table + Download ---
    st.markdown("### Automated Insights Table")
    if "Churn_Flag" in filt.columns:
        insights = filt.groupby(["Channel","Device","Country","AgeGroup","Gender"]).agg(
            Total_Customers=("Customer_ID","count"),
            Churned=("Churn_Flag","sum")
        ).reset_index()
        insights["Retention_Rate"] = 1 - insights["Churned"]/insights["Total_Customers"]
        st.dataframe(insights)
        download_df(insights, "automated_insights.csv")
