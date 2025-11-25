# churn_marketing_lab_marketing_lab_style.py
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
from sklearn.metrics import roc_auc_score, accuracy_score
import math
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Page / App Config
# -------------------------
st.set_page_config(page_title="Customer Retention & Churn Analysis", layout="wide")
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"

# -------------------------
# Marketing Lab CSS (Inter, pure-black text, blue KPI & variable boxes)
# -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }

/* GLOBAL TEXT */
body, [class*="css"] { color:#000 !important; font-size:17px; }

/* MAIN HEADER */
.big-header {
    font-size:36px !important;
    font-weight:700 !important;
    color:#000 !important;
    margin-bottom:12px;
}

/* SECTION TITLE */
.section-title {
    font-size:24px !important;
    font-weight:600 !important;
    margin-top:30px;
    margin-bottom:12px;
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

/* CARD (pure black text) */
.card {
    background:#ffffff;
    padding:22px;
    border-radius:14px;
    border:1px solid #e6e6e6;
    font-size:16.5px;
    color:#000 !important;
    font-weight:500;
    box-shadow:0 3px 14px rgba(0,0,0,0.08);
    transition: all 0.25s ease;
}
.card:hover {
    transform:translateY(-4px);
    box-shadow:0 12px 25px rgba(6,75,134,0.18);
    border-color:#064b86;
}

/* KPI CARDS - blue text */
.kpi {
    background:#ffffff;
    padding:22px;
    border-radius:14px;
    border:1px solid #e2e2e2;
    font-size:20px !important;
    font-weight:600 !important;
    text-align:center;
    color:#064b86 !important;
    box-shadow:0 3px 14px rgba(0,0,0,0.07);
    transition:0.25s ease;
}
.kpi:hover {
    transform:translateY(-4px);
    box-shadow:0 13px 26px rgba(6,75,134,0.20);
    border-color:#064b86;
}

/* VARIABLE BOXES - blue text */
.variable-box {
    padding:18px;
    border-radius:14px;
    background:white;
    border:1px solid #e5e5e5;
    box-shadow:0 2px 10px rgba(0,0,0,0.10);
    transition:0.25s ease;
    text-align:center;
    font-size:17.5px !important;
    font-weight:500 !important;
    color:#064b86 !important;
    margin-bottom:12px;
}
.variable-box:hover {
    transform:translateY(-5px);
    box-shadow:0 12px 22px rgba(6,75,134,0.18);
    border-color:#064b86;
}

/* Table (Plotly/Streamlit fallback) */
.dataframe th {
    background:#064b86 !important;
    color:#fff !important;
    padding:11px !important;
    font-size:15.5px !important;
}
.dataframe td {
    font-size:15.5px !important;
    color:#000 !important;
    padding:9px !important;
    border-bottom:1px solid #efefef !important;
}
.dataframe tbody tr:hover { background:#f4f9ff !important; }

/* index-safe HTML table */
.required-table thead th {
    background:#ffffff !important;
    color:#000 !important;
    font-size:14px !important;
    border-bottom:2px solid #000 !important;
    padding:10px !important;
    text-align:left;
}
.required-table tbody td {
    color:#000 !important;
    font-size:15.5px !important;
    padding:10px !important;
    border-bottom:1px solid #efefef !important;
}
.required-table tbody tr:hover td { background:#f8f8f8 !important; }

/* Buttons */
.stButton>button,
.stDownloadButton>button {
    background:#064b86 !important;
    color:white !important;
    border:none;
    padding:10px 22px;
    border-radius:8px !important;
    font-size:15.5px !important;
    font-weight:600 !important;
    transition:0.25s ease;
}
.stButton>button:hover,
.stDownloadButton>button:hover {
    transform:translateY(-3px);
    background:#0a6eb3 !important;
}

/* Page fade */
.block-container { animation: fadeIn 0.5s ease; }
@keyframes fadeIn { from {opacity:0; transform:translateY(10px);} to {opacity:1; transform:translateY(0);} }

.small-muted { color:#6b6b6b !important; font-size:13px !important; }
</style>
""", unsafe_allow_html=True)



# -------------------------
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display: flex; align-items: center; margin-bottom:16px;">
    <img src="{logo_url}" width="60" style="margin-right:12px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:700;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:700;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='big-header'>Customer Retention & Churn Analysis</div>", unsafe_allow_html=True)

# -------------------------
# Required Columns & Auto-maps
# -------------------------
REQUIRED_CUSTOMER_COLS = [
    "Customer_ID", "SignUp_Date", "Last_Active_Date", "Churn_Flag",
    "Total_Spend", "Total_Orders", "Avg_Order_Value", "Channel",
    "Country", "Device", "AgeGroup", "Gender"
]

AUTO_MAPS = {
    "Customer_ID": ["customer_id", "id", "Customer ID", "cust_id"],
    "SignUp_Date": ["signup_date", "registration_date", "SignUp_Date", "registered_at"],
    "Last_Active_Date": ["last_active_date", "last_seen", "Last_Active_Date", "last_activity"],
    "Churn_Flag": ["churn", "churn_flag", "Churn_Flag", "is_churn"],
    "Total_Spend": ["total_spend", "spend", "Total_Spend", "lifetime_value", "ltv"],
    "Total_Orders": ["orders", "total_orders", "Total_Orders", "num_orders"],
    "Avg_Order_Value": ["avg_order_value", "AOV", "Avg_Order_Value", "aov"],
    "Channel": ["channel", "source", "Channel", "acquisition_channel"],
    "Country": ["country", "Country", "region"],
    "Device": ["device", "Device", "platform"],
    "AgeGroup": ["agegroup", "Age_Group", "AgeGroup", "age_group"],
    "Gender": ["gender", "Gender", "sex"]
}

def auto_map_columns(df):
    rename = {}
    cols = [c.strip() for c in df.columns]
    for req, candidates in AUTO_MAPS.items():
        for c in cols:
            low = c.lower().strip()
            for cand in candidates:
                cand_low = cand.lower().strip()
                if cand_low == low or cand_low in low or low in cand_low:
                    rename[c] = req
                    break
            if c in rename:
                break
    if rename:
        df = df.rename(columns=rename)
    return df

def ensure_datetime(df, col):
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        except:
            pass
    return df

def to_currency(x):
    try:
        return "₹ " + f"{float(x):,.2f}"
    except:
        return "₹ 0.00"

def download_df(df, filename, label="Download CSV"):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")

def render_index_safe_table(df: pd.DataFrame, classes="required-table"):
    if df is None or df.empty:
        st.info("No table data to display.")
        return
    html = df.to_html(index=False, classes=classes, escape=False)
    st.markdown(html, unsafe_allow_html=True)

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Important Attributes", "Application"])

# -------------------------
# TAB 1 — Overview
# -------------------------
with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <b>Purpose:</b> Provide a single pane for customer retention and churn analytics with ML-driven churn probability, segment-level diagnostics, and exportable insights.
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Segment-level churn diagnostics (Channel, Device, Country, AgeGroup, Gender).<br>
        • Churn probability prediction using RandomForestClassifier.<br>
        • Monthly retention trends and cohort-level analysis.<br>
        • Automated identification of at-risk segments and exportable insights.<br>
        • Downloadable ML predictions for downstream workflows.
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Reduce churn by targeting high-risk segments proactively.<br>
        • Improve lifetime value by reactivation campaigns.<br>
        • Inform product and retention playbooks using data-backed segments.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'>Total Customers</div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'>Churned Customers</div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'>Retention Rate</div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'>Avg Order Value</div>", unsafe_allow_html=True)

# -------------------------
# TAB 2 — Important Attributes
# -------------------------
with tab2:
    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)
    required_dict = {
        "Customer_ID": "Unique identifier for the customer.",
        "SignUp_Date": "Customer signup/registration date.",
        "Last_Active_Date": "Last activity or last seen date.",
        "Churn_Flag": "Binary churn flag (1 = churned, 0 = active).",
        "Total_Spend": "Total amount spent by the customer (lifetime).",
        "Total_Orders": "Total number of orders made by the customer.",
        "Avg_Order_Value": "Average order value (Total_Spend / Total_Orders).",
        "Channel": "Acquisition channel (organic, paid, referral, etc.).",
        "Country": "Customer country or region.",
        "Device": "Device/platform (Desktop, Mobile, App, etc.).",
        "AgeGroup": "Customer age group bucket.",
        "Gender": "Customer gender or sex."
    }
    dict_df = pd.DataFrame([{"Attribute": k, "Description": v} for k, v in required_dict.items()])
    render_index_safe_table(dict_df)

    st.markdown('<div class="section-title">Attributes Overview</div>', unsafe_allow_html=True)

    left_col, right_col = st.columns(2)

    # Independent variables on LEFT
    with left_col:
        st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
        indep_vars = [
            "Channel",
            "Country",
            "Device",
            "AgeGroup",
            "Gender",
            "Total_Spend",
            "Total_Orders",
            "Avg_Order_Value",
            "SignUp_Date",
            "Last_Active_Date"
        ]
        for v in indep_vars:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

    # Dependent variables on RIGHT
    with right_col:
        st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
        dep_vars = [
            "Churn_Flag",
            "Retention_Rate",
            "Customer_Lifetime_Value"
        ]
        for v in dep_vars:
            st.markdown(f"<div class='variable-box'>{v}</div>", unsafe_allow_html=True)

# -------------------------
# TAB 3 — Application
# -------------------------
with tab3:
    st.markdown('<div class="section-title">Step 1 — Load dataset</div>', unsafe_allow_html=True)
    mode = st.radio("Dataset option:", ["Default dataset", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)
    df = None
    raw = None

    DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/customer_retention.csv"

    if mode == "Default dataset":
        try:
            df = pd.read_csv(DEFAULT_URL)
            df.columns = df.columns.str.strip()
            df = auto_map_columns(df)
            st.success("Default dataset loaded.")
            st.markdown('<div class="small-muted">Preview (first 5 rows)</div>', unsafe_allow_html=True)
            st.dataframe(df.head(3), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load default dataset: {e}")
            st.stop()

    elif mode == "Upload CSV":
        st.markdown("#### Download sample CSV for reference (first 5 rows)")
        try:
            sample_df = pd.read_csv(DEFAULT_URL).head(5)
            st.download_button("Download Sample CSV", sample_df.to_csv(index=False), "sample_customer_retention.csv", "text/csv")
        except Exception:
            st.info("Sample unavailable.")
        uploaded = st.file_uploader("Upload your dataset (CSV)", type=["csv"], accept_multiple_files=False)
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                df.columns = df.columns.str.strip()
                df = auto_map_columns(df)
                st.success("File uploaded and columns auto-mapped where possible.")
                st.dataframe(df.head(5), use_container_width=True)
            except Exception as e:
                st.error("Failed to read uploaded CSV: " + str(e))

    else:  # Upload + mapping
        uploaded = st.file_uploader("Upload CSV to map (CSV)", type=["csv"], key="mapping_upload")
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.str.strip()
            st.markdown("<div class='small-muted'>Preview (first 5 rows)</div>", unsafe_allow_html=True)
            st.dataframe(raw.head(5), use_container_width=True)

            st.markdown('<div class="section-title">Map your columns to required fields</div>', unsafe_allow_html=True)
            mapping = {}
            cols_list = list(raw.columns)
            for req in REQUIRED_CUSTOMER_COLS:
                mapping[req] = st.selectbox(f"Map → {req}", options=["-- Select --"] + cols_list, key=f"map_{req}")
            if st.button("Apply mapping"):
                missing = [k for k, v in mapping.items() if v == "-- Select --"]
                if missing:
                    st.error("Please map all required columns: " + ", ".join(missing))
                else:
                    inv = {v: k for k, v in mapping.items()}
                    df = raw.rename(columns=inv)
                    st.success("Mapping applied.")
                    st.dataframe(df.head(5), use_container_width=True)

    if df is None:
        st.stop()

    # -------------------------
    # Keep only required columns that exist, coerce types
    # -------------------------
    df = df[[c for c in REQUIRED_CUSTOMER_COLS if c in df.columns]].copy()
    df = ensure_datetime(df, "SignUp_Date")
    df = ensure_datetime(df, "Last_Active_Date")

    for col in ["Total_Spend", "Total_Orders", "Avg_Order_Value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Churn flag ensure int
    if "Churn_Flag" in df.columns:
        try:
            df["Churn_Flag"] = df["Churn_Flag"].astype(int)
        except:
            df["Churn_Flag"] = pd.to_numeric(df["Churn_Flag"], errors="coerce").fillna(0).astype(int)

    # Derive Avg_Order_Value and CLV if possible
    if "Total_Spend" in df.columns and "Total_Orders" in df.columns:
        df["Avg_Order_Value"] = np.where(df["Total_Orders"]>0, df["Total_Spend"]/df["Total_Orders"], df.get("Avg_Order_Value", 0))
    if "Total_Spend" in df.columns:
        df["Customer_Lifetime_Value"] = df["Total_Spend"]

    # -------------------------
    # Step 2 — Filters & Preview
    # -------------------------
    st.markdown('<div class="section-title">Step 2 — Filters & preview</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 2, 1])
    channels = sorted(df["Channel"].dropna().unique()) if "Channel" in df.columns else []
    countries = sorted(df["Country"].dropna().unique()) if "Country" in df.columns else []
    devices = sorted(df["Device"].dropna().unique()) if "Device" in df.columns else []
    with c1:
        sel_channels = st.multiselect("Channel", options=channels, default=channels[:5] if channels else [])
    with c2:
        sel_countries = st.multiselect("Country", options=countries, default=countries[:5] if countries else [])
    with c3:
        try:
            min_d = df["SignUp_Date"].min().date()
            max_d = df["SignUp_Date"].max().date()
            date_range = st.date_input("Signup date range", value=(min_d, max_d))
        except Exception:
            date_range = st.date_input("Signup date range")

    filt = df.copy()
    if sel_channels:
        filt = filt[filt["Channel"].isin(sel_channels)]
    if sel_countries:
        filt = filt[filt["Country"].isin(sel_countries)]
    try:
        if date_range and len(date_range) == 2:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            if "SignUp_Date" in filt.columns:
                filt = filt[(filt["SignUp_Date"] >= start) & (filt["SignUp_Date"] <= end)]
    except Exception:
        pass

    st.markdown('<div class="section-title">Filtered preview (first 10 rows)</div>', unsafe_allow_html=True)
    st.dataframe(filt.head(3), use_container_width=True)
    download_df(filt.head(100), "filtered_customers_preview.csv", label="Download filtered preview (up to 500 rows)")

    # -------------------------
    # Key Metrics
    # -------------------------
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    kcol1, kcol2, kcol3, kcol4 = st.columns(4)

    kcol1.markdown("<div class='kpi'>Total Customers</div>", unsafe_allow_html=True)
    kcol2.markdown("<div class='kpi'>Churned Customers</div>", unsafe_allow_html=True)
    kcol3.markdown("<div class='kpi'>Retention Rate</div>", unsafe_allow_html=True)
    kcol4.markdown("<div class='kpi'>Avg Order Value</div>", unsafe_allow_html=True)

    # numeric metrics below
    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Total Customers", f"{len(filt):,}")
    churned_val = int(filt["Churn_Flag"].sum()) if "Churn_Flag" in filt.columns else 0
    n2.metric("Churned Customers", f"{churned_val:,}")
    retention_rate = (1 - filt["Churn_Flag"].mean()) if "Churn_Flag" in filt.columns and len(filt)>0 else np.nan
    n3.metric("Retention Rate", f"{(retention_rate*100):.2f}%" if not np.isnan(retention_rate) else "N/A")
    n4.metric("Avg Order Value", to_currency(filt["Avg_Order_Value"].mean()) if "Avg_Order_Value" in filt.columns else "N/A")

    # -------------------------
    # Retention over time / Cohort (monthly signup retention)
    # -------------------------
    st.markdown('<div class="section-title">Retention Trend (Monthly Cohort)</div>', unsafe_allow_html=True)
    if "SignUp_Date" in filt.columns and "Churn_Flag" in filt.columns:
        cohort = filt.dropna(subset=["SignUp_Date", "Customer_ID"]).copy()
        cohort["cohort_month"] = cohort["SignUp_Date"].dt.to_period("M").dt.to_timestamp()
        cohort_agg = cohort.groupby("cohort_month").agg(
            total_customers=("Customer_ID", "nunique"),
            churned=("Churn_Flag", "sum")
        ).reset_index().sort_values("cohort_month")
        if not cohort_agg.empty:
            cohort_agg["retention_rate"] = 1 - cohort_agg["churned"] / cohort_agg["total_customers"]
            fig = px.line(cohort_agg, x="cohort_month", y="retention_rate", markers=True, labels={"cohort_month":"Signup Month", "retention_rate":"Retention Rate"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough cohort data to build retention trend.")
    else:
        st.info("SignUp_Date or Churn_Flag missing — cannot compute retention trend.")

    # -------------------------
    # Segment Performance (Channel / Country / Device)
    # -------------------------
    st.markdown('<div class="section-title">Segment Performance</div>', unsafe_allow_html=True)
    seg_cols = ["Channel", "Country", "Device", "AgeGroup", "Gender"]
    for g in seg_cols:
        if g in filt.columns:
            seg = filt.groupby(g).agg(
                Customers=("Customer_ID", "nunique"),
                Churned=("Churn_Flag", "sum")
            ).reset_index()
            seg["Retention_Rate"] = np.where(seg["Customers"]>0, 1 - seg["Churned"]/seg["Customers"], np.nan)
            seg = seg.sort_values("Customers", ascending=False).head(10)
            fig = px.bar(seg, x=g, y="Customers", text="Retention_Rate", labels={"Retention_Rate":"Retention Rate"})
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(title=f"{g} — Customers (top 10)")
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # ML: Churn Prediction (RandomForestClassifier)
    # -------------------------
    st.markdown('<div class="section-title">ML — Predict Churn Probability (RandomForest)</div>', unsafe_allow_html=True)
    ml_msg = st.empty()
    if len(filt) < 40:
        ml_msg.info("Not enough data to train ML model (need at least 40 rows).")
    else:
        feat_cols = ["Total_Spend","Total_Orders","Avg_Order_Value","Channel","Device","Country","AgeGroup","Gender"]
        feat_cols = [c for c in feat_cols if c in filt.columns]
        if len(feat_cols) < 2 or "Churn_Flag" not in filt.columns:
            ml_msg.info("Insufficient features or missing Churn_Flag to train model.")
        else:
            ml_msg.info("Training RandomForest...")

            X = filt[feat_cols].copy()
            y = filt["Churn_Flag"].astype(int)

            cat_cols = [c for c in X.columns if X[c].dtype == "object"]
            num_cols = [c for c in X.columns if c not in cat_cols]

            transformers = []
            if cat_cols:
                transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
            if num_cols:
                transformers.append(("num", StandardScaler(), num_cols))

            preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
            try:
                X_t = preprocessor.fit_transform(X)
            except Exception as e:
                st.error("Preprocessing failed: " + str(e))
                X_t = None

            if X_t is not None:
                X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)
                clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
                with st.spinner("Training RandomForest for churn prediction..."):
                    clf.fit(X_train, y_train)

                preds_proba = clf.predict_proba(X_test)[:, 1]
                preds = clf.predict(X_test)
                auc = roc_auc_score(y_test, preds_proba) if len(np.unique(y_test))>1 else np.nan
                acc = accuracy_score(y_test, preds)
                st.write(f"Model performance — AUC: {auc:.3f}  |  Accuracy: {acc:.3f}")

                # Feature importance: try to obtain feature names
                try:
                    feature_names = []
                    if hasattr(preprocessor, "named_transformers_"):
                        if "cat" in preprocessor.named_transformers_:
                            ohe = preprocessor.named_transformers_["cat"]
                            try:
                                cat_names = list(ohe.get_feature_names_out(cat_cols))
                            except Exception:
                                cat_names = []
                        else:
                            cat_names = []
                        num_names = num_cols
                        feature_names = list(cat_names) + list(num_names)
                    else:
                        feature_names = [f"f_{i}" for i in range(X_t.shape[1])]
                    fi = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False).head(30)
                    st.markdown("Top feature importances")
                    st.dataframe(fi.reset_index().rename(columns={"index":"feature", 0:"importance"}).head(10))
                except Exception:
                    st.info("Feature importance unavailable for this preprocessing pipeline.")

                # Attach predictions back to a sample of original customers (best-effort mapping)
                try:
                    sample_idx = X_test.shape[0]
                    sample_customers = filt.reset_index(drop=True).iloc[:sample_idx].copy()
                    sample_customers = sample_customers.reset_index(drop=True)
                    sample_customers["Churn_Prob"] = preds_proba[:len(sample_customers)]
                    sample_customers["Churn_Pred"] = preds[:len(sample_customers)]
                    st.markdown('<div class="small-muted">Sample predictions</div>', unsafe_allow_html=True)
                    st.dataframe(sample_customers.head(3), use_container_width=True)
                    download_df(sample_customers.head(500), "churn_predictions.csv", label="Download churn predictions (sample)")
                except Exception:
                    st.info("Could not map predictions back to original rows for download. Use the returned predictions object instead.")

    # -------------------------
    # Automated Insights (Segment-level)
    # -------------------------
    st.markdown('<div class="section-title">Automated Insights</div>', unsafe_allow_html=True)
    insights = []
    if "Channel" in filt.columns and "Churn_Flag" in filt.columns:
        ch_perf = filt.groupby("Channel").agg(Customers=("Customer_ID","nunique"), Churned=("Churn_Flag","sum")).reset_index()
        ch_perf["Retention_Rate"] = np.where(ch_perf["Customers"]>0, 1 - ch_perf["Churned"]/ch_perf["Customers"], 0)
        if not ch_perf.empty:
            best = ch_perf.sort_values("Retention_Rate", ascending=False).iloc[0]
            worst = ch_perf.sort_values("Retention_Rate").iloc[0]
            insights.append({"Insight":"Best Channel by Retention", "Channel":best["Channel"], "Retention_Rate":best["Retention_Rate"]})
            insights.append({"Insight":"Worst Channel by Retention", "Channel":worst["Channel"], "Retention_Rate":worst["Retention_Rate"]})

    for seg in ["AgeGroup", "Device", "Country", "Gender"]:
        if seg in filt.columns and "Churn_Flag" in filt.columns:
            seg_df = filt.groupby(seg).agg(Customers=("Customer_ID","nunique"), Churned=("Churn_Flag","sum")).reset_index()
            seg_df["Retention_Rate"] = np.where(seg_df["Customers"]>0, 1 - seg_df["Churned"]/seg_df["Customers"], 0)
            top = seg_df.sort_values("Retention_Rate", ascending=False).head(1)
            bottom = seg_df.sort_values("Retention_Rate").head(1)
            if not top.empty:
                insights.append({"Insight":f"Top {seg} by Retention", seg: str(top.iloc[0][seg]), "Retention_Rate":top.iloc[0]["Retention_Rate"]})
            if not bottom.empty:
                insights.append({"Insight":f"Lowest {seg} by Retention", seg: str(bottom.iloc[0][seg]), "Retention_Rate":bottom.iloc[0]["Retention_Rate"]})

    if not insights:
        st.info("No automated insights generated for the selected filters.")
    else:
        ins_df = pd.DataFrame(insights)
        render_index_safe_table(ins_df)
        download_df(ins_df, "automated_insights.csv", label="Download automated insights (CSV)")
