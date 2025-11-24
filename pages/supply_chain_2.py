# app_order_fulfillment_sla.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO, StringIO
from datetime import datetime, timedelta, date
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
import math
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# App header / page config
# -----------------------------
st.set_page_config(page_title="Order Fulfillment & SLA Analytics", layout="wide", page_icon="🚚")

# -----------------------------
# Styling: left-aligned title, card glow, KPI cards
# -----------------------------
st.markdown("""
<style>
/* Left aligned app title */
.header-row { display:flex; align-items:center; gap:14px; margin-bottom:6px; }
.header-title { font-size:30px; color:#064b86; font-weight:700; margin:0; text-align:left; }
.header-sub { color:#064b86; font-size:14px; margin:0 0 6px 0; text-align:left; }

/* Card / KPI glow */
.card {
  padding:16px;
  border-radius:10px;
  background: #ffffff;
  border: 1px solid #e6eef8;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.card:hover {
  transform: translateY(-6px);
  box-shadow: 0 10px 30px rgba(6,75,134,0.18);
  border-color: #064b86;
}

/* KPI label-only big */
.kpi {
  padding: 28px 16px;
  border-radius: 12px;
  text-align:center;
  font-weight:700;
  font-size:20px;
  color: #064b86;
  background: #fff;
  border: 1px solid #e6eef8;
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.kpi:hover {
  transform: translateY(-6px);
  box-shadow: 0 10px 30px rgba(6,75,134,0.18);
}

/* Left-align text inside card content */
.left-text { text-align: left !important; }

/* Smaller caption */
.small { font-size:13px; color:#666; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header / logo
# -----------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div class="header-row">
  <img src="{logo_url}" width="56" style="margin-right:8px;">
  <div style="line-height:1.05;">
    <div class="header-title">Order Fulfillment & SLA Analytics</div>
    <div class="header-sub">Reduce SLA breaches, improve fulfillment accuracy and speed — data-driven actions for operations.</div>
  </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Helper utilities
# -----------------------------
DEFAULT_URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/supply_chain/order_fulfillment_sla_analytics.csv"
# Replace the above with your raw GitHub CSV URL

def read_csv_safe(url_or_file):
    """
    Read CSV either from a URL or uploaded file-like object.
    If duplicate column names exist, make them unique using suffixes.
    """
    # use pandas to read; StringIO if bytes
    if hasattr(url_or_file, "read"):
        # it's a file-like object from uploader
        content = url_or_file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        df = pd.read_csv(StringIO(content))
    else:
        df = pd.read_csv(url_or_file)
    cols = list(df.columns)
    if len(cols) != len(set(cols)):
        # make unique
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

def download_df(df, filename, label="Download CSV"):
    b = BytesIO()
    b.write(df.to_csv(index=False).encode("utf-8"))
    b.seek(0)
    st.download_button(label, b, file_name=filename, mime="text/csv")

def safe_mean(x):
    try:
        return float(np.nanmean(x))
    except:
        return None

# -----------------------------
# Layout: tabs Overview & Application
# -----------------------------
tabs = st.tabs(["Overview", "Application"])

# -----------------------------
# Overview Tab content
# -----------------------------
with tabs[0]:
    # Purpose card (left aligned)
    st.markdown('<div class="card left-text"><b>Purpose</b>: Provide end-to-end order fulfillment visibility, predict SLA breaches, and optimise processes to meet delivery commitments.</div>', unsafe_allow_html=True)
    st.write("")  # small spacing

    left, right = st.columns(2)
    with left:
        st.markdown("#### Capabilities")
        st.markdown("""
        <div class="card left-text">
        • SLA breach prediction and root-cause attribution.<br>
        • Order-level and warehouse-level EDA & time-series trends.<br>
        • Fulfillment accuracy and short-ship detection.<br>
        • Clustering of order archetypes for operational focus.<br>
        • Exportable prioritized playbooks for operations.
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("#### Business impact")
        st.markdown("""
        <div class="card left-text">
        • Improved on-time delivery and SLA compliance.<br>
        • Reduced rush logistics and penalty costs.<br>
        • Higher customer satisfaction and lower churn.<br>
        • Better capacity planning and inventory alignment.
        </div>
        """, unsafe_allow_html=True)

    # KPI cards: 5 label-only in a single row (responsive)
    st.markdown("### KPIs")
    kcols = st.columns(5)
    klabels = ["Orders Tracked","SLA Breach Rate","Avg Fulfillment Time","Qty Accuracy","Short-Ship Rate"]
    for col, label in zip(kcols, klabels):
        col.markdown(f"<div class='kpi'>{label}</div>", unsafe_allow_html=True)

    st.markdown("### Who should use & How")
    st.markdown("""
    <div class='card left-text'>
    <b>Who</b>: Operations leads, Warehouse managers, Logistics planners, Supply chain analysts.<br><br>
    <b>How</b>: 1) Load dataset (default or upload). 2) Filter by date/warehouse/product/priority. 3) Inspect EDA and ML predictions. 4) Export playbooks (top-risk orders, routes) and schedule corrective actions.
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Application tab: dataset options + EDA + ML + insights
# -----------------------------
with tabs[1]:
    st.header("Application")
    st.markdown("Choose dataset option (Default = GitHub RAW URL). You will replace DEFAULT_URL with your raw CSV URL in the code.")

    mode = st.radio("Dataset option:", ["Default dataset (GitHub RAW URL)", "Upload CSV", "Upload CSV + Column mapping"], horizontal=True)
    df = None

    # canonical columns expected by the app (from your generator)
    EXPECTED_COLS = [
        "Order_ID","Customer_ID","Order_Date","Warehouse","Region","Product_Type","Channel",
        "Processing_Delay_Hrs","Picking_Delay_Hrs","Packing_Delay_Hrs","Dispatch_Delay_Hrs","Transport_Delay_Hrs",
        "Total_Fulfillment_Hours","SLA_Hours","SLA_Breach_Flag","Order_Qty","Fulfilled_Qty","Qty_Accuracy","Short_Ship_Flag",
        "Root_Cause","Priority","Fulfillment_Completed_At","Shipping_Cost","Delay_Cost"
    ]

    if mode == "Default dataset (GitHub RAW URL)":
        try:
            df = read_csv_safe(DEFAULT_URL)
            st.success("Default dataset loaded (DEFAULT_URL).")
            st.dataframe(df.head())
        except Exception as e:
            st.error("Failed to load default dataset. Check DEFAULT_URL or network. Error: " + str(e))
            st.stop()

    elif mode == "Upload CSV":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/supply_chain/order_fulfillment_sla_analytics.csv"
        st.markdown("#### Download Sample CSV for Reference")
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

        
    else:  # Upload + mapping
        uploaded = st.file_uploader("Upload CSV and map columns", type=["csv"])
        if uploaded is not None:
            raw = read_csv_safe(uploaded)
            st.write("Preview (first 5 rows):")
            st.dataframe(raw.head())
            st.markdown("Map your columns to the expected column names (map only existing ones):")
            mapping = {}
            cols = list(raw.columns)
            for expected in EXPECTED_COLS:
                mapping[expected] = st.selectbox(f"Map → {expected}", options=["-- Skip --"] + cols, index=0, key=f"map_{expected}")
            if st.button("Apply mapping"):
                # build rename dict
                rename = {}
                for k,v in mapping.items():
                    if v != "-- Skip --":
                        rename[v] = k
                if rename:
                    df = raw.rename(columns=rename)
                    st.success("Mapping applied.")
                    st.dataframe(df.head())
                else:
                    st.error("No mappings provided.")
            else:
                st.stop()
        else:
            st.stop()

    # Ensure dataset loaded
    if df is None:
        st.stop()

    # -----------------------------
    # Basic cleaning and canonicalization
    # -----------------------------
    df.columns = [str(c).strip() for c in df.columns]

    # If duplicates like Total_Fulfillment_Hours__dup1 exist, prefer first via helper
    def prefer_column(df, base):
        for c in df.columns:
            if c == base:
                return c
        for c in df.columns:
            if c.startswith(base + "__dup"):
                return c
        return None

    # canonical rename map attempt (if user uploaded variant names)
    canonical_map = {}
    for base in EXPECTED_COLS:
        found = prefer_column(df, base)
        if found and found != base:
            canonical_map[found] = base
    if canonical_map:
        df = df.rename(columns=canonical_map)

    # Parse date columns
    for col in ["Order_Date", "Fulfillment_Completed_At"]:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    # Convert numeric columns safely
    numeric_cols = ["Processing_Delay_Hrs","Picking_Delay_Hrs","Packing_Delay_Hrs","Dispatch_Delay_Hrs","Transport_Delay_Hrs",
                    "Total_Fulfillment_Hours","SLA_Hours","SLA_Breach_Flag","Order_Qty","Fulfilled_Qty","Qty_Accuracy",
                    "Short_Ship_Flag","Shipping_Cost","Delay_Cost"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fill simple defaults
    if "SLA_Breach_Flag" not in df.columns:
        df["SLA_Breach_Flag"] = 0
    if "Qty_Accuracy" not in df.columns and "Order_Qty" in df.columns and "Fulfilled_Qty" in df.columns:
        df["Qty_Accuracy"] = (df["Fulfilled_Qty"] / df["Order_Qty"]).fillna(1)

    # -----------------------------
    # Date filter (safe): use date_input to avoid slider typing issues
    # -----------------------------
    st.markdown("### Filters")
    min_dt = df["Order_Date"].min() if "Order_Date" in df.columns and not df["Order_Date"].isna().all() else None
    max_dt = df["Order_Date"].max() if "Order_Date" in df.columns and not df["Order_Date"].isna().all() else None

    # Provide defaults for date_input
    if min_dt is None:
        min_dt = datetime.now() - timedelta(days=30)
    if max_dt is None:
        max_dt = datetime.now()

    date_range = st.date_input("Order date range", value=(min_dt.date(), max_dt.date()))
    date_start = datetime.combine(date_range[0], datetime.min.time())
    date_end = datetime.combine(date_range[1], datetime.max.time())

    # other filters
    warehouses = df["Warehouse"].unique().tolist() if "Warehouse" in df.columns else []
    products = df["Product_Type"].unique().tolist() if "Product_Type" in df.columns else []
    priorities = df["Priority"].unique().tolist() if "Priority" in df.columns else []
    regions = df["Region"].unique().tolist() if "Region" in df.columns else []

    f1, f2, f3, f4 = st.columns(4)
    sel_wh = f1.multiselect("Warehouse", options=warehouses, default=warehouses if warehouses else [])
    sel_prod = f2.multiselect("Product Type", options=products, default=products if products else [])
    sel_pr = f3.multiselect("Priority", options=priorities, default=priorities if priorities else [])
    sel_reg = f4.multiselect("Region", options=regions, default=regions if regions else [])

    # Apply filters
    filt = df.copy()
    if "Order_Date" in filt.columns:
        filt = filt[(filt["Order_Date"] >= date_start) & (filt["Order_Date"] <= date_end)]
    if sel_wh:
        if "Warehouse" in filt.columns:
            filt = filt[filt["Warehouse"].isin(sel_wh)]
    if sel_prod:
        if "Product_Type" in filt.columns:
            filt = filt[filt["Product_Type"].isin(sel_prod)]
    if sel_pr:
        if "Priority" in filt.columns:
            filt = filt[filt["Priority"].isin(sel_pr)]
    if sel_reg:
        if "Region" in filt.columns:
            filt = filt[filt["Region"].isin(sel_reg)]

    st.markdown("Preview (first 10 rows)")
    st.dataframe(filt.head(10), use_container_width=True)
    download_df(filt.head(200), "filtered_preview.csv", label="Download filtered preview (200 rows)")

    # -----------------------------
    # Dynamic KPI values (below the label-only KPIs)
    # -----------------------------
    st.markdown("### KPIs (dynamic values)")
    k1, k2, k3, k4, k5 = st.columns(5)
    orders_count = int(filt.shape[0])
    sla_breach_rate = (filt["SLA_Breach_Flag"].mean()*100) if "SLA_Breach_Flag" in filt.columns and filt.shape[0]>0 else 0
    avg_fulfill = safe_mean(filt["Total_Fulfillment_Hours"]) if "Total_Fulfillment_Hours" in filt.columns else None
    qty_accuracy = safe_mean(filt["Qty_Accuracy"]) if "Qty_Accuracy" in filt.columns else None
    short_ship_rate = (filt["Short_Ship_Flag"].mean()*100) if "Short_Ship_Flag" in filt.columns and filt.shape[0]>0 else 0

    k1.metric("Orders", orders_count)
    k2.metric("SLA Breach Rate", f"{sla_breach_rate:.2f}%")
    k3.metric("Avg Fulfillment Hrs", f"{avg_fulfill:.2f}" if avg_fulfill is not None else "N/A")
    k4.metric("Avg Qty Accuracy", f"{qty_accuracy:.2%}" if qty_accuracy is not None else "N/A")
    k5.metric("Short-Ship Rate", f"{short_ship_rate:.2f}%")

    # -----------------------------
    # EDA: Many varied charts (not repeating same ones)
    # -----------------------------
    st.markdown("## Exploratory Data Analysis (EDA)")

    # 1. Orders by Warehouse (bar)
    if "Warehouse" in filt.columns:
        fig_wh = px.histogram(filt, x="Warehouse", title="Orders by Warehouse", text_auto=True)
        st.plotly_chart(fig_wh, use_container_width=True)

    # 2. SLA breach over time (daily)
    if "Order_Date" in filt.columns and "SLA_Breach_Flag" in filt.columns:
        daily = filt.set_index("Order_Date").resample("D").agg({"SLA_Breach_Flag":"mean","Order_ID":"count"}).reset_index()
        fig_sla = go.Figure()
        fig_sla.add_trace(go.Bar(x=daily["Order_Date"], y=daily["Order_ID"], name="Orders"))
        fig_sla.add_trace(go.Scatter(x=daily["Order_Date"], y=daily["SLA_Breach_Flag"], name="SLA Breach Rate", yaxis="y2", mode="lines+markers"))
        fig_sla.update_layout(title="Daily Orders and SLA Breach Rate", yaxis_title="Orders", yaxis2=dict(title="Breach Rate", overlaying="y", side="right"))
        st.plotly_chart(fig_sla, use_container_width=True)

    # 3. Distribution of Total Fulfillment Hours
    if "Total_Fulfillment_Hours" in filt.columns:
        fig_dist = px.histogram(filt, x="Total_Fulfillment_Hours", nbins=50, title="Distribution of Fulfillment Hours")
        st.plotly_chart(fig_dist, use_container_width=True)

    # 4. Box: Fulfillment by Product Type
    if "Product_Type" in filt.columns and "Total_Fulfillment_Hours" in filt.columns:
        fig_box = px.box(filt, x="Product_Type", y="Total_Fulfillment_Hours", title="Fulfillment Time by Product Type")
        st.plotly_chart(fig_box, use_container_width=True)

    # 5. Scatter: Shipping Cost vs Delay Cost
    if "Shipping_Cost" in filt.columns and "Delay_Cost" in filt.columns:
        fig_sc = px.scatter(filt, x="Shipping_Cost", y="Delay_Cost", color="Priority" if "Priority" in filt.columns else None,
                            hover_data=["Order_ID"], title="Shipping Cost vs Delay Cost")
        st.plotly_chart(fig_sc, use_container_width=True)

    # 6. Heatmap: average fulfillment time by Warehouse x Product_Type
    if "Warehouse" in filt.columns and "Product_Type" in filt.columns and "Total_Fulfillment_Hours" in filt.columns:
        pivot = filt.pivot_table(index="Warehouse", columns="Product_Type", values="Total_Fulfillment_Hours", aggfunc="mean").fillna(0)
        fig_hm = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="Blues"))
        fig_hm.update_layout(title="Avg Fulfillment Hours: Warehouse vs Product Type")
        st.plotly_chart(fig_hm, use_container_width=True)

    # 7. Top root causes stack (bar)
    if "Root_Cause" in filt.columns:
        rc = filt["Root_Cause"].value_counts().reset_index()
        rc.columns = ["Root_Cause","Count"]
        fig_rc = px.bar(rc, x="Root_Cause", y="Count", title="Root Cause Frequency")
        st.plotly_chart(fig_rc, use_container_width=True)

    # 8. Pie: Priority mix
    if "Priority" in filt.columns:
        pr = filt["Priority"].value_counts().reset_index()
        pr.columns = ["Priority","Count"]
        fig_pr = px.pie(pr, names="Priority", values="Count", title="Order Priority Mix")
        st.plotly_chart(fig_pr, use_container_width=True)

    # 9. Time-to-fulfillment CDF (ecdf via histogram cumulative)
    if "Total_Fulfillment_Hours" in filt.columns:
        fig_cdf = px.histogram(filt, x="Total_Fulfillment_Hours", cumulative=True, nbins=100, title="Cumulative Fulfillment Time (CDF)")
        st.plotly_chart(fig_cdf, use_container_width=True)

    # 10. Scatter: Qty accuracy vs Total Fulfillment Hours
    if "Qty_Accuracy" in filt.columns and "Total_Fulfillment_Hours" in filt.columns:
        fig_qa = px.scatter(filt, x="Total_Fulfillment_Hours", y="Qty_Accuracy", color="Short_Ship_Flag" if "Short_Ship_Flag" in filt.columns else None,
                            title="Qty Accuracy vs Fulfillment Time", hover_data=["Order_ID"])
        st.plotly_chart(fig_qa, use_container_width=True)

    # 11. Small multiples: distribution of delays (processing/picking/packing/dispatch/transport)
    delay_fields = [c for c in ["Processing_Delay_Hrs","Picking_Delay_Hrs","Packing_Delay_Hrs","Dispatch_Delay_Hrs","Transport_Delay_Hrs"] if c in filt.columns]
    if delay_fields:
        fig_delays = make_subplots = None
        # use px.histogram with facet_col
        try:
            melt = filt[delay_fields + ["Order_ID"]].melt(id_vars=["Order_ID"], value_vars=delay_fields, var_name="Delay_Type", value_name="Hours")
            fig_multi = px.histogram(melt, x="Hours", color="Delay_Type", facet_col="Delay_Type", nbins=40, title="Delay type distributions (small multiples)")
            fig_multi.update_layout(showlegend=False)
            st.plotly_chart(fig_multi, use_container_width=True)
        except Exception:
            pass

    # 12. Bar: Avg Delay by Region
    if "Region" in filt.columns and "Total_Fulfillment_Hours" in filt.columns:
        region_avg = filt.groupby("Region")["Total_Fulfillment_Hours"].mean().reset_index()
        fig_r = px.bar(region_avg, x="Region", y="Total_Fulfillment_Hours", title="Avg Fulfillment by Region")
        st.plotly_chart(fig_r, use_container_width=True)

    # add any extra charts user may appreciate
    st.markdown("### Additional charts")
    # 13. Top 10 orders by Delay Cost
    if "Delay_Cost" in filt.columns and "Order_ID" in filt.columns:
        top_delay = filt.sort_values("Delay_Cost", ascending=False).head(10)
        fig_td = px.bar(top_delay, x="Order_ID", y="Delay_Cost", title="Top 10 Orders by Delay Cost")
        st.plotly_chart(fig_td, use_container_width=True)

    # -----------------------------
    # Machine Learning section (4 concepts)
    #   1) Classification: SLA breach (RandomForestClassifier)
    #   2) Regression: Predict Total_Fulfillment_Hours (RandomForestRegressor)
    #   3) Clustering: KMeans segments
    #   4) Anomaly detection: IsolationForest (flagging weird orders)
    # -----------------------------
    st.markdown("## Machine Learning & Predictions")

    # Prepare convenient feature matrix (auto-handle categorical features)
    # We'll select a small default feature set (delays + priority + product + warehouse + qty)
    feat_candidates = ["Processing_Delay_Hrs","Picking_Delay_Hrs","Packing_Delay_Hrs","Dispatch_Delay_Hrs","Transport_Delay_Hrs",
                       "Order_Qty","Fulfilled_Qty","Shipping_Cost","Delay_Cost"]
    cat_candidates = ["Warehouse","Product_Type","Priority","Region","Channel"]

    feats = [c for c in feat_candidates if c in filt.columns]
    cats = [c for c in cat_candidates if c in filt.columns]

    st.markdown("### 1) Classification — Predict SLA breach (RandomForestClassifier)")
    if "SLA_Breach_Flag" in filt.columns and len(filt) >= 80 and (len(feats) >= 1 or len(cats) >= 1):
        # build X, y
        X = filt[feats + cats].copy()
        y = filt["SLA_Breach_Flag"].astype(int).copy()

        # simple preprocessing pipeline: one-hot for cats, scaler for numerics
        num_cols = [c for c in feats if c in X.columns]
        cat_cols = [c for c in cats if c in X.columns]
        transformer = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols)
        ], remainder="drop")

        try:
            X_t = transformer.fit_transform(X.fillna(0))
            Xtr, Xte, ytr, yte = train_test_split(X_t, y.values, test_size=0.2, random_state=42, stratify=y.values if len(np.unique(y.values))>1 else None)
            clf = RandomForestClassifier(n_estimators=200, random_state=42)
            clf.fit(Xtr, ytr)
            probs = clf.predict_proba(Xte)[:,1]
            preds = (probs >= 0.5).astype(int)
            acc = accuracy_score(yte, preds)
            try:
                auc = roc_auc_score(yte, probs)
            except:
                auc = None
            st.write(f"Classifier — Accuracy: {acc:.3f}" + (f", ROC AUC: {auc:.3f}" if auc is not None else ""))
            # prepare downloadable results (actual vs predicted + input features)
            # map test indices back to original rows: use train_test_split on indices
            indices = np.arange(X_t.shape[0])
            train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y.values if len(np.unique(y.values))>1 else None)
            X_test_orig = filt.reset_index(drop=True).loc[test_idx, feats + cats].reset_index(drop=True)
            out_clf = X_test_orig.copy()
            out_clf["Actual_SLA_Breach"] = yte
            out_clf["Predicted_Prob"] = probs
            out_clf["Predicted_Label"] = preds
            st.dataframe(out_clf.head(10))
            download_df(out_clf, "ml_sla_classification_results.csv", label="Download classification results")
        except Exception as e:
            st.error("Classification failed: " + str(e))
    else:
        st.info("Not enough data or required columns for SLA classification. Need SLA_Breach_Flag and at least ~80 rows and features.")

    st.markdown("### 2) Regression — Predict Total Fulfillment Hours (RandomForestRegressor)")
    if "Total_Fulfillment_Hours" in filt.columns and len(filt) >= 80 and (len(feats) >= 1 or len(cats) >= 1):
        Xr = filt[feats + cats].copy()
        yr = filt["Total_Fulfillment_Hours"].astype(float).copy()
        num_cols_r = [c for c in feats if c in Xr.columns]
        cat_cols_r = [c for c in cats if c in Xr.columns]
        transformer_r = ColumnTransformer([
            ("num", StandardScaler(), num_cols_r),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols_r)
        ], remainder="drop")
        try:
            Xr_t = transformer_r.fit_transform(Xr.fillna(0))
            Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(Xr_t, yr.values, test_size=0.2, random_state=42)
            from sklearn.ensemble import RandomForestRegressor
            rfr = RandomForestRegressor(n_estimators=200, random_state=42)
            rfr.fit(Xtr_r, ytr_r)
            preds_r = rfr.predict(Xte_r)
            rmse = math.sqrt(mean_squared_error(yte_r, preds_r))
            r2 = r2_score(yte_r, preds_r)
            st.write(f"Regression — RMSE: {rmse:.3f}, R²: {r2:.3f}")
            # prepare downloadable actual vs predicted with features
            indices_r = np.arange(Xr_t.shape[0])
            train_idx_r, test_idx_r = train_test_split(indices_r, test_size=0.2, random_state=42)
            X_test_orig_r = filt.reset_index(drop=True).loc[test_idx_r, feats + cats].reset_index(drop=True)
            out_reg = X_test_orig_r.copy()
            out_reg["Actual_Fulfillment_Hrs"] = yte_r
            out_reg["Pred_Fulfillment_Hrs"] = preds_r
            st.dataframe(out_reg.head(10))
            download_df(out_reg, "ml_fulfillment_regression.csv", label="Download regression results")
        except Exception as e:
            st.error("Regression failed: " + str(e))
    else:
        st.info("Not enough data for fulfillment regression. Need Total_Fulfillment_Hours and sufficient rows.")

    st.markdown("### 3) Clustering — Order archetypes (KMeans)")
    # clustering on numeric delay features + qty + cost
    cluster_fields = [c for c in ["Processing_Delay_Hrs","Picking_Delay_Hrs","Packing_Delay_Hrs","Dispatch_Delay_Hrs","Transport_Delay_Hrs","Total_Fulfillment_Hours","Order_Qty","Delay_Cost"] if c in filt.columns]
    if len(cluster_fields) >= 2 and len(filt) >= 50:
        try:
            scaler = StandardScaler()
            Xcl = scaler.fit_transform(filt[cluster_fields].fillna(0))
            k = st.slider("Choose number of clusters (K)", min_value=2, max_value=8, value=4)
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(Xcl)
            filt["_cluster"] = labels
            st.write("Cluster counts:")
            st.dataframe(pd.Series(labels).value_counts().rename_axis("cluster").reset_index(name="count"))
            # show cluster center summaries
            centers = pd.DataFrame(km.cluster_centers_, columns=cluster_fields)
            centers = pd.DataFrame(scaler.inverse_transform(centers), columns=cluster_fields)
            centers["cluster"] = centers.index
            st.markdown("Cluster centers (approx):")
            st.dataframe(centers.set_index("cluster"))
            # simple scatter of two most important dims if available
            if "Total_Fulfillment_Hours" in filt.columns and "Delay_Cost" in filt.columns:
                fig_k = px.scatter(filt, x="Total_Fulfillment_Hours", y="Delay_Cost", color="_cluster", hover_data=["Order_ID"], title="Clusters: Fulfillment vs Delay Cost")
                st.plotly_chart(fig_k, use_container_width=True)
            download_df(filt[["Order_ID"] + cluster_fields + ["_cluster"]], "clustering_orders.csv", label="Download clustering results")
        except Exception as e:
            st.error("Clustering failed: " + str(e))
    else:
        st.info("Not enough numeric fields or rows for clustering.")

    st.markdown("### 4) Anomaly Detection — IsolationForest (weird orders)")
    # anomaly on numeric delay features + delay cost
    anom_fields = [c for c in ["Processing_Delay_Hrs","Picking_Delay_Hrs","Packing_Delay_Hrs","Dispatch_Delay_Hrs","Transport_Delay_Hrs","Delay_Cost"] if c in filt.columns]
    if len(anom_fields) >= 2 and len(filt) >= 50:
        try:
            iso = IsolationForest(contamination=0.02, random_state=42)
            af = filt[anom_fields].fillna(0)
            iso_pred = iso.fit_predict(af)
            filt["_is_anomaly"] = np.where(iso_pred == -1, 1, 0)
            anomalies = filt.loc[filt["_is_anomaly"]==1, ["Order_ID"] + anom_fields + ["_is_anomaly"]]
            st.markdown(f"Detected {int(anomalies.shape[0])} anomalies (orders).")
            if not anomalies.empty:
                st.dataframe(anomalies.head(10))
                download_df(anomalies, "anomalies_orders.csv", label="Download anomalies (orders)")
        except Exception as e:
            st.error("Anomaly detection failed: " + str(e))
    else:
        st.info("Not enough fields/rows for anomaly detection.")

    # -----------------------------
    # Automated Insights: generate simple rules & recommendations
    # -----------------------------
    st.markdown("## Automated Insights & Playbooks")
    insights = []

    # Top risk warehouses (by SLA breach rate)
    if "Warehouse" in filt.columns and "SLA_Breach_Flag" in filt.columns:
        wh = filt.groupby("Warehouse").agg(total_orders=("Order_ID","count"), breaches=("SLA_Breach_Flag","sum")).reset_index()
        wh["breach_rate"] = np.where(wh["total_orders"]>0, wh["breaches"]/wh["total_orders"], 0)
        top_wh = wh.sort_values("breach_rate", ascending=False).head(5)
        for _, r in top_wh.iterrows():
            insights.append({
                "Insight_Type":"Top-risk Warehouse",
                "Entity": r["Warehouse"],
                "Metric":"SLA breach rate",
                "Value": round(float(r["breach_rate"]),4),
                "Action":"Investigate capacity, staffing and dispatch backlog"
            })

    # Top root causes
    if "Root_Cause" in filt.columns:
        rc = filt["Root_Cause"].value_counts().head(8)
        for rc_name, count in rc.items():
            insights.append({
                "Insight_Type":"Root cause frequency",
                "Entity": rc_name,
                "Metric":"Count",
                "Value": int(count),
                "Action":"Targeted kaizen or supplier review recommended"
            })

    # High delay cost orders
    if "Delay_Cost" in filt.columns and "Order_ID" in filt.columns:
        high_cost = filt.sort_values("Delay_Cost", ascending=False).head(5)
        for _, r in high_cost.iterrows():
            insights.append({
                "Insight_Type":"High delay cost order",
                "Entity": r["Order_ID"],
                "Metric":"Delay_Cost",
                "Value": float(r["Delay_Cost"]),
                "Action":"Expedite or review process to avoid similar costs"
            })

    # Clusters needing manual review (if cluster column exists)
    if "_cluster" in filt.columns:
        cluster_counts = filt["_cluster"].value_counts().to_dict()
        for cluster, cnt in cluster_counts.items():
            insights.append({
                "Insight_Type":"Cluster summary",
                "Entity": f"Cluster {cluster}",
                "Metric":"Order count",
                "Value": int(cnt),
                "Action":"Review cluster center and create playbook"
            })

    # Add simple performance recommendation based on SLA breach rate
    overall_breach = float(filt["SLA_Breach_Flag"].mean()) if "SLA_Breach_Flag" in filt.columns else 0
    insights.append({
        "Insight_Type":"Overall SLA breach",
        "Entity":"All",
        "Metric":"Breach rate",
        "Value": round(overall_breach,4),
        "Action": "If breach rate > 0.05, implement immediate triage of high-priority orders"
    })

    insights_df = pd.DataFrame(insights)
    if insights_df.empty:
        st.info("No automated insights could be produced for the current selection.")
    else:
        st.dataframe(insights_df, use_container_width=True)
        download_df(insights_df, "automated_insights.csv", label="Download automated insights")

    st.markdown("### Export / Next steps")
    st.markdown("""
    - Use exported classification/regression/clustering outputs to operationalize actions.<br>
    - Export playbooks: top-risk orders, warehouses, clusters requiring attention.<br>
    - Hook model predictions into operational systems (WMS/TMS) for real-time alerts.
    """, unsafe_allow_html=True)

# End of app
