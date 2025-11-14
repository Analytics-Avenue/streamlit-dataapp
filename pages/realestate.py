# real_estate_dashboard_final.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import io
import requests

st.set_page_config(page_title="Real Estate Analytics — Final", layout="wide")
st.title("🏡 Real Estate Analytics — Final Dashboard")
st.markdown("Single-page dashboard • 3 dataset options • Column mapping • Charts • KPIs • ML prediction")

# -----------------------
# Config: GitHub raw URL (change this to your raw CSV if you have one)
# -----------------------
GITHUB_RAW = "https://raw.githubusercontent.com/your-username/your-repo/main/real_estate_sample.csv"
# Fallback local file (path generated earlier by the assistant). If not available, synthetic generator will be used.
LOCAL_FALLBACK = "/mnt/data/sample_dataset_5000_rows.csv"

# -----------------------
# Expected internal columns and friendly names (app uses these keys)
# -----------------------
EXPECTED_COLUMNS = {
    "Listing_ID": "Listing_ID",
    "Country": "Country",
    "City": "City",
    "Location": "Location",
    "Property_Type": "Property_Type",
    "Bedrooms": "Bedrooms",
    "Bathrooms": "Bathrooms",
    "Square_Footage": "Square_Footage",
    "Age_of_Property": "Age_of_Property",
    "Listing_Date": "Listing_Date",
    "Price": "Price",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
    "Seller_Type": "Seller_Type",
    "Furnishing": "Furnishing",
    "Parking": "Parking",
    "Demand_Score": "Demand_Score",
    "School_Rating": "School_Rating",
    "Interest_Rate": "Interest_Rate",
    "Economic_Index": "Economic_Index",
    "Days_On_Market": "Days_On_Market",
    "Price_per_SqFt": "Price_per_SqFt"
}

# -----------------------
# Helpers
# -----------------------
def make_sample_csv_bytes(n=200):
    # small sample template with required columns and realistic sample values
    sample = pd.DataFrame({
        "Listing_ID": [f"SMP{i+1:03d}" for i in range(n)],
        "Country": np.random.choice(["India","USA","UAE","Singapore","UK"], n),
        "City": np.random.choice(["Bangalore","Mumbai","Chennai","New York","Dubai","Singapore","London"], n),
        "Location": np.random.choice(["Downtown","Suburban","Uptown","Midtown","Industrial"], n),
        "Property_Type": np.random.choice(["Apartment","Villa","Condo","Townhouse","Studio"], n),
        "Bedrooms": np.random.randint(1,5,n),
        "Bathrooms": np.random.randint(1,4,n),
        "Square_Footage": np.random.randint(350,4000,n),
        "Age_of_Property": np.random.randint(0,30,n),
        "Listing_Date": pd.to_datetime("2024-01-01") + pd.to_timedelta(np.random.randint(0,200,n), unit="D"),
        "Price": (np.random.randint(30,250,n) * 100000).astype(int),
        "Latitude": np.random.uniform(-33,51,n),
        "Longitude": np.random.uniform(-118,151,n),
        "Seller_Type": np.random.choice(["Builder","Agent","Owner"], n),
        "Furnishing": np.random.choice(["Unfurnished","Semi","Fully"], n),
        "Parking": np.random.choice(["Yes","No"], n),
        "Demand_Score": np.random.randint(10,100,n),
        "School_Rating": np.random.randint(1,10,n),
        "Interest_Rate": np.random.uniform(3.0,9.0,n).round(2),
        "Economic_Index": np.random.randint(80,160,n),
        "Days_On_Market": np.random.randint(1,250,n)
    })
    sample["Price_per_SqFt"] = (sample["Price"] / sample["Square_Footage"]).round(2)
    buf = io.StringIO()
    sample.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8"), sample

def try_read_csv_from_url(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    except Exception:
        return None

def safe_transform_transformer(transformer, X):
    """Apply transformer and ensure dense numpy array returned (works with sparse and dense outputs)."""
    Xt = transformer.transform(X)
    # if sparse matrix
    if hasattr(Xt, "toarray"):
        return Xt.toarray()
    # else assume numpy array
    return np.asarray(Xt)

# -----------------------
# Sidebar: Dataset source choice
# -----------------------
st.sidebar.header("1) Data source")
data_source = st.sidebar.radio("Choose dataset source:",
                               ("Use default GitHub dataset", "Upload CSV", "Upload CSV & Map Columns"))

# sample CSV download
csv_bytes, sample_df_small = make_sample_csv_bytes(n=200)
st.sidebar.markdown("Download a **sample CSV** (headers + sample rows):")
st.sidebar.download_button("Download sample CSV", data=csv_bytes, file_name="real_estate_sample.csv", mime="text/csv")

# -----------------------
# Load df_raw depending on choice
# -----------------------
df_raw = None
mapping = {}  # for mapping mode

if data_source == "Use default GitHub dataset":
    st.sidebar.info("Loading dataset from GitHub. If it fails, app will use local fallback or a synthetic sample.")
    df_try = try_read_csv_from_url(GITHUB_RAW)
    if df_try is not None:
        df_raw = df_try
        st.sidebar.success("Loaded dataset from GitHub.")
    else:
        # try local fallback
        try:
            df_local = pd.read_csv(LOCAL_FALLBACK)
            df_raw = df_local
            st.sidebar.warning("GitHub fetch failed — loaded local fallback dataset.")
        except Exception:
            df_raw = sample_df_small.copy()
            st.sidebar.warning("GitHub and local fallback not available — using internal sample dataset.")

elif data_source == "Upload CSV" or data_source == "Upload CSV & Map Columns":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.sidebar.success("Uploaded dataset loaded.")
        except Exception as e:
            st.sidebar.error("Failed to read uploaded CSV.")
            st.sidebar.write(str(e))
            df_raw = None

# If user chose mapping mode, show mapping UI when df_raw present
if data_source == "Upload CSV & Map Columns":
    st.sidebar.subheader("Manual column mapping")
    if df_raw is None:
        st.sidebar.info("Upload a CSV to map columns.")
    else:
        # Provide mapping selectboxes for expected columns
        df_cols = [c for c in df_raw.columns]
        for key in EXPECTED_COLUMNS.keys():
            # default attempt: if friendly name exists in df, choose it
            default_idx = 0
            default_val = None
            if key in df_raw.columns:
                default_val = key
            else:
                # try lowercase match
                for c in df_raw.columns:
                    if c.strip().lower() == key.lower() or key.lower() in c.strip().lower():
                        default_val = c
                        break
            selection = st.sidebar.selectbox(f"Map '{key}' →", options=["<none>"] + df_cols, index=(df_cols.index(default_val)+1 if default_val in df_cols else 0), key=f"map_{key}")
            if selection != "<none>":
                mapping[key] = selection

# -----------------------
# Build mapped_df (app internal columns)
# -----------------------
mapped_df = None
if df_raw is not None:
    if data_source == "Upload CSV & Map Columns" and mapping:
        # create mapped_df using mapping explicitly
        mapped_df = pd.DataFrame()
        for internal in EXPECTED_COLUMNS.keys():
            chosen = mapping.get(internal)
            if chosen and chosen in df_raw.columns:
                mapped_df[internal] = df_raw[chosen]
            else:
                mapped_df[internal] = None
    else:
        # Attempt auto-mapping by column names (case-insensitive)
        mapped_df = pd.DataFrame()
        lower_map = {c.strip().lower(): c for c in df_raw.columns}
        for internal in EXPECTED_COLUMNS.keys():
            # exact match
            if internal in df_raw.columns:
                mapped_df[internal] = df_raw[internal]
            else:
                # look for friendly names or substrings
                found = None
                for col_lower, col_orig in lower_map.items():
                    if internal.lower() == col_lower or internal.lower() in col_lower or col_lower in internal.lower():
                        found = col_orig
                        break
                if found:
                    mapped_df[internal] = df_raw[found]
                else:
                    # no match: create empty column
                    mapped_df[internal] = None

    # Basic type conversions
    # Date
    if "Listing_Date" in mapped_df.columns:
        try:
            mapped_df["Listing_Date"] = pd.to_datetime(mapped_df["Listing_Date"], errors="coerce")
        except Exception:
            pass

    # numerics
    numeric_fields = ["Bedrooms","Bathrooms","Square_Footage","Age_of_Property","Price","Latitude","Longitude","Demand_Score","School_Rating","Interest_Rate","Economic_Index","Days_On_Market","Price_per_SqFt"]
    for nf in numeric_fields:
        if nf in mapped_df.columns:
            mapped_df[nf] = pd.to_numeric(mapped_df[nf], errors="coerce")

else:
    # no df_raw: use internal synthetic sample (200 rows)
    _, sample_small = make_sample_csv_bytes(n=200)
    mapped_df = sample_small.copy()
    mapped_df["Listing_Date"] = pd.to_datetime(mapped_df["Listing_Date"])

# If Price_per_SqFt missing but Price & Square_Footage present, compute
if ("Price_per_SqFt" in mapped_df.columns) and (mapped_df["Price_per_SqFt"].isna().all()):
    if "Price" in mapped_df.columns and "Square_Footage" in mapped_df.columns:
        mapped_df["Price_per_SqFt"] = (mapped_df["Price"] / mapped_df["Square_Footage"]).replace([np.inf, -np.inf], np.nan)

# -----------------------
# Filters (sidebar)
# -----------------------
st.sidebar.markdown("---")
st.sidebar.header("Filters")

# City filter if column available
if "City" in mapped_df.columns and mapped_df["City"].notna().any():
    cities = mapped_df["City"].dropna().unique().tolist()
    sel_cities = st.sidebar.multiselect("City", options=sorted(cities), default=sorted(cities)[:5])
else:
    sel_cities = []

# Property_Type filter
if "Property_Type" in mapped_df.columns and mapped_df["Property_Type"].notna().any():
    types = mapped_df["Property_Type"].dropna().unique().tolist()
    sel_types = st.sidebar.multiselect("Property Type", options=sorted(types), default=sorted(types))
else:
    sel_types = []

# Price range
if "Price" in mapped_df.columns and mapped_df["Price"].notna().any():
    pmin = int(np.nanmin(mapped_df["Price"]))
    pmax = int(np.nanmax(mapped_df["Price"]))
    pr = st.sidebar.slider("Price range", pmin, pmax, (pmin, pmax))
else:
    pr = (0, 0)

# Bedrooms filter
if "Bedrooms" in mapped_df.columns and mapped_df["Bedrooms"].notna().any():
    bmin = int(mapped_df["Bedrooms"].min())
    bmax = int(mapped_df["Bedrooms"].max())
    sel_bed = st.sidebar.slider("Bedrooms", bmin, bmax, (bmin, min(bmin+2, bmax)))
else:
    sel_bed = (0,0)

# Date range
if "Listing_Date" in mapped_df.columns and mapped_df["Listing_Date"].notna().any():
    min_date = mapped_df["Listing_Date"].min().date()
    max_date = mapped_df["Listing_Date"].max().date()
    sel_date = st.sidebar.date_input("Listing Date range", (min_date, max_date), min_value=min_date, max_value=max_date)
else:
    sel_date = None

# -----------------------
# Apply filters to produce filtered_df
# -----------------------
df = mapped_df.copy()
# city
if sel_cities:
    if "City" in df.columns:
        df = df[df["City"].isin(sel_cities)]
# type
if sel_types:
    if "Property_Type" in df.columns:
        df = df[df["Property_Type"].isin(sel_types)]
# price
if pr != (0,0) and "Price" in df.columns:
    df = df[(df["Price"] >= pr[0]) & (df["Price"] <= pr[1])]
# bedrooms
if sel_bed != (0,0) and "Bedrooms" in df.columns:
    df = df[(df["Bedrooms"] >= sel_bed[0]) & (df["Bedrooms"] <= sel_bed[1])]
# listing date
if sel_date and "Listing_Date" in df.columns:
    df = df[(pd.to_datetime(df["Listing_Date"]).dt.date >= sel_date[0]) & (pd.to_datetime(df["Listing_Date"]).dt.date <= sel_date[1])]

# Small guard: if df empty, fallback to small sample
if df.shape[0] == 0:
    st.warning("Filters returned 0 rows — falling back to a small sample so visuals still render.")
    _, sample_small = make_sample_csv_bytes(n=200)
    df = sample_small.copy()
    df["Listing_Date"] = pd.to_datetime(df["Listing_Date"])

# -----------------------
# Page layout: Tabs (single page with sections)
# -----------------------
st.markdown("---")
tab_overview, tab_heatmap, tab_leaderboard, tab_ml, tab_raw = st.tabs(
    ["Overview", "Region Heatmap", "Agent Leaderboard", "ML Price Prediction", "Raw Data / Download"]
)

# -----------------------
# Overview Tab
# -----------------------
with tab_overview:
    st.header("Overview: KPIs & Charts")
    # KPIs row
    c1, c2, c3, c4, c5 = st.columns(5)
    total_listings = len(df)
    avg_price = df["Price"].mean() if "Price" in df.columns else np.nan
    median_pps = df["Price_per_SqFt"].median() if "Price_per_SqFt" in df.columns else np.nan
    avg_days = df["Days_On_Market"].mean() if "Days_On_Market" in df.columns else np.nan
    avg_leads = df["Demand_Score"].mean() if "Demand_Score" in df.columns else np.nan

    c1.metric("Total Listings", f"{total_listings:,}")
    c2.metric("Avg Price", f"₹{avg_price:,.0f}" if not np.isnan(avg_price) else "N/A")
    c3.metric("Median ₹/sqft", f"₹{median_pps:,.0f}" if not np.isnan(median_pps) else "N/A")
    c4.metric("Avg Days on Market", f"{avg_days:.1f}" if not np.isnan(avg_days) else "N/A")
    c5.metric("Avg Demand Score", f"{avg_leads:.1f}" if not np.isnan(avg_leads) else "N/A")

    st.markdown("**Purpose:** High-level KPIs to track supply, price and demand at-a-glance.")
    st.markdown("**Quick Tip:** Hover bars/boxes for exact measures; use filters to narrow to specific cities or types.")

    st.markdown("---")
    # Charts: price distribution, box by city, avg pps by city, property type pie
    st.subheader("Price Distribution")
    if "Price" in df.columns:
        fig_price = px.histogram(df, x="Price", nbins=40, title="Price Distribution", labels={"Price":"Price (local)"})
        fig_price.update_traces(texttemplate="%{y}", textposition="inside")
        st.plotly_chart(fig_price, use_container_width=True)
        st.markdown("**Purpose:** See where most prices lie and detect heavy tails/outliers.")
        st.markdown("**Quick Tip:** If distribution is very skewed, consider log-price transformations for modeling.")
    else:
        st.info("Price column missing — can't show price distribution.")

    st.markdown("---")
    # Box by city (if available)
    if "City" in df.columns and "Price" in df.columns:
        st.subheader("Price by City & Property Type")
        fig_box = px.box(df, x="City", y="Price", color="Property_Type", title="Price by City & Type", points="outliers")
        fig_box.update_traces(boxmean=True)
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown("**Purpose:** Compare pricing across cities and property types.")
        st.markdown("**Quick Tip:** Use this to find premium vs affordable cities at a glance.")
    else:
        st.info("City or Price column not available — skipping city-level boxplot.")

    st.markdown("---")
    # Avg price per sqft by city
    if "City" in df.columns and "Price_per_SqFt" in df.columns:
        st.subheader("Average Price per Sqft by City")
        pps = df.groupby("City", as_index=False)["Price_per_SqFt"].mean().sort_values("Price_per_SqFt", ascending=False)
        fig_pps = px.bar(pps, x="City", y="Price_per_SqFt", title="Avg Price per Sqft by City", text_auto=".2f")
        st.plotly_chart(fig_pps, use_container_width=True)
        st.markdown("**Purpose:** Surface high-value micro-markets.")
        st.markdown("**Quick Tip:** Combine with Days_On_Market to understand overheating vs healthy markets.")
    else:
        st.info("City or Price_per_SqFt column missing — skipping pps chart.")

    st.markdown("---")
    # Property Type Distribution
    if "Property_Type" in df.columns:
        st.subheader("Property Type Distribution")
        prop_counts = df["Property_Type"].value_counts().reset_index()
        prop_counts.columns = ["Property_Type","Count"]
        fig_pie = px.pie(prop_counts, names="Property_Type", values="Count", title="Property Type Mix", hole=0.35)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("**Purpose:** Understand supply mix to align product strategy.")
        st.markdown("**Quick Tip:** If one type dominates, consider diversification strategies.")
    else:
        st.info("Property_Type column missing — skipping property mix chart.")

# -----------------------
# Heatmap Tab
# -----------------------
with tab_heatmap:
    st.header("Region Heatmap")
    st.markdown("**Purpose:** Identify price pockets and demand across micro-locations. Quick Tip: Zoom to inspect cells.")

    # pivot Price_per_SqFt by Location x Property_Type
    if "Location" in df.columns and "Property_Type" in df.columns and "Price_per_SqFt" in df.columns:
        heat_df = df.pivot_table(index="Location", columns="Property_Type", values="Price_per_SqFt", aggfunc="mean")
        # fill NaN with 0 for plotting clarity
        heat_plot_df = heat_df.fillna(0)
        fig_heat = px.imshow(heat_plot_df, title="Avg ₹/sqft by Location and Property Type", labels=dict(x="Property Type", y="Location", color="Avg ₹/sqft"))
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Need Location, Property_Type and Price_per_SqFt columns for heatmap.")

# -----------------------
# Agent Leaderboard Tab
# -----------------------
with tab_leaderboard:
    st.header("Agent Leaderboard")
    st.markdown("**Purpose:** Rank agents by offers / price sold / listings. Quick Tip: Use filters to compare same-city agents.")

    if "Agent" in df.columns:
        # compute metrics for agents (fallback to Listings count if Offers missing)
        metrics = {}
        g = df.groupby("Agent").agg({
            col: "sum" if col in ["Price","Demand_Score","Offers","Leads","Site_Visits"] else "count" for col in ["Price","Offers","Leads","Site_Visits"]
        } , dropna=False)
        # Some datasets won't have Offers/Leads columns; safely calculate basic metrics:
        agent_df = df.groupby("Agent").agg(
            Listings=("Listing_ID","count") if "Listing_ID" in df.columns else ("City","count"),
            Avg_Price=("Price","mean") if "Price" in df.columns else (df.columns[0],"count")
        ).reset_index()
        # add offers if available
        if "Offers" in df.columns:
            agent_df["Offers"] = df.groupby("Agent")["Offers"].sum().values
        else:
            agent_df["Offers"] = 0
        agent_df["Rank_by_Offers"] = agent_df["Offers"].rank(method="min", ascending=False).astype(int)
        agent_df = agent_df.sort_values(["Offers","Avg_Price"], ascending=[False, False]).reset_index(drop=True)
        st.dataframe(agent_df.head(20), use_container_width=True)
        st.markdown("**Quick Tip:** Drill into a top agent's listings to uncover high-converting attributes.")
    else:
        st.info("Agent column not found — cannot compute leaderboard.")

# -----------------------
# ML Price Prediction Tab
# -----------------------
with tab_ml:
    st.header("ML Price Prediction")
    st.markdown("Train a quick RandomForest on the filtered data and predict price for a custom listing.")
    st.markdown("**Quick Tip:** For reliable predictions you need clean, sufficiently large datasets (min ~100 rows).")

    # Choose features (auto-detect)
    # Provide recommended features in order
    recommended_feats = []
    for col in ["City","Location","Property_Type","Square_Footage","Bedrooms","Bathrooms","Age_of_Property","Demand_Score","School_Rating","Price_per_SqFt"]:
        if col in df.columns:
            recommended_feats.append(col)

    st.subheader("Model configuration")
    target_col = st.selectbox("Select target column (price)", options=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])], index=0)
    feat_sel = st.multiselect("Select features (recommended preselected)", options=df.columns.tolist(), default=recommended_feats)

    if len(feat_sel) < 1:
        st.warning("Select at least one feature to train the model.")
    else:
        X = df[feat_sel].copy()
        y = df[target_col].copy()

        # Drop rows with NA in X or y
        data_xy = pd.concat([X,y], axis=1).dropna()
        X = data_xy[feat_sel]
        y = data_xy[target_col]

        if len(X) < 40:
            st.warning(f"Not enough rows ({len(X)}) after filtering/NA removal to train a reliable model. Try broader filters or upload more data.")
        else:
            # Separate categorical and numeric
            cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
            num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]

            # Build transformer (OneHotEncoder handle_unknown only)
            ohe = OneHotEncoder(handle_unknown="ignore")
            preprocessor = ColumnTransformer(transformers=[
                ("ohe", ohe, cat_cols),
                ("num", StandardScaler(), num_cols)
            ], remainder="drop")

            # Fit transform
            try:
                X_t = preprocessor.fit_transform(X)
            except Exception as e:
                st.error("Preprocessing failed: " + str(e))
                X_t = None

            if X_t is not None:
                # ensure dense
                if hasattr(X_t, "toarray"):
                    X_t = X_t.toarray()

                # train test split
                X_train, X_test, y_train, y_test = train_test_split(X_t, y.values, test_size=0.2, random_state=42)

                model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
                with st.spinner("Training model..."):
                    model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                # Compute MSE then RMSE to be compatible with all sklearn versions
                mse = mean_squared_error(y_test, y_pred)
                rmse = mse ** 0.5
                r2 = r2_score(y_test, y_pred)

                st.metric("RMSE", f"₹{rmse:,.0f}")
                st.metric("R²", f"{r2:.3f}")

                # Feature importance: need feature names after OHE
                try:
                    ohe_names = []
                    if len(cat_cols) > 0:
                        ohe_names = list(preprocessor.named_transformers_["ohe"].get_feature_names_out(cat_cols))
                except Exception:
                    ohe_names = []
                feature_names = ohe_names + num_cols
                importances = model.feature_importances_
                fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(30)
                st.subheader("Top feature importances")
                st.dataframe(fi_df.reset_index(drop=True), use_container_width=True)

                st.markdown("### Predict price for a new listing")
                # Build prediction form inputs based on feat_sel
                pred_inputs = {}
                cols_form = st.columns(3)
                for i, f in enumerate(feat_sel):
                    col = cols_form[i % 3]
                    if pd.api.types.is_numeric_dtype(df[f]):
                        pred_inputs[f] = col.number_input(f, value=float(df[f].median() if df[f].median()==df[f].median() else 0))
                    else:
                        pred_inputs[f] = col.selectbox(f, options=sorted(df[f].dropna().unique()))

                if st.button("Predict"):
                    input_df = pd.DataFrame([pred_inputs])
                    # transform via preprocessor (fill missing categories)
                    try:
                        X_in = preprocessor.transform(input_df)
                        if hasattr(X_in, "toarray"):
                            X_in = X_in.toarray()
                        pred_price = model.predict(X_in)[0]
                        st.success(f"Predicted Price: ₹{pred_price:,.0f}")
                        if "Square_Footage" in input_df.columns and input_df["Square_Footage"].iloc[0] > 0:
                            st.info(f"Predicted Price per sqft: ₹{pred_price / input_df['Square_Footage'].iloc[0]:,.2f}")
                    except Exception as e:
                        st.error("Prediction failed: " + str(e))

# -----------------------
# Raw Data / Download Tab
# -----------------------
with tab_raw:
    st.header("Raw Data & Download")
    st.write("Filtered dataset used for charts and ML:")
    st.dataframe(df.reset_index(drop=True), use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", data=csv, file_name="real_estate_filtered.csv", mime="text/csv")

# Footer
st.markdown("---")
st.caption("Dashboard: single-file final. Replace GITHUB_RAW with your raw CSV URL to use a remote default dataset.")
