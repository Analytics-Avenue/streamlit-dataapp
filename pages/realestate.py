# real_estate_dashboard_final_v2.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import requests
import random
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import inspect

st.set_page_config(page_title="Real Estate Analytics — Full (Option 2)", layout="wide")
st.title("🏡 Real Estate Analytics — Full Dashboard (Industry Model)")

# -----------------------
# CONFIG
# -----------------------
# Replace with your hosted CSV raw URL if you have one.
GITHUB_RAW = "https://raw.githubusercontent.com/your-username/your-repo/main/real_estate_full_sample.csv"
LOCAL_FALLBACK = "/mnt/data/sample_dataset_5000_rows.csv"  # assistant-created earlier

# Expected full ML feature set for Option 2
EXPECTED_ML_COLUMNS = [
    "Price", "Square_Footage", "Bedrooms", "Bathrooms",
    "City", "Property_Type", "Year_Built", "Parking", "Furnishing",
    "Latitude", "Longitude"
]

# App helper: make a downloadable sample csv
def make_sample_csv(n=300):
    rng = np.random.default_rng(42)
    countries = ["India", "USA", "UAE", "Singapore", "UK"]
    cities = ["Bangalore","Mumbai","Chennai","Delhi","Pune","Hyderabad","Kolkata",
              "New York","London","Dubai","Singapore"]
    property_types = ["Apartment","Villa","Condo","Townhouse","Studio"]
    furnishing_opts = ["Unfurnished","Semi","Fully"]
    parking_opts = ["Yes","No"]
    rows = []
    for i in range(n):
        city = rng.choice(cities)
        country = "India" if city in ["Bangalore","Mumbai","Chennai","Delhi","Pune","Hyderabad","Kolkata"] else ("USA" if city=="New York" else ("UK" if city=="London" else ("UAE" if city=="Dubai" else "Singapore")))
        sqft = int(rng.integers(300, 3500))
        bedrooms = int(min(6, max(1, sqft // 500)))
        bathrooms = int(max(1, bedrooms - rng.integers(0,2)))
        year_built = int(1980 + int(rng.integers(0,40)))
        price_base = {
            "Bangalore":9000, "Mumbai":18000, "Chennai":6500, "Delhi":12000, "Pune":8500, "Hyderabad":6000, "Kolkata":4000,
            "New York":2000, "London":1700, "Dubai":2000, "Singapore":2200
        }.get(city, 8000)
        price = int(sqft * price_base * rng.normal(1.0, 0.12))
        lat = round(rng.uniform(-33,51),6)
        lon = round(rng.uniform(-118,151),6)
        rows.append({
            "Listing_ID": f"S{1000+i}",
            "Country": country,
            "City": city,
            "Location": f"Loc_{rng.integers(1,50)}",
            "Property_Type": rng.choice(property_types),
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Square_Footage": sqft,
            "Year_Built": year_built,
            "Price": price,
            "Latitude": lat,
            "Longitude": lon,
            "Seller_Type": rng.choice(["Builder","Agent","Owner"]),
            "Furnishing": rng.choice(furnishing_opts),
            "Parking": rng.choice(parking_opts),
            "Demand_Score": int(rng.integers(20,100)),
            "School_Rating": int(rng.integers(1,10)),
            "Interest_Rate": round(rng.uniform(3.0,8.0),2),
            "Economic_Index": int(rng.integers(80,160)),
            "Days_On_Market": int(rng.integers(1,240))
        })
    df = pd.DataFrame(rows)
    df["Price_per_SqFt"] = (df["Price"] / df["Square_Footage"]).round(2)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8"), df

sample_bytes, sample_df = make_sample_csv(n=300)

# -----------------------
# Sidebar: data options + sample download
# -----------------------
st.sidebar.header("Dataset options")
st.sidebar.markdown("Choose one of the three data sources and map columns if needed.")
data_source = st.sidebar.radio("Data source:", ("Default GitHub dataset", "Upload CSV", "Upload CSV & Map Columns"))

st.sidebar.markdown("Download a sample CSV (headers + sample rows):")
st.sidebar.download_button("Download sample CSV", data=sample_bytes, file_name="realestate_sample.csv", mime="text/csv")

# -----------------------
# Load dataset according to selection
# -----------------------
df_raw = None
mapping = {}

def try_read_csv_url(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    except Exception:
        return None

if data_source == "Default GitHub dataset":
    st.sidebar.info("Attempting to load dataset from GitHub (fallbacks enabled).")
    df_try = try_read_csv_url(GITHUB_RAW)
    if df_try is not None:
        df_raw = df_try
        st.sidebar.success("Loaded dataset from GitHub.")
    else:
        # fallback to local file
        try:
            df_local = pd.read_csv(LOCAL_FALLBACK)
            df_raw = df_local
            st.sidebar.warning("GitHub fetch failed; loaded local fallback dataset.")
        except Exception:
            df_raw = sample_df.copy()
            st.sidebar.warning("Using internal synthetic sample dataset (no GitHub/local available).")

elif data_source == "Upload CSV" or data_source == "Upload CSV & Map Columns":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
            st.sidebar.success("Uploaded CSV loaded.")
        except Exception as e:
            st.sidebar.error("Failed to read uploaded CSV: " + str(e))
            df_raw = None
    else:
        st.sidebar.info("No file uploaded yet.")

# -----------------------
# If mapping option chosen, show mapping UI
# -----------------------
if data_source == "Upload CSV & Map Columns":
    st.sidebar.subheader("Manual column mapping (map your CSV columns to expected fields)")
    if df_raw is None:
        st.sidebar.info("Upload a CSV to map columns.")
    else:
        cols = df_raw.columns.tolist()
        for expected in EXPECTED_ML_COLUMNS + ["Listing_ID","Country","Location","Seller_Type","Price_per_SqFt","Days_On_Market","Demand_Score","School_Rating"]:
            default = None
            # try to auto-suggest a matching column name
            for c in cols:
                if c.strip().lower() == expected.strip().lower() or expected.strip().lower() in c.strip().lower() or c.strip().lower() in expected.strip().lower():
                    default = c
                    break
            sel = st.sidebar.selectbox(f"{expected} →", options=["<none>"] + cols, index=(cols.index(default)+1 if default in cols else 0), key=f"map_{expected}")
            if sel != "<none>":
                mapping[expected] = sel

# -----------------------
# Build mapped dataframe (app internal names)
# -----------------------
if df_raw is not None:
    df = pd.DataFrame()
    # if mapping provided, use it
    if mapping:
        for internal in set(list(mapping.keys()) + ["Listing_ID","Country","City","Location","Property_Type","Bedrooms","Bathrooms","Square_Footage","Year_Built","Price","Latitude","Longitude","Furnishing","Parking","Seller_Type","Demand_Score","School_Rating","Days_On_Market","Price_per_SqFt"]):
            src = mapping.get(internal)
            if src and src in df_raw.columns:
                df[internal] = df_raw[src]
            else:
                # try to auto-find
                matched = None
                for c in df_raw.columns:
                    if c.strip().lower() == internal.strip().lower() or internal.strip().lower() in c.strip().lower() or c.strip().lower() in internal.strip().lower():
                        matched = c
                        break
                if matched:
                    df[internal] = df_raw[matched]
                else:
                    df[internal] = None
    else:
        # auto-map by best-effort name matching
        for expected in ["Listing_ID","Country","City","Location","Property_Type","Bedrooms","Bathrooms","Square_Footage","Year_Built","Price","Latitude","Longitude","Furnishing","Parking","Seller_Type","Demand_Score","School_Rating","Days_On_Market","Price_per_SqFt"]:
            chosen = None
            for c in df_raw.columns:
                if c.strip().lower() == expected.strip().lower():
                    chosen = c
                    break
            if chosen:
                df[expected] = df_raw[chosen]
            else:
                # try substring matches
                found = None
                for c in df_raw.columns:
                    if expected.strip().lower() in c.strip().lower() or c.strip().lower() in expected.strip().lower():
                        found = c
                        break
                if found:
                    df[expected] = df_raw[found]
                else:
                    df[expected] = None
else:
    df = sample_df.copy()

# Normalize basic types
for col in df.columns:
    if col in ["Listing_ID","Country","City","Location","Property_Type","Furnishing","Parking","Seller_Type"]:
        df[col] = df[col].astype(str)
    if col in ["Listing_Date"]:
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            pass

# ensure Price_per_SqFt
if "Price_per_SqFt" not in df.columns or df["Price_per_SqFt"].isna().all():
    if "Price" in df.columns and "Square_Footage" in df.columns:
        df["Price_per_SqFt"] = (pd.to_numeric(df["Price"], errors="coerce") / pd.to_numeric(df["Square_Footage"], errors="coerce")).replace([np.inf,-np.inf], np.nan)

# -----------------------
# If Agent column missing, create synthetic 'Agent' column for leaderboard
# -----------------------
if "Agent" not in df.columns or df["Agent"].isnull().all():
    # create Agent column with random agents
    agents = ["Aditi","Manoj","Riya","Kiran","Sanjay","Neha","Vikram","Priya","Rahul","Sunil"]
    df["Agent"] = [random.choice(agents) for _ in range(len(df))]

# -----------------------
# Filters (Sidebar)
# -----------------------
st.sidebar.markdown("---")
st.sidebar.header("Filters")

if "City" in df.columns and df["City"].notna().any():
    city_options = sorted(df["City"].dropna().unique().tolist())
    city_sel = st.sidebar.multiselect("City", city_options, default=city_options[:6])
else:
    city_sel = []

if "Property_Type" in df.columns and df["Property_Type"].notna().any():
    ptype_options = sorted(df["Property_Type"].dropna().unique().tolist())
    ptype_sel = st.sidebar.multiselect("Property Type", ptype_options, default=ptype_options)
else:
    ptype_sel = []

# Price range
if "Price" in df.columns and pd.to_numeric(df["Price"], errors="coerce").notna().any():
    price_min = int(pd.to_numeric(df["Price"], errors="coerce").min())
    price_max = int(pd.to_numeric(df["Price"], errors="coerce").max())
    pr = st.sidebar.slider("Price Range", price_min, price_max, (price_min, price_max))
else:
    pr = (0,0)

# bedrooms range
if "Bedrooms" in df.columns and pd.to_numeric(df["Bedrooms"], errors="coerce").notna().any():
    b_min = int(pd.to_numeric(df["Bedrooms"], errors="coerce").min())
    b_max = int(pd.to_numeric(df["Bedrooms"], errors="coerce").max())
    bed_sel = st.sidebar.slider("Bedrooms", b_min, b_max, (b_min, min(b_min+2, b_max)))
else:
    bed_sel = (0,0)

# Apply filters
df_filtered = df.copy()
if city_sel and "City" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["City"].isin(city_sel)]
if ptype_sel and "Property_Type" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Property_Type"].isin(ptype_sel)]
if pr != (0,0) and "Price" in df_filtered.columns:
    df_filtered = df_filtered[(pd.to_numeric(df_filtered["Price"], errors="coerce") >= pr[0]) & (pd.to_numeric(df_filtered["Price"], errors="coerce") <= pr[1])]
if bed_sel != (0,0) and "Bedrooms" in df_filtered.columns:
    df_filtered = df_filtered[(pd.to_numeric(df_filtered["Bedrooms"], errors="coerce") >= bed_sel[0]) & (pd.to_numeric(df_filtered["Bedrooms"], errors="coerce") <= bed_sel[1])]

if df_filtered.shape[0] == 0:
    st.warning("No rows match your filters — falling back to sample data for visualisation.")
    _, sample_df_sm = make_sample_csv(n=200)
    df_filtered = sample_df_sm.copy()

# -----------------------
# Layout: Tabs
# -----------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview","Region Heatmap","Agent Leaderboard","ML Price Prediction","Raw Data / Download"])

# -----------------------
# Overview Tab
# -----------------------
with tab1:
    st.header("Overview: KPIs & Charts")
    total_listings = len(df_filtered)
    avg_price = pd.to_numeric(df_filtered["Price"], errors="coerce").mean() if "Price" in df_filtered.columns else np.nan
    median_pps = pd.to_numeric(df_filtered["Price_per_SqFt"], errors="coerce").median() if "Price_per_SqFt" in df_filtered.columns else np.nan
    avg_dom = pd.to_numeric(df_filtered["Days_On_Market"], errors="coerce").mean() if "Days_On_Market" in df_filtered.columns else np.nan
    avg_demand = pd.to_numeric(df_filtered["Demand_Score"], errors="coerce").mean() if "Demand_Score" in df_filtered.columns else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Listings", f"{total_listings:,}")
    c2.metric("Avg Price", f"₹{avg_price:,.0f}" if not np.isnan(avg_price) else "N/A")
    c3.metric("Median ₹/sqft", f"₹{median_pps:,.0f}" if not np.isnan(median_pps) else "N/A")
    c4.metric("Avg Days on Market", f"{avg_dom:.1f}" if not np.isnan(avg_dom) else "N/A")

    st.markdown("**Purpose:** Snapshot of supply, price level, and market velocity.")
    st.markdown("**Quick Tip:** Use filters to narrow by city/type and re-check KPIs.")

    st.markdown("---")
    # Price distribution
    if "Price" in df_filtered.columns:
        st.subheader("Price Distribution")
        fig_price = px.histogram(df_filtered, x="Price", nbins=40, title="Price Distribution", labels={"Price":"Price (local)"})
        fig_price.update_traces(texttemplate="%{y}", textposition="inside")
        st.plotly_chart(fig_price, use_container_width=True)
        st.markdown("**Purpose:** See where most listings are priced; detect outliers.")
        st.markdown("**Quick Tip:** Consider log-transforming price for modeling if skewed.")
    else:
        st.info("Price column missing — cannot show price distribution.")

    st.markdown("---")
    # Price by City box
    if "City" in df_filtered.columns and "Price" in df_filtered.columns:
        st.subheader("Price by City & Property Type")
        fig_box = px.box(df_filtered, x="City", y="Price", color="Property_Type", title="Price by City & Type", points="outliers")
        fig_box.update_traces(boxmean=True)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    # Avg price per sqft
    if "City" in df_filtered.columns and "Price_per_SqFt" in df_filtered.columns:
        st.subheader("Average Price per Sqft by City")
        pps = df_filtered.groupby("City", as_index=False)["Price_per_SqFt"].mean().sort_values("Price_per_SqFt", ascending=False)
        fig_pps = px.bar(pps, x="City", y="Price_per_SqFt", title="Avg ₹/sqft by City", text_auto=".2f")
        st.plotly_chart(fig_pps, use_container_width=True)

# -----------------------
# Heatmap Tab
# -----------------------
with tab2:
    st.header("Region Heatmap")
    st.markdown("**Purpose:** Spot high-value vs low-value micro-markets.")
    if "Location" in df_filtered.columns and "Property_Type" in df_filtered.columns and "Price_per_SqFt" in df_filtered.columns:
        heat_df = df_filtered.pivot_table(index="Location", columns="Property_Type", values="Price_per_SqFt", aggfunc="mean")
        heat_df = heat_df.fillna(0)
        fig_heat = px.imshow(heat_df, title="Avg ₹/sqft by Location and Property Type", labels=dict(x="Property Type", y="Location", color="Avg ₹/sqft"))
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Need Location, Property_Type and Price_per_SqFt columns for heatmap.")

# -----------------------
# Agent Leaderboard Tab
# -----------------------
with tab3:
    st.header("Agent Leaderboard")
    st.markdown("**Purpose:** Rank agents by volume and offers. Quick Tip: Focus on top 10 to replicate success patterns.")

    if "Agent" in df_filtered.columns:
        # Listings per agent
        ag = df_filtered.groupby("Agent").agg(
            Listings = ("Listing_ID", "count") if "Listing_ID" in df_filtered.columns else ("City","count"),
            Avg_Price = ("Price","mean") if "Price" in df_filtered.columns else ("Square_Footage","mean"),
            Total_Price = ("Price","sum") if "Price" in df_filtered.columns else ("Square_Footage","sum")
        ).reset_index()
        ag = ag.sort_values("Total_Price", ascending=False)
        st.dataframe(ag.head(20), use_container_width=True)
    else:
        st.info("Agent column not found — but we automatically created synthetic 'Agent' earlier so leaderboard should display.")
        st.write(df_filtered[["Agent"]].value_counts().head(20))

# -----------------------
# ML Price Prediction Tab
# -----------------------
with tab4:
    st.header("ML Price Prediction — Full Industry Model")
    st.markdown("Required model features: " + ", ".join(EXPECTED_ML_COLUMNS))
    st.markdown("**Purpose:** Predict listing price using a RandomForest trained on selected features.")
    st.markdown("**Quick Tip:** Ensure your dataset has consistent units (sqft), correct locations, and clean numeric values.")

    # Verify features exist; if missing, attempt to auto-fill or coerce
    df_ml = df_filtered.copy()

    # Ensure column names for EXPECTED_ML_COLUMNS are present (try case-insensitive matches)
    col_map = {}
    existing_cols = {c.lower(): c for c in df_ml.columns}
    for expected in EXPECTED_ML_COLUMNS:
        if expected in df_ml.columns:
            col_map[expected] = expected
        else:
            el = expected.lower()
            if el in existing_cols:
                col_map[expected] = existing_cols[el]
            else:
                # try partial match
                found = None
                for c in df_ml.columns:
                    if expected.lower() in c.lower() or c.lower() in expected.lower():
                        found = c
                        break
                if found:
                    col_map[expected] = found
                else:
                    col_map[expected] = None

    # Create missing columns with sensible defaults
    for feature in EXPECTED_ML_COLUMNS:
        mapped = col_map.get(feature)
        if mapped is None:
            # auto-create
            if feature == "Price":
                # cannot auto-create price (target) — skip and warn
                st.warning("Target column 'Price' missing in dataset. Predictions will be unreliable.")
                df_ml[feature] = np.nan
            elif feature in ["Square_Footage","Bedrooms","Bathrooms","Year_Built","Latitude","Longitude"]:
                # numeric defaults
                if feature == "Square_Footage":
                    df_ml[feature] = df_ml.get(feature, pd.Series(np.nan)).fillna(1200)
                elif feature == "Bedrooms":
                    df_ml[feature] = df_ml.get(feature, pd.Series(np.nan)).fillna(2)
                elif feature == "Bathrooms":
                    df_ml[feature] = df_ml.get(feature, pd.Series(np.nan)).fillna(2)
                elif feature == "Year_Built":
                    df_ml[feature] = df_ml.get(feature, pd.Series(np.nan)).fillna(2005)
                else:
                    df_ml[feature] = df_ml.get(feature, pd.Series(np.nan)).fillna(0)
            else:
                # categorical defaults
                if feature in df_ml.columns:
                    continue
                if feature == "City":
                    df_ml[feature] = df_ml.get(feature, pd.Series(["Unknown"]*len(df_ml)))
                elif feature == "Property_Type":
                    df_ml[feature] = df_ml.get(feature, pd.Series(["Apartment"]*len(df_ml)))
                else:
                    df_ml[feature] = df_ml.get(feature, pd.Series(["Unknown"]*len(df_ml)))
        else:
            # ensure column exists under expected name by renaming a temp copy
            if mapped != feature:
                df_ml[feature] = df_ml[mapped]

    # Drop rows where target Price missing
    if df_ml["Price"].isna().all():
        st.error("No Price values found after mapping — cannot train model.")
    else:
        df_ml = df_ml.dropna(subset=["Price"])

        # Build features/target
        features = EXPECTED_ML_COLUMNS.copy()  # use their names now present in df_ml
        X = df_ml[features].copy()
        y = pd.to_numeric(df_ml["Price"], errors="coerce")

        # drop rows with NA in X or y
        data_xy = pd.concat([X,y], axis=1).dropna()
        X = data_xy[features]
        y = data_xy["Price"]

        if len(X) < 40:
            st.warning(f"Not enough rows ({len(X)}) to train a reliable model. Need >= 40 rows.")
        else:
            # separate categorical & numeric
            cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(X[c])]
            num_cols = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]

            # create OneHotEncoder with safe parameter depending on sklearn version
            try:
                # check signature for sparse_output
                sig = inspect.signature(OneHotEncoder)
                if 'sparse_output' in sig.parameters:
                    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                else:
                    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            except Exception:
                # fallback
                ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

            preprocessor = ColumnTransformer(transformers=[
                ("ohe", ohe, cat_cols),
                ("scale", StandardScaler(), num_cols)
            ], remainder="drop")

            try:
                X_t = preprocessor.fit_transform(X)
                # convert to dense if sparse matrix returned
                if hasattr(X_t, "toarray"):
                    X_t = X_t.toarray()
            except Exception as e:
                st.error("Error during preprocessing: " + str(e))
                X_t = None

            if X_t is not None:
                X_train, X_test, y_train, y_test = train_test_split(X_t, y.values, test_size=0.2, random_state=42)
                model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
                with st.spinner("Training RandomForest..."):
                    model.fit(X_train, y_train)

                preds = model.predict(X_test)
                mse = mean_squared_error(y_test, preds)
                rmse = mse ** 0.5
                r2 = r2_score(y_test, preds)

                st.subheader("Model Performance")
                st.write(f"RMSE: ₹{rmse:,.0f}   |   R²: {r2:.3f}")

                # feature importance (get names)
                try:
                    ohe_names = list(preprocessor.named_transformers_["ohe"].get_feature_names_out(cat_cols)) if len(cat_cols) > 0 else []
                except Exception:
                    # older sklearn may use get_feature_names
                    try:
                        ohe_names = list(preprocessor.named_transformers_["ohe"].get_feature_names(cat_cols))
                    except Exception:
                        ohe_names = []
                feat_names = ohe_names + num_cols
                fi = model.feature_importances_
                fi_df = pd.DataFrame({"feature": feat_names, "importance": fi}).sort_values("importance", ascending=False).head(40)
                st.subheader("Top Feature Importances")
                st.dataframe(fi_df.reset_index(drop=True), use_container_width=True)

                st.markdown("### Predict price for a new listing")
                # build form inputs
                form_cols = st.columns(3)
                input_vals = {}
                for i, f in enumerate(features):
                    col = form_cols[i % 3]
                    if f in num_cols:
                        input_vals[f] = col.number_input(f, value=float(X[f].median() if X[f].median()==X[f].median() else 0))
                    else:
                        options = sorted(X[f].dropna().unique().tolist())
                        if len(options) == 0:
                            options = ["Unknown"]
                        input_vals[f] = col.selectbox(f, options=options)

                if st.button("Predict Price"):
                    input_df = pd.DataFrame([input_vals])[features]
                    try:
                        X_input_t = preprocessor.transform(input_df)
                        if hasattr(X_input_t, "toarray"):
                            X_input_t = X_input_t.toarray()
                        pred_val = model.predict(X_input_t)[0]
                        st.success(f"Predicted Price: ₹{pred_val:,.0f}")
                        if "Square_Footage" in input_df.columns and input_df["Square_Footage"].iloc[0] > 0:
                            st.info(f"Predicted Price per sqft: ₹{pred_val / input_df['Square_Footage'].iloc[0]:,.2f}")
                    except Exception as e:
                        st.error("Prediction failed: " + str(e))

# -----------------------
# Raw Data / Download Tab
# -----------------------
with tab5:
    st.header("Raw Data / Download")
    st.write("Filtered and mapped dataset used for dashboard and ML.")
    st.dataframe(df_filtered.reset_index(drop=True), use_container_width=True)
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", data=csv, file_name="real_estate_filtered.csv", mime="text/csv")

st.markdown("---")
st.caption("Final dashboard (Option 2: Full industry model). If a GitHub default URL is available, replace GITHUB_RAW at the top of the file.")
