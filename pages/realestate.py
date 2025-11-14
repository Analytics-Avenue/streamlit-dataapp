# real_estate_platform_full.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import io
import random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
import inspect
import math

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Real Estate Intelligence Platform", layout="wide")
st.title("🏢 Real Estate Intelligence — Full Platform")
st.markdown("KPIs • Charts • Clustering • Forecasting • Recommender • Agent Analytics • Leaderboard • Geo Heatmaps • NLP Search • Lead Scoring • Time-to-sell")

# -------------------------
# Helpers & sample data
# -------------------------
GITHUB_RAW = "https://raw.githubusercontent.com/your-username/your-repo/main/real_estate_sample.csv"
LOCAL_FALLBACK = "/mnt/data/sample_dataset_5000_rows.csv"

def make_sample(n=5000, seed=42):
    np.random.seed(seed)
    cities_india = ["Bangalore","Mumbai","Chennai","Delhi","Pune","Hyderabad","Kolkata","Ahmedabad","Coimbatore","Lucknow"]
    cities_global = ["New York","London","Dubai","Singapore","Toronto"]
    cities = cities_india + cities_global
    property_types = ["Apartment","Villa","Condo","Townhouse","Studio"]
    furnishing = ["Unfurnished","Semi","Fully"]
    parking_opts = ["Yes","No"]
    sellers = ["Builder","Agent","Owner"]
    rows = []
    today = datetime.today()
    for i in range(n):
        city = random.choice(cities)
        country = "India" if city in cities_india else random.choice(["USA","UK","UAE","Singapore","Canada"])
        loc = f"Loc_{random.randint(1,200)}"
        ptype = random.choice(property_types)
        sqft = int(max(250, np.random.normal(1100, 450)))
        beds = max(1, sqft//450 if random.random()>0.2 else random.randint(1,5))
        baths = max(1, min(4, int(beds - (random.random()>0.7))))
        year_built = random.randint(1980, 2022)
        price_base = {
            "Bangalore":9000,"Mumbai":18000,"Chennai":6500,"Delhi":12000,"Pune":8500,"Hyderabad":6000,"Kolkata":4500,"Ahmedabad":4000,"Coimbatore":3500,"Lucknow":3000,
            "New York":2000,"London":1700,"Dubai":2000,"Singapore":2200,"Toronto":1300
        }.get(city,8000)
        price = int(sqft * price_base * np.random.normal(1.0, 0.12))
        lat = round(np.random.uniform(-33,51),6)
        lon = round(np.random.uniform(-118,151),6)
        agent = random.choice(["Aditi","Manoj","Riya","Kiran","Sanjay","Neha","Vikram","Priya","Rahul","Sunil"])
        seller = random.choice(sellers)
        furnish = random.choice(furnishing)
        parking = random.choice(parking_opts)
        demand = random.randint(20,100)
        school = random.randint(1,10)
        interest = round(np.random.uniform(3.0,8.0),2)
        econ = random.randint(80,160)
        dom = random.randint(1,240)
        listing_date = today - timedelta(days=random.randint(0,600))
        rows.append({
            "Listing_ID": f"L{i+1:06d}",
            "Country": country,
            "City": city,
            "Location": loc,
            "Property_Type": ptype,
            "Bedrooms": int(beds),
            "Bathrooms": int(baths),
            "Square_Footage": int(sqft),
            "Year_Built": year_built,
            "Price": price,
            "Latitude": lat,
            "Longitude": lon,
            "Agent": agent,
            "Seller_Type": seller,
            "Furnishing": furnish,
            "Parking": parking,
            "Demand_Score": demand,
            "School_Rating": school,
            "Interest_Rate": interest,
            "Economic_Index": econ,
            "Days_On_Market": dom,
            "Listing_Date": listing_date.date()
        })
    df = pd.DataFrame(rows)
    df["Price_per_SqFt"] = (df["Price"] / df["Square_Footage"]).round(2)
    return df

def try_read_csv(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    except Exception:
        return None

def ensure_unique_columns(df):
    # If duplicate column names exist, append _dupX
    cols = list(df.columns)
    seen = {}
    newcols = []
    for c in cols:
        key = c.strip()
        if key in seen:
            seen[key] += 1
            newcols.append(f"{key}_dup{seen[key]}")
        else:
            seen[key] = 0
            newcols.append(key)
    df.columns = newcols
    return df

def safe_ohe_instance():
    # choose correct param for OneHotEncoder depending on sklearn version
    try:
        sig = inspect.signature(OneHotEncoder)
        if 'sparse_output' in sig.parameters:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)
    except Exception:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# -------------------------
# Sidebar: Data source + sample
# -------------------------
st.sidebar.header("Dataset options")
data_mode = st.sidebar.radio("Choose data source:", ("Default GitHub dataset", "Upload CSV", "Upload CSV & Manual Mapping"))
sample_bytes = None
sample_df = make_sample(5000)
buf = io.StringIO()
sample_df.to_csv(buf, index=False)
sample_bytes = buf.getvalue().encode('utf-8')
st.sidebar.download_button("Download sample CSV (5k rows)", data=sample_bytes, file_name="real_estate_5000_sample.csv", mime="text/csv")

df_raw = None
mapping = {}

if data_mode == "Default GitHub dataset":
    st.sidebar.info("Attempting to load default GitHub dataset. If not reachable, app uses local/synth fallback.")
    df_try = try_read_csv(GITHUB_RAW)
    if df_try is not None:
        df_raw = df_try
        st.sidebar.success("Loaded dataset from GitHub.")
    else:
        try:
            df_local = pd.read_csv(LOCAL_FALLBACK)
            df_raw = df_local
            st.sidebar.warning("GitHub unreachable — loaded local fallback.")
        except Exception:
            df_raw = sample_df.copy()
            st.sidebar.warning("Using internal synthetic sample dataset.")

elif data_mode == "Upload CSV" or data_mode == "Upload CSV & Manual Mapping":
    uploaded = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df_tmp = pd.read_csv(uploaded)
            df_raw = df_tmp
            st.sidebar.success("Uploaded file loaded.")
        except Exception as e:
            st.sidebar.error("Failed to read uploaded CSV: " + str(e))
            df_raw = None

# -------------------------
# Manual mapping UI
# -------------------------
EXPECTED = [
    "Listing_ID","Country","City","Location","Property_Type","Bedrooms","Bathrooms",
    "Square_Footage","Year_Built","Price","Latitude","Longitude","Agent","Seller_Type",
    "Furnishing","Parking","Demand_Score","School_Rating","Interest_Rate","Economic_Index",
    "Days_On_Market","Listing_Date","Price_per_SqFt","Region"
]

if data_mode == "Upload CSV & Manual Mapping" and df_raw is not None:
    st.sidebar.subheader("Manual column mapping")
    cols = list(df_raw.columns)
    for exp in EXPECTED:
        default = None
        for c in cols:
            if c.strip().lower() == exp.strip().lower():
                default = c
                break
        mapping[exp] = st.sidebar.selectbox(f"Map '{exp}' →", options=["<none>"] + cols, index=(cols.index(default)+1 if default in cols else 0), key=f"map_{exp}")

# -------------------------
# Build working df (mapped_df)
# -------------------------
if df_raw is None:
    df = sample_df.copy()
else:
    df = df_raw.copy()
    df = ensure_unique_columns(df)
    # If manual mapping provided and not blank, use that
    if mapping and any(v and v!="<none>" for v in mapping.values()):
        df_mapped = pd.DataFrame()
        for exp in EXPECTED:
            sel = mapping.get(exp)
            if sel and sel != "<none>":
                if sel in df.columns:
                    df_mapped[exp] = df[sel]
                else:
                    df_mapped[exp] = np.nan
            else:
                # try auto-match by name
                found = None
                for c in df.columns:
                    if c.strip().lower() == exp.strip().lower() or exp.strip().lower() in c.strip().lower() or c.strip().lower() in exp.strip().lower():
                        found = c
                        break
                if found:
                    df_mapped[exp] = df[found]
                else:
                    df_mapped[exp] = np.nan
        df = df_mapped.copy()

# Normalize some columns, add defaults when missing
for col in ["Price","Square_Footage","Bedrooms","Bathrooms","Latitude","Longitude","Price_per_SqFt","Days_On_Market","Demand_Score","School_Rating"]:
    if col not in df.columns:
        # sensible defaults (keep as numeric)
        if col == "Price":
            df[col] = np.nan
        elif col == "Square_Footage":
            df[col] = 1200
        elif col == "Bedrooms":
            df[col] = 2
        elif col == "Bathrooms":
            df[col] = 2
        elif col == "Latitude":
            df[col] = np.nan
        elif col == "Longitude":
            df[col] = np.nan
        elif col == "Price_per_SqFt":
            df[col] = np.nan
        elif col == "Days_On_Market":
            df[col] = 60
        elif col == "Demand_Score":
            df[col] = 50
        elif col == "School_Rating":
            df[col] = 6

# Attempt parsing dates
if "Listing_Date" in df.columns:
    try:
        df["Listing_Date"] = pd.to_datetime(df["Listing_Date"], errors="coerce")
    except Exception:
        pass

# Compute Price_per_SqFt if possible
if "Price_per_SqFt" not in df.columns or df["Price_per_SqFt"].isna().all():
    if "Price" in df.columns and "Square_Footage" in df.columns:
        try:
            df["Price_per_SqFt"] = pd.to_numeric(df["Price"], errors="coerce") / pd.to_numeric(df["Square_Footage"], errors="coerce")
        except Exception:
            df["Price_per_SqFt"] = np.nan

# Ensure Agent exists
if "Agent" not in df.columns or df["Agent"].isnull().all():
    agents = ["Aditi","Manoj","Riya","Kiran","Sanjay","Neha","Vikram","Priya","Rahul","Sunil"]
    df["Agent"] = [random.choice(agents) for _ in range(len(df))]

# Ensure Region
if "Region" not in df.columns or df["Region"].isnull().all():
    df["Region"] = df["City"].fillna("Unknown") + "_" + df["Location"].fillna("L0").astype(str)

# Filter UI
st.sidebar.header("Filters")
city_opts = sorted(df["City"].dropna().unique().tolist()) if "City" in df.columns else []
city_sel = st.sidebar.multiselect("City", options=city_opts, default=city_opts[:6] if city_opts else [])
ptype_opts = sorted(df["Property_Type"].dropna().unique().tolist()) if "Property_Type" in df.columns else []
ptype_sel = st.sidebar.multiselect("Property Type", options=ptype_opts, default=ptype_opts if ptype_opts else [])
agent_opts = sorted(df["Agent"].dropna().unique().tolist())
agent_sel = st.sidebar.multiselect("Agent", options=agent_opts, default=agent_opts[:10] if agent_opts else [])

# Price slider
if df["Price"].notna().any():
    pmin = int(np.nanmin(pd.to_numeric(df["Price"], errors="coerce")))
    pmax = int(np.nanmax(pd.to_numeric(df["Price"], errors="coerce")))
    price_sel = st.sidebar.slider("Price Range", min_value=pmin, max_value=pmax, value=(pmin,pmax))
else:
    price_sel = (0,0)

# Apply filters
dff = df.copy()
if city_sel:
    dff = dff[dff["City"].isin(city_sel)]
if ptype_sel:
    dff = dff[dff["Property_Type"].isin(ptype_sel)]
if agent_sel:
    dff = dff[dff["Agent"].isin(agent_sel)]
if price_sel != (0,0) and "Price" in dff.columns:
    try:
        dff = dff[(pd.to_numeric(dff["Price"], errors="coerce") >= price_sel[0]) & (pd.to_numeric(dff["Price"], errors="coerce") <= price_sel[1])]
    except Exception:
        pass

# If filtered empty, warn and fallback
if dff.shape[0] == 0:
    st.warning("Filters returned 0 rows; using unfiltered dataset for visuals to avoid blank charts.")
    dff = df.copy()

# -------------------------
# Layout: KPIs row
# -------------------------
st.header("Overview")
k1,k2,k3,k4,k5 = st.columns(5)
total_listings = len(dff)
avg_price = pd.to_numeric(dff["Price"], errors="coerce").mean()
median_pps = pd.to_numeric(dff["Price_per_SqFt"], errors="coerce").median()
avg_dom = pd.to_numeric(dff["Days_On_Market"], errors="coerce").mean()
unique_agents = dff["Agent"].nunique()

k1.metric("Total Listings", f"{total_listings:,}")
k2.metric("Avg Price", f"₹{avg_price:,.0f}" if not math.isnan(avg_price) else "N/A")
k3.metric("Median ₹/sqft", f"₹{median_pps:,.0f}" if not math.isnan(median_pps) else "N/A")
k4.metric("Avg Days on Market", f"{avg_dom:.1f}" if not math.isnan(avg_dom) else "N/A")
k5.metric("Unique Agents", f"{unique_agents}")

st.markdown("**Purpose:** Single-row snapshot of supply, pricing, and market velocity. **Quick Tip:** adjust filters to slice KPIs by geography or product.")

# -------------------------
# Charts: Price distribution, box, pps by city, property mix
# -------------------------
st.subheader("Price Distribution")
if "Price" in dff.columns:
    fig_price = px.histogram(dff, x="Price", nbins=40, title="Price Distribution", labels={"Price":"Price"})
    fig_price.update_traces(texttemplate="%{y}", textposition="inside")
    st.plotly_chart(fig_price, use_container_width=True)
    st.markdown("**Purpose:** Detect skew/outliers. Quick Tip: log-transform if heavily skewed.")
else:
    st.info("Price column missing — distribution not available.")

st.subheader("Price by City & Property Type")
if "City" in dff.columns and "Price" in dff.columns:
    fig_box = px.box(dff, x="City", y="Price", color="Property_Type", title="Price by City & Type", points="outliers")
    fig_box.update_traces(boxmean=True)
    st.plotly_chart(fig_box, use_container_width=True)
else:
    st.info("City or Price missing — skip boxplot.")

st.subheader("Avg Price per Sqft by City")
if "City" in dff.columns and "Price_per_SqFt" in dff.columns:
    pps = dff.groupby("City", as_index=False)["Price_per_SqFt"].mean().sort_values("Price_per_SqFt", ascending=False)
    fig_pps = px.bar(pps, x="City", y="Price_per_SqFt", text_auto=".2f", title="Avg ₹/sqft by City")
    st.plotly_chart(fig_pps, use_container_width=True)
else:
    st.info("Price_per_SqFt or City missing — skip pps chart.")

st.subheader("Property Type Mix")
if "Property_Type" in dff.columns:
    counts = dff["Property_Type"].value_counts().reset_index()
    counts.columns = ["Property_Type","Count"]
    fig_p = px.pie(counts, names="Property_Type", values="Count", title="Property Type Mix", hole=0.35)
    st.plotly_chart(fig_p, use_container_width=True)
else:
    st.info("Property_Type missing — skip mix chart.")

# -------------------------
# Geo Heatmap (scatter_geo)
# -------------------------
st.subheader("Geo Heatmap (Lat/Lon)")
st.markdown("**Purpose:** See spatial price hotspots. Quick Tip: include real lat/lon in your data for meaningful maps.")
if "Latitude" in dff.columns and "Longitude" in dff.columns and dff["Latitude"].notna().any() and dff["Longitude"].notna().any():
    # Sample to reasonable size
    sample_geo = dff.dropna(subset=["Latitude","Longitude"])
    if len(sample_geo) > 1500:
        sample_geo = sample_geo.sample(1500, random_state=42)
    fig_geo = px.scatter_geo(sample_geo, lat="Latitude", lon="Longitude", color="Price", hover_name="City",
                             size="Price", title="Listings Geo Plot (size ~ price)")
    st.plotly_chart(fig_geo, use_container_width=True)
else:
    st.info("Latitude/Longitude not available or mostly null — geo heatmap unavailable.")

# -------------------------
# Agent Leaderboard
# -------------------------
st.subheader("Agent Leaderboard")
st.markdown("**Purpose:** Rank agents by value & activity. Quick Tip: identify top 10 agents to replicate success.")
agent_metrics = dff.groupby("Agent").agg(
    Listings=("Listing_ID","count") if "Listing_ID" in dff.columns else ("Price","count"),
    Total_Sales=("Price","sum"),
    Avg_Price=("Price","mean")
).reset_index().sort_values("Total_Sales", ascending=False)
st.dataframe(agent_metrics.head(20), use_container_width=True)

# -------------------------
# Clustering (KMeans)
# -------------------------
st.subheader("Property Clustering (KMeans)")
st.markdown("**Purpose:** Segmentation of listings by price & size for targeted strategies. Quick Tip: Inspect cluster centers to name segments.")
cluster_btn = st.button("Run Clustering (k=4 default)")
if cluster_btn:
    # prepare features
    feats = []
    if "Price" in dff.columns and "Square_Footage" in dff.columns:
        features = dff[["Price","Square_Footage"]].dropna()
        # take log for price to reduce skew
        features["Price_log"] = np.log1p(features["Price"])
        Xc = features[["Price_log","Square_Footage"]].values
        # scale
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xc)
        k = 4
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(Xs)
        features["cluster"] = labels
        # merge back sample for plotting
        plot_df = features.copy()
        plot_df["Price"] = np.expm1(plot_df["Price_log"])
        fig_cl = px.scatter(plot_df, x="Square_Footage", y="Price", color="cluster", title="KMeans clusters (by sqft & price)", hover_data=["Price"])
        st.plotly_chart(fig_cl, use_container_width=True)
        st.markdown("Cluster counts:")
        st.write(features["cluster"].value_counts())
    else:
        st.info("Need Price and Square_Footage columns to run clustering.")

# -------------------------
# Forecasting (time-aware RandomForest)
# -------------------------
st.subheader("Price Forecasting (simple time-aware model)")
st.markdown("**Purpose:** Short-term price forecast by learning seasonal/time patterns. Quick Tip: For serious forecasting use Prophet/ARIMA with more data.")
if "Listing_Date" in dff.columns and dff["Listing_Date"].notna().any() and "Price" in dff.columns:
    ts = dff.copy()
    ts["Date"] = pd.to_datetime(ts["Listing_Date"], errors="coerce")
    ts = ts.dropna(subset=["Date","Price"])
    # aggregate monthly mean price
    ts["YearMonth"] = ts["Date"].dt.to_period("M").dt.to_timestamp()
    monthly = ts.groupby("YearMonth")["Price"].mean().reset_index().sort_values("YearMonth")
    # features: year, month, lag1
    monthly["year"] = monthly["YearMonth"].dt.year
    monthly["month"] = monthly["YearMonth"].dt.month
    monthly["lag1"] = monthly["Price"].shift(1).fillna(method="bfill")
    X = monthly[["year","month","lag1"]]
    y = monthly["Price"]
    if len(monthly) < 12:
        st.info("Not enough historical months (<12) for meaningful forecast.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = math.sqrt(mse)
        st.write(f"Backtest RMSE: ₹{rmse:,.0f}")
        # forecast next 6 months
        last = monthly.iloc[-1]
        future_rows = []
        last_dt = pd.to_datetime(last["YearMonth"])
        lag = last["Price"]
        for m in range(1,7):
            future_dt = (last_dt + pd.DateOffset(months=m))
            future_rows.append({
                "YearMonth": future_dt,
                "year": future_dt.year,
                "month": future_dt.month,
                "lag1": lag
            })
            # update lag for next iter (use predicted)
            lag = rf.predict(np.array([[future_dt.year, future_dt.month, lag]]))[0]
        fut_df = pd.DataFrame(future_rows)
        fut_preds = rf.predict(fut_df[["year","month","lag1"]])
        forecast = pd.DataFrame({"YearMonth": fut_df["YearMonth"], "Predicted_Price": fut_preds})
        combined = pd.concat([monthly[["YearMonth","Price"]].rename(columns={"Price":"Actual_Price"}), forecast], ignore_index=True, sort=False)
        fig_fore = px.line(combined, x="YearMonth", y=["Actual_Price","Predicted_Price"], title="Monthly Price Forecast (Actual + Predicted)", markers=True)
        st.plotly_chart(fig_fore, use_container_width=True)
else:
    st.info("Listing_Date or Price missing — forecasting not available.")

# -------------------------
# Recommender (Nearest neighbors on feature embeddings)
# -------------------------
st.subheader("Investment Recommendation (similar listings)")
st.markdown("**Purpose:** Find properties similar to a chosen listing. Quick Tip: Use price & sqft filters to narrow candidates.")
if "Price" in dff.columns and "Square_Footage" in dff.columns:
    # features: price, sqft, beds, baths, ptype encoded
    cand = dff.dropna(subset=["Price","Square_Footage"]).copy()
    feat_cols = []
    numeric = ["Price","Square_Footage","Bedrooms","Bathrooms"]
    for c in numeric:
        if c in cand.columns:
            feat_cols.append(c)
    # categorical ptype
    ptype_list = cand["Property_Type"].fillna("Unknown").astype(str)
    # encode ptype by simple one-hot via pandas
    ptype_ohe = pd.get_dummies(ptype_list)
    X_rec = pd.concat([cand[numeric].reset_index(drop=True), ptype_ohe.reset_index(drop=True)], axis=1).fillna(0)
    nn = NearestNeighbors(n_neighbors=6)
    try:
        nn.fit(X_rec)
        pick_index = st.number_input("Pick an index to find similar (0..n-1)", min_value=0, max_value=len(cand)-1, value=0)
        distances, indices = nn.kneighbors([X_rec.iloc[pick_index]])
        sim_idx = indices[0][1:]  # exclude self
        st.write("Top similar listings:")
        st.dataframe(cand.reset_index(drop=True).iloc[sim_idx][["Listing_ID","City","Location","Price","Square_Footage","Bedrooms","Bathrooms"]])
    except Exception as e:
        st.info("Recommender could not run: " + str(e))
else:
    st.info("Price and Square_Footage required for recommender.")

# -------------------------
# NLP lite search
# -------------------------
st.subheader("Keyword Search (NLP-lite)")
st.markdown("Type queries like: '3BHK in Bangalore under 1.5cr with parking'")
query = st.text_input("Search properties by keywords")
if query:
    q = query.lower()
    # naive token matching on columns city, property_type, bedrooms, price range, parking
    res = dff.copy()
    if "City" in res.columns:
        res = res[res["City"].str.lower().str.contains(q.split()[0]) | res["Location"].str.lower().str.contains(q.split()[0]) if "Location" in res.columns else res["City"].str.lower().str.contains(q.split()[0])]
    # further filters
    if "parking" in q and "Parking" in res.columns:
        res = res[res["Parking"].str.lower().str.contains("yes")]
    # bedrooms
    import re
    m = re.search(r'(\d)[ -]*bhk|(\d)[ -]*bdr|(\d)br|(\d)bhk', q)
    # simpler: pick any digit in query and filter bedrooms==digit
    digits = re.findall(r'\d+', q)
    if digits and "Bedrooms" in res.columns:
        try:
            num = int(digits[0])
            res = res[res["Bedrooms"] == num]
        except:
            pass
    # price cap 'under X' pattern
    under = re.search(r'under\s*([0-9\.]+)(cr|lakhs|l|k|m|mn)?', q)
    if under and "Price" in res.columns:
        val = float(under.group(1))
        unit = under.group(2)
        if unit in ['cr']:
            cap = val * 1e7
        elif unit in ['lakhs','l']:
            cap = val * 1e5
        elif unit in ['k']:
            cap = val * 1e3
        elif unit in ['m','mn']:
            cap = val * 1e6
        else:
            cap = val
        res = res[pd.to_numeric(res["Price"], errors="coerce") <= cap]
    st.write(f"Found {len(res)} matches")
    if len(res) > 0:
        st.dataframe(res.head(50)[["Listing_ID","City","Location","Property_Type","Bedrooms","Bathrooms","Square_Footage","Price","Parking"]])

# -------------------------
# Lead scoring & Time-to-sell (if leads columns exist)
# -------------------------
st.subheader("Lead Scoring & Time-to-sell")
if "Leads" in dff.columns or "Lead_Source" in dff.columns:
    st.markdown("Basic lead scoring requires structured lead data. This demo uses any numeric 'Leads' and 'Offers' columns if present.")
    if "Offers" in dff.columns and "Leads" in dff.columns:
        # simple model: predict Offers from Leads and listing features
        ls_df = dff.dropna(subset=["Offers","Leads"])
        features = []
        for c in ["Bedrooms","Bathrooms","Square_Footage","Demand_Score","Price_per_SqFt"]:
            if c in ls_df.columns:
                features.append(c)
        if len(features) == 0:
            st.info("Not enough features to build lead scoring model.")
        else:
            X = ls_df[features]
            y = ls_df["Offers"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            m = RandomForestRegressor(n_estimators=150, random_state=42)
            m.fit(X_train, y_train)
            preds = m.predict(X_test)
            mse = mean_squared_error(y_test, preds); rmse = math.sqrt(mse)
            st.write(f"Lead->Offer model RMSE: {rmse:.2f}")
            st.markdown("Top features used:")
            st.write(features)
    else:
        st.info("Leads/Offers columns not present — cannot build lead scoring model.")
else:
    st.info("No lead-related columns found in dataset.")

# -------------------------
# Time-to-sell model
# -------------------------
st.subheader("Time-to-Sell Prediction")
if "Days_On_Market" in dff.columns:
    tdf = dff.dropna(subset=["Days_On_Market","Price","Square_Footage"])
    if len(tdf) < 50:
        st.info("Not enough rows to train time-to-sell model (need >=50).")
    else:
        features = [c for c in ["Price","Square_Footage","Bedrooms","Bathrooms","Demand_Score"] if c in tdf.columns]
        X = tdf[features]
        y = tdf["Days_On_Market"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        m = RandomForestRegressor(n_estimators=150, random_state=42)
        m.fit(X_train, y_train)
        p = m.predict(X_test)
        rmse = math.sqrt(mean_squared_error(y_test, p))
        st.write(f"Time-to-sell RMSE: {rmse:.1f} days")
        st.markdown("Predict days for a sample input:")
        if st.button("Predict sample time-to-sell (median values)"):
            sample = {c: [tdf[c].median()] for c in features}
            pred_days = m.predict(pd.DataFrame(sample))[0]
            st.success(f"Predicted Days on Market: {pred_days:.0f}")

# -------------------------
# Compare Two Cities mode
# -------------------------
st.subheader("Compare Two Cities")
cities = sorted(dff["City"].dropna().unique().tolist()) if "City" in dff.columns else []
if len(cities) >= 2:
    c1, c2 = st.columns(2)
    with c1:
        city_a = st.selectbox("City A", options=cities, index=0)
    with c2:
        city_b = st.selectbox("City B", options=cities, index=1)
    ca = dff[dff["City"] == city_a]
    cb = dff[dff["City"] == city_b]
    st.write("Avg price:", city_a, pd.to_numeric(ca["Price"], errors="coerce").mean(), " | ", city_b, pd.to_numeric(cb["Price"], errors="coerce").mean())
    st.write("Median ₹/sqft:", city_a, pd.to_numeric(ca["Price_per_SqFt"], errors="coerce").median(), " | ", city_b, pd.to_numeric(cb["Price_per_SqFt"], errors="coerce").median())
    st.write("Property mix (A):"); st.write(ca["Property_Type"].value_counts().head())
    st.write("Property mix (B):"); st.write(cb["Property_Type"].value_counts().head())
else:
    st.info("Not enough distinct cities to compare.")

# -------------------------
# Raw data and download
# -------------------------
st.markdown("---")
st.subheader("Raw Data (filtered) — preview and download")
st.dataframe(dff.reset_index(drop=True).head(200), use_container_width=True)
csv = dff.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=csv, file_name="real_estate_filtered.csv", mime="text/csv")

st.markdown("---")
st.caption("Full platform delivered: clustering, forecasting, recommender, agent analytics, leaderboard, geo heatmaps, NLP search, lead scoring, time-to-sell, and compare mode. If anything errors on your data, upload a CSV and use the Mapping option to align column names to expected fields.")
