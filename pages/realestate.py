# real_estate_dashboard_full.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Real Estate Analytics — All-in-One Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("# Real Estate Analytics — Full Dashboard")
st.markdown("KPIs · Filters · Charts · Agent Leaderboard · Region Heatmap · Price Prediction (ML)\n---")

# ---------------------------
# Synthetic dataset generator (fallback)
# ---------------------------
def generate_synthetic(n=1200, seed=42):
    np.random.seed(seed)
    cities = ["Chennai", "Bangalore", "Hyderabad", "Mumbai", "Pune", "Delhi"]
    property_types = ["Apartment", "Villa", "Plot", "Commercial"]
    statuses = ["Available", "Sold", "Under Offer"]
    agents = ["Aditi", "Manoj", "Riya", "Kiran", "Sanjay", "Neha", "Vikram"]
    neighborhoods = {
        "Chennai": ["Velachery", "Adyar", "Anna Nagar"],
        "Bangalore": ["Whitefield", "Koramangala", "Jayanagar"],
        "Hyderabad": ["Hitech City", "Gachibowli", "Banjara Hills"],
        "Mumbai": ["Andheri", "Bandra", "Powai"],
        "Pune": ["Koregaon Park", "Hinjewadi", "Viman Nagar"],
        "Delhi": ["Saket", "Rohini", "Dwarka"]
    }

    rows = []
    today = datetime.today()
    for i in range(n):
        city = np.random.choice(cities)
        location = np.random.choice(neighborhoods[city])
        ptype = np.random.choice(property_types, p=[0.6, 0.15, 0.15, 0.1])
        area = int(np.random.normal(1200, 400)) if ptype != "Plot" else int(np.random.normal(3000, 800))
        area = max(200, area)
        base_price_per_sqft = {
            "Chennai": 6500, "Bangalore": 9000, "Hyderabad": 6000,
            "Mumbai": 18000, "Pune": 8500, "Delhi": 12000
        }[city]
        price_noise = np.random.normal(1.0, 0.12)
        price = int(area * base_price_per_sqft * price_noise)
        bedrooms = int(np.clip(np.round(area / 400), 1, 6))
        bathrooms = max(1, bedrooms - np.random.randint(0, 2))
        status = np.random.choice(statuses, p=[0.55, 0.35, 0.10])
        agent = np.random.choice(agents)
        listing_age_days = np.random.randint(1, 600)
        listing_date = today - timedelta(days=listing_age_days)
        leads = max(0, int(np.random.poisson(40) * (1 if status != "Sold" else 0.6)))
        site_visits = int(leads * np.random.uniform(0.2, 0.6))
        offers = int(site_visits * np.random.uniform(0.05, 0.25))
        demand_index = int(np.clip(40 + np.random.normal(0, 15), 10, 100))
        school_rating = np.random.randint(1, 6)
        connectivity = np.random.randint(1, 11)
        price_per_sqft = price / area

        rows.append({
            "Listing_ID": i + 1,
            "City": city,
            "Location": location,
            "Property_Type": ptype,
            "Area_sqft": area,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Price": price,
            "Price_per_sqft": price_per_sqft,
            "Status": status,
            "Agent": agent,
            "Listing_Date": listing_date.date(),
            "Leads": leads,
            "Site_Visits": site_visits,
            "Offers": offers,
            "Demand_Index": demand_index,
            "School_Rating": school_rating,
            "Connectivity_Score": connectivity
        })
    return pd.DataFrame(rows)

# ---------------------------
# File upload / load data
# ---------------------------
st.sidebar.header("Data ▶ Upload or Use Sample")
upload = st.sidebar.file_uploader("Upload CSV or Excel (single sheet)", type=["csv", "xlsx"])

if upload:
    try:
        if upload.name.lower().endswith(".csv"):
            raw = pd.read_csv(upload)
        else:
            raw = pd.read_excel(upload)
        st.sidebar.success("File loaded.")
    except Exception as e:
        st.sidebar.error(f"Failed to load file: {e}")
        raw = None
else:
    raw = None

# Normalise column names helper
def normalize_cols(df):
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_").replace(".", "").replace("-", "_") for c in df.columns]
    return df

if raw is None:
    df = generate_synthetic(n=1200)
else:
    df = normalize_cols(raw)
    # try to map common names to our internal names if possible
    mapping = {}
    col_lower = {c.lower(): c for c in df.columns}
    # heuristics - common columns
    heuristics = {
        "price": ["price", "amount", "final_amount", "listing_price"],
        "area_sqft": ["area", "area_sqft", "sqft", "builtup_area", "carpet_area"],
        "city": ["city", "town"],
        "location": ["location", "neighborhood", "locality"],
        "property_type": ["property_type", "propertytype", "ptype"],
        "agent": ["agent", "sales_agent", "broker"],
        "bedrooms": ["bedrooms", "beds", "bhk"],
        "bathrooms": ["bathrooms", "baths"],
        "status": ["status", "listing_status"],
        "listing_date": ["date", "listing_date", "posted_on"],
        "leads": ["leads", "lead_count"],
        "site_visits": ["site_visits", "visits"],
        "offers": ["offers", "offer_count"]
    }
    for target, candidates in heuristics.items():
        for cand in candidates:
            if cand in col_lower:
                mapping[target] = col_lower[cand]
                break
    # keep only mapped columns where present; for missing fall back to synthetic generator
    needed = ["City", "Location", "Property_Type", "Area_sqft", "Bedrooms", "Bathrooms", "Price", "Status", "Agent", "Listing_Date", "Leads", "Site_Visits", "Offers"]
    # if mapping not enough, fallback to synthetic
    if len(mapping) < 8:
        st.sidebar.warning("Uploaded file missing key columns or format mismatch — using synthetic dataset instead.")
        df = generate_synthetic(n=1200)
    else:
        # rename mapped columns
        rename_map = {v: k for k, v in mapping.items()}
        df = df.rename(columns=rename_map)
        # ensure types and defaults
        if "Listing_Date" in df.columns:
            df["Listing_Date"] = pd.to_datetime(df["Listing_Date"], errors='coerce').dt.date
        numeric_cols = ["Area_sqft", "Bedrooms", "Bathrooms", "Price", "Leads", "Site_Visits", "Offers"]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        # fill missing sensible defaults
        for col in needed:
            if col not in df.columns:
                df[col] = generate_synthetic(n=len(df))[col]

# ensure Price_per_sqft present
if "Price_per_sqft" not in df.columns:
    df["Price_per_sqft"] = df["Price"] / df["Area_sqft"]

# ---------------------------
# Sidebar: page navigation & filters
# ---------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Dashboard (All-in-One)", "Raw Data / Download"])

st.sidebar.markdown("---")
st.sidebar.header("Filters")
cities = sorted(df["City"].unique())
city_sel = st.sidebar.multiselect("City", options=cities, default=cities)
ptype_sel = st.sidebar.multiselect("Property Type", options=sorted(df["Property_Type"].unique()), default=sorted(df["Property_Type"].unique()))
status_sel = st.sidebar.multiselect("Status", options=sorted(df["Status"].unique()), default=sorted(df["Status"].unique()))
agent_sel = st.sidebar.multiselect("Agent", options=sorted(df["Agent"].unique()), default=sorted(df["Agent"].unique()))

min_price, max_price = int(df["Price"].min()), int(df["Price"].max())
price_range = st.sidebar.slider("Price Range (₹)", min_price, max_price, (min_price, max_price), step=100000)

area_min, area_max = int(df["Area_sqft"].min()), int(df["Area_sqft"].max())
area_range = st.sidebar.slider("Area (sqft)", area_min, area_max, (area_min, area_max), step=50)

# filtered df
mask = (
    df["City"].isin(city_sel) &
    df["Property_Type"].isin(ptype_sel) &
    df["Status"].isin(status_sel) &
    df["Agent"].isin(agent_sel) &
    (df["Price"] >= price_range[0]) &
    (df["Price"] <= price_range[1]) &
    (df["Area_sqft"] >= area_range[0]) &
    (df["Area_sqft"] <= area_range[1])
)
df_f = df[mask].copy()

# ---------------------------
# Dashboard page
# ---------------------------
if page == "Dashboard (All-in-One)":
    st.header("Dashboard: KPIs · Charts · Leaderboard · Heatmap · ML Prediction")
    # Top KPIs
    total_listings = len(df_f)
    avg_price = df_f["Price"].mean() if total_listings else 0
    median_pps = df_f["Price_per_sqft"].median() if total_listings else 0
    avg_days_on_market = None
    if "Listing_Date" in df_f.columns:
        try:
            days_on_market = [(datetime.today().date() - d).days for d in pd.to_datetime(df_f["Listing_Date"], errors='coerce').dt.date]
            avg_days_on_market = int(np.nanmean(days_on_market))
        except Exception:
            avg_days_on_market = None

    sold_count = (df_f["Status"] == "Sold").sum()
    average_leads = int(df_f["Leads"].mean()) if total_listings else 0
    conv_rate = round((sold_count / total_listings * 100), 2) if total_listings else 0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Listings", f"{total_listings:,}")
    k2.metric("Avg Price", f"₹{avg_price:,.0f}")
    k3.metric("Median ₹/sqft", f"₹{median_pps:,.0f}")
    k4.metric("Sold Count", f"{sold_count:,}")
    k5.metric("Avg Leads", f"{average_leads:,}")
    k6.metric("Conversion Rate", f"{conv_rate}%")

    st.markdown("---")

    # layout: charts left, leaderboard & heatmap right (two-column)
    left, right = st.columns((2, 1))

    # Left column: multiple charts stacked
    with left:
        st.subheader("Price & Supply Visuals")

        # 1. Price distribution by city (box)
        fig_box = px.box(df_f, x="City", y="Price", color="Property_Type", points="outliers",
                         labels={"Price": "Price (₹)"}, title="Price Distribution by City & Type")
        st.plotly_chart(fig_box, use_container_width=True)

        # 2. Avg price per sqft by city (bar)
        pps = df_f.groupby(["City", "Property_Type"], as_index=False)["Price_per_sqft"].mean()
        fig_pps = px.bar(pps, x="City", y="Price_per_sqft", color="Property_Type", barmode="group",
                         title="Average Price per Sqft by City and Property Type",
                         labels={"Price_per_sqft": "₹ / sqft"})
        st.plotly_chart(fig_pps, use_container_width=True)

        # 3. Supply trend by listing date (time-series) — aggregated weekly
        if "Listing_Date" in df_f.columns:
            df_f["Listing_Date_parsed"] = pd.to_datetime(df_f["Listing_Date"], errors="coerce")
            timeseries = df_f.set_index("Listing_Date_parsed").resample("W")["Listing_ID"].count().reset_index().rename(columns={"Listing_ID":"Listings"})
            fig_ts = px.line(timeseries, x="Listing_Date_parsed", y="Listings", title="Weekly New Listings")
            st.plotly_chart(fig_ts, use_container_width=True)

        # 4. Leads vs Site Visits vs Offers scatter
        st.subheader("Lead Funnel Scatter")
        fig_scatter = px.scatter(df_f, x="Leads", y="Offers", size="Site_Visits", color="City",
                                 hover_data=["Agent", "Price", "Area_sqft"],
                                 title="Leads → Site Visits → Offers (bubble size = site visits)")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Right column: leaderboard and heatmap
    with right:
        st.subheader("Agent Leaderboard")
        # compute metrics per agent
        agent_grp = df_f.groupby("Agent").agg({
            "Listing_ID": "count",
            "Leads": "sum",
            "Site_Visits": "sum",
            "Offers": "sum",
            "Price": "sum"
        }).rename(columns={"Listing_ID":"Listings"})
        agent_grp["Conversion_Rate (%)"] = agent_grp["Offers"] / (agent_grp["Leads"].replace(0, np.nan)) * 100
        agent_grp = agent_grp.fillna(0).sort_values(by="Offers", ascending=False).reset_index()

        # show top 10
        st.table(agent_grp.head(10).style.format({
            "Listings":"{:,}",
            "Leads":"{:,}",
            "Site_Visits":"{:,}",
            "Offers":"{:,}",
            "Price":"₹{:,}",
            "Conversion_Rate (%)":"{:.1f}%"
        }))

        st.markdown("---")
        st.subheader("Region Heatmap (Location × Property Type)")

        # pivot: average price per sqft
        heat_df = df_f.pivot_table(index="Location", columns="Property_Type", values="Price_per_sqft", aggfunc="mean")
        # fill NaN with 0 for display
        heat_viz = heat_df.fillna(0)
        fig_heat = px.imshow(heat_viz,
                             labels=dict(x="Property Type", y="Location", color="Avg ₹/sqft"),
                             title="Avg Price per Sqft by Location and Property Type",
                             aspect="auto")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")
    # ML Price Prediction box (train simple model on filtered data or full data)
    st.subheader("ML Price Prediction (Random Forest)")

    # Model features: City, Location, Property_Type, Area_sqft, Bedrooms, Bathrooms, Demand_Index, School_Rating, Connectivity_Score
    ml_df = df_f.copy()
    features = ["City", "Location", "Property_Type", "Area_sqft", "Bedrooms", "Bathrooms"]
    # add optional if present
    for opt in ["Demand_Index", "School_Rating", "Connectivity_Score"]:
        if opt in ml_df.columns:
            features.append(opt)

    # require at least minimal rows
    if len(ml_df) < 40:
        st.info("Not enough data after filters to train a robust model (need at least 40 rows). Use broader filters or upload more data.")
    else:
        # prepare X, y
        X = ml_df[features]
        y = ml_df["Price"]

        # split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # preprocessing: categorical -> onehot, numeric -> scaler
        cat_cols = [c for c in features if X[c].dtype == 'object' or X[c].dtype.name == 'category']
        num_cols = [c for c in features if c not in cat_cols]

        preprocessor = ColumnTransformer(transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols)
        ])

        model = Pipeline([
            ("pre", preprocessor),
            ("rf", RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1))
        ])

        with st.spinner("Training model..."):
            model.fit(X_train, y_train)

        # metrics
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        st.write(f"Model MAE: ₹{mae:,.0f}   |   R²: {r2:.3f}")

        # Prediction UI
        st.markdown("#### Predict price for a custom listing")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            city_i = st.selectbox("City", sorted(df["City"].unique()))
            location_i = st.selectbox("Location", sorted(df[df["City"] == city_i]["Location"].unique()))
            ptype_i = st.selectbox("Property Type", sorted(df["Property_Type"].unique()))
        with col_b:
            area_i = st.number_input("Area (sqft)", min_value=200, max_value=20000, value=1200, step=10)
            beds_i = st.number_input("Bedrooms", min_value=0, max_value=10, value=2, step=1)
        with col_c:
            baths_i = st.number_input("Bathrooms", min_value=0, max_value=10, value=2, step=1)
            demand_i = None
            if "Demand_Index" in df.columns:
                demand_i = st.slider("Demand Index", min_value=int(df["Demand_Index"].min()), max_value=int(df["Demand_Index"].max()), value=int(df["Demand_Index"].median()))
        with col_d:
            school_i = None
            if "School_Rating" in df.columns:
                school_i = st.selectbox("School Rating", sorted(df["School_Rating"].unique()))
            connectivity_i = None
            if "Connectivity_Score" in df.columns:
                connectivity_i = st.slider("Connectivity Score", min_value=int(df["Connectivity_Score"].min()), max_value=int(df["Connectivity_Score"].max()), value=int(df["Connectivity_Score"].median()))

        # build input row
        input_dict = {
            "City": city_i,
            "Location": location_i,
            "Property_Type": ptype_i,
            "Area_sqft": area_i,
            "Bedrooms": beds_i,
            "Bathrooms": baths_i
        }
        if demand_i is not None:
            input_dict["Demand_Index"] = demand_i
        if school_i is not None:
            input_dict["School_Rating"] = school_i
        if connectivity_i is not None:
            input_dict["Connectivity_Score"] = connectivity_i

        input_df = pd.DataFrame([input_dict])[features]  # ensure same order/cols

        if st.button("Predict Price"):
            predicted = model.predict(input_df)[0]
            st.success(f"Predicted Price: ₹{predicted:,.0f}")
            st.info(f"Predicted Price per sqft: ₹{predicted / area_i:,.0f}")

# ---------------------------
# Raw Data page
# ---------------------------
elif page == "Raw Data / Download":
    st.header("Raw Data & Export")
    st.write("You can inspect the filtered raw data below and download it if you want.")
    st.dataframe(df_f.reset_index(drop=True), use_container_width=True)

    # download button
    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button(label="Download filtered data as CSV", data=csv, file_name="real_estate_filtered.csv", mime="text/csv")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Dashboard generated: " + datetime.now().strftime("%d %b %Y") + " • Model: RandomForestRegressor (lightweight demo).")
