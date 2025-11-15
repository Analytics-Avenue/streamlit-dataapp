# market_intel_app.py
"""
Real Estate Market Intelligence Dashboard
Single-file Streamlit app (deployable).
Features:
 - 3 dataset options: Default GitHub, Upload CSV, Upload CSV + Manual Mapping
 - Sample dataset generator (10k rows) and download
 - KPIs, trend charts, city-level bars with labels, property mix pie with labels
 - Geo scatter (lat/lon) and locality heatmap fallback
 - Filters and filtered CSV download
 - Purpose & Quick Tips as expanders for each chart
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import random
import math
import requests
from datetime import datetime, timedelta
import re

# -------------------------
# Page config and styling
# -------------------------
st.set_page_config(page_title="Market Intelligence — Real Estate", layout="wide", initial_sidebar_state="expanded")
st.title("🏙️ Market Intelligence — Real Estate")
st.markdown("Strategic dashboard to monitor city-level and region-level real estate KPIs, trends and hotspots.")

# -------------------------
# Utilities
# -------------------------
def format_inr(value):
    """Format number into Indian style with Lakhs/Crores shorthand where appropriate."""
    try:
        v = float(value)
    except:
        return value
    if math.isnan(v):
        return "N/A"
    abs_v = abs(v)
    if abs_v >= 1e7:
        return f"₹{v/1e7:.2f} Cr"
    if abs_v >= 1e5:
        return f"₹{v/1e5:.2f} L"
    n = int(round(v))
    s = str(abs(n))
    if len(s) <= 3:
        grouped = s
    else:
        last3 = s[-3:]
        rem = s[:-3]
        parts = []
        while len(rem) > 2:
            parts.append(rem[-2:])
            rem = rem[:-2]
        if rem:
            parts.append(rem)
        parts = parts[::-1]
        grouped = ",".join(parts) + "," + last3
    if n < 0:
        grouped = "-" + grouped
    return "₹" + grouped

def ensure_unique_columns(df):
    cols = list(df.columns)
    seen = {}
    new_cols = []
    for c in cols:
        key = c.strip()
        if key in seen:
            seen[key] += 1
            new_cols.append(f"{key}_dup{seen[key]}")
        else:
            seen[key] = 0
            new_cols.append(key)
    df.columns = new_cols
    return df

def try_read_csv_url(url):
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    except Exception:
        return None

# -------------------------
# Sample data generator
# -------------------------
def make_market_sample(n=10000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    india_cities = ["Bangalore","Mumbai","Chennai","Delhi","Pune","Hyderabad","Kolkata","Ahmedabad","Coimbatore","Lucknow"]
    global_cities = ["New York","London","Dubai","Singapore","Toronto"]
    cities = india_cities + global_cities
    property_types = ["Apartment","Villa","Condo","Townhouse","Studio","House"]
    furnishing_opts = ["Unfurnished","Semi-Furnished","Fully-Furnished"]
    parking_opts = ["Yes","No"]
    rows = []
    today = datetime.today()
    for i in range(n):
        city = random.choice(cities)
        region = f"{city}_R{random.randint(1,50)}"
        locality = f"Locality_{random.randint(1,500)}"
        ptype = random.choice(property_types)
        sqft = int(max(250, np.random.normal(1100, 450)))
        bedrooms = max(1, int(sqft//450) if random.random()>0.3 else random.randint(1,5))
        bathrooms = max(1, min(4, bedrooms if bedrooms<=3 else bedrooms-1))
        year_built = random.randint(1980, 2022)
        price_base = {
            "Bangalore":9000,"Mumbai":18000,"Chennai":6500,"Delhi":12000,"Pune":8500,"Hyderabad":6000,"Kolkata":4500,"Ahmedabad":4000,"Coimbatore":3500,"Lucknow":3000,
            "New York":2000,"London":1700,"Dubai":2000,"Singapore":2200,"Toronto":1300
        }.get(city,8000)
        price = int(max(20000, sqft * price_base * np.random.normal(1.0, 0.12)))
        listing_date = (today - timedelta(days=random.randint(0, 900))).date()
        demand_score = random.randint(10,100)
        school_rating = random.randint(1,10)
        days_on_market = random.randint(1,365)
        lat = round(np.random.uniform(-33,51),6)
        lon = round(np.random.uniform(-118,151),6)
        rows.append({
            "Listing_ID": f"L{100000+i}",
            "City": city,
            "Region": region,
            "Locality": locality,
            "Property_Type": ptype,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Square_Footage": sqft,
            "Year_Built": year_built,
            "Price": price,
            "Price_per_SqFt": round(price/sqft,2) if sqft>0 else np.nan,
            "Furnishing": random.choice(furnishing_opts),
            "Parking": random.choice(parking_opts),
            "Listing_Date": listing_date,
            "Demand_Score": demand_score,
            "School_Rating": school_rating,
            "Days_On_Market": days_on_market,
            "Latitude": lat,
            "Longitude": lon
        })
    return pd.DataFrame(rows)

# -------------------------
# Sidebar: Data source options
# -------------------------
st.sidebar.header("Data source & options")
data_mode = st.sidebar.radio("Choose data source:", ("Default GitHub dataset", "Upload CSV", "Upload CSV & Manual Mapping"))

# prepare sample and sample download
sample_df = make_market_sample(10000)
buf = io.StringIO(); sample_df.to_csv(buf, index=False)
st.sidebar.download_button("Download sample CSV (10k rows)", data=buf.getvalue().encode('utf-8'),
                           file_name="market_intel_sample_10000.csv", mime="text/csv")

# You may replace this raw URL with your GitHub raw CSV URL
GITHUB_RAW = "https://raw.githubusercontent.com/your-username/your-repo/main/market_intel_sample_10000.csv"

df_raw = None
mapping = {}

if data_mode == "Default GitHub dataset":
    st.sidebar.info("Loading default dataset from GitHub (falls back to internal sample).")
    df_try = try_read_csv_url(GITHUB_RAW)
    if df_try is not None:
        df_raw = df_try
        st.sidebar.success("Loaded dataset from GitHub.")
    else:
        df_raw = sample_df.copy()
        st.sidebar.warning("Could not fetch GitHub dataset — using internal sample.")

elif data_mode in ("Upload CSV", "Upload CSV & Manual Mapping"):
    uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
            st.sidebar.success("Uploaded CSV loaded.")
        except Exception as e:
            st.sidebar.error("Failed to read uploaded CSV: " + str(e))
            df_raw = None
    else:
        st.sidebar.info("No file uploaded — using internal sample until you upload.")
        df_raw = sample_df.copy()

# Manual mapping UI if selected and upload provided
EXPECTED = ["Listing_ID","City","Region","Locality","Property_Type","Bedrooms","Bathrooms","Square_Footage","Year_Built","Price","Price_per_SqFt","Furnishing","Parking","Listing_Date","Demand_Score","School_Rating","Days_On_Market","Latitude","Longitude"]
if data_mode == "Upload CSV & Manual Mapping" and df_raw is not None:
    st.sidebar.subheader("Manual column mapping")
    cols = list(df_raw.columns)
    for exp in EXPECTED:
        default = None
        for c in cols:
            if c.strip().lower() == exp.strip().lower():
                default = c
                break
        mapping[exp] = st.sidebar.selectbox(f"{exp} →", options=["<none>"] + cols, index=(cols.index(default)+1 if default in cols else 0), key=f"map_{exp}")

# -------------------------
# Build mapped dataframe (auto map if no mapping provided)
# -------------------------
if df_raw is None:
    df = sample_df.copy()
else:
    df_work = df_raw.copy()
    df_work = ensure_unique_columns(df_work)
    if mapping and any(v and v != "<none>" for v in mapping.values()):
        mapped = pd.DataFrame()
        for exp in EXPECTED:
            sel = mapping.get(exp)
            if sel and sel != "<none>" and sel in df_work.columns:
                mapped[exp] = df_work[sel]
            else:
                # attempt auto match by substring
                found = None
                for c in df_work.columns:
                    if exp.strip().lower() == c.strip().lower() or exp.strip().lower() in c.strip().lower() or c.strip().lower() in exp.strip().lower():
                        found = c
                        break
                if found:
                    mapped[exp] = df_work[found]
                else:
                    mapped[exp] = np.nan
        df = mapped.copy()
    else:
        # auto heuristics
        normalized = {c.strip().lower(): c for c in df_work.columns}
        auto = pd.DataFrame()
        for exp in EXPECTED:
            key = exp.strip().lower()
            if key in normalized:
                auto[exp] = df_work[normalized[key]]
            else:
                cand = None
                for c in df_work.columns:
                    if key in c.strip().lower() or c.strip().lower() in key or any(tok in c.strip().lower() for tok in key.split("_")):
                        cand = c
                        break
                if cand:
                    auto[exp] = df_work[cand]
                else:
                    auto[exp] = np.nan
        df = auto.copy()

# -------------------------
# Clean types and compute PPS if missing
# -------------------------
num_cols = ["Bedrooms","Bathrooms","Square_Footage","Year_Built","Price","Price_per_SqFt","Demand_Score","School_Rating","Days_On_Market","Latitude","Longitude"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if "Price_per_SqFt" not in df.columns or df["Price_per_SqFt"].isna().all():
    if "Price" in df.columns and "Square_Footage" in df.columns:
        df["Price_per_SqFt"] = (pd.to_numeric(df["Price"], errors="coerce") / pd.to_numeric(df["Square_Footage"], errors="coerce")).round(2)

if "Listing_Date" in df.columns:
    try:
        df["Listing_Date"] = pd.to_datetime(df["Listing_Date"], errors="coerce")
    except:
        pass

# fallback columns to keep UI safe
required_cols = ["Price","Square_Footage","Bedrooms","Bathrooms","City","Property_Type","Listing_Date"]
for c in required_cols:
    if c not in df.columns:
        if c == "Price":
            df[c] = np.nan
        elif c == "Square_Footage":
            df[c] = 1100
        elif c == "Bedrooms":
            df[c] = 2
        elif c == "Bathrooms":
            df[c] = 2
        else:
            df[c] = "Unknown"

# -------------------------
# Filters
# -------------------------
st.sidebar.header("Filters")
city_vals = sorted(df["City"].dropna().unique().tolist()) if "City" in df.columns else []
city_sel = st.sidebar.multiselect("City", options=city_vals, default=city_vals[:6] if city_vals else [])
ptype_vals = sorted(df["Property_Type"].dropna().unique().tolist()) if "Property_Type" in df.columns else []
ptype_sel = st.sidebar.multiselect("Property Type", options=ptype_vals, default=None)
min_price = int(pd.to_numeric(df["Price"], errors="coerce").min()) if pd.to_numeric(df["Price"], errors="coerce").notna().any() else 0
max_price = int(pd.to_numeric(df["Price"], errors="coerce").max()) if pd.to_numeric(df["Price"], errors="coerce").notna().any() else 10000000
price_sel = st.sidebar.slider("Price range", min_value=min_price, max_value=max_price, value=(min_price, max_price))
bed_min = int(df["Bedrooms"].min()) if "Bedrooms" in df.columns else 1
bed_max = int(df["Bedrooms"].max()) if "Bedrooms" in df.columns else 5
bed_sel = st.sidebar.slider("Bedrooms", min_value=bed_min, max_value=bed_max, value=(bed_min, min(bed_min+2, bed_max)))

date_min = None
date_max = None
if "Listing_Date" in df.columns and df["Listing_Date"].notna().any():
    date_min = df["Listing_Date"].min().date()
    date_max = df["Listing_Date"].max().date()
    date_sel = st.sidebar.date_input("Listing date range", (date_min, date_max), min_value=date_min, max_value=date_max)
else:
    date_sel = None

# Apply filters to df copy
dff = df.copy()
if city_sel:
    dff = dff[dff["City"].isin(city_sel)]
if ptype_sel:
    dff = dff[dff["Property_Type"].isin(ptype_sel)]
if "Price" in dff.columns:
    dff = dff[(pd.to_numeric(dff["Price"], errors="coerce") >= price_sel[0]) & (pd.to_numeric(dff["Price"], errors="coerce") <= price_sel[1])]
if "Bedrooms" in dff.columns:
    dff = dff[(pd.to_numeric(dff["Bedrooms"], errors="coerce") >= bed_sel[0]) & (pd.to_numeric(dff["Bedrooms"], errors="coerce") <= bed_sel[1])]
if date_sel and "Listing_Date" in dff.columns:
    dff = dff[(pd.to_datetime(dff["Listing_Date"]).dt.date >= date_sel[0]) & (pd.to_datetime(dff["Listing_Date"]).dt.date <= date_sel[1])]

if dff.shape[0] == 0:
    st.warning("Filters returned 0 rows. Falling back to whole dataset for visuals.")
    dff = df.copy()

# -------------------------
# KPIs row
# -------------------------
st.header("Overview — KPIs")
k1, k2, k3, k4, k5 = st.columns(5)
total_listings = len(dff)
avg_price = pd.to_numeric(dff["Price"], errors="coerce").mean()
median_pps = pd.to_numeric(dff["Price_per_SqFt"], errors="coerce").median()
avg_dom = pd.to_numeric(dff["Days_On_Market"], errors="coerce").mean()
avg_demand = pd.to_numeric(dff["Demand_Score"], errors="coerce").mean()

k1.metric("Total Listings", f"{total_listings:,}")
k2.metric("Avg Price", format_inr(avg_price))
k3.metric("Median ₹/sqft", format_inr(median_pps))
k4.metric("Avg Days on Market", f"{avg_dom:.1f}" if not math.isnan(avg_dom) else "N/A")
k5.metric("Avg Demand Score", f"{avg_demand:.1f}" if not math.isnan(avg_demand) else "N/A")

with st.expander("Purpose & Quick Tips — KPIs", expanded=False):
    st.markdown("- **Purpose:** Quick snapshot of market health for the selected filters.")
    st.markdown("- **Quick Tips:** Use city & property type filters to focus; median ₹/sqft reduces outlier bias.")

st.markdown("---")

# -------------------------
# Price distribution with labels
# -------------------------
st.subheader("Price Distribution")
if "Price" in dff.columns and pd.to_numeric(dff["Price"], errors="coerce").notna().any():
    fig = px.histogram(dff, x="Price", nbins=60, title="Price distribution (filtered dataset)", labels={"Price":"Price"})
    fig.update_traces(texttemplate="%{y}", textposition="inside")
    fig.update_layout(yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Purpose & Quick Tips — Price Distribution", expanded=False):
        st.markdown("- **Purpose:** See where listings cluster and look for outliers.")
        st.markdown("- **Quick Tips:** Switch price slider to zoom into specific bands.")
else:
    st.info("Price column is missing or empty in the dataset.")

st.markdown("---")

# -------------------------
# Price per sqft by City with data labels — better comparative chart
# -------------------------
st.subheader("Avg ₹/sqft by City — grouped by Property Type (two-category comparison)")
if "City" in dff.columns and "Price_per_SqFt" in dff.columns and "Property_Type" in dff.columns:
    # compute city-property_type averages and pivot top cities
    agg = dff.groupby(["City","Property_Type"], as_index=False)["Price_per_SqFt"].mean()
    top_cities = agg.groupby("City")["Price_per_SqFt"].mean().nlargest(12).index.tolist()
    agg_top = agg[agg["City"].isin(top_cities)]
    fig = px.bar(agg_top, x="City", y="Price_per_SqFt", color="Property_Type", barmode="group",
                 title="Avg ₹/sqft by City and Property Type (top cities)", text_auto=".2f",
                 labels={"Price_per_SqFt":"Avg ₹/sqft"})
    fig.update_traces(textposition="outside")
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Purpose & Quick Tips — Avg ₹/sqft by City & Type", expanded=False):
        st.markdown("- **Purpose:** Directly compare product-type pricing across major cities.")
        st.markdown("- **Quick Tips:** Use 'Property Type' filter to isolate single-product comparisons. Switch barmode to 'stack' for market share view.")
else:
    st.info("Required columns (City, Property_Type, Price_per_SqFt) missing.")

st.markdown("---")

# -------------------------
# City price boxplots (better detail)
# -------------------------
st.subheader("Price by City & Product (Boxplot)")
if "City" in dff.columns and "Price" in dff.columns and "Property_Type" in dff.columns:
    top_cities_price = dff.groupby("City")["Price"].median().nlargest(12).index.tolist()
    box_df = dff[dff["City"].isin(top_cities_price)]
    fig = px.box(box_df, x="City", y="Price", color="Property_Type", points="outliers",
                 title="Price distributions (top cities)")
    fig.update_traces(boxmean=True)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Purpose & Quick Tips — Price by City (Boxplot)", expanded=False):
        st.markdown("- **Purpose:** Understand within-city variance and outliers.")
        st.markdown("- **Quick Tips:** Wide boxes mean high variance — investigate listings with hover.")
else:
    st.info("City, Price, or Property_Type missing for boxplots.")

st.markdown("---")

# -------------------------
# Product mix (pie) with labels
# -------------------------
st.subheader("Product Mix")
if "Property_Type" in dff.columns:
    mix = dff["Property_Type"].value_counts().reset_index()
    mix.columns = ["Property_Type","Count"]
    fig = px.pie(mix, names="Property_Type", values="Count", hole=0.35, title="Property Type Mix")
    fig.update_traces(textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Purpose & Quick Tips — Product Mix", expanded=False):
        st.markdown("- **Purpose:** See which product types dominate supply.")
        st.markdown("- **Quick Tips:** If a product has high share but low demand, it's slow-moving inventory.")
else:
    st.info("Property_Type missing — cannot show product mix.")

st.markdown("---")

# -------------------------
# Top localities by price_per_sqft (horizontal bar labels)
# -------------------------
st.subheader("Top Localities by Avg ₹/sqft")
if "Locality" in dff.columns and "Price_per_SqFt" in dff.columns:
    loc_agg = dff.groupby("Locality", as_index=False)["Price_per_SqFt"].mean().nlargest(25, "Price_per_SqFt")
    fig = px.bar(loc_agg, x="Price_per_SqFt", y="Locality", orientation="h", text_auto=".2f",
                 title="Top Localities by Avg ₹/sqft")
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Purpose & Quick Tips — Top Localities", expanded=False):
        st.markdown("- **Purpose:** Identify premium pockets within the selected filter scope.")
        st.markdown("- **Quick Tips:** Combine with demand score to find high ROI micro-markets.")
else:
    st.info("Locality or Price_per_SqFt missing.")

st.markdown("---")

# -------------------------
# Geo scatter or heatmap fallback
# -------------------------
st.subheader("Geo Visuals / Heatmap")
if "Latitude" in dff.columns and "Longitude" in dff.columns and dff["Latitude"].notna().any() and dff["Longitude"].notna().any():
    geo_sample = dff.dropna(subset=["Latitude","Longitude"])
    if len(geo_sample) > 2000:
        geo_sample = geo_sample.sample(2000, random_state=42)
    fig = px.scatter_geo(geo_sample, lat="Latitude", lon="Longitude", color="Price", size="Price",
                         hover_name="Locality", title="Geo scatter (size ~ price)")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Purpose & Quick Tips — Geo Visuals", expanded=False):
        st.markdown("- **Purpose:** Map price hotspots and spatial clusters.")
        st.markdown("- **Quick Tips:** Use real lat/lon for better accuracy; use subset to improve performance.")
else:
    # heatmap pivot by locality and property_type
    if "Locality" in dff.columns and "Property_Type" in dff.columns and "Price_per_SqFt" in dff.columns:
        heat = dff.pivot_table(index="Locality", columns="Property_Type", values="Price_per_SqFt", aggfunc="mean").fillna(0)
        # limit rows + cols for readability
        if heat.shape[0] > 40:
            heat = heat.sort_values(by=heat.columns[0], ascending=False).head(40)
        fig = px.imshow(heat, title="Avg ₹/sqft by Locality and Property Type", labels={"x":"Property Type","y":"Locality","color":"Avg ₹/sqft"})
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Purpose & Quick Tips — Locality Heatmap", expanded=False):
            st.markdown("- **Purpose:** Compare localities across product types.")
            st.markdown("- **Quick Tips:** Filter to top localities to avoid clutter. Hover for exact values.")
    else:
        st.info("Latitude/Longitude missing and insufficient columns for locality heatmap.")

st.markdown("---")

# -------------------------
# Trend: Price over time (city-level)
# -------------------------
st.subheader("Price Trend Over Time (City-level)")
if "Listing_Date" in dff.columns and "Price" in dff.columns:
    ts = dff.dropna(subset=["Listing_Date","Price"]).copy()
    ts["YearMonth"] = pd.to_datetime(ts["Listing_Date"]).dt.to_period("M").dt.to_timestamp()
    trend = ts.groupby(["YearMonth","City"], as_index=False)["Price"].mean()
    top_city_trend = trend.groupby("City")["Price"].mean().nlargest(6).index.tolist()
    trend_top = trend[trend["City"].isin(top_city_trend)]
    fig = px.line(trend_top, x="YearMonth", y="Price", color="City", markers=True, title="Avg Price over time (top cities)")
    fig.update_traces(mode="lines+markers")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Purpose & Quick Tips — Price Trend", expanded=False):
        st.markdown("- **Purpose:** Detect seasonal trends and price momentum.")
        st.markdown("- **Quick Tips:** Increase data history for smoother trends; compare with interest rates externally.")
else:
    st.info("Listing_Date or Price missing — trend unavailable.")

st.markdown("---")

# -------------------------
# Export & preview filtered data
# -------------------------
st.subheader("Filtered dataset preview & download")
st.dataframe(dff.head(300), use_container_width=True)
csv = dff.to_csv(index=False).encode('utf-8')
st.download_button("Download filtered CSV", data=csv, file_name="market_intel_filtered.csv", mime="text/csv")

st.markdown("---")
st.caption("Market Intelligence — Real Estate. Exported values are in local currency. Want more visuals (choropleth, sunburst, cohort analysis)? Tell me which one to add next.")

