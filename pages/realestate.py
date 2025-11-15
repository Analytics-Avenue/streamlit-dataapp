import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import io
import random
import math
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
import re

st.set_page_config(layout="wide")

hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)


# -------------------------
# Page config & helper utils
# -------------------------
st.set_page_config(page_title="Real Estate Intelligence — Hybrid", layout="wide")
st.title("🏡 Real Estate Intelligence — Hybrid (Property + CRM)")

# ---------- Indian currency formatter ----------
def format_inr(value):
    """Format number into Indian style with Lakhs/Crores shorthand where appropriate."""
    try:
        v = float(value)
    except:
        return value
    if math.isnan(v):
        return "N/A"
    abs_v = abs(v)
    # Crores (>= 1e7)
    if abs_v >= 1e7:
        return f"₹{v/1e7:.2f} Cr"
    # Lakhs (>= 1e5)
    if abs_v >= 1e5:
        return f"₹{v/1e5:.2f} L"
    # Otherwise use Indian comma format for thousands
    # start from raw integer without any commas
    n = int(round(v))
    s = str(abs(n))
    # build indian grouping
    if len(s) <= 3:
        grouped = s
    else:
        # last 3 digits
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

# ---------- OneHotEncoder compatibility ----------
def safe_ohe():
    try:
        sig = inspect.signature(OneHotEncoder)
        if 'sparse_output' in sig.parameters:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)
    except Exception:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# ---------- robust request CSV ----------
def try_read_csv_url(url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))
    except Exception:
        return None

# ---------- ensure unique columns ----------
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

# ---------- synthesize hybrid dataset ----------
def make_hybrid_sample(n=5000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    india_cities = ["Bangalore","Mumbai","Chennai","Delhi","Pune","Hyderabad","Kolkata","Ahmedabad","Coimbatore","Lucknow"]
    global_cities = ["New York","London","Dubai","Singapore","Toronto"]
    cities = india_cities + global_cities
    property_types = ["Apartment","Villa","Condo","Townhouse","Studio"]
    furnishing_opts = ["Unfurnished","Semi","Fully"]
    parking_opts = ["Yes","No"]
    seller_types = ["Builder","Agent","Owner"]
    agents = ["Aditi","Manoj","Riya","Kiran","Sanjay","Neha","Vikram","Priya","Rahul","Sunil"]
    lead_sources = ["Website","Walk-in","Referral","Channel Partner","Social Media","Event"]
    lead_statuses = ["New","Contacted","Site Visit","Negotiation","Booked","Lost"]
    rows = []
    today = datetime.today()
    for i in range(n):
        city = random.choice(cities)
        country = "India" if city in india_cities else random.choice(["USA","UK","UAE","Singapore","Canada"])
        loc = f"Loc_{random.randint(1,300)}"
        region = f"{city}_R{random.randint(1,20)}"
        ptype = random.choice(property_types)
        sqft = int(max(250, np.random.normal(1100, 450)))
        bedrooms = max(1, int(sqft//450) if random.random() > 0.3 else random.randint(1,5))
        bathrooms = max(1, min(4, bedrooms if bedrooms<=3 else bedrooms-1))
        year_built = random.randint(1980, 2022)
        price_base = {
            "Bangalore":9000,"Mumbai":18000,"Chennai":6500,"Delhi":12000,"Pune":8500,"Hyderabad":6000,"Kolkata":4500,"Ahmedabad":4000,"Coimbatore":3500,"Lucknow":3000,
            "New York":2000,"London":1700,"Dubai":2000,"Singapore":2200,"Toronto":1300
        }.get(city,8000)
        price = max(50000, int(sqft * price_base * np.random.normal(1.0, 0.12)))
        lat = round(np.random.uniform(-33,51),6)
        lon = round(np.random.uniform(-118,151),6)
        listing_agent = random.choice(agents)
        agent_experience = random.randint(1,15)
        agent_deals_closed = random.randint(0,200)
        seller_type = random.choice(seller_types)
        furnishing = random.choice(furnishing_opts)
        parking = random.choice(parking_opts)
        demand = random.randint(10,100)
        school = random.randint(1,10)
        interest = round(np.random.uniform(3.0,8.0),2)
        econ = random.randint(80,160)
        days_on_market = random.randint(1,240)
        listing_date = today - timedelta(days=random.randint(0,720))
        # CRM fields (lead-level)
        lead_id = f"LD{i+1:06d}"
        lead_source = random.choice(lead_sources)
        lead_status = random.choice(lead_statuses)
        lead_score = random.randint(1,100)
        time_to_sell = days_on_market + random.randint(-10,30)
        sales_agent = random.choice(agents)
        followup_count = random.randint(0,10)
        rows.append({
            "Property_ID": f"P{i+1:06d}",
            "Listing_ID": f"L{i+1:06d}",
            "Country": country,
            "City": city,
            "Region": region,
            "Locality": loc,
            "Property_Type": ptype,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Carpet_Area": int(sqft*0.8),
            "Builtup_Area": sqft,
            "Square_Footage": sqft,
            "Year_Built": year_built,
            "Price": price,
            "Price_per_SqFt": round(price / sqft, 2) if sqft>0 else np.nan,
            "Parking": parking,
            "Furnishing": furnishing,
            "Facing": random.choice(["North","South","East","West"]),
            "Listing_Agent": listing_agent,
            "Agent_Experience": agent_experience,
            "Agent_Deals_Closed": agent_deals_closed,
            "Seller_Type": seller_type,
            "Listing_Date": listing_date.date(),
            "Demand_Score": demand,
            "School_Rating": school,
            "Interest_Rate": interest,
            "Economic_Index": econ,
            "Days_On_Market": days_on_market,
            # CRM/Lead fields
            "Lead_ID": lead_id,
            "Lead_Source": lead_source,
            "Lead_Status": lead_status,
            "Lead_Score": lead_score,
            "Time_to_Sell": time_to_sell,
            "Sales_Agent": sales_agent,
            "Followup_Count": followup_count,
            # location
            "Latitude": lat,
            "Longitude": lon
        })
    df = pd.DataFrame(rows)
    return df

# -------------------------
# Sidebar: dataset mode, sample download
# -------------------------
st.sidebar.header("Data source options")
mode = st.sidebar.radio("Choose dataset source:", ("Default GitHub dataset", "Upload CSV", "Upload CSV & Manual Mapping"))
sample_df = make_hybrid_sample(5000)
buf = io.StringIO(); sample_df.to_csv(buf, index=False)
st.sidebar.download_button("Download sample hybrid CSV (5k rows)", data=buf.getvalue().encode('utf-8'), file_name="real_estate_hybrid_sample_5000.csv", mime="text/csv")

# GitHub default raw URL placeholder (user can replace)
GITHUB_RAW = "https://raw.githubusercontent.com/your-username/your-repo/main/real_estate_hybrid_5000.csv"

df_raw = None
mapping = {}

if mode == "Default GitHub dataset":
    st.sidebar.info("Attempting to load default GitHub dataset (fallback to local synthethic if not reachable).")
    tmp = try_read_csv_url(GITHUB_RAW)
    if tmp is not None:
        df_raw = tmp
        st.sidebar.success("Loaded dataset from GitHub.")
    else:
        # fallback to local sample created by assistant or generated sample
        try:
            df_local = pd.read_csv("/mnt/data/sample_dataset_5000_rows.csv")
            df_raw = df_local
            st.sidebar.warning("GitHub not reachable — loaded local fallback CSV.")
        except Exception:
            df_raw = sample_df.copy()
            st.sidebar.warning("Using internal synthetic hybrid sample dataset.")

elif mode == "Upload CSV" or mode == "Upload CSV & Manual Mapping":
    uploaded = st.sidebar.file_uploader("Upload CSV (must contain property data)", type=["csv"])
    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
            st.sidebar.success("Uploaded CSV loaded.")
        except Exception as e:
            st.sidebar.error("Failed to read upload: " + str(e))
            df_raw = None
    else:
        st.sidebar.info("No file uploaded yet; using internal sample for visuals.")

# Manual mapping UI
EXPECTED = [
    "Property_ID","Listing_ID","Country","City","Region","Locality","Property_Type","Bedrooms","Bathrooms",
    "Carpet_Area","Builtup_Area","Square_Footage","Year_Built","Price","Price_per_SqFt","Parking","Furnishing",
    "Facing","Listing_Agent","Agent_Experience","Agent_Deals_Closed","Seller_Type","Listing_Date","Demand_Score",
    "School_Rating","Interest_Rate","Economic_Index","Days_On_Market","Lead_ID","Lead_Source","Lead_Status","Lead_Score",
    "Time_to_Sell","Sales_Agent","Followup_Count","Latitude","Longitude"
]

if mode == "Upload CSV & Manual Mapping" and df_raw is not None:
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
# Build mapped dataframe
# -------------------------
if df_raw is None:
    df = sample_df.copy()
else:
    df = df_raw.copy()
    df = ensure_unique_columns(df)
    if mapping and any(v and v != "<none>" for v in mapping.values()):
        # explicit mapping
        mapped = pd.DataFrame()
        for exp in EXPECTED:
            sel = mapping.get(exp)
            if sel and sel != "<none>" and sel in df.columns:
                mapped[exp] = df[sel]
            else:
                # attempt to auto-detect similar column names
                found = None
                for c in df.columns:
                    if c.strip().lower() == exp.strip().lower() or exp.strip().lower() in c.strip().lower() or c.strip().lower() in exp.strip().lower():
                        found = c
                        break
                if found:
                    mapped[exp] = df[found]
                else:
                    mapped[exp] = np.nan
        df = mapped.copy()
    else:
        # auto mapping by reasonable heuristics
        normalized = {c.strip().lower(): c for c in df.columns}
        auto = pd.DataFrame()
        for exp in EXPECTED:
            key = exp.strip().lower()
            if key in normalized:
                auto[exp] = df[normalized[key]]
            else:
                # find best match by substring
                cand = None
                for c in df.columns:
                    if key in c.strip().lower() or c.strip().lower() in key or any(tok in c.strip().lower() for tok in key.split("_")):
                        cand = c
                        break
                if cand:
                    auto[exp] = df[cand]
                else:
                    auto[exp] = np.nan
        df = auto.copy()

# Normalize types and compute price per sqft if missing
numeric_cols = ["Bedrooms","Bathrooms","Carpet_Area","Builtup_Area","Square_Footage","Year_Built","Price","Price_per_SqFt","Agent_Experience","Agent_Deals_Closed","Demand_Score","School_Rating","Interest_Rate","Economic_Index","Days_On_Market","Lead_Score","Time_to_Sell","Followup_Count","Latitude","Longitude"]
for c in numeric_cols:
    if c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except:
            pass

if "Price_per_SqFt" not in df.columns or df["Price_per_SqFt"].isna().all():
    if "Price" in df.columns and "Square_Footage" in df.columns:
        df["Price_per_SqFt"] = (pd.to_numeric(df["Price"], errors="coerce") / pd.to_numeric(df["Square_Footage"], errors="coerce")).round(2)

# parse dates
if "Listing_Date" in df.columns:
    try:
        df["Listing_Date"] = pd.to_datetime(df["Listing_Date"], errors="coerce")
    except:
        pass

# Ensure minimal columns exist, else synthesize demo fields
required_property_cols = ["Price","Square_Footage","Bedrooms","Bathrooms","City","Property_Type"]
for col in required_property_cols:
    if col not in df.columns:
        # fill demo defaults
        if col == "Price":
            df[col] = np.nan
        elif col == "Square_Footage":
            df[col] = 1200
        elif col == "Bedrooms":
            df[col] = 2
        elif col == "Bathrooms":
            df[col] = 2
        else:
            df[col] = "Unknown"

# If CRM columns missing, synthesize demo leads (keeps lead scoring and funnels functional)
crm_cols = ["Lead_ID","Lead_Source","Lead_Status","Lead_Score","Time_to_Sell","Sales_Agent","Followup_Count"]
crm_missing = [c for c in crm_cols if c not in df.columns or df[c].isna().all()]
if crm_missing:
    st.sidebar.info("CRM fields missing — synthesizing demo leads so CRM modules can run. (You can upload a real CRM dataset or map columns.)")
    agents = df["Listing_Agent"].dropna().unique().tolist() if "Listing_Agent" in df.columns and df["Listing_Agent"].notna().any() else ["Aditi","Manoj","Riya","Kiran"]
    for i in range(len(df)):
        df.at[df.index[i], "Lead_ID"] = df.at[df.index[i], "Lead_ID"] if "Lead_ID" in df.columns and pd.notna(df.at[df.index[i], "Lead_ID"]) else f"LD{i+1:06d}"
        df.at[df.index[i], "Lead_Source"] = df.at[df.index[i], "Lead_Source"] if "Lead_Source" in df.columns and pd.notna(df.at[df.index[i], "Lead_Source"]) else random.choice(["Website","Walk-in","Referral","Channel","Event"])
        df.at[df.index[i], "Lead_Status"] = df.at[df.index[i], "Lead_Status"] if "Lead_Status" in df.columns and pd.notna(df.at[df.index[i], "Lead_Status"]) else random.choice(["New","Contacted","Site Visit","Negotiation","Booked","Lost"])
        df.at[df.index[i], "Lead_Score"] = df.at[df.index[i], "Lead_Score"] if "Lead_Score" in df.columns and pd.notna(df.at[df.index[i], "Lead_Score"]) else random.randint(1,100)
        df.at[df.index[i], "Time_to_Sell"] = df.at[df.index[i], "Time_to_Sell"] if "Time_to_Sell" in df.columns and pd.notna(df.at[df.index[i], "Time_to_Sell"]) else int(df.at[df.index[i], "Days_On_Market"] if "Days_On_Market" in df.columns and pd.notna(df.at[df.index[i], "Days_On_Market"]) else random.randint(30,180))
        df.at[df.index[i], "Sales_Agent"] = df.at[df.index[i], "Sales_Agent"] if "Sales_Agent" in df.columns and pd.notna(df.at[df.index[i], "Sales_Agent"]) else random.choice(agents)
        df.at[df.index[i], "Followup_Count"] = df.at[df.index[i], "Followup_Count"] if "Followup_Count" in df.columns and pd.notna(df.at[df.index[i], "Followup_Count"]) else random.randint(0,8)

# Ensure Agent field exists for leaderboard
if "Listing_Agent" not in df.columns or df["Listing_Agent"].isna().all():
    df["Listing_Agent"] = [random.choice(["Aditi","Manoj","Riya","Kiran","Sanjay"]) for _ in range(len(df))]

# Create Region if missing
if "Region" not in df.columns or df["Region"].isna().all():
    df["Region"] = df["City"].fillna("Unknown") + "_" + df["Locality"].fillna("L0").astype(str)

# -------------------------
# Filtering UI
# -------------------------
st.sidebar.header("Filters")
cities = sorted(df["City"].dropna().unique().tolist()) if "City" in df.columns else []
city_sel = st.sidebar.multiselect("City", options=cities, default=cities[:6] if cities else [])
ptype_sel = st.sidebar.multiselect("Property Type", options=sorted(df["Property_Type"].dropna().unique().tolist()) if "Property_Type" in df.columns else [], default=None)
agent_sel = st.sidebar.multiselect("Listing Agent", options=sorted(df["Listing_Agent"].dropna().unique().tolist()), default=None)

# price slider
if df["Price"].notna().any():
    pmin = int(pd.to_numeric(df["Price"], errors="coerce").min())
    pmax = int(pd.to_numeric(df["Price"], errors="coerce").max())
    price_sel = st.sidebar.slider("Price range", min_value=pmin, max_value=pmax, value=(pmin, pmax))
else:
    price_sel = (0,0)

# bedrooms
if "Bedrooms" in df.columns:
    bmin = int(df["Bedrooms"].min())
    bmax = int(df["Bedrooms"].max())
    bed_sel = st.sidebar.slider("Bedrooms", min_value=bmin, max_value=bmax, value=(bmin, min(bmin+2,bmax)))
else:
    bed_sel = None

# date range
if "Listing_Date" in df.columns and df["Listing_Date"].notna().any():
    min_date = df["Listing_Date"].min().date()
    max_date = df["Listing_Date"].max().date()
    date_sel = st.sidebar.date_input("Listing date range", (min_date, max_date), min_value=min_date, max_value=max_date)
else:
    date_sel = None

# Apply filters
dff = df.copy()
if city_sel:
    dff = dff[dff["City"].isin(city_sel)]
if ptype_sel:
    dff = dff[dff["Property_Type"].isin(ptype_sel)]
if agent_sel:
    dff = dff[dff["Listing_Agent"].isin(agent_sel)]
if price_sel != (0,0) and "Price" in dff.columns:
    dff = dff[(pd.to_numeric(dff["Price"], errors="coerce") >= price_sel[0]) & (pd.to_numeric(dff["Price"], errors="coerce") <= price_sel[1])]
if bed_sel is not None:
    dff = dff[(pd.to_numeric(dff["Bedrooms"], errors="coerce") >= bed_sel[0]) & (pd.to_numeric(dff["Bedrooms"], errors="coerce") <= bed_sel[1])]
if date_sel and "Listing_Date" in dff.columns:
    dff = dff[(pd.to_datetime(dff["Listing_Date"]).dt.date >= date_sel[0]) & (pd.to_datetime(dff["Listing_Date"]).dt.date <= date_sel[1])]

if len(dff) == 0:
    st.warning("Filters returned 0 rows — falling back to full dataset for visuals.")
    dff = df.copy()

# -------------------------
# Main layout: KPIs & charts
# -------------------------
st.header("Overview & KPIs")
col1,col2,col3,col4,col5 = st.columns(5)
total_listings = len(dff)
avg_price = pd.to_numeric(dff["Price"], errors="coerce").mean()
median_pps = pd.to_numeric(dff["Price_per_SqFt"], errors="coerce").median()
avg_dom = pd.to_numeric(dff["Days_On_Market"], errors="coerce").mean()
avg_demand = pd.to_numeric(dff["Demand_Score"], errors="coerce").mean()

col1.metric("Total Listings", f"{total_listings:,}")
col2.metric("Avg Price", format_inr(avg_price))
col3.metric("Median ₹/sqft", format_inr(median_pps))
col4.metric("Avg Days on Market", f"{avg_dom:.1f}" if not math.isnan(avg_dom) else "N/A")
col5.metric("Avg Demand Score", f"{avg_demand:.1f}" if not math.isnan(avg_demand) else "N/A")

st.markdown("**Purpose:** Quick snapshot. **Quick Tip:** Use filters to focus on regions or property segments.")

st.markdown("---")
st.subheader("Price Distribution")
if "Price" in dff.columns:
    fig = px.histogram(dff, x="Price", nbins=50, title="Price distribution", labels={"Price":"Price"})
    fig.update_traces(texttemplate="%{y}", textposition="inside")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Purpose & Quick Tips — Price Distribution", expanded=False):
        st.markdown("- **Purpose:** Discover where listings cluster and detect outliers.")
        st.markdown("- **Quick Tips:** Try log-scale or filter by city to reveal local patterns. Use 20–50 bins for typical datasets.")
else:
    st.info("Price column missing.")

st.markdown("---")
st.subheader("Price by City & Property Type")
if "City" in dff.columns and "Price" in dff.columns:
    fig = px.box(dff, x="City", y="Price", color="Property_Type", title="Price by City and Property Type", points="outliers")
    fig.update_traces(boxmean=True)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Purpose & Quick Tips — Price by City & Type", expanded=False):
        st.markdown("- **Purpose:** Compare pricing distributions across cities and product types.")
        st.markdown("- **Quick Tips:** Look for wide boxes (high variance) and outliers—they indicate inconsistent pricing or special listings.")
else:
    st.info("City or Price missing.")

st.markdown("---")
st.subheader("Avg Price per Sqft by City")
if "City" in dff.columns and "Price_per_SqFt" in dff.columns:
    pps = dff.groupby("City", as_index=False)["Price_per_SqFt"].mean().sort_values("Price_per_SqFt", ascending=False)
    fig = px.bar(pps, x="City", y="Price_per_SqFt", title="Avg Price per Sqft by City", text_auto=".2f")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Purpose & Quick Tips — Avg ₹/sqft", expanded=False):
        st.markdown("- **Purpose:** Identify high-value cities and relative pricing per area.")
        st.markdown("- **Quick Tips:** Combine this with demand score to find undervalued markets.")
else:
    st.info("Price_per_SqFt or City missing.")

st.markdown("---")
st.subheader("Property Type Mix")
if "Property_Type" in dff.columns:
    mix = dff["Property_Type"].value_counts().reset_index()
    mix.columns = ["Property_Type","Count"]
    fig = px.pie(mix, names="Property_Type", values="Count", hole=0.35, title="Property Type Mix")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Purpose & Quick Tips — Property Mix", expanded=False):
        st.markdown("- **Purpose:** See product mix (supply composition).")
        st.markdown("- **Quick Tips:** Use this to match inventory to demand; large share + low demand = slow-moving inventory.")
else:
    st.info("Property_Type missing.")

# -------------------------
# Geo map and heatmap (show all property types)
# -------------------------
st.markdown("---")
st.subheader("Region / Geo Visuals")
if dff["Latitude"].notna().any() and dff["Longitude"].notna().any():
    geo_sample = dff.dropna(subset=["Latitude","Longitude"])
    if len(geo_sample) > 1500:
        geo_sample = geo_sample.sample(1500, random_state=42)
    fig = px.scatter_geo(geo_sample, lat="Latitude", lon="Longitude", color="Price", hover_name="City", size="Price", title="Geo scatter (size ~ price)")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Purpose & Quick Tips — Geo Visuals", expanded=False):
        st.markdown("- **Purpose:** Visualize spatial clusters & price hotspots.")
        st.markdown("- **Quick Tips:** Use real lat/lon for accurate maps. Sample large datasets to keep performance smooth.")
else:
    # fallback pivot heatmap: location x property_type
    if "Locality" in dff.columns and "Property_Type" in dff.columns and "Price_per_SqFt" in dff.columns:
        heat = dff.pivot_table(index="Locality", columns="Property_Type", values="Price_per_SqFt", aggfunc="mean").fillna(0)
        fig = px.imshow(heat, title="Avg ₹/sqft by Locality and Property Type", labels={"x":"Property Type","y":"Locality","color":"Avg ₹/sqft"})
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Purpose & Quick Tips — Locality Heatmap", expanded=False):
            st.markdown("- **Purpose:** Compare locality-level price intensity across product types.")
            st.markdown("- **Quick Tips:** Filter to top localities to avoid clutter. Hover cells for exact ₹/sqft.")
    else:
        st.info("Latitude/Longitude missing and insufficient columns for location heatmap.")

# -------------------------
# Agent Leaderboard
# -------------------------
st.markdown("---")
st.subheader("Agent Leaderboard")
st.markdown("**Purpose:** Rank agents by listings and value. **Quick Tip:** Review top agents for replication.")
if "Listing_Agent" in dff.columns:
    agent_df = dff.groupby("Listing_Agent").agg(
        Listings=("Listing_ID","count") if "Listing_ID" in dff.columns else ("Property_ID","count"),
        Total_Value=("Price","sum"),
        Avg_Price=("Price","mean")
    ).reset_index().sort_values("Total_Value", ascending=False)
    agent_df["Total_Value_str"] = agent_df["Total_Value"].apply(format_inr)
    st.dataframe(agent_df.head(20), use_container_width=True)
    with st.expander("Purpose & Quick Tips — Agent Leaderboard", expanded=False):
        st.markdown("- **Purpose:** Identify top-performing agents by value and volume.")
        st.markdown("- **Quick Tips:** Combine with Avg_Price to spot agents selling premium inventory vs. volume sellers.")
else:
    st.info("Listing_Agent missing — leaderboard not available.")

# -------------------------
# Clustering
# -------------------------
st.markdown("---")
st.subheader("Clustering (KMeans)")
st.markdown("**Purpose:** Segment properties into buckets (budget, premium). Quick Tip: Inspect cluster centers to label segments.")
if "Price" in dff.columns and "Square_Footage" in dff.columns:
    cluster_run = st.button("Run KMeans (k=4)")
    if cluster_run:
        clu_df = dff.dropna(subset=["Price","Square_Footage"])[["Price","Square_Footage"]].copy()
        clu_df["Price_log"] = np.log1p(clu_df["Price"])
        Xc = clu_df[["Price_log","Square_Footage"]].values
        scaler = StandardScaler(); Xs = scaler.fit_transform(Xc)
        k = 4
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        clu_df["cluster"] = labels
        plot_df = clu_df.copy(); plot_df["Price"] = np.expm1(plot_df["Price_log"])
        fig = px.scatter(plot_df, x="Square_Footage", y="Price", color="cluster", title="KMeans clusters (sqft vs price)")
        st.plotly_chart(fig, use_container_width=True)
        st.write("Cluster distribution:")
        st.write(clu_df["cluster"].value_counts())
        with st.expander("Purpose & Quick Tips — Clustering", expanded=False):
            st.markdown("- **Purpose:** Create segments to price/position inventory differently.")
            st.markdown("- **Quick Tips:** Try different k (3–6). Label clusters by centroid price & size for business use.")
else:
    st.info("Price and Square_Footage required for clustering.")

# -------------------------
# Forecasting (time-aware RF)
# -------------------------
st.markdown("---")
st.subheader("Forecasting (monthly) — simple time-aware model")
st.markdown("**Purpose:** Short-term price forecasting. Quick Tip: Use more months and external macro data for better forecasts.")
if "Listing_Date" in dff.columns and "Price" in dff.columns:
    ts = dff.copy()
    ts["Date"] = pd.to_datetime(ts["Listing_Date"], errors="coerce")
    ts = ts.dropna(subset=["Date","Price"])
    ts["YearMonth"] = ts["Date"].dt.to_period("M").dt.to_timestamp()
    monthly = ts.groupby("YearMonth")["Price"].mean().reset_index().sort_values("YearMonth")
    if len(monthly) < 12:
        st.info("Not enough months (<12) for meaningful forecasting.")
    else:
        monthly["year"] = monthly["YearMonth"].dt.year
        monthly["month"] = monthly["YearMonth"].dt.month
        monthly["lag1"] = monthly["Price"].shift(1).fillna(method="bfill")
        X = monthly[["year","month","lag1"]]; y = monthly["Price"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        mse = mean_squared_error(y_test, pred); rmse = math.sqrt(mse)
        st.write(f"Backtest RMSE: {format_inr(rmse)}")
        # forecast next 6 months
        last = monthly.iloc[-1]
        last_dt = pd.to_datetime(last["YearMonth"])
        lag = last["Price"]
        fut_rows = []
        for m in range(1,7):
            future_dt = last_dt + pd.DateOffset(months=m)
            fut_rows.append({"YearMonth": future_dt, "year": future_dt.year, "month": future_dt.month, "lag1": lag})
            lag = rf.predict([[future_dt.year, future_dt.month, lag]])[0]
        fut = pd.DataFrame(fut_rows)
        fut_preds = rf.predict(fut[["year","month","lag1"]])
        forecast = pd.DataFrame({"YearMonth": fut["YearMonth"], "Predicted_Price": fut_preds})
        combined = pd.concat([monthly[["YearMonth","Price"]].rename(columns={"Price":"Actual_Price"}), forecast], ignore_index=True, sort=False)
        fig = px.line(combined, x="YearMonth", y=["Actual_Price","Predicted_Price"], title="Price Forecast (monthly)")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Purpose & Quick Tips — Forecasting", expanded=False):
            st.markdown("- **Purpose:** Predict short-term direction for prices.")
            st.markdown("- **Quick Tips:** Use more history and macro indicators (interest rates, CPI) for stability.")
else:
    st.info("Listing_Date or Price missing — forecasting unavailable.")

# -------------------------
# Recommender (NearestNeighbors)
# -------------------------
st.markdown("---")
st.subheader("Recommender: find similar listings")
st.markdown("**Purpose:** Suggest similar properties. Quick Tip: use price & sqft filters before requesting similarity.")
if "Price" in dff.columns and "Square_Footage" in dff.columns:
    cand = dff.dropna(subset=["Price","Square_Footage"]).reset_index(drop=True)
    # build features matrix
    num_feats = cand[["Price","Square_Footage","Bedrooms","Bathrooms"]].fillna(0)
    ptype_ohe = pd.get_dummies(cand["Property_Type"].fillna("Unknown"))
    Xrec = pd.concat([num_feats, ptype_ohe], axis=1).fillna(0)
    if len(Xrec) >= 2:
        nn = NearestNeighbors(n_neighbors=6)
        try:
            nn.fit(Xrec)
            idx = st.number_input("Pick a listing index to find similar (0..n-1)", min_value=0, max_value=len(cand)-1, value=0)
            dists, inds = nn.kneighbors([Xrec.iloc[idx]])
            sim_idx = inds[0][1:]
            st.write("Similar listings:")
            st.dataframe(cand.iloc[sim_idx][["Listing_ID","City","Locality","Property_Type","Price","Square_Footage","Bedrooms","Bathrooms"]])
            with st.expander("Purpose & Quick Tips — Recommender", expanded=False):
                st.markdown("- **Purpose:** Surface comparable listings for buyers or investors.")
                st.markdown("- **Quick Tips:** Pick a representative index (not an outlier) for better recommendations.")
        except Exception as e:
            st.info("Recommender error: " + str(e))
    else:
        st.info("Not enough listings for recommender.")
else:
    st.info("Price & Square_Footage required for recommender.")

# -------------------------
# NLP-lite keyword search (improved)
# -------------------------
st.markdown("---")
st.subheader("Keyword Search (NLP-lite)")
st.markdown("Examples: `3BHK in Bangalore under 1.5cr with parking`, `2BHK Delhi 75L`, `villa in Mumbai`")
query = st.text_input("Search properties by keywords")
def parse_budget(token):
    # accepts formats like 1.5cr, 75L, 75l, 75lakhs, 7500000
    token = token.strip().lower()
    m = re.match(r'([0-9]+(\.[0-9]+)?)\s*(cr|crore|crores)', token)
    if m:
        return float(m.group(1))*1e7
    m = re.match(r'([0-9]+(\.[0-9]+)?)\s*(l|lakhs|lakh|lakhs)', token)
    if m:
        return float(m.group(1))*1e5
    m = re.match(r'([0-9]+(\.[0-9]+)?)\s*(k|thousand)', token)
    if m:
        return float(m.group(1))*1e3
    # plain number (assume rupees)
    try:
        return float(token)
    except:
        return None

def nlp_search(df_input, q):
    ql = q.lower()
    res = df_input.copy()
    # parse bhk / bhk patterns: 3bhk, 3 bhk, 3BHK, '3bdr'
    bhk_match = re.search(r'(\d+)\s*[-]?\s*bhk|\s(\d+)\s*bdr|\s(\d+)\s*br', ql)
    if bhk_match:
        nums = [g for g in bhk_match.groups() if g]
        try:
            bhk = int(nums[0])
            if "Bedrooms" in res.columns:
                res = res[pd.to_numeric(res["Bedrooms"], errors="coerce")==bhk]
        except:
            pass
    # price cap 'under X'
    under_match = re.search(r'under\s*([0-9\.]+(?:\s*(?:cr|crore|l|lakhs|k|m|mn))?)', ql)
    if under_match:
        val = parse_budget(under_match.group(1))
        if val and "Price" in res.columns:
            res = res[pd.to_numeric(res["Price"], errors="coerce") <= val]
    # price range 'between X and Y'
    between_match = re.search(r'between\s*([0-9\.]+(?:\s*(?:cr|l|lakhs|k))?)\s*and\s*([0-9\.]+(?:\s*(?:cr|l|lakhs|k))?)', ql)
    if between_match:
        v1 = parse_budget(between_match.group(1)); v2 = parse_budget(between_match.group(2))
        if v1 and v2 and "Price" in res.columns:
            res = res[(pd.to_numeric(res["Price"], errors="coerce") >= v1) & (pd.to_numeric(res["Price"], errors="coerce") <= v2)]
    # city detection: match any city token present in City column
    if "City" in res.columns:
        for city in res["City"].dropna().unique():
            if city.lower() in ql:
                res = res[res["City"].str.lower() == city.lower()]
                break
    # parking
    if "parking" in ql and "Parking" in res.columns:
        res = res[res["Parking"].astype(str).str.lower().str.contains("yes")]
    # property type
    for pt in ["apartment","villa","condo","townhouse","studio","plot","house"]:
        if pt in ql and "Property_Type" in res.columns:
            res = res[res["Property_Type"].str.lower().str.contains(pt)]
            break
    return res

if query:
    results = nlp_search(dff, query)
    st.write(f"Found {len(results)} matches")
    if len(results) > 0:
        # format price column
        display = results.copy()
        if "Price" in display.columns:
            display["Price_str"] = display["Price"].apply(format_inr)
            cols_show = ["Listing_ID","City","Locality","Property_Type","Bedrooms","Bathrooms","Square_Footage","Price_str","Parking"]
            available = [c for c in cols_show if c in display.columns]
            st.dataframe(display[available].head(200), use_container_width=True)
        with st.expander("Purpose & Quick Tips — Keyword Search", expanded=False):
            st.markdown("- **Purpose:** Quickly find candidate listings using natural keywords.")
            st.markdown("- **Quick Tips:** Use `under 1.5cr`, `3BHK`, or city names. If 0 matches, broaden budget or remove BHK filter.")
    else:
        st.info("No matches found. Try broader terms or check sample CSV.")
        with st.expander("Purpose & Quick Tips — Keyword Search", expanded=False):
            st.markdown("- **Purpose:** Quickly find candidate listings using natural keywords.")
            st.markdown("- **Quick Tips:** Use `under 1.5cr`, `3BHK`, or city names. If 0 matches, broaden budget or remove BHK filter.")

# -------------------------
# Lead scoring & funnel
# -------------------------
st.markdown("---")
st.subheader("CRM Funnel & Lead Scoring")
st.markdown("If you uploaded a CRM table or mapped lead columns, the funnel and lead scoring will be computed; otherwise demo leads were synthesized for functionality.")
# Funnel
if "Lead_Status" in dff.columns:
    funnel = dff["Lead_Status"].value_counts().reset_index()
    funnel.columns = ["Stage","Count"]
    fig = px.bar(funnel, x="Stage", y="Count", title="Lead Funnel")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Purpose & Quick Tips — Lead Funnel", expanded=False):
        st.markdown("- **Purpose:** Visualize lead progression across stages.")
        st.markdown("- **Quick Tips:** Use Lead_Source filter to evaluate channel performance.")
else:
    st.info("Lead_Status missing — cannot show funnel.")

# Lead scoring (simple RF predicting conversion-like target: status == 'Booked')
if "Lead_Status" in dff.columns and "Lead_Score" in dff.columns:
    ls = dff.copy()
    ls["Booked_flag"] = ls["Lead_Status"].apply(lambda x: 1 if str(x).lower().strip() == "booked" else 0)
    features = [c for c in ["Bedrooms","Bathrooms","Price","Square_Footage","Demand_Score","Lead_Score"] if c in ls.columns]
    if len(features) >= 2:
        X = ls[features].fillna(0)
        y = ls["Booked_flag"]
        if y.nunique() > 1 and len(X) >= 40:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = RandomForestRegressor(n_estimators=150, random_state=42)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            mse = mean_squared_error(y_test, preds); rmse = math.sqrt(mse)
            st.write(f"Lead->Booked model RMSE: {rmse:.3f} (lower is better)")
            st.markdown("Top features used:")
            st.write(features)
            with st.expander("Purpose & Quick Tips — Lead Scoring", expanded=False):
                st.markdown("- **Purpose:** Prioritize leads with higher conversion probability.")
                st.markdown("- **Quick Tips:** Use more CRM features (followups, source, agent touchpoints) for accuracy.")
        else:
            st.info("Not enough data or no variance in Booking flag to train lead model.")
    else:
        st.info("Not enough features for lead scoring model.")
else:
    st.info("Lead columns missing; demo leads were generated for presentation.")

# -------------------------
# Time-to-sell model
# -------------------------
st.markdown("---")
st.subheader("Time-to-Sell Prediction")
if "Time_to_Sell" in dff.columns:
    tdf = dff.dropna(subset=["Time_to_Sell","Price","Square_Footage"])
    features = [c for c in ["Price","Square_Footage","Bedrooms","Bathrooms","Demand_Score"] if c in tdf.columns]
    if len(tdf) >= 50 and len(features) >= 2:
        X = tdf[features].fillna(0)
        y = tdf["Time_to_Sell"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        m = RandomForestRegressor(n_estimators=150, random_state=42)
        m.fit(X_train, y_train)
        p = m.predict(X_test)
        rmse = math.sqrt(mean_squared_error(y_test, p))
        st.write(f"Time-to-sell model RMSE: {rmse:.1f} days")
        if st.button("Predict sample time-to-sell (median values)"):
            sample = {c: [tdf[c].median()] for c in features}
            pred_days = m.predict(pd.DataFrame(sample))[0]
            st.success(f"Predicted Days on Market: {pred_days:.0f}")
        with st.expander("Purpose & Quick Tips — Time-to-Sell", expanded=False):
            st.markdown("- **Purpose:** Estimate listing duration to help pricing and incentives.")
            st.markdown("- **Quick Tips:** Combine with agent & locality performance for realistic targets.")
    else:
        st.info("Not enough data or features to train time-to-sell model.")
else:
    st.info("Time_to_Sell column missing; cannot train time-to-sell model.")

# -------------------------
# ML Price Prediction (full feature set)
# -------------------------
st.markdown("---")
st.subheader("ML Price Prediction — Industry model")
st.markdown("Features used: Price ~ Square_Footage + Bedrooms + Bathrooms + City + Property_Type + Year_Built + Parking + Furnishing + Latitude + Longitude")

# Build feature matrix
ml_features = ["Square_Footage","Bedrooms","Bathrooms","City","Property_Type","Year_Built","Parking","Furnishing","Latitude","Longitude"]
# ensure the features exist in dff (use best-effort mapping)
feat_exists = [f for f in ml_features if f in dff.columns]
missing_feats = [f for f in ml_features if f not in dff.columns]
if missing_feats:
    st.sidebar.info(f"Some ML features missing and will be auto-filled: {missing_feats}")

# Prepare data for ML
ml_df = dff.copy()
# fill numeric missing sensibly
for nf in ["Square_Footage","Bedrooms","Bathrooms","Year_Built","Latitude","Longitude"]:
    if nf in ml_df.columns:
        ml_df[nf] = pd.to_numeric(ml_df[nf], errors="coerce").fillna(ml_df[nf].median() if ml_df[nf].median()==ml_df[nf].median() else 0)
# categorical fill
for cf in ["City","Property_Type","Parking","Furnishing"]:
    if cf in ml_df.columns:
        ml_df[cf] = ml_df[cf].fillna("Unknown").astype(str)
# target
if "Price" not in ml_df.columns or ml_df["Price"].isna().all():
    st.error("Price column missing — cannot train price model.")
else:
    # build X,y
    X = ml_df[[c for c in ml_features if c in ml_df.columns]].copy()
    y = pd.to_numeric(ml_df["Price"], errors="coerce")
    data_ml = pd.concat([X,y], axis=1).dropna()
    X = data_ml[X.columns]; y = data_ml[y.name]
    if len(X) < 40:
        st.info(f"Not enough rows ({len(X)}) to train ML model; need >=40.")
    else:
        cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        ohe = safe_ohe()
        preprocessor = ColumnTransformer(transformers=[
            ("ohe", ohe, cat_cols),
            ("scale", StandardScaler(), num_cols)
        ], remainder="drop")
        try:
            X_t = preprocessor.fit_transform(X)
            if hasattr(X_t, "toarray"):
                X_t = X_t.toarray()
        except Exception as e:
            st.error("Preprocessing failed: " + str(e))
            X_t = None

        if X_t is not None:
            X_train, X_test, y_train, y_test = train_test_split(X_t, y.values, test_size=0.2, random_state=42)
            rf = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
            with st.spinner("Training RandomForest model..."):
                rf.fit(X_train, y_train)
            preds = rf.predict(X_test)
            mse = mean_squared_error(y_test, preds); rmse = math.sqrt(mse); r2 = r2_score(y_test, preds)
            st.write(f"Model performance: RMSE = {format_inr(rmse)}   |   R² = {r2:.3f}")

            # Feature importance (get names)
            try:
                ohe_names = list(preprocessor.named_transformers_["ohe"].get_feature_names_out(cat_cols)) if len(cat_cols)>0 else []
            except Exception:
                try:
                    ohe_names = list(preprocessor.named_transformers_["ohe"].get_feature_names(cat_cols))
                except Exception:
                    ohe_names = []
            feat_names = ohe_names + num_cols
            fi = rf.feature_importances_
            fi_df = pd.DataFrame({"feature": feat_names, "importance": fi}).sort_values("importance", ascending=False).head(40)
            st.subheader("Top feature importances")
            st.dataframe(fi_df.reset_index(drop=True), use_container_width=True)

            st.markdown("### Predict price for a new listing")
            # build input form columns
            form_cols = st.columns(3)
            input_vals = {}
            for i, f in enumerate(X.columns):
                col = form_cols[i % 3]
                if pd.api.types.is_numeric_dtype(X[f]):
                    input_vals[f] = col.number_input(f, value=float(X[f].median() if X[f].median()==X[f].median() else 0))
                else:
                    opts = sorted(X[f].dropna().unique().tolist())
                    input_vals[f] = col.selectbox(f, options=opts, index=0)
            if st.button("Predict Price"):
                input_df = pd.DataFrame([input_vals])[X.columns]
                try:
                    Xin = preprocessor.transform(input_df)
                    if hasattr(Xin, "toarray"):
                        Xin = Xin.toarray()
                    pred_price = rf.predict(Xin)[0]
                    st.success(f"Predicted price: {format_inr(pred_price)}")
                    if "Square_Footage" in input_df.columns and float(input_df["Square_Footage"].iloc[0])>0:
                        pps_val = pred_price/float(input_df['Square_Footage'].iloc[0])
                        st.info(f"Predicted ₹/sqft: {format_inr(pps_val)}")
                except Exception as e:
                    st.error("Prediction failed: " + str(e))

# -------------------------
# Compare two cities
# -------------------------
st.markdown("---")
st.subheader("Compare Two Cities")
if "City" in dff.columns:
    city_list = sorted(dff["City"].dropna().unique().tolist())
    if len(city_list) >= 2:
        a, b = st.columns(2)
        with a:
            city_a = st.selectbox("City A", options=city_list, index=0)
        with b:
            city_b = st.selectbox("City B", options=city_list, index=min(1,len(city_list)-1))
        ca = dff[dff["City"]==city_a]
        cb = dff[dff["City"]==city_b]
        st.write(f"{city_a} — Avg price: {format_inr(ca['Price'].mean())} | Median ₹/sqft: {format_inr(ca['Price_per_SqFt'].median())}")
        st.write(f"{city_b} — Avg price: {format_inr(cb['Price'].mean())} | Median ₹/sqft: {format_inr(cb['Price_per_SqFt'].median())}")
        st.write("Top property types A:")
        st.write(ca["Property_Type"].value_counts().head())
        st.write("Top property types B:")
        st.write(cb["Property_Type"].value_counts().head())
    else:
        st.info("Not enough city diversity to compare.")
else:
    st.info("City column missing — compare mode requires City.")

# -------------------------
# Raw data preview & download
# -------------------------
st.markdown("---")
st.subheader("Filtered dataset preview & download")
st.dataframe(dff.head(200), use_container_width=True)
csv = dff.to_csv(index=False).encode('utf-8')
st.download_button("Download filtered CSV", data=csv, file_name="real_estate_filtered.csv", mime="text/csv")

st.markdown("---")
st.caption("Hybrid Real Estate Intelligence platform — Property + CRM. Currency shown in Indian style (Lakhs / Crores). If you want any module upgraded (Prophet forecasting, XGBoost lead scoring, mapbox maps), tell me which one and I will add it.")
