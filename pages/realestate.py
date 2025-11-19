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
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
import re
import inspect

st.set_page_config(layout="wide", page_title="Real Estate Intelligence")

# -------------------------
# Hide sidebar
# -------------------------
hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

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
# Utility functions
# -------------------------
def format_inr(value):
    """Format number to Indian style ₹ with Lakh/Crore shorthand"""
    try:
        v = float(value)
    except:
        return value
    if math.isnan(v):
        return "N/A"
    abs_v = abs(v)
    if abs_v >= 1e7:
        return f"₹{v/1e7:.2f} Cr"
    elif abs_v >= 1e5:
        return f"₹{v/1e5:.2f} L"
    else:
        return f"₹{int(round(v)):,}"

def safe_ohe():
    try:
        sig = inspect.signature(OneHotEncoder)
        if 'sparse_output' in sig.parameters:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)
    except Exception:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def try_read_csv_url(url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))
    except:
        return None

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
            "Lead_ID": lead_id,
            "Lead_Source": lead_source,
            "Lead_Status": lead_status,
            "Lead_Score": lead_score,
            "Time_to_Sell": time_to_sell,
            "Sales_Agent": sales_agent,
            "Followup_Count": followup_count,
            "Latitude": lat,
            "Longitude": lon
        })
    df = pd.DataFrame(rows)
    return df

# -------------------------
# Load dataset
# -------------------------
sample_df = make_hybrid_sample(5000)
df = sample_df.copy()

# -------------------------
# Tabs layout
# -------------------------
tabs = st.tabs(["Overview", "Application"])
tab_overview = tabs[0]
tab_application = tabs[1]

# -------------------------
# Overview Tab (generic KPIs & charts)
# -------------------------
with tab_overview:
    st.header("🏡 Real Estate Overview")
    total_listings = len(df)
    avg_price = df["Price"].mean()
    median_pps = df["Price_per_SqFt"].median()
    avg_dom = df["Days_On_Market"].mean()
    avg_demand = df["Demand_Score"].mean()

    col1,col2,col3,col4,col5 = st.columns(5)
    col1.metric("Total Listings", f"{total_listings:,}")
    col2.metric("Avg Price", format_inr(avg_price))
    col3.metric("Median ₹/sqft", format_inr(median_pps))
    col4.metric("Avg Days on Market", f"{avg_dom:.1f}")
    col5.metric("Avg Demand Score", f"{avg_demand:.1f}")

    st.markdown("---")
    st.subheader("Price Distribution")
    fig = px.histogram(df, x="Price", nbins=50, title="Price distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price by City")
    fig = px.box(df, x="City", y="Price", title="Price by City")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Property Type Mix")
    mix = df["Property_Type"].value_counts().reset_index()
    mix.columns = ["Property_Type","Count"]
    fig = px.pie(mix, names="Property_Type", values="Count", title="Property Type Mix")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Application Tab (all detailed analysis)
# -------------------------
with tab_application:
    st.header("🏢 Real Estate Application Analysis")
    
    # Filter UI
    cities = sorted(df["City"].dropna().unique())
    city_sel = st.multiselect("City", options=cities, default=cities[:6])
    dff = df[df["City"].isin(city_sel)] if city_sel else df.copy()
    
    st.subheader("Agent Leaderboard")
    agent_df = dff.groupby("Listing_Agent").agg(
        Listings=("Listing_ID","count"),
        Total_Value=("Price","sum")
    ).reset_index().sort_values("Total_Value", ascending=False)
    agent_df["Total_Value_str"] = agent_df["Total_Value"].apply(format_inr)
    st.dataframe(agent_df.head(20), use_container_width=True)

    st.subheader("Price Distribution by Property Type")
    fig = px.box(dff, x="Property_Type", y="Price", points="all", title="Price Distribution by Property Type")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Geo Map of Listings")
    if dff["Latitude"].notna().any() and dff["Longitude"].notna().any():
        geo_sample = dff.sample(min(1500,len(dff)), random_state=42)
        fig = px.scatter_geo(geo_sample, lat="Latitude", lon="Longitude", color="Price", hover_name="City", size="Price")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Clustering (KMeans)")
    if st.button("Run KMeans Clustering"):
        clu_df = dff[["Price","Square_Footage"]].dropna()
        clu_df["Price_log"] = np.log1p(clu_df["Price"])
        scaler = StandardScaler(); Xs = scaler.fit_transform(clu_df[["Price_log","Square_Footage"]])
        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        clu_df["cluster"] = labels
        fig = px.scatter(clu_df, x="Square_Footage", y="Price", color="cluster", title="KMeans clusters")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecasting (monthly)")
    ts = dff.copy()
    ts["YearMonth"] = pd.to_datetime(ts["Listing_Date"]).dt.to_period("M").dt.to_timestamp()
    monthly = ts.groupby("YearMonth")["Price"].mean().reset_index().sort_values("YearMonth")
    if len(monthly) >= 12:
        monthly["lag1"] = monthly["Price"].shift(1).fillna(method="bfill")
        X = monthly[["lag1","YearMonth"]].copy()
        X["month_num"] = X["YearMonth"].dt.month
        y = monthly["Price"]
        X_train, X_test, y_train, y_test = train_test_split(X[["lag1","month_num"]], y, test_size=0.2, shuffle=False)
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        monthly["Pred"] = rf.predict(X[["lag1","month_num"]])
        fig = px.line(monthly, x="YearMonth", y=["Price","Pred"], title="Actual vs Predicted Prices")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Keyword Search (NLP-lite)")
    query = st.text_input("Search properties by keywords (e.g., 3BHK Bangalore under 1.5cr)")
    if query:
        res = dff[dff["City"].str.contains(query.split()[1], case=False)] if len(query.split())>1 else dff
        st.write(f"Found {len(res)} matches")
        st.dataframe(res.head(50))

    st.subheader("CRM Funnel & Lead Scoring")
    funnel = dff["Lead_Status"].value_counts().reset_index()
    funnel.columns = ["Stage","Count"]
    fig = px.bar(funnel, x="Stage", y="Count", title="Lead Funnel")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Time-to-Sell Prediction")
    if len(dff) > 50:
        tdf = dff[["Time_to_Sell","Price","Square_Footage","Bedrooms","Bathrooms","Demand_Score"]].dropna()
        features = ["Price","Square_Footage","Bedrooms","Bathrooms","Demand_Score"]
        X = tdf[features].fillna(0)
        y = tdf["Time_to_Sell"]
        rf = RandomForestRegressor(n_estimators=150, random_state=42)
        rf.fit(X, y)
        tdf["Predicted_Time"] = rf.predict(X)
        st.dataframe(tdf.head(50))
