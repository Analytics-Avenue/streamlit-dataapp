# real_estate_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import random

# -------------------------------------------------
# Streamlit Config
# -------------------------------------------------
st.set_page_config(
    page_title="Real Estate Analytics Dashboard",
    layout="wide"
)

# -------------------------------------------------
# Synthetic Real Estate Dataset Generator
# -------------------------------------------------
def generate_real_estate_data():
    np.random.seed(42)

    cities = ["Chennai", "Bangalore", "Hyderabad", "Mumbai", "Pune", "Delhi"]
    property_types = ["Apartment", "Villa", "Plot", "Commercial"]
    furnishing = ["Fully Furnished", "Semi-Furnished", "Unfurnished"]
    status = ["Available", "Sold", "Under Offer"]
    agents = ["Aditi", "Manoj", "Riya", "Kiran", "Sanjay"]
    amenities = ["Gym", "Pool", "Play Area", "Parking", "Security"]

    rows = []
    today = datetime.today()

    for i in range(500):
        city = random.choice(cities)
        prop_type = random.choice(property_types)
        area = np.random.randint(600, 4000)
        price = np.random.randint(20, 250) * 1e5  # 20L to 2.5Cr
        bedrooms = np.random.randint(1, 5)
        bath = np.random.randint(1, 4)
        status_val = random.choice(status)
        agent = random.choice(agents)
        days_ago = np.random.randint(1, 365)
        listing_date = today - timedelta(days=days_ago)

        rows.append([
            i + 1,
            city,
            prop_type,
            furnishing[random.randint(0,2)],
            area,
            price,
            bedrooms,
            bath,
            status_val,
            agent,
            listing_date,
            np.random.choice(amenities),
            np.random.randint(50, 1000),  # Lead count
            np.random.randint(10, 300),   # Site visits
            np.random.randint(5, 50),     # Offers
        ])

    df = pd.DataFrame(rows, columns=[
        "Listing_ID", "City", "Property_Type", "Furnishing",
        "Area_sqft", "Price", "Bedrooms", "Bathrooms",
        "Status", "Agent", "Listing_Date", "Amenity",
        "Leads", "Site_Visits", "Offers"
    ])

    df["Price_per_sqft"] = df["Price"] / df["Area_sqft"]

    return df


df = generate_real_estate_data()

# -------------------------------------------------
# Sidebar Filters
# -------------------------------------------------
st.sidebar.title("Filters")

city_filter = st.sidebar.multiselect("Select City", df["City"].unique(), default=df["City"].unique())
type_filter = st.sidebar.multiselect("Property Type", df["Property_Type"].unique(), default=df["Property_Type"].unique())
status_filter = st.sidebar.multiselect("Status", df["Status"].unique(), default=df["Status"].unique())
agent_filter = st.sidebar.multiselect("Agent", df["Agent"].unique(), default=df["Agent"].unique())

df_filtered = df[
    (df["City"].isin(city_filter)) &
    (df["Property_Type"].isin(type_filter)) &
    (df["Status"].isin(status_filter)) &
    (df["Agent"].isin(agent_filter))
]

# -------------------------------------------------
# Navigation Buttons
# -------------------------------------------------
col1, col2, col3 = st.columns([1,1,1])

if col1.button("Home"):
    st.session_state["page"] = "Home"

if col2.button("Previous"):
    st.session_state["page"] = "KPIs"

if col3.button("Next"):
    st.session_state["page"] = "Charts"

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------
if st.session_state["page"] == "Home":
    st.title("🏠 Real Estate Analytics Dashboard")
    st.markdown("Comprehensive insights across pricing, supply, demand, agents, and conversions.")


# -------------------------------------------------
# KPI PAGE
# -------------------------------------------------
if st.session_state["page"] == "KPIs":
    st.title("📊 Key Performance Metrics")

    total_listings = len(df_filtered)
    avg_price = round(df_filtered["Price"].mean(), 2)
    median_ppsqft = round(df_filtered["Price_per_sqft"].median(), 2)
    sold_count = len(df_filtered[df_filtered["Status"] == "Sold"])
    avg_leads = round(df_filtered["Leads"].mean(), 2)
    conv_rate = round((sold_count / total_listings) * 100, 2) if total_listings > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Listings", total_listings)
    c2.metric("Avg Property Price", f"₹{avg_price:,.0f}")
    c3.metric("Median Price/Sqft", f"₹{median_ppsqft:,.0f}")

    d1, d2, d3 = st.columns(3)
    d1.metric("Total Sold", sold_count)
    d2.metric("Avg Leads per Listing", avg_leads)
    d3.metric("Sale Conversion Rate", f"{conv_rate}%")

# -------------------------------------------------
# CHARTS PAGE
# -------------------------------------------------
if st.session_state["page"] == "Charts":
    st.title("📈 Dashboard Visualizations")

    # Price Distribution
    fig1 = px.box(df_filtered, x="City", y="Price", color="Property_Type")
    st.subheader("Price Distribution by City")
    st.plotly_chart(fig1, use_container_width=True)

    # Avg Price per Sqft
    fig2 = px.bar(
        df_filtered.groupby("City")["Price_per_sqft"].mean().reset_index(),
        x="City", y="Price_per_sqft",
        title="Average Price per Sqft by City"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Property Supply
    fig3 = px.pie(
        df_filtered, names="Property_Type",
        title="Property Type Distribution"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Agent Performance
    fig4 = px.bar(
        df_filtered.groupby("Agent")["Offers"].sum().reset_index(),
        x="Agent", y="Offers",
        title="Agent Performance (Total Offers)"
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Leads vs Site Visits vs Offers
    fig5 = px.scatter(
        df_filtered,
        x="Leads", y="Offers", size="Site_Visits",
        color="City",
        title="Lead Funnel Performance"
    )
    st.plotly_chart(fig5, use_container_width=True)

# END
