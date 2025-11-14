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

st.title("🏠 Real Estate Analytics Dashboard")

# -------------------------------------------------
# Synthetic Dataset Generator
# -------------------------------------------------
def generate_data():
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
        price = np.random.randint(20, 250) * 1e5
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
            np.random.randint(50, 1000),
            np.random.randint(10, 300),
            np.random.randint(5, 50),
        ])

    df = pd.DataFrame(rows, columns=[
        "Listing_ID", "City", "Property_Type", "Furnishing",
        "Area_sqft", "Price", "Bedrooms", "Bathrooms",
        "Status", "Agent", "Listing_Date", "Amenity",
        "Leads", "Site_Visits", "Offers"
    ])

    df["Price_per_sqft"] = df["Price"] / df["Area_sqft"]

    return df

# -------------------------------------------------
# File Upload
# -------------------------------------------------
st.sidebar.header("Upload Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload Real Estate Dataset (CSV or Excel)", 
    type=["csv", "xlsx"]
)

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
else:
    df = generate_data()

# -------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "KPIs", "Dashboard Charts", "Raw Data"]
)

# -------------------------------------------------
# Sidebar Filters
# -------------------------------------------------
st.sidebar.title("Filters")

city_filter = st.sidebar.multiselect(
    "City", df["City"].unique(), default=df["City"].unique()
)
type_filter = st.sidebar.multiselect(
    "Property Type", df["Property_Type"].unique(), default=df["Property_Type"].unique()
)
status_filter = st.sidebar.multiselect(
    "Status", df["Status"].unique(), default=df["Status"].unique()
)
agent_filter = st.sidebar.multiselect(
    "Agent", df["Agent"].unique(), default=df["Agent"].unique()
)

df_f = df[
    (df["City"].isin(city_filter)) &
    (df["Property_Type"].isin(type_filter)) &
    (df["Status"].isin(status_filter)) &
    (df["Agent"].isin(agent_filter))
]

# -------------------------------------------------
# PAGES
# -------------------------------------------------

# HOME
if page == "Home":
    st.subheader("Overview")
    st.write("""
    This dashboard provides end-to-end analytics for the real estate industry, including:
    - Property pricing insights  
    - City-wise supply trends  
    - Agent performance  
    - Lead funnel performance  
    - Full data upload support  
    """)

# KPIs
elif page == "KPIs":
    st.subheader("📊 Key Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Listings", len(df_f))
    col2.metric("Avg Price", f"₹{df_f['Price'].mean():,.0f}")
    col3.metric("Median Price/Sqft", f"₹{df_f['Price_per_sqft'].median():,.0f}")

    col4, col5, col6 = st.columns(3)
    sold_count = len(df_f[df_f["Status"] == "Sold"])
    conv_rate = (sold_count / len(df_f) * 100) if len(df_f) > 0 else 0

    col4.metric("Sold Properties", sold_count)
    col5.metric("Avg Leads", round(df_f["Leads"].mean(), 2))
    col6.metric("Conversion Rate", f"{round(conv_rate, 2)}%")

# CHARTS
elif page == "Dashboard Charts":
    st.subheader("📈 Visual Dashboard")

    st.write("### Price Distribution")
    fig1 = px.box(df_f, x="City", y="Price", color="Property_Type")
    st.plotly_chart(fig1, use_container_width=True)

    st.write("### Avg Price per Sqft by City")
    fig2 = px.bar(
        df_f.groupby("City")["Price_per_sqft"].mean().reset_index(),
        x="City", y="Price_per_sqft"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.write("### Property Type Distribution")
    fig3 = px.pie(df_f, names="Property_Type")
    st.plotly_chart(fig3, use_container_width=True)

    st.write("### Agent Performance")
    fig4 = px.bar(
        df_f.groupby("Agent")["Offers"].sum().reset_index(),
        x="Agent", y="Offers"
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.write("### Lead-to-Offer Funnel")
    fig5 = px.scatter(
        df_f,
        x="Leads", y="Offers", size="Site_Visits", color="City"
    )
    st.plotly_chart(fig5, use_container_width=True)

# RAW DATA
elif page == "Raw Data":
    st.subheader("📄 Raw Dataset")
    st.dataframe(df_f, use_container_width=True)
