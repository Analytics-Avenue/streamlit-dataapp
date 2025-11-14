import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# -------------------------------------------------------------
# APP CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Market Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# PAGE NAVIGATION STATE
# -------------------------------------------------------------
if "page_index" not in st.session_state:
    st.session_state.page_index = 0

PAGES = [
    "Home",
    "Market Overview",
    "Price Trends",
    "Demand & Ratings",
    "Property Mix",
    "Location Insights"
]

PAGE_MAP = {
    "Overview": ["Home", "Market Overview"],
    "Market": ["Price Trends", "Demand & Ratings"],
    "Inventory": ["Property Mix"],
    "Location": ["Location Insights"]
}

# -------------------------------------------------------------
# NAV FUNCTIONS
# -------------------------------------------------------------
def go_prev():
    if st.session_state.page_index > 0:
        st.session_state.page_index -= 1

def go_next():
    if st.session_state.page_index < len(PAGES) - 1:
        st.session_state.page_index += 1

def go_home():
    st.session_state.page_index = 0

def go_to_page(page):
    st.session_state.page_index = PAGES.index(page)

# -------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------
st.sidebar.title("Navigation")

category = st.sidebar.selectbox("Section", list(PAGE_MAP.keys()))
page_choice = st.sidebar.selectbox("Page", PAGE_MAP[category])
go_to_page(page_choice)

colA, colB, colC = st.sidebar.columns(3)
colA.button("⟵ Prev", on_click=go_prev)
colB.button("🏠 Home", on_click=go_home)
colC.button("Next ⟶", on_click=go_next)

current_page = PAGES[st.session_state.page_index]

# -------------------------------------------------------------
# SIDEBAR DATA UPLOAD
# -------------------------------------------------------------
st.sidebar.markdown("### Upload Dataset")

use_sample = st.sidebar.radio("Choose Data Source", ["Use sample dataset", "Upload CSV"])

df = None

if use_sample == "Use sample dataset":
    sample_data = {
        "Date": ["2022-01-01","2022-01-02","2022-01-03","2022-01-04"],
        "Property_Type": ["Condo","Townhouse","Apartment","Condo"],
        "Location": ["Suburban","Suburban","Suburban","Rural"],
        "Price": [271305,276780,233677,357729],
        "Bedrooms": [5,5,3,5],
        "Bathrooms": [3,2,2,3],
        "Square_Footage": [823,1146,1565,1206],
        "Days_On_Market": [93,118,127,46],
        "Interest_Rate": [52,62,35,57],
        "Economic_Index": [82,58,56,57],
        "School_Rating": [8,8,1,2],
        "Demand_Score": [116,85,101,103],
        "Year": [2022,2022,2022,2022],
        "Month": [1,1,1,1],
        "Price_per_SqFt": [329,241,149,296]
    }
    df = pd.DataFrame(sample_data)
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        df.columns = df.columns.str.strip()
        st.sidebar.success("Uploaded successfully!")

# -------------------------------------------------------------
# BASIC CLEANING
# -------------------------------------------------------------
if df is not None:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# -------------------------------------------------------------
# PAGE CONTENT
# -------------------------------------------------------------
# --------------------------- HOME ---------------------------
if current_page == "Home":
    st.title("Real Estate Market Dashboard")
    st.markdown("""
    Explore key insights from real estate listings including:
    • Price trends  
    • Market demand  
    • Location-based analysis  
    • Property types  
    • School ratings impact  
    """)

# --------------------------- MARKET OVERVIEW ---------------------------
elif current_page == "Market Overview":
    st.title("Market Overview")

    if df is None:
        st.warning("Upload a dataset to continue.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Listings", len(df))
        c2.metric("Avg Price", f"{df['Price'].mean():,.0f}")
        c3.metric("Avg Days on Market", f"{df['Days_On_Market'].mean():.1f}")

        fig = px.box(df, y="Price", title="Price Distribution")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------- PRICE TRENDS ---------------------------
elif current_page == "Price Trends":
    st.title("Price Trends")

    if df is None:
        st.warning("Upload a dataset.")
    else:
        fig = px.line(df, x="Date", y="Price", color="Property_Type", title="Daily Price Trend")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(df, x="Square_Footage", y="Price", color="Location",
                          size="Bedrooms", title="Price vs SqFt")
        st.plotly_chart(fig2, use_container_width=True)

# --------------------------- DEMAND & RATINGS ---------------------------
elif current_page == "Demand & Ratings":
    st.title("Demand & Ratings")

    if df is None:
        st.warning("Upload a dataset.")
    else:
        fig = px.scatter(df, x="School_Rating", y="Demand_Score",
                         color="Location", title="School Rating vs Demand")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(df, x="Property_Type", y="Demand_Score", 
                      title="Demand by Property Type", barmode="group")
        st.plotly_chart(fig2, use_container_width=True)

# --------------------------- PROPERTY MIX ---------------------------
elif current_page == "Property Mix":
    st.title("Property Type Mix")

    if df is None:
        st.warning("Upload a dataset.")
    else:
        mix = df["Property_Type"].value_counts().reset_index()
        mix.columns = ["Type", "Count"]

        fig = px.pie(mix, names="Type", values="Count", title="Property Distribution")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------- LOCATION INSIGHTS ---------------------------
elif current_page == "Location Insights":
    st.title("Location Insights")

    if df is None:
        st.warning("Upload a dataset.")
    else:
        fig = px.bar(df.groupby("Location")["Price"].mean().reset_index(),
                     x="Location", y="Price", title="Avg Price by Location")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.box(df, x="Location", y="Days_On_Market",
                      title="Days on Market by Location")
        st.plotly_chart(fig2, use_container_width=True)
