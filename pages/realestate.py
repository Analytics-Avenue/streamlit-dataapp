import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from PIL import Image
import os
import io

# -------------------------------------------------------------
# APP CONFIG
# -------------------------------------------------------------
st.set_page_config(page_title="Real Estate Analytics Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")

# -------------------------------------------------------------
# PAGE NAVIGATION STATE
# -------------------------------------------------------------
if "page_index" not in st.session_state:
    st.session_state.page_index = 0

PAGES = [
    "Home",
    "About Real Estate Analytics",
    "Inventory Overview",
    "Lead Funnel Analysis",
    "Sales Performance",
    "Project-Level Metrics",
    "Market Insights"
]

PAGE_MAP = {
    "Overview": ["Home", "About Real Estate Analytics"],
    "Operations": ["Inventory Overview", "Lead Funnel Analysis"],
    "Sales": ["Sales Performance"],
    "Projects": ["Project-Level Metrics"],
    "Market": ["Market Insights"]
}

# -------------------------------------------------------------
# PAGE SWITCH HANDLERS
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
# NAVIGATION UI
# -------------------------------------------------------------
st.sidebar.title("Navigation")

# Dropdown (hierarchical)
category = st.sidebar.selectbox("Select Section", list(PAGE_MAP.keys()))
page_choice = st.sidebar.selectbox("Select Page", PAGE_MAP[category])
go_to_page(page_choice)

# Buttons
colA, colB, colC = st.sidebar.columns(3)
colA.button("⟵ Prev", on_click=go_prev)
colB.button("🏠 Home", on_click=go_home)
colC.button("Next ⟶", on_click=go_next)

current_page = PAGES[st.session_state.page_index]

# -------------------------------------------------------------
# SAMPLE CSV TEMPLATE
# -------------------------------------------------------------
DEFAULT_COLUMNS = {
    "project": "Project Name",
    "city": "City",
    "unit_type": "Unit Type",
    "inventory": "Available Inventory",
    "leads": "Total Leads",
    "qualified": "Qualified Leads",
    "site_visits": "Site Visits",
    "bookings": "Bookings",
    "revenue": "Revenue (INR)",
    "launch_date": "Launch Date",
}

def make_sample_csv():
    df = pd.DataFrame(columns=list(DEFAULT_COLUMNS.values()))
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

# -------------------------------------------------------------
# SIDEBAR DATA INPUT
# -------------------------------------------------------------
st.sidebar.markdown("### Upload Real Estate Dataset")
data_source_selector = st.sidebar.radio(
    "Choose Data Source",
    ["Use sample dataset", "Upload CSV"]
)

df_raw = None

if data_source_selector == "Use sample dataset":
    # Empty dataset template
    st.sidebar.download_button(
        "Download sample CSV (headers only)",
        data=make_sample_csv(),
        file_name="real_estate_sample_template.csv",
        mime="text/csv"
    )
    st.sidebar.info("Upload your dataset to see realistic dashboards.")

else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        df_raw.columns = df_raw.columns.str.strip()
        st.sidebar.success("Uploaded successfully!")

# -------------------------------------------------------------
# CLEANING + BASIC MAPPING
# -------------------------------------------------------------
mapped_df = None
if df_raw is not None:
    mapped_df = pd.DataFrame()
    for key, friendly in DEFAULT_COLUMNS.items():
        mapped_df[key] = df_raw[friendly] if friendly in df_raw.columns else None

    # Convert dates
    if "launch_date" in mapped_df:
        mapped_df["launch_date"] = pd.to_datetime(mapped_df["launch_date"], errors="coerce")

    # Numeric fields
    numeric_cols = ["inventory", "leads", "qualified", "site_visits", "bookings", "revenue"]
    for col in numeric_cols:
        if col in mapped_df.columns:
            mapped_df[col] = pd.to_numeric(mapped_df[col], errors="coerce")

# -------------------------------------------------------------
# PAGE CONTENT
# -------------------------------------------------------------

# --------------------------- HOME --------------------------------
if current_page == "Home":
    st.title("Real Estate Analytics Dashboard")
    st.markdown("""
    Welcome to **Real Estate Analytics**,  
    a complete analytical suite for evaluating:
    - Project performance  
    - Lead funnel efficiency  
    - Sales velocity  
    - Revenue tracking  
    - City-wise market trends  
    """)
    st.image("https://i.imgur.com/0ZfPPrf.jpeg", caption="Real Estate Market Intelligence", use_column_width=True)

# --------------------------- ABOUT --------------------------------
elif current_page == "About Real Estate Analytics":
    st.title("About Real Estate Analytics")
    st.markdown("""
    Real Estate Analytics tracks **end-to-end real estate performance** including:
    - Project demand  
    - Lead quality  
    - Sales pipeline  
    - Inventory aging  
    - Market shifts  
    """)

# --------------------------- INVENTORY -----------------------------
elif current_page == "Inventory Overview":
    st.title("Inventory Overview")
    if mapped_df is None:
        st.warning("Upload a dataset to view inventory metrics.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Inventory", f"{mapped_df['inventory'].sum():,}")
        col2.metric("Avg Unit Launch Date", mapped_df['launch_date'].min().strftime("%Y-%m-%d"))
        col3.metric("Total Projects", mapped_df['project'].nunique())

        # city-wise inventory
        inv = mapped_df.groupby("city")["inventory"].sum().reset_index()
        fig = px.bar(inv, x="city", y="inventory", title="City-wise Inventory")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------- LEADS FUNNEL --------------------------
elif current_page == "Lead Funnel Analysis":
    st.title("Lead Funnel Analysis")
    if mapped_df is None:
        st.warning("Upload a dataset first.")
    else:
        funnel = [
            mapped_df["leads"].sum(),
            mapped_df["qualified"].sum(),
            mapped_df["site_visits"].sum(),
            mapped_df["bookings"].sum()
        ]

        funnel_df = pd.DataFrame({
            "Stage": ["Leads", "Qualified", "Site Visits", "Bookings"],
            "Count": funnel
        })

        fig = px.funnel(funnel_df, x="Count", y="Stage", title="Lead Funnel Overview")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------- SALES --------------------------
elif current_page == "Sales Performance":
    st.title("Sales Performance")
    if mapped_df is None:
        st.warning("Upload a dataset.")
    else:
        city_sales = mapped_df.groupby("city")["revenue"].sum().reset_index()
        fig = px.bar(city_sales, x="city", y="revenue", text="revenue",
                     title="Revenue by City")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------- PROJECT METRICS -----------------------
elif current_page == "Project-Level Metrics":
    st.title("Project-Level Metrics")
    if mapped_df is None:
        st.warning("Upload a dataset.")
    else:
        fig = px.scatter(mapped_df, x="inventory", y="bookings", color="city",
                         size="revenue", hover_name="project",
                         title="Project Performance Bubble Chart")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------- MARKET INSIGHTS -----------------------
elif current_page == "Market Insights":
    st.title("Market Insights")
    st.markdown("Additional market intelligence visuals can go here.")


