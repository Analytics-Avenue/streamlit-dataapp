import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# -------------------------------------------------------------
# APP CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# PAGE STATE
# -------------------------------------------------------------
if "page_index" not in st.session_state:
    st.session_state.page_index = 0

PAGES = [
    "Home",
    "Project Overview",
    "Inventory Analysis",
    "Lead Funnel Analytics",
    "Sales Performance",
    "Market Intelligence"
]

PAGE_MAP = {
    "Overview": ["Home", "Project Overview"],
    "Inventory": ["Inventory Analysis"],
    "Sales": ["Lead Funnel Analytics", "Sales Performance"],
    "Market": ["Market Intelligence"]
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
# SIDEBAR DATA LOAD
# -------------------------------------------------------------
st.sidebar.markdown("### Upload Real Estate Dataset")

data_choice = st.sidebar.radio("Select Data Source:", ["Use Sample Industry Dataset", "Upload CSV"])

df_projects = df_inventory = df_leads = df_market = None

# ---------------- SAMPLE INDUSTRY DATASET ----------------
if data_choice == "Use Sample Industry Dataset":

    df_projects = pd.DataFrame({
        "Project_ID": [1, 2],
        "Project_Name": ["Skyline Heights", "Urban Crest"],
        "Builder_Name": ["ABC Developers", "Prime Estates"],
        "City": ["Chennai", "Bangalore"],
        "Location": ["Velachery", "Whitefield"],
        "Launch_Date": ["2021-06-01", "2020-09-15"],
        "RERA_Approval": ["Yes", "Yes"],
        "Land_Area_Acres": [4.5, 6.2],
        "Total_Units": [450, 620],
        "Construction_Status": ["Ongoing", "Completed"]
    })

    df_inventory = pd.DataFrame({
        "Unit_ID": [101,102,103,201,202],
        "Project_ID": [1,1,2,2,2],
        "Unit_Type": ["2BHK","3BHK","2BHK","3BHK","Villa"],
        "Carpet_Area": [950, 1250, 900, 1400, 2400],
        "Builtup_Area": [1200,1500,1100,1600,3000],
        "Price": [72,95,68,110,260],
        "Floor": [3,10,2,7,1],
        "Facing": ["East","North","West","East","South"],
        "Status": ["Available","Booked","Available","Hold","Sold"]
    })

    df_leads = pd.DataFrame({
        "Lead_ID": [1001,1002,1003,1004],
        "Project_ID": [1,1,2,2],
        "Channel": ["Website","Broker","Referral","Walk-in"],
        "Lead_Date": ["2023-01-01","2023-01-05","2023-02-01","2023-02-10"],
        "Qualified": ["Yes","No","Yes","Yes"],
        "Site_Visit_Date": ["2023-01-10",None,"2023-02-05","2023-02-15"],
        "Booking_Date": ["2023-01-20",None,"2023-02-20",None],
        "Booking_Amount": [200000,0,300000,0],
        "Final_Amount": [7200000,0,6800000,0],
        "Cancellation_Status": ["No","No","No","Yes"]
    })

    df_market = pd.DataFrame({
        "City": ["Chennai","Bangalore"],
        "Location": ["Velachery","Whitefield"],
        "Avg_Market_Price": [7800,9200],
        "Inventory_Overhang_Months": [8,12],
        "Demand_Index": [78,85],
        "Rental_Yield": [3.5,4.1],
        "School_Rating": [4,5],
        "Connectivity_Score": [8,9]
    })


# ------------- USER UPLOAD -----------------
else:
    st.sidebar.info("Upload all 4 datasets: Projects, Inventory, Leads, Market.")

    p = st.sidebar.file_uploader("Projects.csv", type=["csv"])
    i = st.sidebar.file_uploader("Inventory.csv", type=["csv"])
    l = st.sidebar.file_uploader("Leads.csv", type=["csv"])
    m = st.sidebar.file_uploader("Market.csv", type=["csv"])

    if p and i and l and m:
        df_projects = pd.read_csv(p)
        df_inventory = pd.read_csv(i)
        df_leads = pd.read_csv(l)
        df_market = pd.read_csv(m)
        st.sidebar.success("All datasets uploaded!")

# -------------------------------------------------------------
# PAGE CONTENT
# -------------------------------------------------------------
# --------------------------- HOME ---------------------------
if current_page == "Home":
    st.title("🏙️ Real Estate Analytics Suite")
    st.markdown("""
    A complete industry-grade analytics dashboard covering:
    • Project performance  
    • Inventory status  
    • Lead funnel & conversion  
    • Sales insights  
    • Market intelligence  
    """)

# ---------------- PROJECT OVERVIEW ----------------
elif current_page == "Project Overview":
    st.title("📌 Project Overview")

    if df_projects is not None:
        st.dataframe(df_projects)

        fig = px.bar(df_projects, x="Project_Name", y="Total_Units", title="Total Units per Project")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- INVENTORY ----------------
elif current_page == "Inventory Analysis":
    st.title("🏘️ Inventory Analysis")

    if df_inventory is not None:
        st.dataframe(df_inventory)

        fig = px.histogram(df_inventory, x="Price", color="Unit_Type", title="Price Distribution by Unit Type")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- LEAD FUNNEL ----------------
elif current_page == "Lead Funnel Analytics":
    st.title("🧭 Lead Funnel Analytics")

    if df_leads is not None:
        df_leads["Qualified_Flag"] = df_leads["Qualified"].apply(lambda x: 1 if x=="Yes" else 0)
        df_leads["Booked_Flag"] = df_leads["Booking_Date"].notnull().astype(int)

        funnel_data = {
            "Stage": ["Leads","Qualified Leads","Bookings"],
            "Count": [
                len(df_leads),
                df_leads["Qualified_Flag"].sum(),
                df_leads["Booked_Flag"].sum()
            ]
        }

        funnel_df = pd.DataFrame(funnel_data)
        fig = px.funnel(funnel_df, x="Count", y="Stage", title="Lead Funnel")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- SALES ----------------
elif current_page == "Sales Performance":
    st.title("💰 Sales Performance")

    if df_leads is not None:
        booked = df_leads[df_leads["Booking_Date"].notnull()]

        st.metric("Total Sales Value", f"{booked['Final_Amount'].sum():,.0f}")
        st.metric("Total Bookings", len(booked))

        fig = px.bar(booked, x="Project_ID", y="Final_Amount",
                     title="Sales Value by Project", text="Final_Amount")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- MARKET ----------------
elif current_page == "Market Intelligence":
    st.title("🌍 Market & Competition Analysis")

    if df_market is not None:
        fig = px.bar(df_market, x="Location", y="Avg_Market_Price",
                     title="Average Market Price by Location")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(df_market, x="Demand_Index", y="Rental_Yield",
                          color="City", size="Connectivity_Score",
                          title="Demand vs Rental Yield")
        st.plotly_chart(fig2, use_container_width=True)
