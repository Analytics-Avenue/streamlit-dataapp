import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# ----------------------------------------
# App Configuration
# ----------------------------------------
st.set_page_config(
    page_title="Marketing Campaign Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------
# Default Column Names
# ----------------------------------------
DEFAULT_COLUMNS = {
    'campaign': 'Campaign name',
    'city': 'City',
    'age': 'Age',
    'gender': 'Gender',
    'start': 'Reporting starts',
    'end': 'Reporting ends',
    'spent': 'Amount spent (INR)',
    'impressions': 'Impressions',
    'reach': 'Reach',
    'clicks': 'Link clicks',
    'ctr': 'CTR (all)',
    'cpm': 'CPM (cost per 1,000 impressions)',
    'ad_name': 'Ad name',
    'thruplays': 'ThruPlays',
    'cost_per_thruplay': 'Cost per ThruPlay',
    'video_25': 'Video plays at 25%',
    'video_50': 'Video plays at 50%',
    'video_75': 'Video plays at 75%',
    'video_95': 'Video plays at 95%',
    'video_100': 'Video plays at 100%',
    'cpc': 'CPC (cost per link click)'
}

# ----------------------------------------
# Sidebar: Upload or Map
# ----------------------------------------
st.sidebar.title("Upload or Map Your Data")
upload_option = st.sidebar.radio(
    "Choose how to provide data:",
    ["Direct Upload (default columns)", "Upload and Map Columns"]
)

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded!")

    # Strip column whitespaces
    df.columns = df.columns.str.strip()

    # Initialize mapping
    user_columns = {}
    if upload_option == "Upload and Map Columns":
        st.sidebar.subheader("Map your columns")
        for key, default_name in DEFAULT_COLUMNS.items():
            if default_name not in df.columns:
                user_columns[key] = st.sidebar.selectbox(
                    f"Select column for '{default_name}' (or leave blank)",
                    options=[None] + list(df.columns),
                    index=0
                )
            else:
                user_columns[key] = default_name
    else:
        # Direct upload, assume default columns
        for key, default_name in DEFAULT_COLUMNS.items():
            user_columns[key] = default_name if default_name in df.columns else None

    # Create a standard dataframe with mapped columns
    mapped_df = pd.DataFrame()
    for key, col in user_columns.items():
        if col in df.columns:
            mapped_df[key] = df[col]
        else:
            mapped_df[key] = None  # leave blank if column missing

    # Convert date columns to datetime
    for date_col in ['start', 'end']:
        if mapped_df[date_col] is not None:
            mapped_df[date_col] = pd.to_datetime(mapped_df[date_col], errors='coerce')

else:
    df = None
    mapped_df = None
    st.warning("Please upload your CSV to begin analysis.")

# ----------------------------------------
# Example of how to handle missing columns in a chart
# ----------------------------------------
if mapped_df is not None:
    st.title("Campaign Overview Example")
    
    if mapped_df['spent'] is not None:
        total_spent = mapped_df['spent'].sum()
        st.metric("Total Spent (INR)", f"₹{total_spent:,.0f}")
    else:
        st.info("Column 'Amount spent (INR)' not found. Chart cannot be displayed.")

    if mapped_df['impressions'] is not None:
        total_impressions = mapped_df['impressions'].sum()
        st.metric("Total Impressions", f"{total_impressions:,.0f}")
    else:
        st.info("Column 'Impressions' not found. Chart cannot be displayed.")
