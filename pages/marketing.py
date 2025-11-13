import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# ------------------------
# App Configuration
# ------------------------
st.set_page_config(
    page_title="Marketing Campaign Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------
# Default Columns
# ------------------------
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

# ------------------------
# Sidebar Upload / Mapping
# ------------------------
st.sidebar.title("Upload or Map Your Data")
upload_option = st.sidebar.radio("Choose how to provide data:", ["Direct Upload", "Upload & Map Columns"])
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

mapped_df = None
filtered_df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    st.sidebar.success("File uploaded!")

    if upload_option == "Upload & Map Columns":
        st.sidebar.subheader("Map your columns")
        user_columns = {}
        for key, default_name in DEFAULT_COLUMNS.items():
            col = st.sidebar.selectbox(
                f"Map '{default_name}'",
                options=[None] + list(df.columns),
                index=0
            )
            user_columns[key] = col
    else:
        user_columns = {}
        for key, default_name in DEFAULT_COLUMNS.items():
            user_columns[key] = default_name if default_name in df.columns else None

    # Create mapped dataframe
    mapped_df = pd.DataFrame()
    for key, col in user_columns.items():
        if col in df.columns:
            mapped_df[key] = df[col]
        else:
            mapped_df[key] = None

    # Convert dates
    for date_col in ['start', 'end']:
        if mapped_df[date_col] is not None:
            mapped_df[date_col] = pd.to_datetime(mapped_df[date_col], errors='coerce')

    # ------------------------
    # Filters / Slicers
    # ------------------------
    filtered_df = mapped_df.copy()

    st.sidebar.subheader("Filters")
    # Campaign filter
    if mapped_df['campaign'] is not None:
        campaign_filter = st.sidebar.multiselect("Campaign Name", options=mapped_df['campaign'].dropna().unique())
        if campaign_filter:
            filtered_df = filtered_df[filtered_df['campaign'].isin(campaign_filter)]
    # City filter
    if mapped_df['city'] is not None:
        city_filter = st.sidebar.multiselect("City", options=mapped_df['city'].dropna().unique())
        if city_filter:
            filtered_df = filtered_df[filtered_df['city'].isin(city_filter)]
    # Age filter
    if mapped_df['age'] is not None:
        age_filter = st.sidebar.multiselect("Age Group", options=mapped_df['age'].dropna().unique())
        if age_filter:
            filtered_df = filtered_df[filtered_df['age'].isin(age_filter)]
    # Gender filter
    if mapped_df['gender'] is not None:
        gender_filter = st.sidebar.multiselect("Gender", options=mapped_df['gender'].dropna().unique())
        if gender_filter:
            filtered_df = filtered_df[filtered_df['gender'].isin(gender_filter)]
    # Date filter
    if mapped_df['start'] is not None and mapped_df['end'] is not None:
        start_date = st.sidebar.date_input("Start Date", mapped_df['start'].min().date())
        end_date = st.sidebar.date_input("End Date", mapped_df['end'].max().date())
        filtered_df = filtered_df[
            (filtered_df['start'] >= pd.to_datetime(start_date)) &
            (filtered_df['end'] <= pd.to_datetime(end_date))
        ]

else:
    st.warning("Upload your CSV to begin.")

# ------------------------
# Helper: Style axes
# ------------------------
def style_axes(fig):
    fig.update_layout(
        xaxis=dict(title_font=dict(size=14, color='black', family='Arial Black')),
        yaxis=dict(title_font=dict(size=14, color='black', family='Arial Black')),
        title_font=dict(size=16, color='black', family='Arial Black')
    )
    return fig

# ------------------------
# Sidebar Navigation
# ------------------------
st.sidebar.title("Dashboard Pages")
page = st.sidebar.radio("Navigate to", ["Case Study Overview", "Campaign Overview", "Audience Insights", "Ad Performance", "Video Metrics"])

# ------------------------
# CASE STUDY OVERVIEW
# ------------------------
if page == "Case Study Overview":
    st.title("Marketing Campaign Case Study")
    st.markdown("""
    Explore campaign performance interactively.

    **Use Case Highlights:**
    - Evaluate budget effectiveness and ROI trends
    - Analyze audience segments driving conversions
    - Compare creatives for engagement and reach
    - Identify performance gaps to improve delivery
    """)
    with st.expander("Watch Case Study Preview"):
        st.video("https://www.youtube.com/watch?v=0d6oY8G5e5c")
    st.markdown("---")
    st.caption("Created by Vibin | " + datetime.now().strftime("%d %B %Y"))

# ------------------------
# CAMPAIGN OVERVIEW
# ------------------------
elif page == "Campaign Overview" and filtered_df is not None:
    st.title("Campaign Overview")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    # Metrics
    metrics_map = {
        'Total Spent (INR)': 'spent',
        'Total Impressions': 'impressions',
        'Total Reach': 'reach',
        'Total Clicks': 'clicks',
        'Avg CTR': 'ctr',
        'Avg CPM (₹)': 'cpm'
    }

    for idx, (label, col) in enumerate(metrics_map.items()):
        if filtered_df[col] is not None:
            value = filtered_df[col].sum() if 'Total' in label else filtered_df[col].mean()
            if 'INR' in label:
                value = f"₹{value:,.0f}" if 'Total' in label else f"₹{value:,.2f}"
            else:
                value = f"{value:,.0f}" if 'Total' in label else f"{value:.2f}%"
            [col1, col2, col3, col4, col5, col6][idx].metric(label, value)
        else:
            [col1, col2, col3, col4, col5, col6][idx].info("Missing column")

    st.markdown("---")
    # Spend Trend
    if filtered_df['start'] is not None and filtered_df['spent'] is not None:
        spend_over_time = filtered_df.groupby('start', as_index=False)['spent'].sum()
        fig = px.line(spend_over_time, x='start', y='spent', markers=True,
                      text=spend_over_time['spent'].apply(lambda x: f"₹{x:,.0f}"),
                      title="Campaign Spend Over Time", color_discrete_sequence=['#2E86C1'])
        fig.update_traces(textposition='top right')
        fig = style_axes(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Purpose:**")
        st.markdown("""
        - Track daily/weekly campaign spend patterns
        - Identify overspending or underspending periods
        - Pinpoint trends to optimize future budget allocation
        - Quickly spot anomalies or spikes in ad spend
        """)
        with st.expander("Quick Tips"):
            st.markdown("""
            - Hover on points to see exact spend
            - Use filters to focus on specific campaigns or dates
            - Compare trends over multiple campaigns
            """)

    # Top Campaigns
    if filtered_df['campaign'] is not None and filtered_df['spent'] is not None:
        top_campaigns = filtered_df.groupby('campaign', as_index=False)['spent'].sum().nlargest(10,'spent')
        fig = px.bar(top_campaigns, x='spent', y='campaign', orientation='h',
                     text=top_campaigns['spent'].apply(lambda x: f"₹{x:,.0f}"), color='spent', color_continuous_scale="Blues",
                     title="Top Campaigns by Spend")
        fig.update_traces(textposition='outside')
        fig = style_axes(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Purpose:**")
        st.markdown("""
        - See which campaigns consume the most budget
        - Prioritize analysis on high-investment campaigns
        - Compare spend vs ROI for top campaigns
        - Identify underperforming campaigns despite high spend
        """)
        with st.expander("Quick Tips"):
            st.markdown("""
            - Focus analysis on top-spend campaigns first
            - Adjust budgets based on ROI, not just spend
            - Check for seasonal spikes in spending
            """)
