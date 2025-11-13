import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from PIL import Image
import os

# ------------------------
# App Config
# ------------------------
st.set_page_config(page_title="Marketing Campaign Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------
# Default Columns
# ------------------------
DEFAULT_COLUMNS = {
    'campaign': 'Campaign name',
    'city': 'City',
    'age': 'Age',
    'gender': 'Gender',
    'adset_name': 'Ad set name',
    'ad_name': 'Ad name',
    'start': 'Reporting starts',
    'end': 'Reporting ends',
    'spent': 'Amount spent (INR)',
    'impressions': 'Impressions',
    'reach': 'Reach',
    'clicks': 'Link clicks',
    'ctr': 'CTR (all)',
    'cpm': 'CPM (cost per 1,000 impressions)',
    'cpc': 'CPC (cost per link click)',
    'thruplays': 'ThruPlays',
    'cost_per_thruplay': 'Cost per ThruPlay',
    'video_25': 'Video plays at 25%',
    'video_50': 'Video plays at 50%',
    'video_75': 'Video plays at 75%',
    'video_95': 'Video plays at 95%',
    'video_100': 'Video plays at 100%',
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

    # Column Mapping
    user_columns = {}
    if upload_option == "Upload & Map Columns":
        st.sidebar.subheader("Map your columns")
        for key, default_name in DEFAULT_COLUMNS.items():
            col = st.sidebar.selectbox(f"Map '{default_name}'", options=[None] + list(df.columns), index=0)
            user_columns[key] = col
    else:
        for key, default_name in DEFAULT_COLUMNS.items():
            user_columns[key] = default_name if default_name in df.columns else None

    # Create mapped dataframe
    mapped_df = pd.DataFrame()
    for key, col in user_columns.items():
        if col in df.columns:
            mapped_df[key] = df[col]
        else:
            mapped_df[key] = None

    # Convert date columns
    for date_col in ['start', 'end']:
        if mapped_df[date_col] is not None:
            mapped_df[date_col] = pd.to_datetime(mapped_df[date_col], errors='coerce')

    # Filters / Slicers
    filtered_df = mapped_df.copy()
    st.sidebar.subheader("Filters")
    filter_columns = ['campaign', 'city', 'age', 'gender', 'adset_name', 'ad_name']
    for col in filter_columns:
        if mapped_df.get(col) is not None:
            vals = mapped_df[col].dropna().unique()
            selected = st.sidebar.multiselect(f"{DEFAULT_COLUMNS.get(col, col).title()}", options=vals)
            if selected:
                filtered_df = filtered_df[filtered_df[col].isin(selected)]

    # Date filter
    if mapped_df['start'] is not None and mapped_df['end'] is not None:
        start_date = st.sidebar.date_input("Start Date", mapped_df['start'].min().date())
        end_date = st.sidebar.date_input("End Date", mapped_df['end'].max().date())
        filtered_df = filtered_df[(filtered_df['start'] >= pd.to_datetime(start_date)) &
                                  (filtered_df['end'] <= pd.to_datetime(end_date))]
else:
    st.warning("Upload your CSV to begin analysis.")

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
# Pages
# ------------------------
pages = [
    "About Marketing Analytics",
    "Case Study Overview",
    "Campaign Overview",
    "Audience Insights",
    "Ad Performance",
    "Video Metrics"
]
st.sidebar.title("Dashboard Pages")
page = st.sidebar.radio("Navigate to", pages)

# ------------------------
# PAGE 1: About Marketing Analytics
# ------------------------
if page == "About Marketing Analytics":
    st.title("About Marketing Analytics")
    st.markdown("""
    Marketing Analytics is the science of measuring, managing, and analyzing marketing performance.  
    It enables brands to **understand what works, optimize campaigns, and maximize ROI**.

    **Applications:**
    - Target the right audience (age, city, interests)  
    - Design better creatives and messaging  
    - Spend efficiently to maximize returns  
    - Prove campaign effectiveness to stakeholders
    """)

    # Architecture Image
    try:
        arch_img = Image.open("images/marketing_analytics_architecture.jpg")
        st.image(arch_img, caption="End-to-End Marketing Analytics Architecture", use_column_width=True)
    except:
        st.warning("Architecture image not found!")

    # Funnel Image
    try:
        funnel_img = Image.open("images/marketing_funnel.jpg")
        st.image(funnel_img, caption="Marketing Funnel Stages: Awareness → Consideration → Conversion", use_column_width=True)
    except:
        st.warning("Funnel image not found!")

# ------------------------
# PAGE 2: Case Study Overview
# ------------------------
elif page == "Case Study Overview":
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
    st.caption("Created by Vibin | " + datetime.now().strftime("%d %B %Y"))

# ------------------------
# PAGE 3: Campaign Overview
# ------------------------
elif page == "Campaign Overview" and filtered_df is not None:
    st.title("Campaign Overview")
    cols = st.columns(6)
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
            value = f"₹{value:,.0f}" if 'INR' in label else (f"{value:.2f}%" if 'Avg' in label else f"{value:,.0f}")
            cols[idx].metric(label, value)
        else:
            cols[idx].info("Missing column")

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
        - Spot anomalies in spend quickly
        """)
        with st.expander("Quick Tips"):
            st.markdown("""
            - Hover on points to see exact spend
            - Use filters to focus on specific campaigns or dates
            - Compare trends across multiple campaigns
            """)

    # Top Cities by Spend
    if filtered_df['city'] is not None and filtered_df['spent'] is not None:
        city_perf = filtered_df.groupby('city', as_index=False)['spent'].sum().nlargest(10, 'spent')
        fig = px.bar(city_perf, x='city', y='spent', color='city', text='spent',
                     title="Top Cities by Ad Spend", color_discrete_sequence=px.colors.sequential.Viridis)
        fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        fig = style_axes(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Purpose:**")
        st.markdown("""
        - Identify high-spend regions
        - Allocate budget efficiently across cities
        - Recognize markets with maximum engagement
        """)
        with st.expander("Quick Tips"):
            st.markdown("- Hover on bars to see exact spend per city")

# ------------------------
# PAGE 4: Audience Insights
# ------------------------
elif page == "Audience Insights" and filtered_df is not None:
    st.title("Audience Insights")
    filtered_df['gender_std'] = filtered_df['gender'].str.strip().str.title()
    agg = filtered_df.groupby(['age','gender_std'], as_index=False)['clicks'].sum()
    gender_colors = {'Female': 'pink', 'Male': 'blue'}
    fig = px.bar(
        agg,
        x='age',
        y='clicks',
        color='gender_std',
        color_discrete_map=gender_colors,
        barmode='group',
        text='clicks',
        title="Clicks by Age & Gender"
    )
    fig.update_traces(textposition='outside')
    fig = style_axes(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Purpose:**")
    st.markdown("""
    - Identify which age and gender segments engage most
    - Guide audience targeting for campaigns
    - Compare male vs female engagement visually
    - Optimize creative messaging per segment
    """)
    with st.expander("Quick Tips"):
        st.markdown("- Use filters to drill down by campaign, city, adset, or date")

# ------------------------
# PAGE 5: Ad Performance
# ------------------------
elif page == "Ad Performance" and filtered_df is not None:
    st.title("Ad Performance")
    ad_perf = filtered_df.groupby('ad_name', as_index=False).agg({
        'clicks':'sum', 'impressions':'sum', 'spent':'sum', 'ctr':'mean', 'cpc':'mean'
    }).nlargest(10,'clicks')

    # Top Ads by Clicks
    fig1 = px.bar(ad_perf, x='clicks', y='ad_name', orientation='h', color='ctr', text='clicks',
                  title="Top Ads by Clicks", color_continuous_scale="Agsunset")
    fig1.update_traces(textposition='outside')
    fig1 = style_axes(fig1)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("**Purpose:**")
    st.markdown("""
    - Identify top-performing ads
    - Compare CTR to optimize creatives
    - Compare CPC to control ad cost
    """)
    with st.expander("Quick Tips"):
        st.markdown("- Focus on high clicks but low CPC ads for better ROI")

    # CPC vs CTR Bubble Chart
    fig2 = px.scatter(ad_perf, x='cpc', y='ctr', size='clicks', color='spent', hover_name='ad_name',
                      title="CPC vs CTR Efficiency", color_continuous_scale='Blues')
    fig2 = style_axes(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Purpose:**")
    st.markdown("""
    - Compare cost vs efficiency of each ad
    - Larger bubbles indicate ads with more clicks
    - Identify ads delivering clicks at lower cost
    """)
    with st.expander("Quick Tips"):
        st.markdown("- Hover to see exact CPC, CTR, and click count")

# ------------------------
# PAGE 6: Video Metrics
# ------------------------
elif page == "Video Metrics" and filtered_df is not None:
    st.title("Video Metrics")
    video_cols = ['video_25','video_50','video_75','video_95','video_100']
    melted = filtered_df.melt(id_vars=['ad_name'], value_vars=video_cols, var_name='Stage', value_name='Plays')
    melted['Plays'] = melted['Plays'].fillna(0).astype(int)
    fig1 = px.bar(melted, x='Stage', y='Plays', color='Stage', text='Plays',
                  title="Video Completion Funnel", color_discrete_sequence=px.colors.qualitative.Safe)
    fig1.update_traces(texttemplate='%{text}', textposition='inside')
    fig1 = style_axes(fig1)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("**Purpose:**")
    st.markdown("""
    - Measure engagement at different video stages
    - Identify drop-off points
    - Optimize video content length and messaging
    """)
    with st.expander("Quick Tips"):
        st.markdown("- Filter by ad or campaign to see segment-level engagement")

    # ThruPlay Efficiency Bubble Chart
    if filtered_df['thruplays'] is not None and filtered_df['cost_per_thruplay'] is not None:
        fig2 = px.scatter(filtered_df, x='thruplays', y='cost_per_thruplay', size='thruplays', color='spent',
                          hover_name='ad_name', title="ThruPlay Efficiency Bubble Chart", color_continuous_scale='Blues')
        fig2 = style_axes(fig2)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Purpose:**")
        st.markdown("""
        - Evaluate cost efficiency of ThruPlays
        - Larger bubbles indicate more ThruPlays
        - Compare campaigns/adsets on efficiency vs volume
        """)
        with st.expander("Quick Tips"):
            st.markdown("- Hover to see exact cost per ThruPlay and ThruPlays count")
