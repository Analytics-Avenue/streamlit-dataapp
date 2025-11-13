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

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    st.sidebar.success("File uploaded!")

    if upload_option == "Upload & Map Columns":
        st.sidebar.subheader("Map your columns")
        # Dropdown for each default column
        user_columns = {}
        for key, default_name in DEFAULT_COLUMNS.items():
            col = st.sidebar.selectbox(
                f"Map '{default_name}'",
                options=[None] + list(df.columns),
                index=0
            )
            user_columns[key] = col
    else:
        # Direct upload assumes default columns
        user_columns = {}
        for key, default_name in DEFAULT_COLUMNS.items():
            user_columns[key] = default_name if default_name in df.columns else None

    # Create mapped dataframe
    mapped_df = pd.DataFrame()
    for key, col in user_columns.items():
        if col in df.columns:
            mapped_df[key] = df[col]
        else:
            mapped_df[key] = None  # leave blank if column missing

    # Convert dates if available
    for date_col in ['start', 'end']:
        if mapped_df[date_col] is not None:
            mapped_df[date_col] = pd.to_datetime(mapped_df[date_col], errors='coerce')

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
elif page == "Campaign Overview" and mapped_df is not None:
    st.title("Campaign Overview")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    # Total Spent
    if mapped_df['spent'] is not None:
        col1.metric("Total Spent (INR)", f"₹{mapped_df['spent'].sum():,.0f}")
    else:
        col1.info("Missing column")

    # Total Impressions
    if mapped_df['impressions'] is not None:
        col2.metric("Total Impressions", f"{mapped_df['impressions'].sum():,.0f}")
    else:
        col2.info("Missing column")

    # Total Reach
    if mapped_df['reach'] is not None:
        col3.metric("Total Reach", f"{mapped_df['reach'].sum():,.0f}")
    else:
        col3.info("Missing column")

    # Total Clicks
    if mapped_df['clicks'] is not None:
        col4.metric("Total Clicks", f"{mapped_df['clicks'].sum():,.0f}")
    else:
        col4.info("Missing column")

    # Avg CTR
    if mapped_df['ctr'] is not None:
        col5.metric("Avg CTR", f"{mapped_df['ctr'].mean():.2f}%")
    else:
        col5.info("Missing column")

    # Avg CPM
    if mapped_df['cpm'] is not None:
        col6.metric("Avg CPM (₹)", f"₹{mapped_df['cpm'].mean():,.2f}")
    else:
        col6.info("Missing column")

    st.markdown("---")
    # Spend Trend
    if mapped_df['start'] is not None and mapped_df['spent'] is not None:
        spend_over_time = mapped_df.groupby('start', as_index=False)['spent'].sum()
        fig = px.line(spend_over_time, x='start', y='spent', markers=True,
                      text=spend_over_time['spent'].apply(lambda x: f"₹{x:,.0f}"),
                      title="Campaign Spend Over Time", color_discrete_sequence=['#2E86C1'])
        fig.update_traces(textposition='top right')
        fig = style_axes(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Purpose:** Visualize spend trends over time.")
        st.info("**Quick Tip:** Identify peaks/drops to evaluate budget allocation.")

    # Top Campaigns
    if mapped_df['campaign'] is not None and mapped_df['spent'] is not None:
        top_campaigns = mapped_df.groupby('campaign', as_index=False)['spent'].sum().nlargest(10,'spent')
        fig = px.bar(top_campaigns, x='spent', y='campaign', orientation='h',
                     text=top_campaigns['spent'].apply(lambda x: f"₹{x:,.0f}"), color='spent', color_continuous_scale="Blues",
                     title="Top Campaigns by Spend")
        fig.update_traces(textposition='outside')
        fig = style_axes(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Purpose:** Highlights campaigns with highest spend.")
        st.info("**Quick Tip:** Focus analysis on top-spend campaigns.")

# ------------------------
# AUDIENCE INSIGHTS
# ------------------------
elif page == "Audience Insights" and mapped_df is not None:
    st.title("Audience Insights")
    # Clicks by Age & Gender
    if mapped_df['age'] is not None and mapped_df['gender'] is not None and mapped_df['clicks'] is not None:
        agg = mapped_df.groupby(['age','gender'], as_index=False)['clicks'].sum()
        fig = px.bar(agg, x='age', y='clicks', color='gender', barmode='group', text='clicks', title="Clicks by Age & Gender")
        fig.update_traces(textposition='outside')
        fig = style_axes(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Purpose:** Shows which audience segments engage most.")
        st.info("**Quick Tip:** Use for targeted campaigns.")

    # Top Cities by Spend
    if mapped_df['city'] is not None and mapped_df['spent'] is not None:
        city_perf = mapped_df.groupby('city', as_index=False)['spent'].sum().nlargest(10,'spent')
        fig = px.bar(city_perf, x='city', y='spent', text=city_perf['spent'].apply(lambda x: f"₹{x:,.0f}"), color='city',
                     title="Top Cities by Ad Spend", color_discrete_sequence=px.colors.sequential.Viridis)
        fig.update_traces(textposition='outside')
        fig = style_axes(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Purpose:** Highlights cities with highest spend.")
        st.info("**Quick Tip:** Prioritize high-spend cities for analysis.")

# ------------------------
# AD PERFORMANCE
# ------------------------
elif page == "Ad Performance" and mapped_df is not None:
    st.title("Ad Performance")
    if mapped_df['ad_name'] is not None and mapped_df['clicks'] is not None:
        ad_perf = mapped_df.groupby('ad_name', as_index=False).agg({
            'clicks':'sum', 'impressions':'sum', 'spent':'sum', 'ctr':'mean', 'cpc':'mean'
        }).nlargest(10,'clicks')
        fig = px.bar(ad_perf, x='clicks', y='ad_name', orientation='h', color='ctr', text='clicks', title="Top Ads by Clicks", color_continuous_scale="Agsunset")
        fig.update_traces(textposition='outside')
        fig = style_axes(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Purpose:** Highlights high-engagement ads.")
        st.info("**Quick Tip:** Focus on high CTR & click ads.")

        # CPC vs CTR
        if 'cpc' in ad_perf.columns and 'ctr' in ad_perf.columns:
            fig = px.scatter(ad_perf, x='cpc', y='ctr', color='spent', text=ad_perf['spent'].apply(lambda x: f"₹{x:,.0f}"),
                             hover_name='ad_name', title="CPC vs CTR")
            fig.update_traces(marker=dict(size=10), textposition='top center')
            fig = style_axes(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Purpose:** Compare cost efficiency vs engagement.")
            st.info("**Quick Tip:** Low CPC + high CTR = best-performing ads.")

# ------------------------
# VIDEO METRICS
# ------------------------
elif page == "Video Metrics" and mapped_df is not None:
    st.title("Video Metrics")
    video_cols = ['video_25','video_50','video_75','video_95','video_100']
    if all(mapped_df[col] is not None for col in video_cols) and mapped_df['ad_name'] is not None:
        melted = mapped_df.melt(id_vars=['ad_name'], value_vars=video_cols, var_name='Stage', value_name='Plays')
        melted['Plays'] = melted['Plays'].fillna(0).astype(int)
        fig = px.bar(melted, x='Stage', y='Plays', color='Stage', text='Plays', title="Video Completion Funnel", color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_traces(texttemplate='%{text}', textposition='inside')
        fig = style_axes(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Purpose:** Shows viewer drop-off through video stages.")
        st.info("**Quick Tip:** Identify stages with highest drop-off to optimize content.")

    # ThruPlay Efficiency Bubble Chart
    if mapped_df['thruplays'] is not None and mapped_df['cost_per_thruplay'] is not None and mapped_df['impressions'] is not None:
        mapped_df['thruplays'] = mapped_df['thruplays'].fillna(0).astype(int)
        mapped_df['cost_per_thruplay'] = mapped_df['cost_per_thruplay'].fillna(0).astype(int)
        mapped_df['impressions'] = mapped_df['impressions'].fillna(0).astype(int)
        fig = px.scatter(mapped_df, x='thruplays', y='cost_per_thruplay', size='impressions', color='campaign',
                         hover_name='ad_name', size_max=40, text=mapped_df['cost_per_thruplay'].apply(lambda x: f"₹{x:,}"),
                         title="ThruPlays vs Cost per ThruPlay")
        fig.update_traces(marker=dict(sizemode='area', sizeref=2.*max(mapped_df['impressions'])/(40.**2), line=dict(width=1, color='DarkSlateGrey')), textposition='top center')
        fig = style_axes(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Purpose:** Shows cost-efficiency of video completions, bubble size = impressions.")
        st.info("**Quick Tip:** Focus on low Cost per ThruPlay with high ThruPlays and impressions.")

