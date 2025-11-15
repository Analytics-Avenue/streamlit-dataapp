# main/pages/marketing.py
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from PIL import Image
import os
import io

st.set_page_config(layout="wide")

hide_sidebar = """
<style>
[data-testid="stSidebarNav"] {display: none;}
section[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# ------------------------
# App Config
# ------------------------
st.set_page_config(page_title="Marketing Campaign Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------
# Default Columns (internal names -> friendly header names)
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
# Helper functions
# ------------------------
def style_axes(fig):
    fig.update_layout(
        xaxis=dict(title_font=dict(size=14, color='black', family='Arial Black')),
        yaxis=dict(title_font=dict(size=14, color='black', family='Arial Black')),
        title_font=dict(size=16, color='black', family='Arial Black'),
        legend_title_font=dict(size=12)
    )
    return fig

def safe_get(df, col):
    return col in df.columns and df[col].notna().any()

def format_inr_whole(x):
    try:
        return f"₹{int(round(x)):,}"
    except:
        return x

def make_sample_csv_bytes():
    # produce headers-only sample CSV using friendly column names
    cols = list(DEFAULT_COLUMNS.values())
    sample_df = pd.DataFrame(columns=cols)
    buf = io.StringIO()
    sample_df.to_csv(buf, index=False)
    return buf.getvalue().encode('utf-8')

# ------------------------
# Paths (images)
# ------------------------
arch_path = "./images/marketing_analytics_architecture.jpg"
funnel_path = "./images/marketing_funnel.jpg"


# ------------------------
# Default GitHub dataset raw URL (converted from user's repo URL)
# Provided input:
# https://github.com/Analytics-Avenue/streamlit-dataapp/blob/main/datasets/Marketing_Analytics.csv
GITHUB_RAW = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/Marketing_Analytics.csv"

# ------------------------
# Sidebar: Data source & upload options
# ------------------------
st.sidebar.title("Get data")
data_source = st.sidebar.radio("Choose data source:",
                               ["Use default GitHub dataset", "Upload CSV", "Upload CSV & Map Columns"])

# sample download button
st.sidebar.markdown("Download a **sample CSV** (headers only) to see expected columns:")
st.sidebar.download_button(
    label="Download sample CSV",
    data=make_sample_csv_bytes(),
    file_name="marketing_analytics_sample_headers.csv",
    mime="text/csv"
)

uploaded_file = None
df_raw = None
mapping = {}  # user mapping (internal->actual column name)

if data_source == "Use default GitHub dataset":
    st.sidebar.info("Loading sample dataset from GitHub for visual/demo purposes.")
    try:
        df_raw = pd.read_csv(GITHUB_RAW)
        df_raw.columns = df_raw.columns.str.strip()
        st.sidebar.success("Loaded default dataset from GitHub.")
    except Exception as e:
        st.sidebar.error("Could not load default dataset from GitHub. Check network or URL.")
        st.sidebar.write(str(e))

elif data_source == "Upload CSV" or data_source == "Upload CSV & Map Columns":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            df_raw.columns = df_raw.columns.str.strip()
            st.sidebar.success("File uploaded.")
        except Exception as e:
            st.sidebar.error("Failed to read uploaded CSV.")
            st.sidebar.write(str(e))

# ------------------------
# If mapping mode, let user map columns
# ------------------------
if data_source == "Upload CSV & Map Columns":
    st.sidebar.subheader("Column mapping (map app fields to your CSV columns)")
    if df_raw is None:
        st.sidebar.info("Upload a CSV on the sidebar to map columns.")
    else:
        for key, friendly in DEFAULT_COLUMNS.items():
            options = [None] + list(df_raw.columns)
            # try to pick default if column name exactly matches expected friendly header
            default_index = 0
            if friendly in df_raw.columns:
                try:
                    default_index = options.index(friendly)
                except:
                    default_index = 0
            selection = st.sidebar.selectbox(f"{friendly}", options=options, index=default_index, key=f"map_{key}")
            mapping[key] = selection

# ------------------------
# Build mapped_df (app internal column names)
# ------------------------
mapped_df = None
if df_raw is not None:
    # If user provided mapping (upload & map), use it; otherwise attempt auto-map using DEFAULT_COLUMNS values
    if data_source == "Upload CSV & Map Columns" and mapping:
        # create mapped_df with internal keys
        mapped_df = pd.DataFrame()
        for key in DEFAULT_COLUMNS.keys():
            colname = mapping.get(key)
            if colname in df_raw.columns:
                mapped_df[key] = df_raw[colname]
            else:
                mapped_df[key] = None
    else:
        # Auto mapping: find columns by friendly header names (DEFAULT_COLUMNS values)
        mapped_df = pd.DataFrame()
        for key, friendly in DEFAULT_COLUMNS.items():
            if friendly in df_raw.columns:
                mapped_df[key] = df_raw[friendly]
            elif key in df_raw.columns:
                # maybe the CSV already uses internal names
                mapped_df[key] = df_raw[key]
            else:
                mapped_df[key] = None

    # Convert date fields if present
    for date_col in ['start', 'end']:
        if date_col in mapped_df.columns and mapped_df[date_col].notna().any():
            mapped_df[date_col] = pd.to_datetime(mapped_df[date_col], errors='coerce')

    # Basic cleaning for numeric columns (fillna for plotting)
    numeric_cols = ['spent', 'impressions', 'reach', 'clicks', 'ctr', 'cpm', 'cpc', 'thruplays', 'cost_per_thruplay',
                    'video_25', 'video_50', 'video_75', 'video_95', 'video_100']
    for nc in numeric_cols:
        if nc in mapped_df.columns:
            # coerce to numeric where possible
            mapped_df[nc] = pd.to_numeric(mapped_df[nc], errors='coerce')

# filtered_df = mapped_df after applying filters
filtered_df = None
if mapped_df is not None:
    filtered_df = mapped_df.copy()
    # Sidebar filters (reactive)
    st.sidebar.subheader("Filters")
    # Campaign
    if safe_get(mapped_df, 'campaign'):
        sel = st.sidebar.multiselect("Campaign", options=mapped_df['campaign'].dropna().unique())
        if sel:
            filtered_df = filtered_df[filtered_df['campaign'].isin(sel)]
    # City
    if safe_get(mapped_df, 'city'):
        sel = st.sidebar.multiselect("City", options=mapped_df['city'].dropna().unique())
        if sel:
            filtered_df = filtered_df[filtered_df['city'].isin(sel)]
    # Age
    if safe_get(mapped_df, 'age'):
        sel = st.sidebar.multiselect("Age", options=mapped_df['age'].dropna().unique())
        if sel:
            filtered_df = filtered_df[filtered_df['age'].isin(sel)]
    # Gender
    if safe_get(mapped_df, 'gender'):
        sel = st.sidebar.multiselect("Gender", options=mapped_df['gender'].dropna().unique())
        if sel:
            filtered_df = filtered_df[filtered_df['gender'].isin(sel)]
    # Ad set
    if safe_get(mapped_df, 'adset_name'):
        sel = st.sidebar.multiselect("Ad Set Name", options=mapped_df['adset_name'].dropna().unique())
        if sel:
            filtered_df = filtered_df[filtered_df['adset_name'].isin(sel)]
    # Ad name
    if safe_get(mapped_df, 'ad_name'):
        sel = st.sidebar.multiselect("Ad Name", options=mapped_df['ad_name'].dropna().unique())
        if sel:
            filtered_df = filtered_df[filtered_df['ad_name'].isin(sel)]
    # Date range
    if 'start' in mapped_df.columns and mapped_df['start'].notna().any():
        min_date = mapped_df['start'].min().date()
        max_date = mapped_df['start'].max().date()
        start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        filtered_df = filtered_df[(filtered_df['start'] >= pd.to_datetime(start_date)) & (filtered_df['start'] <= pd.to_datetime(end_date))]

# ------------------------
# Page navigation
# ------------------------
pages = [
    "About Marketing Analytics",
    "Case Study Overview",
    "Campaign Overview",
    "Audience Insights",
    "Ad Performance",
    "Video Metrics"
]
st.sidebar.title("Pages")
page = st.sidebar.radio("Go to", pages)


# ------------------------
# PAGE: About Marketing Analytics
# ------------------------
if page == "About Marketing Analytics":
    st.title("About Marketing Analytics")
    st.markdown("""
    **Marketing analytics** measures, manages and analyzes marketing performance to maximize ROI and optimize campaigns.

    Every ad interaction is tracked — impressions, clicks, views and conversions — and analytics explains why things happened.
    """)

    st.subheader("How it works — architecture")

    if os.path.exists(arch_path):
        st.image(Image.open(arch_path),
                 caption="End-to-End Marketing Analytics Architecture",
                 use_column_width=True)
    else:
        st.info("Architecture image not found at: " + arch_path)

    if os.path.exists(funnel_path):
        st.image(Image.open(funnel_path),
                 caption="Marketing Funnel: Awareness → Consideration → Conversion",
                 use_column_width=True)
    else:
        st.info("Funnel image not found at: " + funnel_path)

    st.markdown("### Why this matters")
    st.markdown("""
    - Track Reach, CTR, CPC, CAC and engagement to measure campaign health.
    - Use data to target audiences, optimize creatives, and allocate budget.
    - Visualize funnel drop-offs and act where conversions are lost.
    """)

# ------------------------
# PAGE: Case Study Overview
# ------------------------
elif page == "Case Study Overview":
    st.title("Marketing Campaign Case Study")
    st.markdown("""Short case study overview and embedded video demonstration.""")
    with st.expander("Watch Case Study Preview"):
        st.video("https://www.youtube.com/watch?v=0d6oY8G5e5c")
    st.caption("Created by Vibin | " + datetime.now().strftime("%d %B %Y"))

# ------------------------
# PAGE: Campaign Overview
# ------------------------
elif page == "Campaign Overview":
    st.title("Campaign Overview")
    if filtered_df is None:
        st.info("No dataset loaded. Use the sidebar to load the GitHub sample or upload your CSV.")
    else:
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
            if safe_get(filtered_df, col):
                if 'Total' in label or 'Spent' in label:
                    val = filtered_df[col].sum(skipna=True)
                    display = format_inr_whole(val)
                else:
                    if 'Avg' in label:
                        display = f"{filtered_df[col].mean():.2f}%"
                    else:
                        display = f"{int(filtered_df[col].sum()):,}"
                cols[idx].metric(label, display)
            else:
                cols[idx].info("Missing")

        st.markdown("---")
        # Spend trend
        if 'start' in filtered_df.columns and safe_get(filtered_df, 'spent'):
            spend_over_time = filtered_df.groupby('start', as_index=False)['spent'].sum()
            fig = px.line(spend_over_time, x='start', y='spent', markers=True,
                          text=spend_over_time['spent'].apply(lambda x: format_inr_whole(x)),
                          title="Campaign Spend Over Time", color_discrete_sequence=['#2E86C1'],
                          hover_data={'start': True, 'spent': True})
            fig.update_traces(textposition='top right')
            fig = style_axes(fig)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Purpose:**")
            st.markdown("""
            - Track spend patterns across time.  
            - Detect overspend windows and anomalies.  
            - Inform budget reallocation decisions.
            """)
            with st.expander("Quick Tips"):
                st.markdown("- Hover for exact spend. Use date filters to zoom.")

        # Top campaigns by spend
        if safe_get(filtered_df, 'campaign') and safe_get(filtered_df, 'spent'):
            top_campaigns = filtered_df.groupby('campaign', as_index=False)['spent'].sum().nlargest(10, 'spent')
            top_campaigns['spent_label'] = top_campaigns['spent'].apply(format_inr_whole)
            fig = px.bar(top_campaigns, x='spent', y='campaign', orientation='h',
                         text='spent_label', title="Top Campaigns by Spend", color='spent',
                         color_continuous_scale='Blues')
            fig.update_traces(textposition='outside')
            fig = style_axes(fig)
            st.plotly_chart(fig, use_container_width=True)

# ------------------------
# PAGE: Audience Insights
# ------------------------
elif page == "Audience Insights":
    st.title("Audience Insights")
    if filtered_df is None:
        st.info("No dataset loaded.")
    else:
        # standardize gender
        if 'gender' in filtered_df.columns:
            filtered_df['gender_std'] = filtered_df['gender'].astype(str).str.strip().str.title()
        else:
            filtered_df['gender_std'] = 'Unknown'

        # Clicks by Age & Gender
        if safe_get(filtered_df, 'clicks') and 'age' in filtered_df.columns:
            agg = filtered_df.groupby(['age', 'gender_std'], as_index=False)['clicks'].sum()
            gender_colors = {'Female': '#ff66b2', 'Male': '#1f77b4'}  # pink and blue hex
            fig = px.bar(agg, x='age', y='clicks', color='gender_std', barmode='group',
                         color_discrete_map=gender_colors,
                         text='clicks', title="Clicks by Age & Gender",
                         hover_data={'age': True, 'gender_std': True, 'clicks': True})
            fig.update_traces(textposition='outside')
            fig = style_axes(fig)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Purpose:**")
            st.markdown("""
            - Identify high-engagement age/gender segments.  
            - Prioritize targeting and creative variants.  
            - Reveal underperforming demographics to adjust messaging.
            """)
            with st.expander("Quick Tips"):
                st.markdown("- Use filters to drill down by city/campaign/ad set.")

        # Top cities by spend
        if safe_get(filtered_df, 'spent') and safe_get(filtered_df, 'city'):
            city_perf = filtered_df.groupby('city', as_index=False)['spent'].sum().nlargest(10, 'spent')
            city_perf['spent_label'] = city_perf['spent'].apply(format_inr_whole)
            fig = px.bar(city_perf, x='city', y='spent', text='spent_label', title="Top Cities by Ad Spend",
                         color='city', color_discrete_sequence=px.colors.sequential.Viridis)
            fig.update_traces(textposition='outside')
            fig = style_axes(fig)
            st.plotly_chart(fig, use_container_width=True)

# ------------------------
# PAGE: Ad Performance
# ------------------------
elif page == "Ad Performance":
    st.title("Ad Performance")
    if filtered_df is None:
        st.info("No dataset loaded.")
    else:
        required_cols = ['ad_name', 'clicks', 'impressions', 'spent', 'ctr', 'cpc']
        if all(c in filtered_df.columns for c in required_cols) and filtered_df[required_cols].notna().any().any():
            ad_perf = filtered_df.groupby('ad_name', as_index=False).agg({
                'clicks':'sum', 'impressions':'sum', 'spent':'sum', 'ctr':'mean', 'cpc':'mean'
            }).nlargest(10, 'clicks')

            # Top Ads by Clicks
            ad_perf['clicks_label'] = ad_perf['clicks'].astype(int)
            fig1 = px.bar(ad_perf, x='clicks', y='ad_name', orientation='h', color='ctr', text='clicks_label',
                          title="Top Ads by Clicks", color_continuous_scale="Agsunset",
                          hover_data={'clicks': True, 'ctr': True, 'cpc': True, 'spent': True})
            fig1.update_traces(textposition='outside')
            fig1 = style_axes(fig1)
            st.plotly_chart(fig1, use_container_width=True)

            st.markdown("**Purpose:**")
            st.markdown("""
            - Surface top-performing creatives.  
            - Use CTR and CPC to evaluate efficiency.  
            - Select ads to scale or pause.
            """)
            with st.expander("Quick Tips"):
                st.markdown("- Cross-check spend to evaluate ROI for top ads.")

            # CPC vs CTR bubble: size = clicks, color = spent
            if 'cpc' in ad_perf.columns and 'ctr' in ad_perf.columns:
                max_clicks = ad_perf['clicks'].max() if ad_perf['clicks'].max() > 0 else 1
                sizeref = 2.*max_clicks/(40.**2)
                fig2 = px.scatter(ad_perf, x='cpc', y='ctr', size='clicks', color='spent', hover_name='ad_name',
                                  title="CPC vs CTR Efficiency", size_max=40, color_continuous_scale='Blues',
                                  hover_data={'clicks': True, 'cpc': True, 'ctr': True, 'spent': True})
                fig2.update_traces(marker=dict(sizemode='area', sizeref=sizeref, line=dict(width=1, color='DarkSlateGrey')))
                fig2 = style_axes(fig2)
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Ad performance requires columns: Ad name, Link clicks, Impressions, Amount spent, CTR, CPC.")

# ------------------------
# PAGE: Video Metrics
# ------------------------
elif page == "Video Metrics":
    st.title("Video Metrics")
    if filtered_df is None:
        st.info("No dataset loaded.")
    else:
        video_cols = ['video_25','video_50','video_75','video_95','video_100']
        if all(c in filtered_df.columns for c in video_cols) and safe_get(filtered_df, 'ad_name'):
            melted = filtered_df.melt(id_vars=['ad_name'], value_vars=video_cols, var_name='Stage', value_name='Plays')
            melted['Plays'] = pd.to_numeric(melted['Plays'], errors='coerce').fillna(0).astype(int)
            funnel = melted.groupby('Stage', as_index=False)['Plays'].sum().reindex(index=[0,1,2,3,4]) if False else melted.groupby('Stage', as_index=False)['Plays'].sum()
            fig1 = px.bar(melted, x='Stage', y='Plays', color='Stage', text='Plays',
                          title="Video Completion Funnel", color_discrete_sequence=px.colors.qualitative.Safe,
                          hover_data={'Stage': True, 'Plays': True})
            fig1.update_traces(texttemplate='%{text}', textposition='inside')
            fig1 = style_axes(fig1)
            st.plotly_chart(fig1, use_container_width=True)

            st.markdown("**Purpose:**")
            st.markdown("""
            - Track viewer drop-off across video watch stages.  
            - Identify where attention is lost.  
            - Improve hooks and creative pacing.
            """)
            with st.expander("Quick Tips"):
                st.markdown("- Compare funnel across campaigns or creatives.")

        # ThruPlay efficiency bubble
        if 'thruplays' in filtered_df.columns and 'cost_per_thruplay' in filtered_df.columns:
            filtered_df['thruplays_num'] = pd.to_numeric(filtered_df['thruplays'], errors='coerce').fillna(0).astype(int)
            filtered_df['cost_per_thruplay_num'] = pd.to_numeric(filtered_df['cost_per_thruplay'], errors='coerce').fillna(0)
            # size by impressions if available otherwise by thruplays
            size_col = 'impressions' if 'impressions' in filtered_df.columns and filtered_df['impressions'].notna().any() else 'thruplays_num'
            max_size = filtered_df[size_col].max() if filtered_df[size_col].max() > 0 else 1
            sizeref = 2.*max_size/(40.**2)
            fig2 = px.scatter(filtered_df, x='thruplays_num', y='cost_per_thruplay_num',
                              size=size_col, color='spent' if 'spent' in filtered_df.columns else None,
                              hover_name='ad_name' if 'ad_name' in filtered_df.columns else None,
                              title="ThruPlay Efficiency", size_max=40,
                              hover_data={'thruplays_num': True, 'cost_per_thruplay_num': True, size_col: True})
            fig2.update_traces(marker=dict(sizemode='area', sizeref=sizeref, line=dict(width=1, color='DarkSlateGrey')))
            fig2 = style_axes(fig2)
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("**Purpose:**")
            st.markdown("""
            - Evaluate cost efficiency of full video plays.  
            - Bubble size = impressions (reach) or thruplays.  
            - Identify videos with high engagement at low cost.
            """)
            with st.expander("Quick Tips"):
                st.markdown("- Hover bubbles to see exact values. Focus on large low-cost bubbles.")

# ------------------------
# Footer / Notes
# ------------------------
st.markdown("---")
st.caption("Dashboard generated: " + datetime.now().strftime("%d %b %Y") + " • Make sure your CSV headers match the sample CSV downloaded from the sidebar.")
