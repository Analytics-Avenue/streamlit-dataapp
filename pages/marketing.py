import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# ------------------------
# App Configuration
# ------------------------
st.set_page_config(page_title="Marketing Campaign Dashboard", layout="wide", initial_sidebar_state="expanded")

# ------------------------
# Default Columns
# ------------------------
DEFAULT_COLUMNS = {
    'campaign':'Campaign name', 'city':'City', 'age':'Age', 'gender':'Gender',
    'adset_name':'Ad set name', 'ad_name':'Ad name',
    'start':'Reporting starts','end':'Reporting ends',
    'spent':'Amount spent (INR)','impressions':'Impressions','reach':'Reach','clicks':'Link clicks',
    'ctr':'CTR (all)','cpm':'CPM (cost per 1,000 impressions)','cpc':'CPC (cost per link click)',
    'thruplays':'ThruPlays','cost_per_thruplay':'Cost per ThruPlay',
    'video_25':'Video plays at 25%','video_50':'Video plays at 50%','video_75':'Video plays at 75%',
    'video_95':'Video plays at 95%','video_100':'Video plays at 100%'
}

# ------------------------
# Upload / Map Columns
# ------------------------
st.sidebar.title("Upload or Map Data")
upload_option = st.sidebar.radio("Choose how to provide data:", ["Direct Upload", "Upload & Map Columns"])
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

mapped_df = None
filtered_df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    st.sidebar.success("File uploaded!")

    # Column mapping
    user_columns = {}
    if upload_option == "Upload & Map Columns":
        st.sidebar.subheader("Map your columns")
        for key, default_name in DEFAULT_COLUMNS.items():
            col = st.sidebar.selectbox(f"Map '{default_name}'", options=[None]+list(df.columns))
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

    # Convert dates
    for date_col in ['start','end']:
        if date_col in mapped_df.columns:
            mapped_df[date_col] = pd.to_datetime(mapped_df[date_col], errors='coerce')

    # Filters
    filtered_df = mapped_df.copy()
    st.sidebar.subheader("Filters")
    filter_columns = ['campaign','city','age','gender','adset_name','ad_name']
    for col in filter_columns:
        if col in filtered_df.columns and filtered_df[col].notna().any():
            vals = filtered_df[col].dropna().unique()
            selected = st.sidebar.multiselect(f"{DEFAULT_COLUMNS.get(col, col)}", options=vals)
            if selected:
                filtered_df = filtered_df[filtered_df[col].isin(selected)]

    # Date filter
    if 'start' in filtered_df.columns and 'end' in filtered_df.columns:
        start_date = st.sidebar.date_input("Start Date", filtered_df['start'].min().date())
        end_date = st.sidebar.date_input("End Date", filtered_df['end'].max().date())
        filtered_df = filtered_df[(filtered_df['start'] >= pd.to_datetime(start_date)) & (filtered_df['end'] <= pd.to_datetime(end_date))]

else:
    st.warning("Upload your CSV to begin.")

# ------------------------
# Helper to style axes
# ------------------------
def style_axes(fig):
    fig.update_layout(
        xaxis=dict(title_font=dict(size=14,color='black',family='Arial Black')),
        yaxis=dict(title_font=dict(size=14,color='black',family='Arial Black')),
        title_font=dict(size=16,color='black',family='Arial Black')
    )
    return fig

# ------------------------
# Navigation
# ------------------------
st.sidebar.title("Dashboard Pages")
page = st.sidebar.radio("Navigate to", ["Case Study Overview","Campaign Overview","Audience Insights","Ad Performance","Video Metrics"])

# ------------------------
# Case Study Overview
# ------------------------
if page=="Case Study Overview":
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
# Campaign Overview
# ------------------------
elif page=="Campaign Overview" and filtered_df is not None:
    st.title("Campaign Overview")
    cols = st.columns(6)
    metrics = {
        'Total Spent (INR)':'spent','Total Impressions':'impressions','Total Reach':'reach',
        'Total Clicks':'clicks','Avg CTR':'ctr','Avg CPM (₹)':'cpm'
    }
    for idx,(label,col_name) in enumerate(metrics.items()):
        col_obj = cols[idx]
        if col_name in filtered_df.columns and filtered_df[col_name].notna().any():
            value = filtered_df[col_name].sum() if 'Total' in label else filtered_df[col_name].mean()
            if 'INR' in label:
                value = f"₹{value:,.0f}" if 'Total' in label else f"₹{value:,.2f}"
            else:
                value = f"{value:,.0f}" if 'Total' in label else f"{value:.2f}%"
            col_obj.metric(label,value)
        else:
            col_obj.info("Missing column")

    st.markdown("---")
    # Spend Trend
    if 'start' in filtered_df.columns and 'spent' in filtered_df.columns:
        spend_over_time = filtered_df.groupby('start',as_index=False)['spent'].sum()
        fig = px.line(spend_over_time, x='start',y='spent',markers=True,
                      text=spend_over_time['spent'].apply(lambda x:f"₹{x:,.0f}"),
                      title="Campaign Spend Over Time",color_discrete_sequence=['#2E86C1'],
                      hover_data={'start':True,'spent':True})
        fig.update_traces(textposition='top right')
        fig = style_axes(fig)
        st.plotly_chart(fig,use_container_width=True)

        st.markdown("**Purpose:**")
        st.markdown("""
        - Track daily/weekly campaign spend patterns
        - Identify overspending or underspending periods
        - Pinpoint trends to optimize future budget allocation
        - Quickly spot anomalies or spikes in ad spend
        """)
        with st.expander("Quick Tips"):
            st.markdown("- Hover on points to see exact spend\n- Use filters to focus on specific campaigns or dates\n- Compare trends over multiple campaigns")

# ------------------------
# Audience Insights
# ------------------------
elif page=="Audience Insights" and filtered_df is not None:
    st.title("Audience Insights")
    # Standardize gender
    if 'gender' in filtered_df.columns:
        filtered_df['gender_std'] = filtered_df['gender'].str.strip().str.title()
    else:
        filtered_df['gender_std'] = 'Unknown'

    # Clicks by Age & Gender
    if all(x in filtered_df.columns for x in ['age','clicks']):
        agg = filtered_df.groupby(['age','gender_std'],as_index=False)['clicks'].sum()
        gender_colors = {'Female':'pink','Male':'blue'}
        fig = px.bar(agg,x='age',y='clicks',color='gender_std',
                     color_discrete_map=gender_colors,barmode='group',
                     text='clicks',title="Clicks by Age & Gender")
        fig.update_traces(textposition='outside')
        fig = style_axes(fig)
        st.plotly_chart(fig,use_container_width=True)
        st.markdown("**Purpose:**\n- Identify top engaging age/gender segments\n- Guide audience targeting\n- Optimize creative for top segments\n- Compare male vs female engagement")
        with st.expander("Quick Tips"):
            st.markdown("- Filter by campaign/city/adset for segment-level insights")

    # Top Cities by Spend
    if all(x in filtered_df.columns for x in ['city','spent']):
        city_perf = filtered_df.groupby('city',as_index=False)['spent'].sum().nlargest(10,'spent')
        fig = px.bar(city_perf,x='city',y='spent',color='city',text='spent',
                     color_discrete_sequence=px.colors.sequential.Viridis)
        fig.update_traces(texttemplate='%{text:.0f}',textposition='outside')
        fig = style_axes(fig)
        st.plotly_chart(fig,use_container_width=True)
        st.markdown("**Purpose:**\n- Identify high-spend regions\n- Allocate budget efficiently\n- Recognize top engagement markets")
        with st.expander("Quick Tips"):
            st.markdown("- Hover on bars to see exact spend")

# ------------------------
# Ad Performance
# ------------------------
elif page=="Ad Performance" and filtered_df is not None:
    st.title("Ad Performance")
    if all(x in filtered_df.columns for x in ['ad_name','clicks','impressions','spent','ctr','cpc']):
        ad_perf = filtered_df.groupby('ad_name',as_index=False).agg({
            'clicks':'sum','impressions':'sum','spent':'sum','ctr':'mean','cpc':'mean'
        }).nlargest(10,'clicks')

        # Top Ads
        fig1 = px.bar(ad_perf,x='clicks',y='ad_name',orientation='h',color='ctr',text='clicks',title="Top Ads by Clicks",
                      color_continuous_scale="Agsunset")
        fig1.update_traces(textposition='outside')
        fig1 = style_axes(fig1)
        st.plotly_chart(fig1,use_container_width=True)
        st.markdown("**Purpose:**\n- Identify top-performing ads\n- Analyze CTR to optimize creatives\n- Compare CPC for cost efficiency")
        with st.expander("Quick Tips"):
            st.markdown("- Focus on high clicks & low CPC ads")

        # CPC vs CTR Bubble
        fig2 = px.scatter(ad_perf,x='cpc',y='ctr',size='clicks',color='spent',hover_name='ad_name',
                          title="CPC vs CTR Efficiency",color_continuous_scale='Blues')
        fig2 = style_axes(fig2)
        st.plotly_chart(fig2,use_container_width=True)
        st.markdown("**Purpose:**\n- Compare cost vs efficiency\n- Identify ads delivering clicks at lower cost\n- Bubble size = popularity by clicks")
        with st.expander("Quick Tips"):
            st.markdown("- Larger bubbles = more clicks")

# ------------------------
# Video Metrics
# ------------------------
elif page=="Video Metrics" and filtered_df is not None:
    st.title("Video Metrics")
    video_cols = ['video_25','video_50','video_75','video_95','video_100']
    if all(x in filtered_df.columns for x in video_cols+['ad_name']):
        melted = filtered_df.melt(id_vars=['ad_name'],value_vars=video_cols,var_name='Stage',value_name='Plays')
        melted['Plays'] = melted['Plays'].fillna(0).astype(int)
        fig1 = px.bar(melted,x='Stage',y='Plays',color='Stage',text='Plays',
                      color_discrete_sequence=px.colors.qualitative.Safe)
        fig1.update_traces(textposition='inside')
        fig1 = style_axes(fig1)
        st.plotly_chart(fig1,use_container_width=True)
        st.markdown("**Purpose:**\n- Measure engagement at video stages\n- Identify drop-off points\n- Optimize video length/content")
        with st.expander("Quick Tips"):
            st.markdown("- Filter by ad/campaign for segment insights")

    # ThruPlay Bubble
    if all(x in filtered_df.columns for x in ['thruplays','cost_per_thruplay','ad_name','spent']):
        fig2 = px.scatter(filtered_df,x='thruplays',y='cost_per_thruplay',size='thruplays',color='spent',
                          hover_name='ad_name',title="ThruPlay Efficiency Bubble Chart",color_continuous_scale='Blues')
        fig2 = style_axes(fig2)
        st.plotly_chart(fig2,use_container_width=True)
        st.markdown("**Purpose:**\n- Evaluate ThruPlay cost efficiency\n- Larger bubbles = more ThruPlays\n- Compare campaigns/adsets on volume vs cost")
        with st.expander("Quick Tips"):
            st.markdown("- Hover to see exact cost per ThruPlay and ThruPlays count")
