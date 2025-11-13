import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# ----------------------------------------
# App Configuration
# ----------------------------------------
st.set_page_config(
    page_title="Marketing Campaign Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------
# Sidebar Navigation
# ----------------------------------------
st.sidebar.title("Campaign Analytics Dashboard")
st.sidebar.markdown("Built by **Data Experts**")
st.sidebar.markdown(f"Created on: {datetime.now().strftime('%d %b %Y')}")

page = st.sidebar.radio(
    "Navigate to",
    [
        "Case Study Overview",
        "Campaign Overview",
        "Audience Insights",
        "Ad Performance",
        "Video Metrics"
    ]
)

# ----------------------------------------
# File Upload
# ----------------------------------------
st.sidebar.subheader("Upload Campaign Data")
uploaded_file = st.sidebar.file_uploader("Upload your Meta Ads CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")

    df.columns = df.columns.str.strip()
    df['Reporting starts'] = pd.to_datetime(df['Reporting starts'], errors='coerce')
    df['Reporting ends'] = pd.to_datetime(df['Reporting ends'], errors='coerce')

    # Filters
    st.sidebar.subheader("Filters")
    campaign_filter = st.sidebar.multiselect("Campaign Name", options=df['Campaign name'].unique())
    city_filter = st.sidebar.multiselect("City", options=df['City'].dropna().unique())
    age_filter = st.sidebar.multiselect("Age Group", options=df['Age'].dropna().unique())
    gender_filter = st.sidebar.multiselect("Gender", options=df['Gender'].dropna().unique())
    start_date = st.sidebar.date_input("Start Date", df['Reporting starts'].min().date())
    end_date = st.sidebar.date_input("End Date", df['Reporting ends'].max().date())

    filtered_df = df.copy()
    if campaign_filter:
        filtered_df = filtered_df[filtered_df['Campaign name'].isin(campaign_filter)]
    if city_filter:
        filtered_df = filtered_df[filtered_df['City'].isin(city_filter)]
    if age_filter:
        filtered_df = filtered_df[filtered_df['Age'].isin(age_filter)]
    if gender_filter:
        filtered_df = filtered_df[filtered_df['Gender'].isin(gender_filter)]
    filtered_df = filtered_df[
        (filtered_df['Reporting starts'] >= pd.to_datetime(start_date)) &
        (filtered_df['Reporting ends'] <= pd.to_datetime(end_date))
    ]
else:
    df = None
    st.warning("Please upload your dataset from the sidebar to begin analysis.")

# ----------------------------------------
# Helper function for axes styling
# ----------------------------------------
def style_axes(fig):
    fig.update_layout(
        xaxis=dict(title_font=dict(size=14, color='black', family='Arial Black')),
        yaxis=dict(title_font=dict(size=14, color='black', family='Arial Black')),
        title_font=dict(size=16, color='black', family='Arial Black')
    )
    return fig

# ----------------------------------------
# PAGE 1: Case Study Overview
# ----------------------------------------
if page == "Case Study Overview":
    st.title("Marketing Campaign Case Study")
    st.markdown("""
    Welcome to the **Marketing Campaign Performance Dashboard** — an interactive workspace to explore Meta Ads performance across regions, demographics, and ad sets.

    **Use Case Highlights:**
    - Evaluate **budget effectiveness** and **ROI trends**  
    - Analyze audience segments driving conversions  
    - Compare creatives for engagement and reach  
    - Identify performance gaps to improve delivery  

    **Tools Used:** Streamlit | Plotly | Pandas | Python
    """)
    with st.expander("Watch Case Study Preview"):
        st.video("https://www.youtube.com/watch?v=0d6oY8G5e5c")
    st.markdown("---")
    st.caption("Created by **Vibin**, Senior Business Analyst | Date: " + datetime.now().strftime("%d %B %Y"))

    if df is not None:
        st.success("Dataset uploaded successfully. Navigate through the tabs on the left to explore dashboards.")
    else:
        st.info("Upload your campaign CSV file using the sidebar to begin exploration.")

# ----------------------------------------
# PAGE 2: Campaign Overview
# ----------------------------------------
elif page == "Campaign Overview" and df is not None:
    st.title("Campaign Overview")

    total_spent = filtered_df['Amount spent (INR)'].sum()
    total_impressions = filtered_df['Impressions'].sum()
    total_reach = filtered_df['Reach'].sum()
    total_clicks = filtered_df['Link clicks'].sum()
    avg_ctr = filtered_df['CTR (all)'].mean()
    avg_cpm = filtered_df['CPM (cost per 1,000 impressions)'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Spent (INR)", f"₹{total_spent:,.0f}")
    col2.metric("Total Impressions", f"{total_impressions:,.0f}")
    col3.metric("Total Reach", f"{total_reach:,.0f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Total Clicks", f"{total_clicks:,.0f}")
    col5.metric("Avg CTR", f"{avg_ctr:.2f}%")
    col6.metric("Avg CPM", f"₹{avg_cpm:,.0f}")

    st.markdown("---")
    st.subheader("Spend Trend Over Time")
    spend_over_time = filtered_df.groupby('Reporting starts', as_index=False)['Amount spent (INR)'].sum()
    fig = px.line(
        spend_over_time, x='Reporting starts', y='Amount spent (INR)',
        markers=True, title="Campaign Spend Over Time",
        color_discrete_sequence=['#2E86C1'],
        text=spend_over_time['Amount spent (INR)'].apply(lambda x: f"₹{x:,.0f}")
    )
    fig.update_traces(textposition='top right')
    fig = style_axes(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Quick Insight: This chart shows how your campaign budget is being spent over time. Use it to see patterns, peaks, and trends in spending.")

    st.subheader("Top 10 Campaigns by Spend")
    top_campaigns = filtered_df.groupby("Campaign name", as_index=False)['Amount spent (INR)'].sum().nlargest(10, 'Amount spent (INR)')
    fig = px.bar(
        top_campaigns, x="Amount spent (INR)", y="Campaign name", orientation='h',
        color="Amount spent (INR)", color_continuous_scale="Blues",
        title="Top Campaigns by Spend",
        text=top_campaigns['Amount spent (INR)'].apply(lambda x: f"₹{x:,.0f}")
    )
    fig.update_traces(textposition='outside')
    fig = style_axes(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Quick Insight: Identify which campaigns are taking most of your budget and assess if spending aligns with strategy.")

# ----------------------------------------
# PAGE 3: Audience Insights
# ----------------------------------------
elif page == "Audience Insights" and df is not None:
    st.title("Audience Insights")

    st.subheader("Clicks by Age & Gender")
    agg = filtered_df.groupby(["Age", "Gender"], as_index=False)["Link clicks"].sum()
    fig = px.bar(
        agg, x="Age", y="Link clicks", color="Gender", barmode="group",
        title="Clicks Distribution by Age and Gender",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        text='Link clicks'
    )
    fig.update_traces(textposition='outside')
    fig = style_axes(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Quick Insight: See which age and gender groups interact most with ads. Helps refine targeting strategies.")

    st.subheader("Top 10 Cities by Ad Spend")
    city_perf = filtered_df.groupby("City", as_index=False)['Amount spent (INR)'].sum().nlargest(10, 'Amount spent (INR)')
    fig = px.bar(
        city_perf, x="City", y="Amount spent (INR)", color="City",
        title="Top Cities by Ad Spend",
        color_discrete_sequence=px.colors.sequential.Viridis,
        text=city_perf['Amount spent (INR)'].apply(lambda x: f"₹{x:,.0f}")
    )
    fig.update_traces(textposition='outside')
    fig = style_axes(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Quick Insight: Highlights cities with highest spend. Useful for geographic focus and allocation of resources.")

# ----------------------------------------
# PAGE 4: Ad Performance
# ----------------------------------------
elif page == "Ad Performance" and df is not None:
    st.title("Ad-Level Performance")

    ad_perf = filtered_df.groupby("Ad name", as_index=False).agg({
        "Link clicks": "sum",
        "Impressions": "sum",
        "Amount spent (INR)": "sum",
        "CTR (all)": "mean",
        "CPC (cost per link click)": "mean"
    }).nlargest(10, 'Link clicks')

    st.subheader("Top 10 Ads by Clicks")
    fig = px.bar(
        ad_perf, x="Link clicks", y="Ad name", orientation='h',
        color="CTR (all)", color_continuous_scale="Agsunset",
        title="Top Performing Ads",
        text='Link clicks'
    )
    fig.update_traces(textposition='outside')
    fig = style_axes(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Quick Insight: Identify top-performing ads by clicks. Helps decide which creatives to scale or optimize.")

    st.subheader("CPC vs CTR Performance")
    fig = px.scatter(
        ad_perf,
        x="CPC (cost per link click)",
        y="CTR (all)",
        color="Amount spent (INR)",
        hover_name="Ad name",
        title="CPC vs CTR Efficiency",
        text=ad_perf['Amount spent (INR)'].apply(lambda x: f"₹{x:,.0f}")
    )
    fig.update_traces(textposition='top center', marker=dict(size=10))
    fig = style_axes(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Quick Insight: Compare cost per click with engagement rate. Useful to identify most cost-efficient ads.")

# ----------------------------------------
# PAGE 5: Video Metrics
# ----------------------------------------
elif page == "Video Metrics" and df is not None:
    st.title("Video Metrics")

    st.subheader("Video Completion Funnel")
    video_cols = ["Video plays at 25%", "Video plays at 50%", "Video plays at 75%", "Video plays at 95%", "Video plays at 100%"]
    melted = filtered_df.melt(id_vars=["Ad name"], value_vars=video_cols, var_name="Play Stage", value_name="Plays")
    melted['Plays'] = melted['Plays'].fillna(0).astype(int)

    fig = px.bar(
        melted, x="Play Stage", y="Plays", color="Play Stage",
        title="Video Completion Funnel",
        color_discrete_sequence=px.colors.qualitative.Safe,
        text='Plays'
    )
    fig.update_traces(texttemplate='%{text}', textposition='inside')
    fig = style_axes(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Quick Insight: Track drop-off points in video ads. Helps improve creative flow and messaging.")

    if "ThruPlays" in filtered_df.columns and "Cost per ThruPlay" in filtered_df.columns:
        st.subheader("ThruPlay Efficiency Bubble Chart")

        filtered_df['Cost per ThruPlay'] = filtered_df['Cost per ThruPlay'].fillna(0).astype(int)
        filtered_df['ThruPlays'] = filtered_df['ThruPlays'].fillna(0).astype(int)
        filtered_df['Impressions'] = filtered_df['Impressions'].fillna(0).astype(int)

        fig = px.scatter(
            filtered_df,
            x="ThruPlays",
            y="Cost per ThruPlay",
            size="Impressions",
            color="Campaign name",
            hover_name="Ad name",
            hover_data={'ThruPlays': True, 'Cost per ThruPlay': True, 'Impressions': True},
            text=filtered_df['Cost per ThruPlay'].apply(lambda x: f"₹{x:,}"),
            size_max=40,
            title="ThruPlays vs Cost per ThruPlay"
        )
        fig.update_traces(
            textposition='top center',
            marker=dict(sizemode='area',
                        sizeref=2.*max(filtered_df['Impressions'])/(40.**2),
                        line=dict(width=1, color='DarkSlateGrey'))
        )
        fig = style_axes(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.info("Quick Insight: Evaluates video ad efficiency. Large bubbles = high impressions, position shows cost vs completed plays. Helps optimize budget and creative performance.")
