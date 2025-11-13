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
st.sidebar.title("📊 Campaign Analytics Dashboard")
st.sidebar.markdown("Built with ❤️ by **Vibin — Senior Business Analyst**")
st.sidebar.markdown(f"🗓️ Created on: {datetime.now().strftime('%d %b %Y')}")

page = st.sidebar.radio(
    "Navigate to",
    [
        "🏠 Case Study Overview",
        "📈 Campaign Overview",
        "👥 Audience Insights",
        "📢 Ad Performance",
        "🎥 Video Metrics"
    ]
)

# ----------------------------------------
# File Upload
# ----------------------------------------
st.sidebar.subheader("📂 Upload Campaign Data")
uploaded_file = st.sidebar.file_uploader("Upload your Meta Ads CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")

    # Clean & prep data
    df.columns = df.columns.str.strip()
    df['Reporting starts'] = pd.to_datetime(df['Reporting starts'], errors='coerce')
    df['Reporting ends'] = pd.to_datetime(df['Reporting ends'], errors='coerce')

    # ----------------------------------------
    # Filters / Slicers
    # ----------------------------------------
    st.sidebar.subheader("🎚️ Filters")
    campaign_filter = st.sidebar.multiselect("Campaign Name", options=df['Campaign name'].unique())
    city_filter = st.sidebar.multiselect("City", options=df['City'].dropna().unique())
    age_filter = st.sidebar.multiselect("Age Group", options=df['Age'].dropna().unique())
    gender_filter = st.sidebar.multiselect("Gender", options=df['Gender'].dropna().unique())

    start_date = st.sidebar.date_input(
        "Start Date",
        df['Reporting starts'].min().date() if not df.empty else None
    )
    end_date = st.sidebar.date_input(
        "End Date",
        df['Reporting ends'].max().date() if not df.empty else None
    )

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
    st.warning("⚠️ Please upload your dataset from the sidebar to begin analysis.")


# ----------------------------------------
# PAGE 1: Case Study Overview
# ----------------------------------------
if page == "🏠 Case Study Overview":
    st.title("📚 Marketing Campaign Case Study")
    st.markdown("""
    Welcome to the **Marketing Campaign Performance Dashboard** — an interactive workspace to explore Meta Ads performance across regions, demographics, and ad sets.

    **Use Case Highlights:**
    - Evaluate **budget effectiveness** and **ROI trends**  
    - Analyze audience segments driving conversions  
    - Compare creatives for engagement and reach  
    - Identify performance gaps to improve delivery  

    **Tools Used:** Streamlit | Plotly | Pandas | Python
    """)

    # YouTube preview embedded in an expandable container
    with st.expander("▶️ Watch Case Study Preview"):
        st.video("https://www.youtube.com/watch?v=0d6oY8G5e5c")

    st.markdown("---")
    st.caption("👨‍💻 Created by **Vibin**, Senior Business Analyst | Date: " + datetime.now().strftime("%d %B %Y"))

    if df is not None:
        st.success("✅ Dataset uploaded successfully. Navigate through the tabs on the left to explore dashboards.")
    else:
        st.info("👈 Upload your campaign CSV file using the sidebar to begin exploration.")


# ----------------------------------------
# PAGE 2: Campaign Overview
# ----------------------------------------
elif page == "📈 Campaign Overview" and df is not None:
    st.title("📈 Campaign Overview")

    total_spent = filtered_df['Amount spent (INR)'].sum()
    total_impressions = filtered_df['Impressions'].sum()
    total_reach = filtered_df['Reach'].sum()
    total_clicks = filtered_df['Link clicks'].sum()
    avg_ctr = filtered_df['CTR (all)'].mean()
    avg_cpm = filtered_df['CPM (cost per 1,000 impressions)'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Spent (INR)", f"₹{total_spent:,.0f}")
    col2.metric("📣 Total Impressions", f"{total_impressions:,.0f}")
    col3.metric("👥 Total Reach", f"{total_reach:,.0f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("🔗 Total Clicks", f"{total_clicks:,.0f}")
    col5.metric("📈 Avg CTR", f"{avg_ctr:.2f}%")
    col6.metric("🎯 Avg CPM", f"₹{avg_cpm:.2f}")

    st.markdown("---")
    st.subheader("📊 Spend Trend Over Time")
    spend_over_time = filtered_df.groupby('Reporting starts', as_index=False)['Amount spent (INR)'].sum()
    fig = px.line(
        spend_over_time, x='Reporting starts', y='Amount spent (INR)',
        markers=True, title="Campaign Spend Over Time",
        color_discrete_sequence=['#2E86C1']
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏆 Top 10 Campaigns by Spend")
    top_campaigns = filtered_df.groupby("Campaign name", as_index=False)['Amount spent (INR)'].sum().nlargest(10, 'Amount spent (INR)')
    fig = px.bar(
        top_campaigns, x="Amount spent (INR)", y="Campaign name", orientation='h',
        color="Amount spent (INR)", color_continuous_scale="Blues",
        title="Top Campaigns by Spend"
    )
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------
# PAGE 3: Audience Insights
# ----------------------------------------
elif page == "👥 Audience Insights" and df is not None:
    st.title("👥 Audience Insights")

    st.subheader("🎯 Clicks by Age & Gender")
    agg = filtered_df.groupby(["Age", "Gender"], as_index=False)["Link clicks"].sum()
    fig = px.bar(
        agg, x="Age", y="Link clicks", color="Gender", barmode="group",
        title="Clicks Distribution by Age and Gender",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏙️ Top 10 Cities by Ad Spend")
    city_perf = filtered_df.groupby("City", as_index=False)['Amount spent (INR)'].sum().nlargest(10, 'Amount spent (INR)')
    fig = px.bar(
        city_perf, x="City", y="Amount spent (INR)", color="City",
        title="Top Cities by Ad Spend", color_discrete_sequence=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------
# PAGE 4: Ad Performance
# ----------------------------------------
elif page == "📢 Ad Performance" and df is not None:
    st.title("📢 Ad-Level Performance")

    ad_perf = filtered_df.groupby("Ad name", as_index=False).agg({
        "Link clicks": "sum",
        "Impressions": "sum",
        "Amount spent (INR)": "sum",
        "CTR (all)": "mean",
        "CPC (cost per link click)": "mean"
    }).nlargest(10, 'Link clicks')

    st.subheader("🏅 Top 10 Ads by Clicks")
    fig = px.bar(
        ad_perf, x="Link clicks", y="Ad name", orientation='h',
        color="CTR (all)", color_continuous_scale="Agsunset",
        title="Top Performing Ads"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("⚖️ CPC vs CTR Performance")
    fig = px.scatter(
        ad_perf, x="CPC (cost per link click)", y="CTR (all)",
        color="Amount spent (INR)", size="Link clicks",
        hover_name="Ad name", title="CPC vs CTR Efficiency"
    )
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------
# PAGE 5: Video Metrics
# ----------------------------------------
elif page == "🎥 Video Metrics" and df is not None:
    st.title("🎥 Video Metrics")

    st.subheader("📺 Video Completion Funnel")
    video_cols = ["Video plays at 25%", "Video plays at 50%", "Video plays at 75%", "Video plays at 95%", "Video plays at 100%"]
    melted = filtered_df.melt(id_vars=["Ad name"], value_vars=video_cols, var_name="Play Stage", value_name="Plays")
    fig = px.bar(
        melted, x="Play Stage", y="Plays", color="Play Stage",
        title="Video Completion Funnel", color_discrete_sequence=px.colors.qualitative.Safe
    )
    st.plotly_chart(fig, use_container_width=True)

    if "ThruPlays" in filtered_df.columns and "Cost per ThruPlay" in filtered_df.columns:
        st.subheader("🎞️ ThruPlay Efficiency")
        fig = px.scatter(
            filtered_df, x="ThruPlays", y="Cost per ThruPlay",
            color="Campaign name", hover_name="Ad name", size="Impressions",
            title="ThruPlays vs Cost per ThruPlay"
        )
        st.plotly_chart(fig, use_container_width=True)
