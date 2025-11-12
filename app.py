import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(
    page_title="Marketing Campaign Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("📊 Campaign Analytics Dashboard")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Case Study Overview", "📈 Campaign Overview", "👥 Audience Insights", "📢 Ad Performance", "🎥 Video Metrics"]
)

# -------------------------------
# File Upload
# -------------------------------
st.sidebar.subheader("📂 Upload Campaign Data")
uploaded_file = st.sidebar.file_uploader("Upload your Facebook Ads CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")

    # Basic cleanup
    df.columns = df.columns.str.strip()
    df['Reporting starts'] = pd.to_datetime(df['Reporting starts'], errors='coerce')
    df['Reporting ends'] = pd.to_datetime(df['Reporting ends'], errors='coerce')

    # -------------------------------
    # Filters (Slicers)
    # -------------------------------
    st.sidebar.subheader("🎚️ Filters")
    campaign_filter = st.sidebar.multiselect("Campaign Name", options=df['Campaign name'].unique())
    city_filter = st.sidebar.multiselect("City", options=df['City'].dropna().unique())
    age_filter = st.sidebar.multiselect("Age", options=df['Age'].dropna().unique())
    gender_filter = st.sidebar.multiselect("Gender", options=df['Gender'].dropna().unique())
    start_date = st.sidebar.date_input("Start Date", df['Reporting starts'].min().date() if not df.empty else None)
    end_date = st.sidebar.date_input("End Date", df['Reporting ends'].max().date() if not df.empty else None)

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
    st.warning("⚠️ Upload a campaign dataset to begin.")


# -------------------------------
# PAGE 1: Case Study Overview
# -------------------------------
if page == "🏠 Case Study Overview":
    st.title("📚 Marketing Campaign Case Study")
    st.markdown("""
    This dashboard helps visualize **Meta Ads performance** across campaigns, age groups, genders, and cities.  
    Get actionable insights into how each campaign performs, which demographics respond best, and which creatives actually drive conversions.

    **Use Case:**  
    Performance monitoring, budget allocation, creative testing, and engagement optimization.

    **Built with:** Streamlit | Plotly | Pandas | Python  
    """)

    st.video("https://www.youtube.com/watch?v=0d6oY8G5e5c")  # Replace with your case study video link

    if df is not None:
        st.success("✅ Dataset uploaded. Explore tabs on the left to view interactive dashboards.")
    else:
        st.info("👈 Upload your campaign data from the sidebar to start exploring.")

# -------------------------------
# PAGE 2: Campaign Overview
# -------------------------------
elif page == "📈 Campaign Overview" and df is not None:
    st.title("📈 Campaign Overview")

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
    col5.metric("Average CTR", f"{avg_ctr:.2f}%")
    col6.metric("Average CPM", f"₹{avg_cpm:.2f}")

    st.markdown("### Campaign Spend vs Results")
    fig = px.scatter(filtered_df, x="Amount spent (INR)", y="Results", color="Campaign name",
                     size="Impressions", hover_data=["Result type"])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Impressions by Campaign")
    fig = px.bar(filtered_df, x="Campaign name", y="Impressions", color="Campaign name", title="Impressions by Campaign")
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------
# PAGE 3: Audience Insights
# -------------------------------
elif page == "👥 Audience Insights" and df is not None:
    st.title("👥 Audience Insights")

    st.markdown("### Performance by Age & Gender")
    if "Age" in filtered_df.columns and "Gender" in filtered_df.columns:
        agg = filtered_df.groupby(["Age", "Gender"], as_index=False).agg({
            "Amount spent (INR)": "sum",
            "Link clicks": "sum",
            "Impressions": "sum",
            "CTR (all)": "mean"
        })
        fig = px.bar(agg, x="Age", y="Link clicks", color="Gender", barmode="group",
                     title="Clicks by Age & Gender")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### City-Wise Performance")
    if "City" in filtered_df.columns:
        city_perf = filtered_df.groupby("City", as_index=False).agg({
            "Amount spent (INR)": "sum",
            "Link clicks": "sum",
            "CTR (all)": "mean"
        }).sort_values(by="Amount spent (INR)", ascending=False).head(10)
        fig = px.bar(city_perf, x="City", y="Amount spent (INR)", color="CTR (all)",
                     title="Top 10 Cities by Spend and CTR")
        st.plotly_chart(fig, use_container_width=True)


# -------------------------------
# PAGE 4: Ad Performance
# -------------------------------
elif page == "📢 Ad Performance" and df is not None:
    st.title("📢 Ad-Level Performance")

    st.markdown("### Top Performing Ads")
    ad_perf = filtered_df.groupby("Ad name", as_index=False).agg({
        "Link clicks": "sum",
        "Impressions": "sum",
        "Amount spent (INR)": "sum",
        "CTR (all)": "mean",
        "CPC (cost per link click)": "mean"
    }).sort_values(by="Link clicks", ascending=False).head(10)

    fig = px.bar(ad_perf, x="Ad name", y="Link clicks", color="CTR (all)", title="Top 10 Ads by Clicks")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Cost Efficiency")
    fig = px.scatter(ad_perf, x="CPC (cost per link click)", y="CTR (all)",
                     size="Link clicks", color="Amount spent (INR)",
                     hover_name="Ad name", title="Cost per Click vs CTR")
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------
# PAGE 5: Video Metrics
# -------------------------------
elif page == "🎥 Video Metrics" and df is not None:
    st.title("🎥 Video Performance Metrics")

    st.markdown("### Video Completion Rates")
    video_cols = ["Video plays at 25%", "Video plays at 50%", "Video plays at 75%", "Video plays at 95%", "Video plays at 100%"]
    melted = filtered_df.melt(id_vars=["Ad name"], value_vars=video_cols, var_name="Play Stage", value_name="Plays")

    fig = px.bar(melted, x="Play Stage", y="Plays", color="Play Stage",
                 title="Video Completion Progress Across Ads")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ThruPlay Efficiency")
    if "ThruPlays" in filtered_df.columns and "Cost per ThruPlay" in filtered_df.columns:
        fig = px.scatter(filtered_df, x="ThruPlays", y="Cost per ThruPlay",
                         color="Campaign name", hover_name="Ad name",
                         size="Impressions", title="ThruPlays vs Cost per ThruPlay")
        st.plotly_chart(fig, use_container_width=True)
