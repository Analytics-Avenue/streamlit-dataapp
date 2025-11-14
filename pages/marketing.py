import streamlit as st
import pandas as pd
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Marketing Analytics Hub", layout="wide")

# -----------------------------------------------------------
# PAGE LIST
# -----------------------------------------------------------
PAGES = [
    "About Marketing Analytics",
    "Case Study Overview",
    "Campaign Overview",
    "Audience Insights",
    "Ad Performance",
    "Video Metrics"
]

page = st.sidebar.selectbox("Navigate", PAGES)

# -----------------------------------------------------------
# GITHUB DEFAULT DATA
# -----------------------------------------------------------
DEFAULT_DATA_URL = "https://github.com/Analytics-Avenue/streamlit-dataapp/blob/main/datasets/Marketing_Analytics.csv"

def load_default():
    try:
        return pd.read_csv(DEFAULT_DATA_URL)
    except:
        st.error("Could not load default dataset from GitHub.")
        return None

# -----------------------------------------------------------
# SAMPLE CSV (Column names only)
# -----------------------------------------------------------
REQUIRED_COLUMNS = [
    "Campaign Name", "Adset Name", "Ad Name",
    "Impressions", "Reach", "Clicks", "Spend",
    "ThruPlays", "Video Plays",
    "City", "Age", "Gender"
]

sample_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
st.sidebar.download_button(
    "Download Sample CSV",
    data=sample_df.to_csv(index=False),
    file_name="sample_template.csv",
    mime="text/csv"
)

# -----------------------------------------------------------
# UPLOAD DATASET OR LOAD DEFAULT
# -----------------------------------------------------------
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = load_default()

if df is None:
    st.stop()

# -----------------------------------------------------------
# COLUMN MAPPING
# -----------------------------------------------------------
st.sidebar.write("### Column Mapping")

mapped = {}
for col in REQUIRED_COLUMNS:
    options = ["-- Not Available --"] + list(df.columns)
    sel = st.sidebar.selectbox(col, options)
    mapped[col] = None if sel == "-- Not Available --" else sel

def col(name):
    return mapped.get(name)

# -----------------------------------------------------------
# FILTER SETUP (used in multiple pages)
# -----------------------------------------------------------
filtered = df.copy()

if col("Campaign Name"):
    campaign_filter = st.sidebar.multiselect(
        "Campaign", sorted(df[col("Campaign Name")].unique())
    )
    if campaign_filter:
        filtered = filtered[filtered[col("Campaign Name")].isin(campaign_filter)]

if col("Adset Name"):
    adset_filter = st.sidebar.multiselect(
        "Adset", sorted(df[col("Adset Name")].unique())
    )
    if adset_filter:
        filtered = filtered[filtered[col("Adset Name")].isin(adset_filter)]

if col("Ad Name"):
    ad_filter = st.sidebar.multiselect(
        "Ad Name", sorted(df[col("Ad Name")].unique())
    )
    if ad_filter:
        filtered = filtered[filtered[col("Ad Name")].isin(ad_filter)]

# -----------------------------------------------------------
# PAGE 1: ABOUT MARKETING ANALYTICS
# -----------------------------------------------------------
if page == "About Marketing Analytics":
    st.title("About Marketing Analytics")

    st.markdown("""
    Marketing analytics helps organizations evaluate their marketing performance using data.
    
    Key questions answered:
    • How many people saw my ad?  
    • How many engaged with it?  
    • Which audience performed the best?  
    • What was the cost of acquiring a customer?  
    • How effective was the video content?  

    Benefits:
    • Better targeting  
    • Lower ad spend  
    • Improved ROI  
    • Data-driven creative optimization  
    """)

    st.subheader("Architecture of a Modern Marketing Analytics System")
    try:
        img = Image.open("images/architecture.jpg")
        st.image(img, use_column_width=True)
    except:
        st.info("Place architecture.jpg inside /images")

    st.subheader("Marketing Funnel")
    try:
        img2 = Image.open("images/marketing_funnel.jpg")
        st.image(img2, use_column_width=True)
    except:
        st.info("Place marketing_funnel.jpg inside /images")

# -----------------------------------------------------------
# PAGE 2: CASE STUDY OVERVIEW
# -----------------------------------------------------------
elif page == "Case Study Overview":
    st.title("Case Study Overview")

    st.markdown("""
    This dashboard demonstrates:
    • Campaign hierarchy  
    • Audience behavior  
    • Engagement funnel  
    • Creative effectiveness  
    • Video performance  
    """)

# -----------------------------------------------------------
# PAGE 3: CAMPAIGN OVERVIEW
# -----------------------------------------------------------
elif page == "Campaign Overview":
    st.title("Campaign Overview")

    col1, col2, col3, col4 = st.columns(4)

    if col("Impressions"):
        col1.metric("Total Impressions", filtered[col("Impressions")].sum())

    if col("Reach"):
        col2.metric("Total Reach", filtered[col("Reach")].sum())

    if col("Clicks"):
        col3.metric("Total Clicks", filtered[col("Clicks")].sum())

    if col("Spend"):
        col4.metric("Total Spend", f"₹{filtered[col('Spend')].sum():,.0f}")

    st.subheader("Campaign Insights Image")
    try:
        img = Image.open("images/campaign_overview.jpg")
        st.image(img, use_column_width=True)
    except:
        st.info("Place campaign_overview.jpg in /images folder")

# -----------------------------------------------------------
# PAGE 4: AUDIENCE INSIGHTS
# -----------------------------------------------------------
elif page == "Audience Insights":
    st.title("Audience Insights")

    # Gender Chart
    if col("Gender") and col("Impressions"):
        st.subheader("Gender Split")

        gender_df = filtered.groupby(col("Gender"))[col("Impressions")].sum()

        # Required: female = pink, male = blue
        color_map = {
            "Male": "#3b82f6",
            "Female": "#ec4899"
        }
        colors = [color_map.get(g, "#7f7f7f") for g in gender_df.index]

        st.bar_chart(gender_df, color=colors)

    # Age Chart
    if col("Age") and col("Impressions"):
        st.subheader("Age Distribution")
        age_df = filtered.groupby(col("Age"))[col("Impressions")].sum()
        st.bar_chart(age_df)

    # City Chart
    if col("City") and col("Impressions"):
        st.subheader("Top Cities")
        city_df = filtered.groupby(col("City"))[col("Impressions")].sum().sort_values(ascending=False)
        st.bar_chart(city_df)

# -----------------------------------------------------------
# PAGE 5: AD PERFORMANCE
# -----------------------------------------------------------
elif page == "Ad Performance":
    st.title("Ad Performance")

    # CTR
    if col("Clicks") and col("Impressions"):
        st.subheader("CTR Trend")

        filtered["CTR"] = (filtered[col("Clicks")] / filtered[col("Impressions")]) * 100
        st.line_chart(filtered["CTR"])

    # CPC
    if col("Clicks") and col("Spend"):
        st.subheader("Cost Per Click")
        filtered["CPC"] = filtered[col("Spend")] / filtered[col("Clicks")]
        st.line_chart(filtered["CPC"])

# -----------------------------------------------------------
# PAGE 6: VIDEO METRICS
# -----------------------------------------------------------
elif page == "Video Metrics":
    st.title("Video Metrics")

    # ThruPlay Rate
    if col("ThruPlays") and col("Video Plays"):
        st.subheader("ThruPlay Rate")
        filtered["ThruPlay Rate"] = (filtered[col("ThruPlays")] / filtered[col("Video Plays")]) * 100
        st.area_chart(filtered["ThruPlay Rate"])

    # Funnel Chart (Video Completion)
    if col("Video Plays") and col("ThruPlays"):
        st.subheader("Video Completion Funnel")

        funnel_df = pd.DataFrame({
            "Stage": ["Video Played", "ThruPlays"],
            "Count": [
                filtered[col("Video Plays")].sum(),
                filtered[col("ThruPlays")].sum()
            ]
        })

        st.bar_chart(funnel_df.set_index("Stage"))

