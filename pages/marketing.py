import streamlit as st
import plotly.express as px
import pandas as pd

st.title("📊 Marketing Analytics Dashboard")

st.sidebar.header("Filters")
channel = st.sidebar.multiselect("Select Channel", ["Instagram", "YouTube", "Google Ads"])
metric = st.sidebar.selectbox("Metric", ["CTR", "CPC", "Conversion Rate"])

# Sample data
data = {
    "Channel": ["Instagram", "YouTube", "Google Ads"],
    "CTR": [5.2, 7.1, 4.6],
    "CPC": [1.2, 2.0, 1.5],
    "Conversion Rate": [3.1, 4.5, 2.8],
}
df = pd.DataFrame(data)

fig = px.bar(df, x="Channel", y=metric, color="Channel", title=f"{metric} by Channel")
st.plotly_chart(fig, use_container_width=True)
