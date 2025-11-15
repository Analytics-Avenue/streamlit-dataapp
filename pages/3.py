import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
    return pd.read_csv(url)

df = load_data()

# ----------------------------------------------------------
# DERIVE AGE CATEGORY
# ----------------------------------------------------------
def categorize_age(x):
    if x <= 5: return "0-5 Years"
    elif x <= 10: return "6-10 Years"
    elif x <= 20: return "11-20 Years"
    return "20+ Years"

df["Age_Category"] = df["Age_Years"].apply(categorize_age)

# ----------------------------------------------------------
# PAGE TITLE
# ----------------------------------------------------------
st.title("Application 3: Price vs Property Features Analyzer")

# ----------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------
city = st.sidebar.multiselect("City", df["City"].unique())
ptype = st.sidebar.multiselect("Property Type", df["Property_Type"].unique())
bhk = st.sidebar.multiselect("BHK", sorted(df["BHK"].unique()))
bath = st.sidebar.multiselect("Bathrooms", sorted(df["Bathroom_Count"].unique()))
parking = st.sidebar.multiselect("Parking", df["Parking"].unique())
agecat = st.sidebar.multiselect("Age Category", df["Age_Category"].unique())

price_range = st.sidebar.slider(
    "Price Range",
    int(df["Price"].min()),
    int(df["Price"].max()),
    (int(df["Price"].min()), int(df["Price"].max()))
)

filtered = df.copy()

if city:
    filtered = filtered[filtered["City"].isin(city)]
if ptype:
    filtered = filtered[filtered["Property_Type"].isin(ptype)]
if bhk:
    filtered = filtered[filtered["BHK"].isin(bhk)]
if bath:
    filtered = filtered[filtered["Bathroom_Count"].isin(bath)]
if parking:
    filtered = filtered[filtered["Parking"].isin(parking)]
if agecat:
    filtered = filtered[filtered["Age_Category"].isin(agecat)]

filtered = filtered[
    (filtered["Price"] >= price_range[0]) &
    (filtered["Price"] <= price_range[1])
]

# ----------------------------------------------------------
# KPIs
# ----------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)

k1.metric("Average Price", f"{filtered['Price'].mean():,.0f}")
k2.metric("Avg Price per SqFt", f"{(filtered['Price'] / filtered['Area_sqft']).mean():,.0f}")

k3.metric("Top Property Type",
          filtered["Property_Type"].mode()[0] if not filtered.empty else "NA")

k4.metric("Top City by Listings",
          filtered["City"].mode()[0] if not filtered.empty else "NA")

st.markdown("---")

# ----------------------------------------------------------
# CHART 1 — PRICE VS BHK
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
group = filtered.groupby("BHK")["Price"].mean()

ax.bar(group.index.astype(str), group.values, color=["#2471A3", "#AF7AC5", "#48C9B0", "#F5B041", "#CD6155"])
ax.set_xlabel("BHK", fontweight="bold")
ax.set_ylabel("Avg Price", fontweight="bold")
ax.set_title("Price vs BHK", fontsize=13, fontweight="bold")

for i, v in enumerate(group.values):
    ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)

ax.spines["bottom"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)

st.pyplot(fig)
st.markdown("**Purpose:** Identify how BHK variations impact pricing.")
st.markdown("**Quick Tip:** 3BHK and 2BHK units usually show strongest price stability.")

st.markdown("---")

# ----------------------------------------------------------
# CHART 2 — PRICE VS AGE CATEGORY
# ----------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(7, 5))
group2 = filtered.groupby("Age_Category")["Price"].mean()

ax2.plot(group2.index, group2.values, marker="o", linewidth=3, color="#1F618D")
ax2.set_xlabel("Age Category", fontweight="bold")
ax2.set_ylabel("Avg Price", fontweight="bold")
ax2.set_title("Impact of Property Age on Price", fontsize=13, fontweight="bold")

for i, val in enumerate(group2.values):
    ax2.text(i, val, f"{int(val):,}", ha="center", va="bottom", fontsize=9)

ax2.spines["bottom"].set_linewidth(2)
ax2.spines["left"].set_linewidth(2)

st.pyplot(fig2)

st.markdown("**Purpose:** Understand depreciation patterns across property lifecycle.")

st.markdown("---")

# ----------------------------------------------------------
# CHART 3 — SCATTER PRICE VS AREA
# ----------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(7, 5))
ax3.scatter(filtered["Area_sqft"], filtered["Price"], alpha=0.6, color="#5DADE2")

ax3.set_xlabel("Area (SqFt)", fontweight="bold")
ax3.set_ylabel("Price", fontweight="bold")
ax3.set_title("Price vs Built-up Area", fontsize=13, fontweight="bold")

ax3.spines["bottom"].set_linewidth(2)
ax3.spines["left"].set_linewidth(2)

st.pyplot(fig3)

st.markdown("**Purpose:** Analyse relationship between built-up area and valuation.")

st.markdown("---")

# ----------------------------------------------------------
# DATA TABLE + DOWNLOAD
# ----------------------------------------------------------
st.subheader("Filtered Dataset")
st.dataframe(filtered)

csv = filtered.to_csv(index=False)
st.download_button("Download Filtered Data", csv, "app3_filtered_data.csv", "text/csv")
