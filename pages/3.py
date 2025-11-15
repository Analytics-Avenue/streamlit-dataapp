import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/RealEstate/real_estate_data.csv"
    return pd.read_csv(url)

df = load_data()

st.title("Application 3: Price vs Property Features Analyzer")

# ----------------------------
# Filters
# ----------------------------
city = st.sidebar.multiselect("City", df["city"].unique())
ptype = st.sidebar.multiselect("Property Type", df["property_type"].unique())
bed = st.sidebar.multiselect("Bedrooms", sorted(df["bedrooms"].unique()))
bath = st.sidebar.multiselect("Bathrooms", sorted(df["bathrooms"].unique()))
parking = st.sidebar.multiselect("Parking", df["parking"].unique())
age = st.sidebar.multiselect("Age Category", df["age_category"].unique())
price_range = st.sidebar.slider("Price Range", int(df.price.min()), int(df.price.max()), (int(df.price.min()), int(df.price.max())))

filtered = df.copy()

if city:
    filtered = filtered[filtered.city.isin(city)]
if ptype:
    filtered = filtered[filtered.property_type.isin(ptype)]
if bed:
    filtered = filtered[filtered.bedrooms.isin(bed)]
if bath:
    filtered = filtered[filtered.bathrooms.isin(bath)]
if parking:
    filtered = filtered[filtered.parking.isin(parking)]
if age:
    filtered = filtered[filtered.age_category.isin(age)]

filtered = filtered[(filtered.price >= price_range[0]) & (filtered.price <= price_range[1])]

# ----------------------------
# KPIs
# ----------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Average Price", f"{filtered.price.mean():,.0f}")
kpi2.metric("Avg Price per SqFt", f"{(filtered.price / filtered.sqft).mean():.0f}")
kpi3.metric("Top Property Type", filtered.property_type.mode()[0] if not filtered.empty else "NA")
kpi4.metric("Top City by Listings", filtered.city.mode()[0] if not filtered.empty else "NA")

st.markdown("---")

# ----------------------------
# CHART 1 – Price vs Bedrooms
# ----------------------------
fig, ax = plt.subplots(figsize=(8, 5))
grouped = filtered.groupby("bedrooms")["price"].mean()
ax.bar(grouped.index.astype(str), grouped.values)
ax.set_xlabel("Bedrooms", fontweight="bold")
ax.set_ylabel("Avg Price", fontweight="bold")
ax.set_title("Price vs Bedrooms")

for i, v in enumerate(grouped.values):
    ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)

st.pyplot(fig)
st.markdown("**Purpose:** Identify how room count impacts valuation.")
st.markdown("**Quick Tip:** 3BHK units usually have strongest resale value.")

st.markdown("---")

# ----------------------------
# CHART 2 – Price vs Age Category
# ----------------------------
fig2, ax2 = plt.subplots(figsize=(8, 5))
grouped2 = filtered.groupby("age_category")["price"].mean()
ax2.plot(grouped2.index, grouped2.values, marker="o", linewidth=2)
ax2.set_xlabel("Age Category", fontweight="bold")
ax2.set_ylabel("Avg Price", fontweight="bold")
ax2.set_title("Price vs Property Age")

for i, val in enumerate(grouped2.values):
    ax2.text(i, val, f"{int(val):,}", ha="center", va="bottom", fontsize=9)

st.pyplot(fig2)
st.markdown("**Purpose:** Understand depreciation/appreciation patterns.")

st.markdown("---")

# ----------------------------
# CHART 3 – Scatter Price vs Sqft
# ----------------------------
fig3, ax3 = plt.subplots(figsize=(8, 5))
scatter = ax3.scatter(filtered.sqft, filtered.price, alpha=0.6)
ax3.set_xlabel("SqFt", fontweight="bold")
ax3.set_ylabel("Price", fontweight="bold")
ax3.set_title("Price vs SqFt")

st.pyplot(fig3)

st.markdown("**Purpose:** Visualize relationship between size and price.")
st.markdown("---")

# ----------------------------
# Data Table + Download Option
# ----------------------------
st.subheader("Filtered Dataset")
st.dataframe(filtered)

csv = filtered.to_csv(index=False)
st.download_button("Download Filtered Data", csv, "app3_filtered_data.csv", "text/csv")

