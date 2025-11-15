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
    df = pd.read_csv(url)

    # Create Age Category Bins
    df["Age_Years_Category"] = pd.cut(
        df["Age_Years"],
        bins=[0, 5, 10, 20, 50, 200],
        labels=["0-5 yrs", "5-10 yrs", "10-20 yrs", "20-50 yrs", "50+ yrs"],
        include_lowest=True
    )

    return df

df = load_data()

st.title("Application 3: Price vs Property Features Analyzer")

# ----------------------------
# Filters (Mapped to dataset)
# ----------------------------
city = st.sidebar.multiselect("City", df["City"].unique())
ptype = st.sidebar.multiselect("Property Type", df["Property_Type"].unique())
bed = st.sidebar.multiselect("Bedrooms (BHK)", sorted(df["BHK"].unique()))
bath = st.sidebar.multiselect("Bathrooms", sorted(df["Bathroom_Count"].unique()))
parking = st.sidebar.multiselect("Parking", df["Parking"].unique())
age = st.sidebar.multiselect("Age Category", df["Age_Years_Category"].unique())
price_range = st.sidebar.slider(
    "Price Range (INR)",
    int(df.Price.min()),
    int(df.Price.max()),
    (int(df.Price.min()), int(df.Price.max()))
)

filtered = df.copy()

# Apply Filters
if city:
    filtered = filtered[filtered.City.isin(city)]
if ptype:
    filtered = filtered[filtered.Property_Type.isin(ptype)]
if bed:
    filtered = filtered[filtered.BHK.isin(bed)]
if bath:
    filtered = filtered[filtered.Bathroom_Count.isin(bath)]
if parking:
    filtered = filtered[filtered.Parking.isin(parking)]
if age:
    filtered = filtered[filtered.Age_Years_Category.isin(age)]

filtered = filtered[
    (filtered.Price >= price_range[0]) & (filtered.Price <= price_range[1])
]

# ----------------------------
# KPIs
# ----------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Average Price", f"{filtered.Price.mean():,.0f}")
kpi2.metric("Avg Price per SqFt", f"{filtered.Price_per_sqft.mean():,.0f}")
kpi3.metric("Top Property Type", filtered.Property_Type.mode()[0] if not filtered.empty else "NA")
kpi4.metric("Top City by Listings", filtered.City.mode()[0] if not filtered.empty else "NA")

st.markdown("---")

# ----------------------------
# CHART 1 – Price vs Bedrooms (BHK)
# ----------------------------
fig, ax = plt.subplots(figsize=(8, 5))
grouped = filtered.groupby("BHK")["Price"].mean()
ax.bar(grouped.index.astype(str), grouped.values, color="skyblue")

ax.set_xlabel("BHK", fontweight="bold")
ax.set_ylabel("Avg Price", fontweight="bold")
ax.set_title("Price vs Bedrooms (BHK)")

# Data labels
for i, v in enumerate(grouped.values):
    ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)

st.pyplot(fig)
st.markdown("**Purpose:** Identify how BHK count influences price.")
st.markdown("**Quick Tip:** 3BHK units show highest stability in valuation.")

st.markdown("---")

# ----------------------------
# CHART 2 – Price vs Age Category
# ----------------------------
fig2, ax2 = plt.subplots(figsize=(8, 5))
grouped2 = filtered.groupby("Age_Years_Category")["Price"].mean()

ax2.plot(grouped2.index.astype(str), grouped2.values, marker="o", linewidth=2, color="orange")
ax2.set_xlabel("Age Category", fontweight="bold")
ax2.set_ylabel("Avg Price", fontweight="bold")
ax2.set_title("Price vs Property Age")

# Data labels
for i, val in enumerate(grouped2.values):
    ax2.text(i, val, f"{int(val):,}", ha="center", va="bottom", fontsize=9)

st.pyplot(fig2)
st.markdown("**Purpose:** Understand depreciation or premium based on age.")

st.markdown("---")

# ----------------------------
# CHART 3 – Scatter Price vs SqFt
# ----------------------------
fig3, ax3 = plt.subplots(figsize=(8, 5))

ax3.scatter(
    filtered.Area_sqft,
    filtered.Price,
    alpha=0.6,
    color="purple"
)

ax3.set_xlabel("SqFt", fontweight="bold")
ax3.set_ylabel("Price", fontweight="bold")
ax3.set_title("Price vs SqFt")

st.pyplot(fig3)

st.markdown("**Purpose:** Shows price variation with property size.")

st.markdown("---")

# ----------------------------
# Data Table + Download Option
# ----------------------------
st.subheader("Filtered Dataset")
st.dataframe(filtered)

csv = filtered.to_csv(index=False)
st.download_button(
    "Download Filtered Data",
    csv,
    "app3_filtered_data.csv",
    "text/csv"
)
