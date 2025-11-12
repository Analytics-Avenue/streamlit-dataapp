import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Visualizer", layout="wide")

st.title("Marketing Analytics")
st.write("Marketing analytics is basically how brands use data to figure out what’s working and what’s not in their marketing. Every time you see an ad on Instagram, YouTube, or Google — that ad is being tracked, measured, and analyzed somewhere")

# Default column names (your data dictionary)
default_columns = {
    "Date": "Date when ad performance data was recorded",
    "Amount Spent (INR)": "Total spend for the campaign/ad",
    "Campaign": "Marketing campaign name on social platforms",
    "Region": "Geographical or segmentation field"
}

st.sidebar.header("🧭 Data Dictionary (Default Columns)")
for key, desc in default_columns.items():
    st.sidebar.write(f"**{key}** → {desc}")

# Upload section
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📁 Uploaded Data Preview")
    st.dataframe(df.head())

    # Column mapping
    st.subheader("🧩 Map Your Columns to Default Fields")
    column_mapping = {}
    for default_col in default_columns.keys():
        selected_col = st.selectbox(
            f"Select column for '{default_col}'",
            options=["None"] + list(df.columns),
            index=0
        )
        if selected_col != "None":
            column_mapping[default_col] = selected_col

    if st.button("Generate Visuals"):
        try:
            df_renamed = df.rename(columns={v: k for k, v in column_mapping.items()})
            st.success("✅ Columns mapped successfully!")
            st.subheader("📋 Mapped Data Preview")
            st.dataframe(df_renamed.head())

            st.subheader("📈 Generated Visuals")

            # 1️⃣ Sales by Category
            if "Category" in df_renamed.columns and "Sales" in df_renamed.columns:
                st.write("**Sales by Category**")
                cat_sales = df_renamed.groupby("Category")["Sales"].sum()
                st.bar_chart(cat_sales)
            else:
                st.warning("Missing either 'Category' or 'Sales' columns.")

            # 2️⃣ Sales over Time
            if "Date" in df_renamed.columns and "Sales" in df_renamed.columns:
                st.write("**Sales over Time**")
                df_renamed["Date"] = pd.to_datetime(df_renamed["Date"], errors="coerce")
                date_sales = df_renamed.groupby("Date")["Sales"].sum().sort_index()
                st.line_chart(date_sales)
            else:
                st.warning("Missing either 'Date' or 'Sales' columns.")

            # 3️⃣ Sales by Region (optional)
            if "Region" in df_renamed.columns and "Sales" in df_renamed.columns:
                st.write("**Sales by Region**")
                region_sales = df_renamed.groupby("Region")["Sales"].sum()
                st.bar_chart(region_sales)
            else:
                st.info("Add a 'Region' column mapping to enable regional chart.")

        except Exception as e:
            st.error(f"⚠️ Error generating visuals: {e}")

else:
    st.info("⬆️ Please upload a CSV file to get started.")
