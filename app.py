import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Marketing Analytics", layout="wide")

st.title("Marketing Analytics")
st.write("Marketing analytics helps brands understand which campaigns actually drive results. Upload your campaign data to analyze spend trends and performance patterns.")

# Default column names (your data dictionary)
default_columns = {
    "Date": "Date when ad performance data was recorded",
    "Amount Spent (INR)": "Total spend for the campaign/ad",
    "Campaign": "Marketing campaign name on social platforms",
    "Region": "Geographical or segmentation field"
}

st.sidebar.header("Data Dictionary")
for key, desc in default_columns.items():
    st.sidebar.write(f"**{key}** → {desc}")

# Upload section
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
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
            # Rename columns based on mapping
            df_renamed = df.rename(columns={v: k for k, v in column_mapping.items()})
            st.success("✅ Columns mapped successfully!")
            st.subheader("📋 Mapped Data Preview")
            st.dataframe(df_renamed.head())

            st.subheader("Generated Visuals")

            # 1️⃣ Spend by Campaign
            if "Campaign" in df_renamed.columns and "Amount Spent (INR)" in df_renamed.columns:
                st.write("**Amount Spent by Campaign**")
                campaign_spend = df_renamed.groupby("Campaign")["Amount Spent (INR)"].sum().sort_values(ascending=False)
                st.bar_chart(campaign_spend)
            else:
                st.warning("Missing either 'Campaign' or 'Amount Spent (INR)' columns.")

            # 2️⃣ Spend over Time
            if "Date" in df_renamed.columns and "Amount Spent (INR)" in df_renamed.columns:
                st.write("**Amount Spent over Time**")
                df_renamed["Date"] = pd.to_datetime(df_renamed["Date"], errors="coerce")
                date_spend = df_renamed.groupby("Date")["Amount Spent (INR)"].sum().sort_index()
                st.line_chart(date_spend)
            else:
                st.warning("Missing either 'Date' or 'Amount Spent (INR)' columns.")

            # 3️⃣ Spend by Region
            if "Region" in df_renamed.columns and "Amount Spent (INR)" in df_renamed.columns:
                st.write("**Amount Spent by Region**")
                region_spend = df_renamed.groupby("Region")["Amount Spent (INR)"].sum().sort_values(ascending=False)
                st.bar_chart(region_spend)
            else:
                st.info("Add a 'Region' column mapping to enable regional spend chart.")

        except Exception as e:
            st.error(f"⚠️ Error generating visuals: {e}")

else:
    st.info("⬆️ Please upload a CSV file to get started.")
