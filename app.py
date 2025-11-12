import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Smart Data Visualizer")

# --- Step 1: Define your default data dictionary ---
data_dictionary = {
    "Date": "Transaction or event date",
    "Sales": "Total sales value",
    "Category": "Product category",
    "Region": "Geographic region"
}

st.sidebar.header("Default Data Dictionary")
st.sidebar.write(pd.DataFrame(list(data_dictionary.items()), columns=["Column", "Description"]))

# --- Step 2: Upload section ---
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of uploaded data:")
    st.dataframe(df.head())

    # --- Step 3: Align columns with default data dictionary ---
    st.subheader("Column Mapping")
    column_mapping = {}
    for default_col in data_dictionary.keys():
        selected_col = st.selectbox(f"Select column for '{default_col}'", df.columns, key=default_col)
        column_mapping[default_col] = selected_col

    if st.button("Generate Visuals"):
        df_renamed = df.rename(columns=column_mapping)

        # --- Step 4: Generate visuals ---
        st.subheader("Generated Visuals")

        # Example 1: Sales by Category
        if "Category" in df_renamed.columns and "Sales" in df_renamed.columns:
            st.write("**Sales by Category**")
            cat_sales = df_renamed.groupby("Category")["Sales"].sum()
            st.bar_chart(cat_sales)

        # Example 2: Sales over Time
        if "Date" in df_renamed.columns and "Sales" in df_renamed.columns:
            st.write("**Sales over Time**")
            df_renamed["Date"] = pd.to_datetime(df_renamed["Date"], errors="coerce")
            st.line_chart(df_renamed.groupby("Date")["Sales"].sum())

else:
    st.info("Please upload a CSV file to continue.")
