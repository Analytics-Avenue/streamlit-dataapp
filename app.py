import streamlit as st
import pandas as pd

st.title("My First Streamlit App")

st.write("Upload a CSV file and I’ll show what’s inside:")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head(10))
    st.success("File uploaded successfully!")
else:
    st.info("Please upload a CSV file to begin.")
