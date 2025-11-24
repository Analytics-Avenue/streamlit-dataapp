import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO

st.set_page_config(page_title="Marketing Campaign Performance Analyzer", layout="wide")

# Hide default sidebar navigation (optional)
st.markdown("""<style>[data-testid="stSidebarNav"]{display:none;}</style>""", unsafe_allow_html=True)



# -------------------------
# Company Logo + Name
# -------------------------
logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
st.markdown(f"""
<div style="display: flex; align-items: center;">
    <img src="{logo_url}" width="60" style="margin-right:10px;">
    <div style="line-height:1;">
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0; padding:0;">Analytics Avenue &</div>
        <div style="color:#064b86; font-size:36px; font-weight:bold; margin:0; padding:0;">Advanced Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# STYLING
# -------------------------------
st.markdown("""
<style>
.big-header {
    font-size: 36px;
    font-weight: 900;
    color: black !important;
    background: none !important;
    -webkit-background-clip: unset !important;
    -webkit-text-fill-color: unset !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

    /* Global font + smoothness */
    body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }

    /* Section headers with animated underline */
    .section-title {
        font-size: 26px;
        font-weight: 800;
        padding-bottom: 6px;
        margin-top: 20px;
        position: relative;
        color: #064b86;
    }
    .section-title::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        height: 3px;
        width: 0%;
        background: #064b86;
        transition: width 0.4s;
    }
    .section-title:hover::after {
        width: 35%;
    }

    /* Soft fade-in animation for tab content */
    .fade-in {
        animation: fadeIn 0.5s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Minimal cards */
    .dict-card, .kpi-card {
        background: white;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #e5eaf0;
        box-shadow: 0 3px 14px rgba(0,0,0,0.06);
        margin-bottom: 15px;
        transition: all 0.2s ease-in-out;
    }
    .dict-card:hover, .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 18px rgba(0,0,0,0.12);
    }

    /* Table styling */
    .dataframe th {
        background: #064b86 !important;
        color: white !important;
        padding: 10px !important;
        text-align: left !important;
    }

    .dataframe td {
        padding: 8px !important;
        border-bottom: 1px solid #e6e6e6 !important;
    }

    .dataframe tbody tr:hover {
        background: #f3faff !important;
        transition: background 0.25s ease-in-out;
    }

    /* Button styling */
    .stButton>button {
        background: #064b86;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        border: none;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: #0b6ab8;
        transform: translateY(-2px);
    }

    /* Radio buttons & checkbox styling */
    .stRadio label, .stCheckbox label {
        font-weight: 600 !important;
    }

    /* Download button styling */
    .stDownloadButton>button {
        background: #064b86 !important;
        color: white !important;
        border-radius: 8px;
        padding: 8px 16px !important;
        border: none;
        font-weight: 600;
    }
    .stDownloadButton>button:hover {
        background: #0b6ab8 !important;
    }

</style>
""", unsafe_allow_html=True)


st.markdown("<div class='big-header'>Marketing Campaign Performance Analyzer</div>", unsafe_allow_html=True)

# -------------------------------
# REQUIRED COLUMNS
# -------------------------------
REQUIRED_COLS = [
    'Campaign','Channel','Date','Impressions',
    'Clicks','Leads','Conversions','Spend'
]

# Auto-map helper (improved)
FB_MAP = {
    "Campaign": ["Campaign name", "campaign_name"],
    "Channel": ["Page Name", "page_name", "Channel"],
    "Date": ["Date", "Day"],
    "Impressions": ["Impressions", "impressions"],
    "Clicks": ["Link clicks", "clicks"],
    "Leads": ["Results", "leads"],
    "Conversions": ["Conversions", "Website conversions"],
    "Spend": ["Amount spent (INR)", "Spend"]
}

def auto_map_columns(df):
    rename_dict = {}
    for req, possible_cols in FB_MAP.items():
        for col in df.columns:
            if col.strip() in possible_cols:
                rename_dict[col] = req
                break
    return df.rename(columns=rename_dict)


# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["Overview","Important Attributes","Application"])

# -------------------------------
# TAB 1: OVERVIEW
# -------------------------------
with tab1:
    st.markdown("### Overview")
    st.markdown(
        "<div class='card'>This app tracks marketing campaign performance across channels, helping measure engagement, conversions, and ROI.</div>",
        unsafe_allow_html=True
    )

    st.markdown("### KPIs")
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown("<div class='metric-card'>Total Impressions</div>", unsafe_allow_html=True)
    k2.markdown("<div class='metric-card'>Total Clicks</div>", unsafe_allow_html=True)
    k3.markdown("<div class='metric-card'>Total Leads</div>", unsafe_allow_html=True)
    k4.markdown("<div class='metric-card'>Total Spend</div>", unsafe_allow_html=True)

# -------------------------------
with tab2:
    st.markdown("## Important Attributes")
    st.markdown("This section explains the meaning of the mandatory columns and identifies the independent and dependent variables that drive analysis.")

    st.markdown("### Required Column Data Dictionary")
    
    data_dict = {
        "Campaign": "Name of the marketing campaign or ad set being analyzed.",
        "Channel": "Marketing channel such as Facebook, Google, Instagram, Email, etc.",
        "Date": "Date of the marketing activity logged.",
        "Impressions": "Total number of times the ad was shown.",
        "Clicks": "Number of users who clicked the ad.",
        "Leads": "Number of users who submitted interest forms or became prospects.",
        "Conversions": "Number of leads who completed the target action (purchase, signup, etc.).",
        "Spend": "Amount spent on the marketing campaign for that period."
    }

    df_dict = pd.DataFrame(
        [{"Attribute": k, "Description": v} for k, v in data_dict.items()]
    )

    st.dataframe(df_dict, use_container_width=True)

    # -------------------------------
    # VARIABLE CLASSIFICATION
    # -------------------------------
    st.markdown("### Independent & Dependent Variables")

    st.markdown("#### Independent Variables (Inputs)")
    indep = [
        "Campaign",
        "Channel",
        "Date",
        "Impressions",
        "Clicks",
        "Spend"
    ]

    st.markdown("""
    These variables influence marketing outcomes and are considered controllable or measurable inputs.
    """)

    st.write(pd.DataFrame({"Independent Variables": indep}))

    st.markdown("#### Dependent Variables (Outputs)")
    dep = [
        "Leads",
        "Conversions"
    ]

    st.markdown("""
    These represent the outcomes of the marketing activities and are directly affected by changes in the independent variables.
    """)

    st.write(pd.DataFrame({"Dependent Variables": dep}))

    st.markdown('<div class="fade-in">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Important Attributes</div>', unsafe_allow_html=True)
    st.markdown("This section explains the meaning of the mandatory columns...")

    st.markdown('<div class="section-title">Required Column Data Dictionary</div>', unsafe_allow_html=True)
    st.markdown('<div class="dict-card">', unsafe_allow_html=True)
    st.dataframe(df_dict, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Independent Variables</div>', unsafe_allow_html=True)
    st.markdown('<div class="dict-card">', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"Independent Variables": indep}), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Dependent Variables</div>', unsafe_allow_html=True)
    st.markdown('<div class="dict-card">', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"Dependent Variables": dep}), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)



# TAB 2: APPLICATION
# -------------------------------
with tab3:

    st.markdown("### Step 1: Load Dataset")

    df = None

    mode = st.radio("Select Dataset Option:",
        ["Default Dataset","Upload CSV","Upload CSV + Column Mapping"],
        horizontal=True
    )

    # -------------------------------
    # DEFAULT DATASET
    # -------------------------------
    if mode == "Default Dataset":
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/marketing.csv"

        try:
            df = pd.read_csv(URL)
            df.columns = df.columns.str.strip()

            df = auto_map_columns(df)

            # If any required column missing, create empty safe columns
            for col in REQUIRED_COLS:
                if col not in df.columns:
                    df[col] = 0

            st.success("Default dataset loaded successfully.")

        except Exception as e:
            st.error(f"Failed to load default dataset: {e}")

    # -------------------------------
    # UPLOAD CSV
    # -------------------------------
    elif mode == "Upload CSV":
        st.markdown("#### Download Sample CSV for Reference")
        URL = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/datasets/marketing_analytics/marketing.csv"
        try:
            # Load default dataset
            sample_df = pd.read_csv(URL).head(5)  # Take first 5 rows
            sample_csv = sample_df.to_csv(index=False)
            st.download_button("Download Sample CSV", sample_csv, "sample_dataset.csv", "text/csv")
        except Exception as e:
            st.info(f"Sample CSV unavailable: {e}")
        
        file = st.file_uploader("Upload your CSV file", type=["csv"])

        if file:
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()

            df = auto_map_columns(df)

            # Fill missing required columns
            for col in REQUIRED_COLS:
                if col not in df.columns:
                    df[col] = 0

            st.success("File uploaded successfully.")
            sample_csv = df.head(5).to_csv(index=False)
            st.download_button(
                "Download Sample CSV (5 rows)",
                sample_csv, "sample_marketing.csv", "text/csv"
            )

    # -------------------------------
    # UPLOAD CSV + COLUMN MAPPING
    # -------------------------------
    elif mode == "Upload CSV + Column Mapping":
        file = st.file_uploader("Upload your CSV file", type=["csv"])

        if file:
            raw = pd.read_csv(file)
            raw.columns = raw.columns.str.strip()
            st.write("Uploaded Data Preview", raw.head())

            mapping = {}
            for col in REQUIRED_COLS:
                mapping[col] = st.selectbox(
                    f"Map to required column: {col}",
                    ["-- Select --"] + list(raw.columns)
                )

            if st.button("Apply Mapping"):
                missing = [req for req,val in mapping.items() if val == "-- Select --"]
                if missing:
                    st.error(f"Please map all required columns: {missing}")
                else:
                    df = raw.rename(columns=mapping)
                    st.success("Column mapping applied successfully.")

    # -------------------------------
    # VALIDATION
    # -------------------------------
    if df is None:
        st.warning("Please upload or select a dataset.")
        st.stop()

    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        st.error(f"The following required columns are missing: {missing}")
        st.stop()

    df = df.dropna(subset=["Campaign","Channel"])

    # -------------------------------
    # FILTERS
    # -------------------------------
    campaign = st.multiselect("Campaign", df["Campaign"].unique())
    channel = st.multiselect("Channel", df["Channel"].unique())

    filt = df.copy()
    if campaign:
        filt = filt[filt["Campaign"].isin(campaign)]
    if channel:
        filt = filt[filt["Channel"].isin(channel)]

    st.markdown("### Data Preview")
    st.dataframe(filt.head(), use_container_width=True)

    # -------------------------------
    # -------------------------------
    # KPIs (with currency formatting)
    # -------------------------------
    
    def inr(x):
        return f"₹{x:,.2f}"
    
    k1, k2, k3, k4 = st.columns(4)
    
    k1.metric("Total Impressions", f"{int(filt['Impressions'].sum()):,}")
    k2.metric("Total Clicks", f"{int(filt['Clicks'].sum()):,}")
    k3.metric("Total Leads", f"{int(filt['Leads'].sum()):,}")
    k4.metric("Total Spend", inr(filt["Spend"].sum()))


    # -------------------------------
    # CHARTS
    # -------------------------------
    st.markdown("### Campaign-wise Clicks")
    fig1 = px.bar(filt, x="Campaign", y="Clicks", color="Campaign", text="Clicks")
    fig1.update_traces(textposition="outside")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### Channel-wise Leads")
    fig2 = px.pie(filt, names="Channel", values="Leads")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Spend vs Conversions")
    fig3 = px.scatter(
        filt, x="Spend", y="Conversions",
        size="Impressions", color="Channel", hover_data=["Campaign"]
    )
    st.plotly_chart(fig3, use_container_width=True)

    # -------------------------------
    # DOWNLOAD FILTERED DATA
    # -------------------------------
    csv = filt.to_csv(index=False)
    st.download_button("Download Filtered Dataset", csv, "marketing_filtered.csv", "text/csv")
