import streamlit as st
import base64

st.set_page_config(page_title="Real Estate Insights – App 1", layout="wide")

# ---------------------------------------------------------
# Utility for modal popup
# ---------------------------------------------------------
def popup_youtube(video_url, project_link):
    with st.modal("Application Preview", width=900):
        st.video(video_url)
        st.write("")
        st.link_button("🚀 Go to Full Project", project_link)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown("""
<style>
.header {
    font-size: 40px;
    font-weight: 700;
    color: #1A237E;
    padding-bottom: 0px;
}
.sub {
    font-size: 20px;
    color: #555;
}
.card {
    background-color: #f8f9fa;
    padding: 18px;
    border-radius: 10px;
    border: 1px solid #ddd;
}
.preview-btn {
    font-size: 15px;
    font-weight: 600;
    padding: 8px 14px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">🏙 Real Estate Application 1: Property Valuation Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">A complete consulting-grade showcase for enterprise clients</div>', unsafe_allow_html=True)
st.write("")

st.divider()

# ---------------------------------------------------------
# HIERARCHICAL OVERVIEW SECTIONS
# ---------------------------------------------------------

with st.expander("📌 1. Overview", expanded=True):
    st.markdown("""
    This application delivers automated property valuation driven by structured data,  
    machine learning models, and market benchmarks across India and global regions.  
    It helps enterprises understand fair market pricing using robust data pipelines.

    It acts as a decision-making enabler for brokers, investors, consultants, and banks.
    """)

with st.expander("🎯 2. Purpose of the Application"):
    st.markdown("""
    - Improve accuracy of property pricing  
    - Reduce manual appraisal effort  
    - Enable data-backed negotiation and pricing strategy  
    - Standardize valuation across branches and markets  
    """)

with st.expander("🧩 3. Our Capabilities"):
    st.markdown("""
    **Core Technical Capabilities**
    - Automated ML-driven price prediction  
    - Multi-city data ingestion pipelines  
    - Market benchmarking analytics  
    - Feature engineering based on locality, amenities, demand score  
    - Scalable architecture supporting 10,000+ property records

    **Consulting Capabilities**
    - Requirement gathering  
    - Market metrics definition  
    - Model validation  
    - Deployment & integration  
    """)

with st.expander("📈 4. Business Impact Delivered"):
    st.markdown("""
    - 35 to 50 percent faster pricing cycles  
    - 20 percent improvement in negotiation leverage  
    - Reduction of overpricing & underpricing  
    - Better forecast accuracy across markets  
    """)

with st.expander("📊 5. Key KPIs"):
    st.markdown("""
    - RMSE of valuation model  
    - Market avg price alignment score  
    - Price deviation %  
    - Prediction confidence  
    - Locality demand index  
    """)

st.divider()

# ---------------------------------------------------------
# APPLICATION PREVIEW SECTION
# ---------------------------------------------------------

st.markdown("## 🎥 Application Preview")

col1, col2 = st.columns([1.4, 3])

with col1:
    # Thumbnail image
    st.image("https://i.ibb.co/QNmDgGY/real-estate-dashboard.jpg", use_column_width=True)
    if st.button("Preview App", key="p1"):
        popup_youtube("https://www.youtube.com/watch?v=6Dh-RL__uN4",
                      "https://your-project-link.com")

with col2:
    st.markdown("""
    ### Property Valuation Engine  
    A modern data-driven valuation tool providing:  
    - Automated pricing  
    - Locality scoring  
    - Property insights  
    - Investment-grade recommendations  

    Use this for client demos, investor presentations, and enterprise onboarding.
    """)

st.divider()

st.success("App 1 structure completed. Tell me when to generate App 2.")
