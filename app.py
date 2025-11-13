import streamlit as st
from datetime import date

st.set_page_config(page_title="Analytics Use Case Hub", layout="wide")

st.title("Analytics Avenue")
st.markdown("Explore real-world data analytics case studies built by **the team of Data Experts**.")

# --- Style ---
st.markdown("""
<style>
    .card {
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 15px;
        margin: 10px;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .card:hover {
        background-color: #f7f9fc;
        box-shadow: 4px 4px 10px rgba(0,0,0,0.15);
    }
    .card img {
        border-radius: 10px;
        width: 100%;
        height: 180px;
        object-fit: cover;
    }
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

# Use Case 1: Marketing Analytics
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image("assets/marketing_preview.jpg", use_column_width=True)
    st.markdown("### Use Case 1: Marketing Analytics")
    st.markdown("Understand Meta Ads performance, audience insights, and campaign impact.")
    if st.button("Preview Marketing Analytics"):
        st.session_state["show_modal"] = "marketing"
    st.markdown('</div>', unsafe_allow_html=True)

# Use Case 2: Healthcare Analytics (future)
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image("assets/healthcare_preview.jpg", use_column_width=True)
    st.markdown("### Use Case 2: Healthcare Analytics")
    st.markdown("Analyze patient data and treatment effectiveness using dashboards.")
    if st.button("Preview Healthcare Analytics"):
        st.session_state["show_modal"] = "healthcare"
    st.markdown('</div>', unsafe_allow_html=True)

# Use Case 3 Placeholder
with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image("assets/real_estate_preview.jpg", use_column_width=True)
    st.markdown("### Use Case 3: Retail Conversion Analytics")
    st.markdown("Measure footfall-to-sales conversion rates across stores.")
    if st.button("Preview Retail Analytics"):
        st.session_state["show_modal"] = "retail"
    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------
# POPUP MODAL (Custom Popup Simulation)
# ------------------------------
if "show_modal" in st.session_state:
    use_case = st.session_state["show_modal"]

    if use_case == "marketing":
        st.markdown("---")
        st.subheader("#1 Marketing Analytics")
        st.video("https://www.youtube.com/watch?v=0d6oY8G5e5c")
        st.markdown(f"**Author:** Vibin  \n**Created on:** {date.today().strftime('%B %d, %Y')}")
        st.markdown("""
        **Project Summary:**  
        Visualizes Meta Ad campaign performance with insights on reach, engagement, audience, and cost efficiency.  
        Built with **Streamlit, Plotly, and Pandas**.
        """)
        if st.button("Go to Project"):
            st.switch_page("pages/usecase1_marketing.py")

    elif use_case == "healthcare":
        st.markdown("---")
        st.subheader("#2 Healthcare Analytics")
        st.video("https://www.youtube.com/watch?v=x8tYGIvK32Q")
        st.markdown(f"**Author:** Vibin — Senior Business Analyst  \n**Created on:** {date.today().strftime('%B %d, %Y')}")
        st.markdown("""
        **Project Summary:**  
        Analyzing patient treatment outcomes using interactive data dashboards.  
        Built with Streamlit, Plotly, and advanced statistical analytics.
        """)
        if st.button("Go to Project"):
            st.switch_page("pages/usecase1_marketing.py")

    else:
        st.warning("🚧 Use Case under development.")
