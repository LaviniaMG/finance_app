"""
FinPlan — Platformă de planificare financiară
Entry point: run with `streamlit run streamlit_app/main.py`
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from streamlit_app.utils.state import init_session_state

st.set_page_config(
    page_title="FinPlan",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()

# Custom CSS — minimal, matches Figma dark sidebar + white content
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #1a1a2e; }
[data-testid="stSidebar"] .st-emotion-cache-1cypcdb { color: #ffffff; }
[data-testid="stSidebar"] label { color: #aaaacc !important; font-size: 0.75rem; }
[data-testid="stSidebar"] .stSelectbox > div { color: white; }
.metric-card {
    background: white;
    border-radius: 8px;
    padding: 16px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.alert-red { border-left: 4px solid #F44336; background: #fff5f5; padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
.alert-yellow { border-left: 4px solid #FF9800; background: #fffbf0; padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
.alert-green { border-left: 4px solid #4CAF50; background: #f5fff5; padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
.step-badge { background: #E91E63; color: white; border-radius: 50%; padding: 2px 8px; font-weight: bold; margin-right: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("📊 FinPlan")
st.markdown("**Platformă de planificare financiară** — selectează o pagină din meniu.")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**Navigare**: Folosește meniul din stânga pentru a naviga între pagini.")
with col2:
    st.info("**Scenariu activ**: Setează scenariul din orice pagină prin selectboxul din sidebar.")
with col3:
    scenario = st.session_state.get("active_scenario", "—")
    st.success(f"**Scenariu curent**: {scenario or 'Nesetat'}")
