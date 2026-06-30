import streamlit as st
import os

# Set page config FIRST
st.set_page_config(
    page_title="SentiVerse AI - Social Media Analytics",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Injection for Premium Cyberpunk Dark Theme Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800;900&display=swap');

    /* Core Application Settings */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0b0f1e 0%, #05060b 100%) !important;
        color: #cbd5e1 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #030408 !important;
        border-right: 1px solid rgba(0, 242, 254, 0.15) !important;
    }
    [data-testid="stSidebar"] * {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Headers & Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Metrics panel cards overrides */
    [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        font-size: 2.4rem !important;
        color: #00f2fe !important;
        text-shadow: 0 0 12px rgba(0, 242, 254, 0.3) !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        color: #94a3b8 !important;
        letter-spacing: 0.08em !important;
    }
    
    /* Glassmorphic Metric Card Containers */
    .metric-card {
        background: rgba(15, 23, 42, 0.65) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(0, 242, 254, 0.15) !important;
        border-radius: 14px !important;
        padding: 24px !important;
        text-align: center !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.35) !important;
        transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s !important;
    }
    .metric-card:hover {
        transform: translateY(-4px) !important;
        border-color: rgba(0, 242, 254, 0.4) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 242, 254, 0.2) !important;
    }
    
    /* Cyan Gradient Primary Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00f2fe, #4facfe) !important;
        color: #05060c !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(0, 242, 254, 0.45) !important;
        color: #05060c !important;
    }
    .stButton>button:active {
        transform: translateY(0px) !important;
    }
    
    /* Secondary Action Buttons */
    button[kind="secondary"] {
        background: rgba(30, 41, 59, 0.5) !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(5px) !important;
        transition: all 0.2s ease !important;
    }
    button[kind="secondary"]:hover {
        background: rgba(30, 41, 59, 0.8) !important;
        border-color: rgba(0, 242, 254, 0.3) !important;
        color: #ffffff !important;
    }
    
    /* Cyberpunk Styled Custom Tabs */
    button[data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: #64748b !important;
        background-color: transparent !important;
        border: none !important;
        transition: color 0.2s, border-color 0.2s !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #00f2fe !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00f2fe !important;
        border-bottom: 3px solid #00f2fe !important;
        text-shadow: 0 0 10px rgba(0, 242, 254, 0.2) !important;
    }
    
    /* Interactive Inputs & Fields styling */
    input, textarea, div[data-baseweb="select"], div[data-baseweb="input"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    
    /* Styled Markdown tables */
    table {
        border-collapse: collapse;
        width: 100%;
        color: #cbd5e1;
        background: rgba(15, 23, 42, 0.4);
        border-radius: 8px;
        overflow: hidden;
    }
    th {
        background-color: rgba(30, 41, 59, 0.85) !important;
        color: #00f2fe !important;
        text-align: left;
        padding: 14px 16px;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700;
    }
    td {
        padding: 14px 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    tr:hover {
        background-color: rgba(0, 242, 254, 0.05) !important;
    }

    /* Premium Chat Message bubble overrides */
    [data-testid="stChatMessage"] {
        background: rgba(15, 23, 42, 0.55) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 14px !important;
        margin-bottom: 12px !important;
        padding: 16px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15) !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    [data-testid="stChatMessage"]:hover {
        border-color: rgba(0, 242, 254, 0.2) !important;
        box-shadow: 0 4px 20px rgba(0, 242, 254, 0.1) !important;
    }
    [data-testid="stChatMessageContent"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Imports for DB & Sub-pages
from src.utils.db import DBManager
from src.dashboard.pages.home import render_home
from src.dashboard.pages.ingestion import render_ingestion
from src.dashboard.pages.preprocessing import render_preprocessing
from src.dashboard.pages.eda import render_eda
from src.dashboard.pages.workbench import render_workbench
from src.dashboard.pages.predictions import render_predictions
from src.dashboard.pages.explain import render_explain
from src.dashboard.pages.topics import render_topics
from src.dashboard.pages.trends import render_trends
from src.dashboard.pages.bi import render_bi
from src.dashboard.pages.chat import render_chat

# Initialize Database Manager
db_manager = DBManager(db_url="sqlite:///data/sentiverse.db")

# Optional/Mock Authentication Block
def show_auth_form() -> bool:
    """
    Renders login/registration panel in Streamlit.
    """
    st.markdown("""
    <div style='text-align: center; margin-top: 50px;'>
        <h2 style='color:#00F2FE;'>SentiVerse AI Portal Login</h2>
        <p style='color:#64748b;'>Enter credentials to access the analytics workspace.</p>
    </div>
    """, unsafe_allow_html=True)
    
    auth_opt = st.radio("Access Mode", ["Login", "Register"], horizontal=True)
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if auth_opt == "Login":
        if st.button("Access Workspace"):
            user = db_manager.authenticate_user(username, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.role = user.role
                st.success("Access Granted! Loading workspace...")
                st.rerun()
            else:
                st.error("Invalid credentials.")
    else:
        role = st.selectbox("Role", ["Data Analyst", "Data Scientist", "Marketing Manager", "Administrator"])
        if st.button("Register Account"):
            if username and password:
                user = db_manager.create_user(username, password, role=role.lower())
                if user:
                    st.success("Account successfully created! Please switch to Login.")
                else:
                    st.error("Registration failed. Username may already exist.")
            else:
                st.error("Please fill in all fields.")
    return False

# Manage Session Auth state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Toggle Authentication (Set False if you want to bypass auth entirely, or True to enforce it)
ENFORCE_AUTH = False # Keep False for recruiter portfolios to allow instant click-through access!

if ENFORCE_AUTH and not st.session_state.authenticated:
    show_auth_form()
else:
    # Sidebar Navigation Menu
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 10px 0;'>
        <h2 style='color: #00F2FE; margin-bottom: 2px;'>SentiVerse AI</h2>
        <span style='color: #475569; font-size:0.8rem; font-weight:bold;'>WORKSPACE</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.authenticated:
        st.sidebar.write(f"👤 **User:** {st.session_state.username} ({st.session_state.get('role', 'user').upper()})")
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()
            
    page = st.sidebar.selectbox(
        "Navigation",
        [
            "Home",
            "Data Ingestion",
            "Preprocessing & NLP",
            "Exploratory Data Analysis",
            "Model Workbench",
            "Predictions Portal",
            "Explain Predictions (XAI)",
            "Topic Modeling",
            "Trends & Forecasting",
            "Business Intelligence",
            "AI Chat Assistant"
        ]
    )

    # Database Status in Sidebar Footer
    session = db_manager.get_session()
    try:
        from src.utils.db import Post
        cnt = session.query(Post).count()
    except Exception:
        cnt = 0
    finally:
        session.close()
        
    st.sidebar.write("---")
    st.sidebar.markdown(f"""
    <div style='font-size:0.8rem; color:#64748b;'>
        <div>Database Status: <b>{cnt} posts</b></div>
        <div>Model Active: <b>{st.session_state.get('active_model_name', 'None')}</b></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Route Pages
    if page == "Home":
        render_home(db_manager)
    elif page == "Data Ingestion":
        render_ingestion(db_manager)
    elif page == "Preprocessing & NLP":
        render_preprocessing(db_manager)
    elif page == "Exploratory Data Analysis":
        render_eda(db_manager)
    elif page == "Model Workbench":
        render_workbench(db_manager)
    elif page == "Predictions Portal":
        render_predictions(db_manager)
    elif page == "Explain Predictions (XAI)":
        render_explain(db_manager)
    elif page == "Topic Modeling":
        render_topics(db_manager)
    elif page == "Trends & Forecasting":
        render_trends(db_manager)
    elif page == "Business Intelligence":
        render_bi(db_manager)
    elif page == "AI Chat Assistant":
        render_chat(db_manager)
