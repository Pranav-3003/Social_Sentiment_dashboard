import streamlit as st
import pandas as pd
from src.utils.db import DBManager

def render_home(db_manager: DBManager):
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #00F2FE; font-family: "Outfit", sans-serif; font-size: 3rem; margin-bottom: 5px;'>SentiVerse AI</h1>
        <p style='color: #8E9EAB; font-size: 1.2rem; font-weight: 300;'>End-to-End AI-Powered Social Media Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Statistics
    session = db_manager.get_session()
    try:
        from src.utils.db import Post
        total_posts = session.query(Post).count()
        platforms = session.query(Post.platform).distinct().all()
        platforms_list = [p[0] for p in platforms]
        
        # Avg sentiment score
        avg_score_res = session.query(Post.sentiment_score).all()
        avg_score = sum(s[0] for s in avg_score_res) / len(avg_score_res) if avg_score_res else 0.0
    except Exception:
        total_posts = 0
        platforms_list = []
        avg_score = 0.0
    finally:
        session.close()

    # Dynamic metrics display using custom styled HTML boxes
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #334155; padding: 20px; border-radius: 12px; text-align: center;'>
            <div style='color: #94a3b8; font-size: 0.9rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;'>Total Ingested Posts</div>
            <div style='color: #38bdf8; font-size: 2.5rem; font-weight: 700; margin: 10px 0;'>{total_posts}</div>
            <div style='color: #475569; font-size: 0.8rem;'>Across all channels</div>
        </div>
        """, unsafe_allow_html=True)
        
    with cols[1]:
        sentiment_label = "Neutral"
        sentiment_color = "#38bdf8"
        if avg_score > 0.05:
            sentiment_label = "Positive"
            sentiment_color = "#4ade80"
        elif avg_score < -0.05:
            sentiment_label = "Negative"
            sentiment_color = "#f87171"
            
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #334155; padding: 20px; border-radius: 12px; text-align: center;'>
            <div style='color: #94a3b8; font-size: 0.9rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;'>Global Sentiment Polarity</div>
            <div style='color: {sentiment_color}; font-size: 2.5rem; font-weight: 700; margin: 10px 0;'>{avg_score:.2f}</div>
            <div style='color: {sentiment_color}; font-size: 0.9rem; font-weight: 600;'>{sentiment_label}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with cols[2]:
        platform_count = len(platforms_list)
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #334155; padding: 20px; border-radius: 12px; text-align: center;'>
            <div style='color: #94a3b8; font-size: 0.9rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;'>Active Ingest Sources</div>
            <div style='color: #38bdf8; font-size: 2.5rem; font-weight: 700; margin: 10px 0;'>{platform_count}</div>
            <div style='color: #475569; font-size: 0.8rem;'>{", ".join([p.capitalize() for p in platforms_list]) if platforms_list else "None"}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")
    
    st.markdown("### Platform Concept & Features")
    st.write(
        "SentiVerse AI simplifies social media intelligence into a single dashboard. "
        "It provides modular pipelines for cleaning, exploring, training machine learning classifiers, "
        "detecting multi-category emotions, extracting topics, and forecasting sentiment trends with state-of-the-art explainability."
    )
    
    st.markdown("#### System Architecture Flow")
    st.markdown("""
    ```mermaid
    graph TD
        A[Social Media Feeds: Reddit, YouTube, Files, Manual] --> B[Data Collection & Translation]
        B --> C[Data Cleaning Pipeline]
        C --> D[Exploratory Data Analysis EDA]
        D --> E[NLP Processing: POS, NER, Tokenizer]
        E --> F[Feature Matrix: TF-IDF, Embeddings]
        F --> G[Model Workbench: Classical ML, DL, BERT]
        G --> H[Model Predictor & SHAP/LIME Explainers]
        H --> I[Business Intelligence Engine]
        I --> J[Actionable Executive PDF/Word Reports]
    ```
    """)
    
    st.markdown("#### Feature Overview")
    st.markdown("""
    - **Interactive Workbench**: Train 9 classical models (XGBoost, Naive Bayes, LightGBM, Random Forest, etc.) and deep learning networks.
    - **Explainable Predictions**: Visual LIME and SHAP word-level contribution chips.
    - **Sub-Analysis Engines**: 7-class Emotion Detector, Topic modeling (LDA/NMF/c-TF-IDF K-Means), Sarcasm/Toxicity/Spam/Fake News filters.
    - **Business Intelligence**: Brand reputation tracker, Estimated CSAT, auto-recommendations, and professional doc generators.
    """)
