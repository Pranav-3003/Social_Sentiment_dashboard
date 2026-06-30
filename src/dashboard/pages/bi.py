import streamlit as st
import pandas as pd
from src.business.bi_engine import BIEngine
from src.utils.report_gen import ReportGenerator
from src.utils.db import DBManager, Post
from textblob import TextBlob

def render_bi(db_manager: DBManager):
    st.markdown("## Business Intelligence & Executive Reporting")
    st.write("Extract operational insights, calculate KPIs, audit crises, and download summaries.")
    
    session = db_manager.get_session()
    posts = session.query(Post).all()
    session.close()
    
    if not posts:
        st.warning("No data found in the database. Ingest a dataset first to generate business intelligence insights!")
        return

    # Ingest posts into dataframe
    df = pd.DataFrame([{
        "text": p.cleaned_text or p.text,
        "platform": p.platform,
        "timestamp": p.timestamp
    } for p in posts])

    # Run quick sentiment tagging
    sent_scores = []
    sentiments = []
    for text in df['text']:
        score = TextBlob(text).sentiment.polarity
        sent_scores.append(score)
        sentiments.append("Positive" if score > 0.05 else ("Negative" if score < -0.05 else "Neutral"))
        
    df['sentiment_score'] = sent_scores
    df['sentiment'] = sentiments

    bi_engine = BIEngine()
    
    # 1. Calculate KPIs
    reputation = bi_engine.calculate_brand_reputation_score(df)
    csat = bi_engine.calculate_csat_score(df)
    
    # 2. Crisis Check
    is_crisis, crisis_msg = bi_engine.detect_crisis(df, window_hours=24)
    
    if is_crisis:
        st.error(f"🚨 **CRISIS ALERT:** {crisis_msg}")
    else:
        st.success("✅ **Brand Health Status:** Favorable. Reputation patterns are within normal thresholds.")

    # Render CSAT and Reputation Cards
    col_kpi1, col_kpi2 = st.columns(2)
    
    with col_kpi1:
        rep_color = "#2ecc71" if reputation > 75 else ("#e67e22" if reputation > 50 else "#e74c3c")
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #334155; padding: 25px; border-radius: 12px; text-align: center; border-top: 5px solid {rep_color};'>
            <div style='color: #94a3b8; font-size: 1rem; text-transform: uppercase; font-weight: bold; letter-spacing: 0.05em;'>Brand Reputation Score</div>
            <div style='color: {rep_color}; font-size: 3rem; font-weight: 800; margin: 15px 0;'>{reputation} <span style='font-size:1.5rem; color:#475569;'>/ 100</span></div>
            <div style='color: #64748b; font-size: 0.9rem;'>Volume-weighted time-decay formula</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi2:
        csat_color = "#2ecc71" if csat > 70 else ("#e67e22" if csat > 40 else "#e74c3c")
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #334155; padding: 25px; border-radius: 12px; text-align: center; border-top: 5px solid {csat_color};'>
            <div style='color: #94a3b8; font-size: 1rem; text-transform: uppercase; font-weight: bold; letter-spacing: 0.05em;'>Estimated CSAT Score</div>
            <div style='color: {csat_color}; font-size: 3rem; font-weight: 800; margin: 15px 0;'>{csat}%</div>
            <div style='color: #64748b; font-size: 0.9rem;'>Percentage of positive sentiment reviews</div>
        </div>
        """, unsafe_allow_html=True)

    # 3. Actionable Suggestions
    st.write("---")
    st.markdown("### Actionable Product & Customer Support Suggestions")
    suggestions = bi_engine.generate_product_suggestions(df)
    
    sug_df = pd.DataFrame(suggestions)
    # Render styled table
    st.table(sug_df)

    # 4. Top Feedback Extract
    st.write("---")
    st.markdown("### Highlighted Customer Testimonials")
    feedback = bi_engine.extract_top_feedback(df, count=2)
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("<h4 style='color:#2ecc71;'>Top Positive Comments</h4>", unsafe_allow_html=True)
        if feedback["positive"]:
            for item in feedback["positive"]:
                st.info(f"**@{item['username']} ({item['platform'].upper()}):** \"{item['text']}\" (Score: +{item['score']})")
        else:
            st.info("No positive feedback found.")
            
    with col_f2:
        st.markdown("<h4 style='color:#e74c3c;'>Top Critical Comments</h4>", unsafe_allow_html=True)
        if feedback["negative"]:
            for item in feedback["negative"]:
                st.error(f"**@{item['username']} ({item['platform'].upper()}):** \"{item['text']}\" (Score: {item['score']})")
        else:
            st.info("No negative feedback found.")

    # 5. Report Downloader
    st.write("---")
    st.markdown("### Download Executive Reports")
    st.write("Generate and download a finalized intelligence report including KPIs, summaries, and actions.")
    
    exec_summary = bi_engine.generate_executive_summary(df)
    
    st.markdown("#### Executive Summary Preview")
    st.code(exec_summary, language="text")
    
    # Generate files on request
    col_d1, col_d2 = st.columns(2)
    
    kpis = {
        "dataset_name": "SentiVerse Social Media Feed",
        "total_posts": len(df),
        "brand_reputation": reputation,
        "csat": csat
    }
    
    # Model performance stats for the report
    active_model_name = st.session_state.get("active_model_name", "TextBlob Lexicon Model")
    model_perf = {active_model_name: "Active Model"}
    
    rep_gen = ReportGenerator()
    
    with col_d1:
        st.write("#### Download PDF Document")
        if st.button("Compile PDF Report"):
            with st.spinner("Compiling ReportLab PDF..."):
                pdf_bytes = rep_gen.generate_pdf_report(
                    kpis=kpis,
                    executive_summary=exec_summary,
                    recommendations=suggestions,
                    model_perf=model_perf
                )
                st.download_button(
                    label="Save PDF File",
                    data=pdf_bytes,
                    file_name="sentiverse_executive_report.pdf",
                    mime="application/pdf"
                )
                st.success("PDF Compiled!")
                
    with col_d2:
        st.write("#### Download Word Document")
        if st.button("Compile Word Report"):
            with st.spinner("Compiling DOCX document..."):
                docx_bytes = rep_gen.generate_word_report(
                    kpis=kpis,
                    executive_summary=exec_summary,
                    recommendations=suggestions,
                    model_perf=model_perf
                )
                st.download_button(
                    label="Save Word File",
                    data=docx_bytes,
                    file_name="sentiverse_executive_report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
                st.success("Word Document Compiled!")
