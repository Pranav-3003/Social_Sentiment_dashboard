import streamlit as st
import pandas as pd
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.nlp_pipeline import NLPPipeline
from src.utils.db import DBManager, Post

def render_preprocessing(db_manager: DBManager):
    st.markdown("## Data Cleaning & NLP Pipeline Workbench")
    
    # Load all posts
    session = db_manager.get_session()
    posts = session.query(Post).all()
    session.close()
    
    if not posts:
        st.warning("No data found in the database. Please go to the **Data Ingestion** page to load or collect data first!")
        return

    # Create dataframe
    df = pd.DataFrame([{
        "id": p.id,
        "text": p.text,
        "username": p.username,
        "platform": p.platform,
        "timestamp": p.timestamp
    } for p in posts])

    tab1, tab2 = st.tabs(["Dataset Cleaning Pipeline", "NLP Processing Sandbox"])

    # TAB 1: Bulk Dataset Cleaning
    with tab1:
        st.markdown("### Cleaning Configuration")
        st.write("Tune how the text cleaning pipeline processes the raw social media feed.")
        
        cols = st.columns(3)
        with cols[0]:
            lowercase = st.checkbox("Lowercase Text", value=True)
            remove_urls = st.checkbox("Strip URLs (http/www)", value=True)
            remove_html = st.checkbox("Strip HTML Tags", value=True)
        with cols[1]:
            expand_contractions = st.checkbox("Expand Contractions (don't -> do not)", value=True)
            remove_punctuation = st.checkbox("Strip Punctuation", value=True)
            remove_numbers = st.checkbox("Strip Numbers", value=False)
        with cols[2]:
            remove_emojis = st.checkbox("Strip Emojis", value=False)
            process_hashtags = st.selectbox("Process Hashtags", ["extract", "remove", "keep"])
            unicode_normal = st.checkbox("Normalize Unicode", value=True)
            
        clean_config = {
            "lowercase": lowercase,
            "remove_urls": remove_urls,
            "remove_html": remove_html,
            "expand_contractions": expand_contractions,
            "remove_punctuation": remove_punctuation,
            "remove_numbers": remove_numbers,
            "remove_emojis": remove_emojis,
            "process_hashtags": process_hashtags,
            "unicode_normalization": unicode_normal,
            "whitespace_cleanup": True
        }
        
        cleaner = TextCleaner(clean_config)
        
        if st.button("Run Bulk Data Cleaning Pipeline", type="primary"):
            cleaned_texts = []
            stats_list = []
            
            with st.spinner("Cleaning posts and compiling statistics..."):
                # Clean each post
                for idx, row in df.iterrows():
                    cleaned, stats = cleaner.clean_text(row['text'])
                    cleaned_texts.append(cleaned)
                    stats_list.append(stats)
                    
                df['cleaned_text'] = cleaned_texts
                
                # Update cleaned text in SQLite database
                session = db_manager.get_session()
                try:
                    for idx, row in df.iterrows():
                        db_post = session.query(Post).filter(Post.id == int(row['id'])).first()
                        if db_post:
                            db_post.cleaned_text = row['cleaned_text']
                    session.commit()
                except Exception as e:
                    session.rollback()
                    st.error(f"Error updating cleaned text in database: {e}")
                finally:
                    session.close()
                    
                st.success("Pipeline executed successfully and committed to the database!")
                
                # Show before/after statistics
                stats_df = pd.DataFrame(stats_list)
                summary = TextCleaner.get_cleaning_pipeline_summary(df['text'], df['cleaned_text'])
                
                st.markdown("#### Before vs After Cleaning Metrics")
                cols_m = st.columns(3)
                with cols_m[0]:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; font-weight: bold;'>Avg Original Length</div>
                        <div style='color: #00f2fe; font-size: 1.8rem; font-weight: 800; margin: 10px 0;'>{summary['average_original_length']} <span style='font-size: 0.9rem; color: #475569;'>chars</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                with cols_m[1]:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; font-weight: bold;'>Avg Cleaned Length</div>
                        <div style='color: #38bdf8; font-size: 1.8rem; font-weight: 800; margin: 10px 0;'>{summary['average_cleaned_length']} <span style='font-size: 0.9rem; color: #475569;'>chars</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                with cols_m[2]:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; font-weight: bold;'>Size Reduction</div>
                        <div style='color: #4ade80; font-size: 1.8rem; font-weight: 800; margin: 10px 0;'>{summary['reduction_percentage']}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>#### Extracted Noise Counters", unsafe_allow_html=True)
                col_c = st.columns(4)
                with col_c[0]:
                    st.markdown(f"""
                    <div class='metric-card' style='padding: 15px !important;'>
                        <div style='color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; font-weight: bold;'>URLs Removed</div>
                        <div style='color: #f87171; font-size: 1.6rem; font-weight: 800; margin: 5px 0;'>{int(stats_df['urls_removed'].sum())}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_c[1]:
                    st.markdown(f"""
                    <div class='metric-card' style='padding: 15px !important;'>
                        <div style='color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; font-weight: bold;'>HTML Tags</div>
                        <div style='color: #f87171; font-size: 1.6rem; font-weight: 800; margin: 5px 0;'>{int(stats_df['html_tags_removed'].sum())}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_c[2]:
                    st.markdown(f"""
                    <div class='metric-card' style='padding: 15px !important;'>
                        <div style='color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; font-weight: bold;'>Contractions</div>
                        <div style='color: #60a5fa; font-size: 1.6rem; font-weight: 800; margin: 5px 0;'>{int(stats_df['contractions_expanded'].sum())}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_c[3]:
                    st.markdown(f"""
                    <div class='metric-card' style='padding: 15px !important;'>
                        <div style='color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; font-weight: bold;'>Punctuation</div>
                        <div style='color: #e2e8f0; font-size: 1.6rem; font-weight: 800; margin: 5px 0;'>{int(stats_df['punctuation_removed'].sum())}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>#### Sample Comparison Table", unsafe_allow_html=True)
                comp_df = df[['text', 'cleaned_text']].head(10)
                st.dataframe(comp_df)

    # TAB 2: NLP Analysis Sandbox
    with tab2:
        st.markdown("### Interactive Single Post Parser")
        st.write("Extract linguistic structural components of any post using NLTK and spaCy features.")
        
        with st.expander("📚 NLP Concept Guide: What do these terms mean?", expanded=False):
            st.markdown("""
            Here is a quick reference to the linguistic operations running in the background:
            
            - 🔤 **Tokenization**: Splitting a continuous stream of text into individual units (tokens) like words and punctuation.
            - 🏷️ **Part-of-Speech (POS) Tagging**: Analyzing the grammatical role of each word in the sentence (e.g., Noun, Verb, Adjective, Pronoun).
            - 🌱 **Lemmatization**: Reducing inflected words to their root/base form (e.g., *\"crying\"* and *\"cries\"* simplify to *\"cry\"*).
            - 🛑 **Stopwords Removal**: Filtering out common words (e.g., *\"the\"*, *\"is\"*, *\"and\"*) that carry little standalone semantic meaning.
            - 🏢 **Named Entity Recognition (NER)**: Identifying and classifying key real-world objects in the text (e.g., People, Organizations, Geopolitical Locations).
            """)
        
        # Dropdown to select a post, or custom text input
        post_options = {f"Post #{row['id']}: {row['text'][:60]}...": row['text'] for idx, row in df.iterrows()}
        selected_key = st.selectbox("Select a post from database", list(post_options.keys()))
        custom_text_opt = st.checkbox("Write custom text instead")
        
        raw_input_text = ""
        if custom_text_opt:
            raw_input_text = st.text_area("Input text to parse", value="Google has released an amazing new AI model. I can't wait to test it tomorrow!")
        else:
            raw_input_text = post_options[selected_key]
            
        nlp_engine = st.radio("Choose NLP Engine Backend", ["NLTK (Default)", "spaCy (Linguistic parsing)"], horizontal=True)
        use_spacy = (nlp_engine == "spaCy (Linguistic parsing)")
        
        nlp_pipeline = NLPPipeline(use_spacy=use_spacy)
        cleaner_sandbox = TextCleaner(clean_config)
        
        if st.button("Run NLP Pipeline Analysis", key="nlp_sandbox_run"):
            # 1. Clean details
            cleaned_text, clean_stats = cleaner_sandbox.clean_text(raw_input_text)
            
            st.markdown("#### 1. Data Cleaning Stage")
            st.write(f"**Original Text:** `{raw_input_text}`")
            st.write(f"**Cleaned Text:** `{cleaned_text}`")
            
            c1, c2, c3 = st.columns(3)
            c1.write(f"Contractions expanded: {clean_stats['contractions_expanded']}")
            c2.write(f"Emojis removed: {clean_stats['emojis_removed']}")
            c3.write(f"Mentions removed: {clean_stats['mentions_removed']}")
            
            # 2. NLP Pipeline details
            nlp_res = nlp_pipeline.process_pipeline(cleaned_text, remove_stops=True, do_lemmatize=True)
            
            st.write("---")
            st.markdown("#### 2. Tokenization & Lemmatization Stage")
            st.write(f"**Tokens (Lemmatized & Stopwords Removed):**")
            st.code(str(nlp_res['tokens']))
            
            # 3. POS Tagging
            st.write("---")
            st.markdown("#### 3. Part-of-Speech (POS) Tags")
            pos_tags = nlp_res['pos_tags']
            # Render visual POS tags
            pos_html = "<div style='display:flex; flex-wrap:wrap; gap:6px;'>"
            for word, tag in pos_tags:
                pos_html += f"<span style='background-color:#1e293b; border: 1px solid #3b82f6; padding: 2px 8px; border-radius:4px; font-size:12px; color:#93c5fd;'><b>{word}</b> <span style='font-size:10px; color:#60a5fa;'>({tag})</span></span>"
            pos_html += "</div>"
            st.markdown(pos_html, unsafe_allow_html=True)
            
            # 4. Named Entity Recognition
            st.write("---")
            st.markdown("#### 4. Named Entity Recognition (NER)")
            entities = nlp_res['entities']
            if len(entities) > 0:
                ent_html = "<div style='display:flex; flex-wrap:wrap; gap:10px;'>"
                for ent_text, label in entities:
                    ent_html += f"<div style='background-color:#0f172a; border-left: 4px solid #10b981; padding: 6px 12px; border-radius:0 6px 6px 0;'><span style='color:#a7f3d0; font-weight:bold;'>{ent_text}</span> <span style='color:#34d399; font-size:11px; text-transform:uppercase;'>({label})</span></div>"
                ent_html += "</div>"
                st.markdown(ent_html, unsafe_allow_html=True)
            else:
                st.info("No named entities detected in the text.")
