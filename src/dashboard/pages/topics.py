import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from src.business.topic_modeling import TopicModeler
from src.features.vectorizers import TextVectorizer
from src.utils.db import DBManager, Post

def render_topics(db_manager: DBManager):
    st.markdown("## Topic Modeling & Document Clustering")
    st.write("Unsupervised mining to discover dominant discussion themes across social media posts.")
    
    session = db_manager.get_session()
    posts = session.query(Post).all()
    session.close()
    
    if not posts:
        st.warning("No data found in the database. Ingest and Clean a dataset first before running topic modeling!")
        return

    # Prepare DataFrame
    df = pd.DataFrame([{
        "text": p.cleaned_text or p.text,
        "raw_text": p.text,
        "username": p.username,
        "platform": p.platform,
        "timestamp": p.timestamp
    } for p in posts])

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        model_type = st.selectbox("Topic Modeling Algorithm", ["LDA (Latent Dirichlet Allocation)", "NMF (Non-Negative Matrix Factorization)", "BERTopic Fallback (c-TF-IDF Embeddings)"])
    with col_t2:
        n_topics = st.slider("Number of Topics to Extract", 2, 10, 5)

    if st.button("Run Topic Extraction Pipeline", type="primary"):
        with st.spinner("Extracting topic clusters..."):
            modeler = TopicModeler(n_topics=n_topics)
            corpus = df['text'].tolist()
            
            topics_keywords = []
            doc_topics = None
            
            # Run selected model
            if "LDA" in model_type:
                topics_keywords, doc_topics = modeler.fit_lda(corpus)
            elif "NMF" in model_type:
                topics_keywords, doc_topics = modeler.fit_nmf(corpus)
            else:
                # BERTopic simulated
                # 1. Extract embeddings (using TF-IDF as base embeddings representation for Kmeans to make it fast & robust)
                vectorizer = TextVectorizer(method="tfidf", max_features=1000)
                embeddings = vectorizer.fit_transform(corpus)
                topics_keywords, doc_topics = modeler.fit_bertopic_fallback(corpus, embeddings)
                
            # Assign dominant topic to each row
            dominant_topics = np.argmax(doc_topics, axis=1)
            df['assigned_topic'] = [f"Topic {t}" for t in dominant_topics]
            df['topic_confidence'] = np.max(doc_topics, axis=1)
            
            # Commit assigned topics back to database
            session = db_manager.get_session()
            try:
                for idx, post_db in enumerate(session.query(Post).order_by(Post.id).all()):
                    if idx < len(dominant_topics):
                        post_db.topic = f"Topic {dominant_topics[idx]}"
                session.commit()
            except Exception as e:
                session.rollback()
                st.error(f"Error updating topics in database: {e}")
            finally:
                session.close()

            st.success("Topic modeling pipeline completed successfully! Database updated.")
            
            # Show Keywords
            st.markdown("### Discovered Topic Keywords")
            cols_grid = st.columns(min(3, n_topics))
            
            for idx, tk in enumerate(topics_keywords):
                col_idx = idx % min(3, n_topics)
                with cols_grid[col_idx]:
                    st.write(f"#### Topic {tk['topic_id']}")
                    # Plot keywords in Plotly bar chart
                    kw_df = pd.DataFrame({
                        "Keyword": tk["keywords"],
                        "Weight": tk["weights"]
                    }).sort_values(by="Weight", ascending=True)
                    
                    fig_kw = px.bar(kw_df, x="Weight", y="Keyword", orientation="h",
                                     title=f"Topic {tk['topic_id']} Top Terms", template="plotly_dark")
                    # Hide axis labels for neat styling
                    fig_kw.update_yaxes(title="")
                    fig_kw.update_xaxes(title="")
                    st.plotly_chart(fig_kw, use_container_width=True)
            
            # Topic Clusters Visualizer (using PCA of TF-IDF vectors for dimensionality reduction)
            st.write("---")
            st.markdown("### Topic Spatial Visualizer")
            st.write("A 2D projection (PCA) of posts, color-coded by their dominant topic cluster.")
            
            try:
                vectorizer = TextVectorizer(method="tfidf", max_features=1000)
                embs = vectorizer.fit_transform(corpus)
                coords_df = modeler.get_topic_coordinates(embs, doc_topics)
                coords_df["Text"] = df["raw_text"].apply(lambda x: x[:80] + "...")
                
                fig_clusters = px.scatter(coords_df, x="x", y="y", color="topic",
                                          hover_data=["Text", "confidence"],
                                          title="2D Topic Document Map", template="plotly_dark")
                st.plotly_chart(fig_clusters, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not build spatial coordinates: {e}")
                
            # Lookup table
            st.write("---")
            st.markdown("### Document Topic Browser")
            selected_t = st.selectbox("Select Topic to View Posts", [f"Topic {i}" for i in range(n_topics)])
            filtered_df = df[df['assigned_topic'] == selected_t].sort_values(by="topic_confidence", ascending=False)
            st.dataframe(filtered_df[['raw_text', 'platform', 'timestamp', 'topic_confidence']].head(15))
