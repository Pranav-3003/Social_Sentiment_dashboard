import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from src.utils.db import DBManager, Post
from src.features.text_stats import TextStatsExtractor
from textblob import TextBlob

def render_eda(db_manager: DBManager):
    st.markdown("## Exploratory Data Analysis (EDA)")
    
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
        "cleaned_text": p.cleaned_text or p.text,
        "username": p.username,
        "platform": p.platform,
        "timestamp": p.timestamp
    } for p in posts])

    # Run quick sentiment analyzer (TextBlob) on dataset to provide EDA classifications
    sentiment_scores = []
    sentiments = []
    
    for text in df['cleaned_text']:
        analysis = TextBlob(text)
        score = analysis.sentiment.polarity
        sentiment_scores.append(score)
        
        if score > 0.05:
            sentiments.append("Positive")
        elif score < -0.05:
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")
            
    df['sentiment_score'] = sentiment_scores
    df['sentiment'] = sentiments

    # Extract Text Stats
    stats_extractor = TextStatsExtractor()
    df = stats_extractor.extract_features_df(df, text_column='text')

    tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Text Stats & Distributions", "Advanced NLP Analytics"])

    # TAB 1: Dataset Overview
    with tab1:
        st.markdown("### Dataset Summary & Health Report")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ingested Documents", len(df))
        c2.metric("Platforms Analyzed", len(df['platform'].unique()))
        c3.metric("User Accounts", len(df['username'].unique()))
        # Check duplicate percent
        dup_cnt = df.duplicated(subset=['text']).sum()
        c4.metric("Duplicates Detected", f"{dup_cnt} ({round(dup_cnt/len(df)*100, 1)}%)")
        
        # Missing values report
        missing_report = df[['text', 'platform', 'timestamp']].isna().sum().reset_index()
        missing_report.columns = ['Column', 'Missing Values Count']
        
        st.write("#### Missing Values & Schema Completeness")
        st.table(missing_report)
        
        # Ingest volume by platform
        st.write("#### Document Volume by Platform Source")
        platform_counts = df['platform'].value_counts().reset_index()
        fig_platform = px.bar(platform_counts, x='platform', y='count', color='platform', 
                             title="Document Counts by Platform", template="plotly_dark")
        st.plotly_chart(fig_platform, use_container_width=True)

    # TAB 2: Text Stats & Distributions
    with tab2:
        st.markdown("### Text Metrics & Sentiment Distributions")
        
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.write("#### Sentiment Classification Breakdown")
            sent_counts = df['sentiment'].value_counts().reset_index()
            fig_pie = px.pie(sent_counts, values='count', names='sentiment', 
                             color='sentiment', color_discrete_map={'Positive': '#4ade80', 'Neutral': '#60a5fa', 'Negative': '#f87171'},
                             title="Sentiment Distribution Ratio", hole=0.4, template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_s2:
            st.write("#### Document Word Count Histogram")
            fig_hist = px.histogram(df, x='word_count', nbins=20, 
                                    color='sentiment', color_discrete_map={'Positive': '#4ade80', 'Neutral': '#60a5fa', 'Negative': '#f87171'},
                                    title="Distribution of Word Counts", template="plotly_dark")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        st.write("---")
        st.write("#### Sentiment Polarity vs Character Length by Platform")
        fig_scatter = px.scatter(df, x='char_count', y='sentiment_score', color='platform',
                                 hover_data=['text'], size='word_count', size_max=20,
                                 title="Sentiment Score vs Character Count", template="plotly_dark")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.write("---")
        st.write("#### Platform Comparison (Violin Plot of Polarity)")
        fig_violin = px.violin(df, x='platform', y='sentiment_score', color='platform',
                               box=True, points="all", title="Linguistic Polarity Spread by Platform", template="plotly_dark")
        st.plotly_chart(fig_violin, use_container_width=True)

    # TAB 3: Advanced NLP Analytics
    with tab3:
        st.markdown("### N-Grams, Word Cloud & Timelines")
        
        col_nlp1, col_nlp2 = st.columns(2)
        
        with col_nlp1:
            st.write("#### Word Frequency Cloud")
            # Generate word cloud on cleaned text
            text_corpus = " ".join(df['cleaned_text'].astype(str))
            
            if text_corpus.strip():
                wordcloud = WordCloud(width=800, height=400, background_color='#0f172a', 
                                      colormap='viridis', max_words=100).generate(text_corpus)
                fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0f172a')
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.info("Insufficient text data to generate Word Cloud.")
                
        with col_nlp2:
            st.write("#### Top 10 Trending Keywords")
            # Calculate top words
            from src.business.trend_analyzer import TrendAnalyzer
            analyzer = TrendAnalyzer()
            top_words = analyzer.extract_trending_terms(df['cleaned_text'], top_n=10)
            if top_words:
                words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
                fig_word_bar = px.bar(words_df, x='Frequency', y='Word', orientation='h',
                                      color='Frequency', title="Top Keywords", template="plotly_dark")
                st.plotly_chart(fig_word_bar, use_container_width=True)
            else:
                st.info("No frequent words to show.")
                
        # N-grams Selection
        st.write("---")
        st.write("#### N-Gram Analysis")
        ngram_val = st.selectbox("Select N-Gram size", ["1-gram (Unigram)", "2-gram (Bigram)", "3-gram (Trigram)"])
        
        # Build n-grams
        from sklearn.feature_extraction.text import CountVectorizer
        n_size = 1 if "1-gram" in ngram_val else (2 if "2-gram" in ngram_val else 3)
        try:
            vec = CountVectorizer(ngram_range=(n_size, n_size), stop_words='english', max_features=10)
            dtm = vec.fit_transform(df['cleaned_text'])
            counts = dtm.sum(axis=0).A1
            ngram_features = vec.get_feature_names_out()
            ngram_df = pd.DataFrame({"N-Gram": ngram_features, "Count": counts}).sort_values(by="Count", ascending=True)
            
            fig_ngram = px.bar(ngram_df, x="Count", y="N-Gram", orientation="h", title=f"Top {ngram_val} Combinations", template="plotly_dark")
            st.plotly_chart(fig_ngram, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not build N-Grams: {e}. Corpus may be too small.")

        # Sentiment timeline
        st.write("---")
        st.write("#### Daily Sentiment Timeline")
        try:
            timeline_df = analyzer.aggregate_sentiment_timeline(df, freq='D')
            if not timeline_df.empty:
                fig_time = px.line(timeline_df, x='timestamp', y='average_polarity', 
                                   title="Sentiment Timeline (Daily Average Polarity)", template="plotly_dark")
                # Highlight zero line
                fig_time.add_hline(y=0.0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.info("Insufficient timeline data.")
        except Exception as e:
            st.error(f"Error drawing timeline chart: {e}")
            
        # Correlation Matrix
        st.write("---")
        st.write("#### Numeric Features Correlation Analysis")
        numeric_cols = ['char_count', 'word_count', 'sentence_count', 'avg_word_length', 'punctuation_count', 'emoji_count', 'capital_letter_ratio', 'sentiment_score']
        corr_matrix = df[numeric_cols].corr()
        
        # Draw Heatmap
        fig_heat = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', 
                             title="Correlation Matrix Heatmap", template="plotly_dark")
        st.plotly_chart(fig_heat, use_container_width=True)
