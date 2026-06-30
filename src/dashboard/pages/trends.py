import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from src.business.trend_analyzer import TrendAnalyzer
from src.business.forecaster import SentimentForecaster
from src.preprocessing.nlp_pipeline import NLPPipeline
from src.preprocessing.cleaner import TextCleaner
from src.utils.db import DBManager, Post
from textblob import TextBlob

def render_trends(db_manager: DBManager):
    st.markdown("## Trend Intelligence & Predictive Forecasting Center")
    
    session = db_manager.get_session()
    posts = session.query(Post).all()
    session.close()
    
    if not posts:
        st.warning("No data found in the database. Load or collect a dataset first before analyzing trends!")
        return

    # Load into DataFrame
    df = pd.DataFrame([{
        "text": p.cleaned_text or p.text,
        "raw_text": p.text,
        "username": p.username,
        "platform": p.platform,
        "timestamp": p.timestamp
    } for p in posts])

    # Extract sentiment values
    sent_scores = []
    sent_labels = []
    for text in df['text']:
        score = TextBlob(text).sentiment.polarity
        sent_scores.append(score)
        sent_labels.append("Positive" if score > 0.05 else ("Negative" if score < -0.05 else "Neutral"))
        
    df['sentiment_score'] = sent_scores
    df['sentiment'] = sent_labels

    tab1, tab2 = st.tabs(["Linguistic & Named Entity Trends", "Sentiment Forecasting"])

    # TAB 1: Trends
    with tab1:
        st.markdown("### Top Keywords, Hashtags & Entity Tracking")
        
        analyzer = TrendAnalyzer()
        
        col_tr1, col_tr2 = st.columns(2)
        
        with col_tr1:
            st.write("#### Most Frequent Words")
            top_words = analyzer.extract_trending_terms(df['text'], top_n=10)
            if top_words:
                st.dataframe(pd.DataFrame(top_words, columns=["Word", "Occurrence Count"]))
            else:
                st.info("No words to display.")
                
            st.write("#### Most Frequent Hashtags")
            # Extract hashtags list from raw texts using cleaner
            cleaner = TextCleaner()
            all_hashtags = []
            for t in df['raw_text']:
                _, stats = cleaner.clean_text(t)
                all_hashtags.append(stats["hashtags_extracted"])
                
            top_tags = analyzer.extract_trending_hashtags(all_hashtags, top_n=10)
            if top_tags:
                st.dataframe(pd.DataFrame(top_tags, columns=["Hashtag", "Occurrence Count"]))
            else:
                st.info("No hashtags detected in the dataset posts.")

        with col_tr2:
            st.write("#### Top Mentioned Entities (NER)")
            # Extract entities using NLTK/spaCy on sample of posts to avoid high CPU latency
            nlp = NLPPipeline(use_spacy=False) # Fallback to NLTK for speed/stability
            
            entities_list = []
            # Sample at most 50 posts for entity extraction in real-time
            sample_size = min(len(df), 50)
            sampled_texts = df['text'].sample(sample_size, random_state=42).tolist()
            
            with st.spinner("Extracting named entities from sample corpus..."):
                for txt in sampled_texts:
                    ents = nlp.get_named_entities(txt)
                    entities_list.append(ents)
                    
            entity_trends = analyzer.extract_trending_entities(entities_list, top_n=8)
            
            st.write("**Organizations (ORG)**")
            if entity_trends["organizations"]:
                st.dataframe(pd.DataFrame(entity_trends["organizations"], columns=["Organization", "Frequency"]))
            else:
                st.info("No organizations detected.")
                
            st.write("**People (PERSON)**")
            if entity_trends["persons"]:
                st.dataframe(pd.DataFrame(entity_trends["persons"], columns=["Person Name", "Frequency"]))
            else:
                st.info("No people detected.")
                
            st.write("**Geopolitical Places (GPE)**")
            if entity_trends["locations"]:
                st.dataframe(pd.DataFrame(entity_trends["locations"], columns=["Location", "Frequency"]))
            else:
                st.info("No locations/GPE detected.")

    # TAB 2: Forecasting
    with tab2:
        st.markdown("### Predictive Sentiment Forecasting")
        st.write("Fit a time-series model on historical sentiment average to forecast polarity over the next week.")
        
        forecast_days = st.slider("Forecast Period (Days)", 3, 14, 7)
        
        # Resample daily averages
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        timeline_df = analyzer.aggregate_sentiment_timeline(df, freq='D')
        
        if len(timeline_df) < 3:
            st.warning("Insufficient timeline dates in history to run predictive models. Please load posts spread across multiple days.")
            # Show a mock timeline to demonstrate forecast UI
            st.info("Displaying Mock 14-day timeline & forecast preview.")
            mock_dates = pd.date_range(end=datetime.now(), periods=10)
            timeline_df = pd.DataFrame({
                "timestamp": mock_dates,
                "average_polarity": np.sin(np.linspace(0, 3, 10)) * 0.3 + np.random.uniform(-0.1, 0.1, 10),
                "post_count": [10] * 10
            })
            
        forecaster = SentimentForecaster()
        forecast_df, model_desc = forecaster.forecast_sentiment(timeline_df, target_col='average_polarity', steps=forecast_days)
        
        st.info(f"**Fitted Model:** {model_desc}")
        
        # Plotly chart with bounds
        fig = go.Figure()
        
        # 1. Historical Polarity
        fig.add_trace(go.Scatter(
            x=timeline_df['timestamp'], y=timeline_df['average_polarity'],
            name="Historical Polarity", line=dict(color="#3b82f6", width=3)
        ))
        
        # 2. Shaded Confidence Interval bounds
        # Lower bound
        fig.add_trace(go.Scatter(
            x=forecast_df['timestamp'], y=forecast_df['lower_bound'],
            showlegend=False, line=dict(color="rgba(16, 185, 129, 0)"),
        ))
        # Upper bound filled to lower bound
        fig.add_trace(go.Scatter(
            x=forecast_df['timestamp'], y=forecast_df['upper_bound'],
            name="95% Confidence Interval", fill='tonexty',
            fillcolor='rgba(16, 185, 129, 0.15)', line=dict(color="rgba(16, 185, 129, 0)")
        ))
        
        # 3. Forecast values
        fig.add_trace(go.Scatter(
            x=forecast_df['timestamp'], y=forecast_df['forecast'],
            name="Predicted Polarity", line=dict(color="#10b981", width=3, dash='dash')
        ))
        
        fig.add_hline(y=0.0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title="Future Sentiment Direction Forecast",
            xaxis_title="Date",
            yaxis_title="Average Sentiment Polarity",
            template="plotly_dark",
            yaxis=dict(range=[-1.05, 1.05])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("#### Forecast Values Table")
        st.dataframe(forecast_df)
