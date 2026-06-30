import streamlit as st
import pandas as pd
import numpy as np
import io
from textblob import TextBlob
from src.business.emotion_detector import EmotionDetector
from src.business.advanced_detectors import AdvancedDetectors
from src.utils.db import DBManager

def render_predictions(db_manager: DBManager):
    st.markdown("## Sentiment & Attribute Prediction Portal")
    
    # Check if a custom model is active
    active_model_name = st.session_state.get("active_model_name", "TextBlob Lexicon Engine (Default)")
    active_model = st.session_state.get("active_model", None)
    active_vect = st.session_state.get("active_vectorizer", None)
    
    st.info(f"**Activated Classifier:** {active_model_name}")

    # Initialize detectors
    emotion_detector = EmotionDetector()
    advanced_detectors = AdvancedDetectors()

    tab1, tab2 = st.tabs(["Single Text Profiler", "Batch Processing Portal"])

    # TAB 1: Single Text Profiler
    with tab1:
        st.markdown("### Profile a Single Document")
        input_text = st.text_area("Input Post Text", value="I am extremely happy with SentiVerse, but sometimes it runs a bit slow.")
        
        if st.button("Analyze Content", type="primary", key="btn_single_predict"):
            if input_text.strip():
                with st.spinner("Analyzing linguistic attributes..."):
                    
                    # 1. Sentiment Prediction
                    sentiment_label = "Neutral"
                    score = 0.0
                    
                    if active_model is not None:
                        try:
                            # If it's Keras DL workbench
                            if hasattr(active_model, "predict"):
                                if active_vect.method in ['word2vec', 'fasttext']:
                                    seq = active_vect.transform([input_text])
                                    pred_idx = active_model.predict(active_model_name, seq)[0]
                                    probs = active_model.predict_proba(active_model_name, seq)[0]
                                else: # sklearn vectorizer
                                    X = active_vect.transform([input_text])
                                    pred_idx = active_model.predict(X)[0]
                                    probs = active_model.predict_proba(X)[0] if hasattr(active_model, "predict_proba") else None
                                
                                labels_map = ["Negative", "Neutral", "Positive"]
                                sentiment_label = labels_map[pred_idx]
                                score = float(probs[2] - probs[0]) if probs is not None else (0.5 if sentiment_label == "Positive" else (-0.5 if sentiment_label == "Negative" else 0.0))
                            # If it's Transformer workbench
                            elif hasattr(active_model, "predict_proba") is False and hasattr(active_model, "tokenizer"):
                                pred_idx, conf = active_model.predict([input_text])
                                labels_map = ["Negative", "Neutral", "Positive"]
                                sentiment_label = labels_map[pred_idx[0]]
                                score = float(conf[0]) if sentiment_label == "Positive" else (float(-conf[0]) if sentiment_label == "Negative" else 0.0)
                            else: # Sklearn
                                X = active_vect.transform([input_text])
                                pred_idx = active_model.predict(X)[0]
                                probs = active_model.predict_proba(X)[0] if hasattr(active_model, "predict_proba") else None
                                labels_map = ["Negative", "Neutral", "Positive"]
                                sentiment_label = labels_map[pred_idx]
                                score = float(probs[2] - probs[0]) if probs is not None else (0.5 if sentiment_label == "Positive" else (-0.5 if sentiment_label == "Negative" else 0.0))
                        except Exception as e:
                            st.warning(f"Error predicting with custom model: {e}. Falling back to default lexicon engine.")
                            # Fallback
                            analysis = TextBlob(input_text)
                            score = analysis.sentiment.polarity
                            sentiment_label = "Positive" if score > 0.05 else ("Negative" if score < -0.05 else "Neutral")
                    else:
                        # Fallback Lexicon
                        analysis = TextBlob(input_text)
                        score = analysis.sentiment.polarity
                        sentiment_label = "Positive" if score > 0.05 else ("Negative" if score < -0.05 else "Neutral")

                    # 2. Emotion Prediction
                    emotion, emotion_conf = emotion_detector.predict_emotion(input_text, sentiment_score=score)
                    
                    # 3. Advanced attributes
                    adv_res = advanced_detectors.analyze_post(input_text, sentiment_score=score)

                    # RENDER RESULTS
                    st.write("---")
                    st.markdown("#### Profiler Findings")
                    
                    # Sentiment Display
                    sent_color = "#2ecc71" if sentiment_label == "Positive" else ("#e74c3c" if sentiment_label == "Negative" else "#3498db")
                    st.markdown(f"""
                    <div style='background-color:#1e293b; border-left: 8px solid {sent_color}; padding:15px; border-radius: 4px; margin-bottom:15px;'>
                        <span style='font-size:0.9rem; text-transform:uppercase; color:#94a3b8;'>Sentiment Polarity</span>
                        <div style='font-size:2rem; font-weight:bold; color:{sent_color};'>{sentiment_label} <span style='font-size:1.1rem; color:#64748b;'>({score:.2f})</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Emotion Display
                    st.markdown(f"""
                    <div style='background-color:#1e293b; border-left: 8px solid #9b59b6; padding:15px; border-radius: 4px; margin-bottom:15px;'>
                        <span style='font-size:0.9rem; text-transform:uppercase; color:#94a3b8;'>Primary Emotion</span>
                        <div style='font-size:2rem; font-weight:bold; color:#d98880;'>{emotion.upper()} <span style='font-size:1.1rem; color:#64748b;'>({int(emotion_conf*100)}% Confidence)</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Advanced Flags Display
                    st.markdown("#### Quality and Content Attributes")
                    cols_adv = st.columns(5)
                    
                    with cols_adv[0]:
                        val = "⚠️ Yes" if adv_res["is_toxic"] else "✅ Safe"
                        st.metric("Toxic Comment", val, f"{int(adv_res['toxic_confidence']*100)}% conf")
                    with cols_adv[1]:
                        val = "⚠️ Yes" if adv_res["is_sarcastic"] else "❌ No"
                        st.metric("Sarcasm Detected", val, f"{int(adv_res['sarcasm_confidence']*100)}% conf")
                    with cols_adv[2]:
                        val = "⚠️ Yes" if adv_res["is_spam"] else "❌ No"
                        st.metric("Spam Classification", val, f"{int(adv_res['spam_confidence']*100)}% conf")
                    with cols_adv[3]:
                        val = "⚠️ Yes" if adv_res["is_fake"] else "❌ No"
                        st.metric("Fake News Indicator", val, f"{int(adv_res['fake_confidence']*100)}% conf")
                    with cols_adv[4]:
                        val = "🤖 Bot" if adv_res["is_bot"] else "👤 Human"
                        st.metric("Account Profiler", val, f"{int(adv_res['bot_confidence']*100)}% conf")
            else:
                st.error("Please enter some text.")

    # TAB 2: Batch Processing
    with tab2:
        st.markdown("### Process a Batch File")
        st.write("Upload a file containing social media posts, run predictions, and export the annotated results.")
        
        batch_file = st.file_uploader("Upload batch file (CSV/JSON/Excel)", type=["csv", "xlsx", "xls", "json"], key="batch_pred_uploader")
        
        if batch_file:
            try:
                if batch_file.name.endswith('.csv'):
                    batch_df = pd.read_csv(batch_file)
                elif batch_file.name.endswith(('.xlsx', '.xls')):
                    batch_df = pd.read_excel(batch_file)
                else:
                    batch_df = pd.read_json(batch_file)
                    
                # Locate text column
                text_col = None
                for col in batch_df.columns:
                    if col.lower() in ['text', 'post', 'comment', 'tweet', 'content', 'body']:
                        text_col = col
                        break
                        
                if not text_col:
                    st.error("Uploaded file must contain a column named 'text' or similar.")
                    return
                    
                st.success("File ingested. Ready to predict.")
                
                if st.button("Run Batch Predictions", type="primary", key="btn_batch_predict"):
                    batch_df['text'] = batch_df[text_col].astype(str).fillna("")
                    
                    sentiments = []
                    polarity_scores = []
                    emotions = []
                    toxicity = []
                    sarcasms = []
                    spams = []
                    
                    progress_bar = st.progress(0)
                    total_rows = len(batch_df)
                    
                    for idx, row in batch_df.iterrows():
                        text = row['text']
                        
                        # 1. Sentiment
                        score = TextBlob(text).sentiment.polarity
                        if active_model is not None:
                            try:
                                if hasattr(active_model, "predict"):
                                    if active_vect.method in ['word2vec', 'fasttext']:
                                        seq = active_vect.transform([text])
                                        p_idx = active_model.predict(active_model_name, seq)[0]
                                    else:
                                        X = active_vect.transform([text])
                                        p_idx = active_model.predict(X)[0]
                                    labels_map = ["Negative", "Neutral", "Positive"]
                                    sentiment = labels_map[p_idx]
                                else:
                                    sentiment = "Positive" if score > 0.05 else ("Negative" if score < -0.05 else "Neutral")
                            except Exception:
                                sentiment = "Positive" if score > 0.05 else ("Negative" if score < -0.05 else "Neutral")
                        else:
                            sentiment = "Positive" if score > 0.05 else ("Negative" if score < -0.05 else "Neutral")
                            
                        # 2. Emotion
                        emo, _ = emotion_detector.predict_emotion(text, sentiment_score=score)
                        
                        # 3. Advanced
                        adv = advanced_detectors.analyze_post(text, sentiment_score=score)
                        
                        sentiments.append(sentiment)
                        polarity_scores.append(score)
                        emotions.append(emo)
                        toxicity.append(adv["is_toxic"])
                        sarcasms.append(adv["is_sarcastic"])
                        spams.append(adv["is_spam"])
                        
                        # Update progress
                        progress_bar.progress((idx + 1) / total_rows)
                        
                    batch_df['predicted_sentiment'] = sentiments
                    batch_df['sentiment_score'] = polarity_scores
                    batch_df['predicted_emotion'] = emotions
                    batch_df['is_toxic'] = toxicity
                    batch_df['is_sarcastic'] = sarcasms
                    batch_df['is_spam'] = spams
                    
                    st.success("Batch predictions completed!")
                    st.dataframe(batch_df.head(10))
                    
                    # Export options
                    st.write("#### Download Annotated File")
                    col_ex1, col_ex2, col_ex3 = st.columns(3)
                    
                    with col_ex1:
                        csv_data = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download CSV", csv_data, "sentiment_predictions.csv", "text/csv")
                    with col_ex2:
                        # Write to excel bytes
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            batch_df.to_excel(writer, index=False, sheet_name='Predictions')
                        excel_data = output.getvalue()
                        st.download_button("Download Excel", excel_data, "sentiment_predictions.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    with col_ex3:
                        json_data = batch_df.to_json(orient='records').encode('utf-8')
                        st.download_button("Download JSON", json_data, "sentiment_predictions.json", "application/json")
            except Exception as e:
                st.error(f"Error running batch predictions: {e}")
