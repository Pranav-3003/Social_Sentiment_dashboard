import streamlit as st
import pandas as pd
from textblob import TextBlob
from src.business.bi_engine import BIEngine
from src.utils.db import DBManager, Post

def render_chat(db_manager: DBManager):
    st.markdown("## AI Chat Assistant")
    st.write("Ask questions about the current social media dataset and get instant analytical answers.")
    
    # Session state to store conversation history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hello! I am SentiVerse Bot. Ask me anything about your social media data. E.g., 'Why is sentiment negative?', 'Which topics are trending?', or 'What should the company improve?'"}
        ]
        
    # Render chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    # User Input
    user_input = st.chat_input("Type your question here...")
    
    if user_input:
        # Append User message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
            
        # Generate Answer
        with st.chat_message("assistant"):
            with st.spinner("Analyzing dataset query..."):
                answer = generate_agent_response(user_input, db_manager)
                st.write(answer)
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})

def generate_agent_response(query: str, db_manager: DBManager) -> str:
    """
    Analyzes the user query intent and constructs a response based on the SQLite database state,
    supporting statistics, explanations, technical troubleshooting, and general conversation.
    """
    session = db_manager.get_session()
    posts = session.query(Post).all()
    session.close()
    
    q_lower = query.lower().strip()
    
    # 1. Handle Greetings & Basic Conversations
    if any(greet in q_lower for greet in ["hello", "hi", "hey", "greetings", "who are you", "what is your name", "help"]):
        return (
            "🤖 **Hello! I am SentiVerse Bot, your AI Social Media Analyst.**\n\n"
            "I can help you explore your dataset, explain NLP metrics, and troubleshoot platform features.\n\n"
            "**Try asking me:**\n"
            "- *'What are the dataset statistics?'*\n"
            "- *'Why is sentiment negative?'* or *'What should we improve?'*\n"
            "- *'Explain LIME and SHAP'* or *'What is TF-IDF?'*\n"
            "- *'How do I train a machine learning model?'*"
        )
        
    if not posts:
        return "⚠️ **I can't analyze the dataset because the database is currently empty.** Please navigate to the **Data Ingestion** page to load a demo feed or upload files first!"

    # Load dataframe
    df = pd.DataFrame([{
        "text": p.cleaned_text or p.text,
        "platform": p.platform,
        "topic": p.topic or "General",
        "timestamp": p.timestamp
    } for p in posts])
    
    # Sentiment scores
    sent_scores = []
    sentiments = []
    for text in df['text']:
        score = TextBlob(text).sentiment.polarity
        sent_scores.append(score)
        sentiments.append("Positive" if score > 0.05 else ("Negative" if score < -0.05 else "Neutral"))
    df['sentiment'] = sentiments
    df['sentiment_score'] = sent_scores

    # 2. INTENT: Dataset Statistics / Counts / Platforms
    if any(w in q_lower for w in ["stats", "statistics", "count", "how many", "dataset", "volume", "platform", "reputation", "summary", "overview"]):
        platform_counts = df['platform'].value_counts()
        platform_str = ", ".join([f"**{p.capitalize()}** ({c} posts)" for p, c in platform_counts.items()])
        
        pos_cnt = len(df[df['sentiment'] == 'Positive'])
        neg_cnt = len(df[df['sentiment'] == 'Negative'])
        neu_cnt = len(df[df['sentiment'] == 'Neutral'])
        avg_score = df['sentiment_score'].mean()
        
        bi = BIEngine()
        rep = bi.calculate_brand_reputation_score(df)
        csat = bi.calculate_csat_score(df)
        
        return (
            f"📊 **Here is the active SentiVerse Dataset Profile:**\n\n"
            f"- 📝 **Total Ingested Comments:** {len(df)} posts\n"
            f"- 🌐 **Distribution by Channel:** {platform_str}\n"
            f"- 🧠 **Average Polarity Score:** `{avg_score:.3f}`\n"
            f"- 🟢 **Positive Comments:** {pos_cnt} ({pos_cnt/len(df)*100:.1f}%)\n"
            f"- 🟡 **Neutral Comments:** {neu_cnt} ({neu_cnt/len(df)*100:.1f}%)\n"
            f"- 🔴 **Negative Comments:** {neg_cnt} ({neg_cnt/len(df)*100:.1f}%)\n"
            f"- 🏆 **Brand Reputation Score:** `{rep}/100`\n"
            f"- 😊 **Estimated Customer Satisfaction (CSAT):** `{csat}%`\n\n"
            f"Navigate to the **Exploratory Data Analysis (EDA)** tab to view detailed visual graphs of these distributions!"
        )

    # 3. INTENT: Why is sentiment negative? / What are complaints?
    if any(w in q_lower for w in ["negative", "complaint", "bad", "hate", "criticism", "crashed", "issue"]):
        neg_df = df[df['sentiment'] == 'Negative']
        if neg_df.empty:
            return "🎉 **Excellent news!** There are no negative posts in the database, indicating highly favorable sentiment."
            
        # Top words in negative comments
        from src.business.trend_analyzer import TrendAnalyzer
        analyzer = TrendAnalyzer()
        top_neg_words = analyzer.extract_trending_terms(neg_df['text'].tolist(), top_n=5)
        words_str = ", ".join([f"**'{w}'** ({count} times)" for w, count in top_neg_words])
        
        # Sample negative text
        sample_neg = neg_df['text'].iloc[0]
        if len(sample_neg) > 120:
            sample_neg = sample_neg[:120] + "..."
            
        return (
            f"🔴 **Negative Sentiment Deep-Dive:**\n\n"
            f"Linguistic clustering shows the primary keywords in customer complaints are: {words_str}.\n\n"
            f"💬 **Typical critical review:**\n"
            f"> \"{sample_neg}\"\n\n"
            f"For automated recommendations on how to address these performance, design, or billing issues, visit the **Business Intelligence** tab!"
        )

    # 4. INTENT: What should we improve? / Recommendations / Action
    if any(w in q_lower for w in ["improve", "recommend", "suggest", "action", "fix", "todo"]):
        bi = BIEngine()
        sugs = bi.generate_product_suggestions(df)
        
        if not sugs:
            return "No specific improvement guidelines could be compiled from the current comments."
            
        sugs_str = ""
        for s in sugs:
            sugs_str += f"- 🛠️ **{s['category']}**:\n"
            sugs_str += f"  - *Issue:* {s['issue']}\n"
            sugs_str += f"  - *Action:* {s['action']}\n"
            
        return (
            f"📋 **Actionable Business Recommendations:**\n\n"
            f"Based on semantic filtering of client complaints, I recommend the following tactical sprints:\n\n{sugs_str}"
        )

    # 5. INTENT: Technical Glossaries (LIME, SHAP, TF-IDF, POS)
    if "lime" in q_lower or "shap" in q_lower:
        return (
            "🔍 **Explainable AI (XAI) Frameworks:**\n\n"
            "- **SHAP (SHapley Additive exPlanations)**: Based on game theory, SHAP calculates the exact impact of each word by comparing predictions with and without that word. It provides globally consistent word attribution highlights.\n"
            "- **LIME (Local Interpretable Model-agnostic Explanations)**: LIME trains a local linear surrogate model around a specific text by perturbing it (deleting words). It shows which words were highly influential *locally* for that particular classification.\n\n"
            "You can test both methods interactively on custom comments in the **Explain Predictions (XAI)** tab!"
        )
        
    if "tf-idf" in q_lower or "tfidf" in q_lower or "vectoriz" in q_lower:
        return (
            "🔢 **Text Representation (Vectorization):**\n\n"
            "Machine Learning models cannot read raw text; they require numbers. **TF-IDF (Term Frequency-Inverse Document Frequency)** converts text into numerical arrays by calculating:\n"
            "1. **Term Frequency (TF)**: How often a word appears in a document.\n"
            "2. **Inverse Document Frequency (IDF)**: How unique a word is across the whole corpus.\n\n"
            "This highlights important, specific words while discounting common filler terms like *'the'* or *'is'*."
        )

    # 6. INTENT: User Guide / Troubleshooting ("how do I train", "how do I ingest", "how to use")
    if "train" in q_lower or "model" in q_lower or "workbench" in q_lower:
        return (
            "⚙️ **How to Train models in SentiVerse:**\n\n"
            "1. Go to the **Model Workbench** page using the sidebar.\n"
            "2. Under the **Classical Machine Learning** tab, select your vectorizer (TF-IDF/Count) and classifier (e.g. SVM, Logistic Regression, XGBoost).\n"
            "3. Click **Train Model** to run training and evaluate accuracies with learning curves and confusion matrices.\n"
            "4. Once trained, your model is active! You can test it on new text inside the **Predictions Portal** and interpret it inside **Explain Predictions (XAI)**."
        )
        
    if "ingest" in q_lower or "load" in q_lower or "add data" in q_lower:
        return (
            "📥 **How to Ingest Social Media Feeds:**\n\n"
            "1. Navigate to the **Data Ingestion** tab.\n"
            "2. Choose your ingestion channel (e.g. upload a CSV file, fetch comments from a Subreddit or YouTube video, or type manually).\n"
            "3. Click **Commit Dataset to Database** to save the records.\n"
            "4. *(Pro-tip)*: You can quickly clear the database or load a demo product feed containing 80 pre-populated posts at the bottom of the page."
        )

    # 7. Fallback Response
    return (
        "🤖 **I'm not sure how to answer that specific query.**\n\n"
        "As an AI Social Media Analyst, I can assist you with:\n"
        "1. **Dataset Overview**: *'Give me dataset statistics'* or *'What is our volume?'*\n"
        "2. **Customer Feedback**: *'Why is sentiment negative?'* or *'How can we improve?'*\n"
        "3. **Linguistic Terms**: *'What is TF-IDF?'* or *'Explain LIME and SHAP'* \n"
        "4. **User Manual**: *'How do I ingest data?'* or *'How do I train a classifier?'*"
    )
