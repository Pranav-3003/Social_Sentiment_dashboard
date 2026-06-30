import streamlit as st
import pandas as pd
from datetime import datetime
from src.data.collector import SocialDataCollector
from src.utils.db import DBManager

def render_ingestion(db_manager: DBManager):
    st.markdown("## Data Ingestion Portal")
    st.write("Collect social media comments and posts from multiple channels, translate them, and commit them to the database.")
    
    collector = SocialDataCollector()
    
    ingest_source = st.radio(
        "Choose Ingestion Channel",
        ["File Upload (CSV/Excel/JSON)", "Reddit Subreddit", "YouTube Video Comments", "X / Twitter Query", "Manual Text Entry"]
    )
    
    df_loaded = None
    source_platform = "unknown"
    
    # 1. File Upload
    if ingest_source == "File Upload (CSV/Excel/JSON)":
        uploaded_file = st.file_uploader("Upload comment feed", type=["csv", "xlsx", "xls", "json"])
        if uploaded_file:
            try:
                # Save file to a temporary location to load it, or load directly using pandas
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_json(uploaded_file)
                    
                df_loaded, stats = collector.validate_dataset(df)
                source_platform = "file_upload"
                st.success(f"Successfully validated dataset! Ingested {len(df_loaded)} unique posts.")
                if stats["duplicates_removed"] > 0:
                    st.info(f"Duplicate check: Removed {stats['duplicates_removed']} duplicate records.")
                for issue in stats["issues"]:
                    st.warning(issue)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                
    # 2. Reddit Ingestion
    elif ingest_source == "Reddit Subreddit":
        cols = st.columns(2)
        with cols[0]:
            subreddit = st.text_input("Subreddit (e.g. technology, movies)", value="technology")
            limit = st.slider("Max Posts", 10, 200, 50)
        with cols[1]:
            st.markdown("**Reddit API Credentials (Optional)**")
            client_id = st.text_input("Client ID", type="password")
            client_secret = st.text_input("Client Secret", type="password")
            
        if st.button("Collect Reddit Comments"):
            with st.spinner("Fetching Reddit posts..."):
                df_loaded = collector.collect_reddit(
                    subreddit=subreddit, 
                    limit=limit, 
                    client_id=client_id, 
                    client_secret=client_secret
                )
                source_platform = "reddit"
                if not client_id:
                    st.info("No credentials provided. Mock Reddit feed generated for testing.")
                st.success(f"Collected {len(df_loaded)} Reddit posts!")
                
    # 3. YouTube Ingestion
    elif ingest_source == "YouTube Video Comments":
        cols = st.columns(2)
        with cols[0]:
            video_id = st.text_input("YouTube Video ID (e.g. dQw4w9WgXcQ)", value="dQw4w9WgXcQ")
            limit = st.slider("Max Comments", 10, 200, 50)
        with cols[1]:
            st.markdown("**YouTube Developer Credentials (Optional)**")
            api_key = st.text_input("YouTube API Key", type="password")
            
        if st.button("Collect YouTube Comments"):
            with st.spinner("Fetching YouTube comments..."):
                df_loaded = collector.collect_youtube(
                    video_id=video_id,
                    limit=limit,
                    api_key=api_key
                )
                source_platform = "youtube"
                if not api_key:
                    st.info("No API key provided. Mock YouTube comments generated for testing.")
                st.success(f"Collected {len(df_loaded)} YouTube comments!")
                
    # 4. X Ingestion
    elif ingest_source == "X / Twitter Query":
        cols = st.columns(2)
        with cols[0]:
            query = st.text_input("Search Term / Hashtag", value="AI")
            limit = st.slider("Max Tweets", 10, 200, 50)
        with cols[1]:
            st.markdown("**X API Credentials (Optional)**")
            bearer_token = st.text_input("Bearer Token", type="password")
            
        if st.button("Collect Tweets"):
            with st.spinner("Fetching tweets..."):
                df_loaded = collector.collect_x(
                    query=query,
                    limit=limit,
                    bearer_token=bearer_token
                )
                source_platform = "x"
                if not bearer_token:
                    st.info("No Bearer Token provided. Mock Twitter feed generated for testing.")
                st.success(f"Collected {len(df_loaded)} tweets!")
                
    # 5. Manual Entry
    else:
        username = st.text_input("Username / Handle", value="guest_user")
        manual_text = st.text_area("Write Post Content", placeholder="Type social media content here...")
        translate_opt = st.checkbox("Automatically translate to English")
        
        if st.button("Submit Manual Post"):
            if manual_text.strip():
                final_text = manual_text
                if translate_opt:
                    with st.spinner("Translating text..."):
                        final_text = collector.translate_text(manual_text)
                        
                df_loaded = pd.DataFrame([{
                    "text": final_text,
                    "timestamp": datetime.utcnow(),
                    "username": username,
                    "platform": "manual"
                }])
                source_platform = "manual"
                st.success("Manual post validated!")
            else:
                st.error("Please enter some text.")

    # Commit Collected Data
    if df_loaded is not None and not df_loaded.empty:
        st.write("---")
        st.markdown("### Raw Dataset Preview")
        st.dataframe(df_loaded[['text', 'username', 'platform', 'timestamp']].head(10))
        
        # Translation Options for entire dataset if upload
        st.write("### Data Prep Pipeline")
        translate_dataset = st.checkbox("Run English Translation Pipeline on Ingestion")
        
        if st.button("Commit Dataset to Database", key="commit_db_btn"):
            with st.spinner("Processing translations and database commit..."):
                if translate_dataset:
                    # Translate non-English items
                    translated_texts = []
                    for t in df_loaded['text']:
                        translated_texts.append(collector.translate_text(t))
                    df_loaded['text'] = translated_texts
                    st.info("Dataset translation checks completed.")
                
                # Convert DataFrame to list of dictionaries
                posts_data = df_loaded.to_dict('records')
                
                # Commit
                success = db_manager.add_posts_batch(posts_data)
                if success:
                    st.success(f"Successfully saved {len(posts_data)} posts to SentiVerse database!")
                    # Force page refresh or update stats
                    st.rerun()
                else:
                    st.error("Failed to commit posts to database.")

    # Database Maintenance Section
    st.write("---")
    st.markdown("### Database Operations")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("Clear all posts to start a clean session with a new dataset.")
        if st.button("Clear SQLite Database", type="secondary"):
            db_manager.clear_all_posts()
            st.warning("Database cleared!")
            st.rerun()
            
    with col2:
        st.info("Load pre-configured demo dataset containing mixed positive, neutral, and negative posts.")
        if st.button("Load Mock Product Feed", type="primary"):
            # Generate mock dataset for typical company product
            mock_df = collector._generate_mock_posts(platform="all_channels", count=80, query="SentiVerse App")
            posts_data = mock_df.to_dict('records')
            db_manager.add_posts_batch(posts_data)
            st.success("Demo dataset loaded!")
            st.rerun()
