import os
import logging
import random
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from deep_translator import GoogleTranslator

logger = logging.getLogger("Collector")

class SocialDataCollector:
    def __init__(self):
        pass

    def validate_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates the dataset, ensuring a 'text' and 'timestamp' columns exist.
        Handles duplicates and missing values.
        """
        issues = []
        original_shape = df.shape
        
        # 1. Check for text column
        text_col = None
        for col in df.columns:
            if col.lower() in ['text', 'post', 'comment', 'tweet', 'content', 'body']:
                text_col = col
                break
        
        if text_col:
            # Rename to standard 'text'
            df = df.rename(columns={text_col: 'text'})
        else:
            # If no text column found, check if there's any object/string column we can use
            string_cols = df.select_dtypes(include=['object']).columns
            if len(string_cols) > 0:
                text_col = string_cols[0]
                df = df.rename(columns={text_col: 'text'})
                issues.append(f"No explicit text column found. Guessed and renamed '{text_col}' to 'text'.")
            else:
                raise ValueError("Dataset must contain a text/content column.")

        # 2. Check for timestamp column
        ts_col = None
        for col in df.columns:
            if col.lower() in ['timestamp', 'date', 'created_at', 'time', 'created']:
                ts_col = col
                break
                
        if ts_col:
            df = df.rename(columns={ts_col: 'timestamp'})
            # Try parsing timestamp
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            except Exception:
                issues.append("Failed to parse timestamp column. Generating timestamps.")
                df['timestamp'] = datetime.utcnow()
        else:
            issues.append("No timestamp column found. Automatically generated timestamps.")
            # Generate random timestamps in the last 7 days
            dates = [datetime.utcnow() - timedelta(days=random.randint(0, 7), hours=random.randint(0, 23)) for _ in range(len(df))]
            df['timestamp'] = dates

        # Ensure text is string and fill missing values
        df['text'] = df['text'].astype(str).fillna("")
        df = df[df['text'].str.strip() != ""] # Remove empty rows
        
        # Drop duplicates
        dups_count = df.duplicated(subset=['text']).sum()
        if dups_count > 0:
            df = df.drop_duplicates(subset=['text'])
            issues.append(f"Removed {dups_count} duplicate posts.")
            
        # Ensure user column exists
        user_col = None
        for col in df.columns:
            if col.lower() in ['user', 'username', 'author', 'screen_name']:
                user_col = col
                break
        if user_col:
            df = df.rename(columns={user_col: 'username'})
        else:
            df['username'] = "anonymous"
            
        # Ensure platform column exists
        platform_col = None
        for col in df.columns:
            if col.lower() in ['platform', 'source']:
                platform_col = col
                break
        if platform_col:
            df = df.rename(columns={platform_col: 'platform'})
        else:
            df['platform'] = "file_upload"

        # Final cleanup: keep relevant columns
        cols_to_keep = ['text', 'timestamp', 'username', 'platform']
        for col in df.columns:
            if col not in cols_to_keep:
                df[col] = df[col] # Keep other metadata in dataframe, but we focus on these
                
        validation_stats = {
            "original_rows": original_shape[0],
            "final_rows": df.shape[0],
            "duplicates_removed": dups_count,
            "issues": issues
        }
        
        return df, validation_stats

    def load_file(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Loads CSV, Excel, or JSON files.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif ext == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV, Excel, or JSON.")
            
        return self.validate_dataset(df)

    def translate_text(self, text: str, source_lang: str = 'auto') -> str:
        """
        Translates text to English if not already in English.
        """
        if not text.strip():
            return text
        try:
            # We assume GoogleTranslator translates to English ('en')
            translator = GoogleTranslator(source=source_lang, target='en')
            return translator.translate(text)
        except Exception as e:
            logger.warning(f"Translation failed: {e}. Returning original text.")
            return text

    def collect_reddit(self, subreddit: str, limit: int = 50, 
                       client_id: str = "", client_secret: str = "", 
                       user_agent: str = "SentiVerseAI:v1.0.0") -> pd.DataFrame:
        """
        Fetches posts from Reddit. Falls back to mock data if credentials are empty.
        """
        if not client_id or not client_secret:
            logger.info("Reddit credentials missing. Fetching mock Reddit posts.")
            return self._generate_mock_posts(platform="reddit", count=limit, query=subreddit)
            
        try:
            import praw
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            posts = []
            for submission in reddit.subreddit(subreddit).hot(limit=limit):
                posts.append({
                    "text": f"{submission.title} {submission.selftext}",
                    "timestamp": datetime.utcfromtimestamp(submission.created_utc),
                    "username": submission.author.name if submission.author else "deleted",
                    "platform": "reddit"
                })
            df = pd.DataFrame(posts)
            df, _ = self.validate_dataset(df)
            return df
        except Exception as e:
            logger.error(f"Reddit collection error: {e}. Falling back to mock data.")
            return self._generate_mock_posts(platform="reddit", count=limit, query=subreddit)

    def collect_youtube(self, video_id: str, limit: int = 50, api_key: str = "") -> pd.DataFrame:
        """
        Fetches YouTube comments. Falls back to mock data if api_key is empty.
        """
        if not api_key:
            logger.info("YouTube API key missing. Fetching mock YouTube comments.")
            return self._generate_mock_posts(platform="youtube", count=limit, query=video_id)
            
        try:
            from googleapiclient.discovery import build
            youtube = build('youtube', 'v3', developerKey=api_key)
            
            # Fetch comments
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(limit, 100),
                textFormat="plainText"
            )
            response = request.execute()
            
            comments = []
            for item in response.get('items', []):
                snippet = item['snippet']['topLevelComment']['snippet']
                published_str = snippet['publishedAt'].replace('Z', '')
                try:
                    published = datetime.fromisoformat(published_str)
                except ValueError:
                    published = datetime.utcnow()
                    
                comments.append({
                    "text": snippet['textDisplay'],
                    "timestamp": published,
                    "username": snippet.get('authorDisplayName', 'anonymous'),
                    "platform": "youtube"
                })
                
            df = pd.DataFrame(comments)
            df, _ = self.validate_dataset(df)
            return df
        except Exception as e:
            logger.error(f"YouTube collection error: {e}. Falling back to mock data.")
            return self._generate_mock_posts(platform="youtube", count=limit, query=video_id)

    def collect_x(self, query: str, limit: int = 50, bearer_token: str = "") -> pd.DataFrame:
        """
        Fetches X/Twitter posts. Falls back to mock data if bearer_token is empty.
        """
        if not bearer_token:
            logger.info("X Bearer Token missing. Fetching mock X posts.")
            return self._generate_mock_posts(platform="x", count=limit, query=query)
            
        try:
            import tweepy
            client = tweepy.Client(bearer_token=bearer_token)
            response = client.search_recent_tweets(query=query, max_results=min(limit, 100), tweet_fields=['created_at', 'author_id'])
            
            tweets = []
            if response.data:
                for tweet in response.data:
                    tweets.append({
                        "text": tweet.text,
                        "timestamp": tweet.created_at or datetime.utcnow(),
                        "username": f"user_{tweet.author_id}",
                        "platform": "x"
                    })
            df = pd.DataFrame(tweets)
            df, _ = self.validate_dataset(df)
            return df
        except Exception as e:
            logger.error(f"X collection error: {e}. Falling back to mock data.")
            return self._generate_mock_posts(platform="x", count=limit, query=query)

    def _generate_mock_posts(self, platform: str, count: int = 50, query: str = "") -> pd.DataFrame:
        """
        Helper to generate realistic mock posts for testing the dashboard.
        """
        # Sentiment templates
        pos_templates = [
            "Just bought the new {query}! It is absolutely amazing and the battery life is stellar. Highly recommend it!",
            "I am in love with {query}. The design is so sleek and the performance is unmatched.",
            "Shoutout to the team behind {query}, they resolved my support ticket in 5 minutes! Incredible customer service.",
            "Wow, {query} is a game changer. Can't imagine my daily workflow without it now.",
            "Super happy with my purchase of {query}. Worth every single penny!",
            "The customer experience for {query} is just top tier. Loving the new updates too.",
            "A solid 10/10 for {query}. It is super intuitive and clean."
        ]
        
        neu_templates = [
            "Has anyone tried using {query} for data science tasks? Looking for some feedback.",
            "Checking out {query} today. It looks interesting, but let's see how it goes.",
            "They just released a new patch for {query}. Mostly minor bug fixes and UI updates.",
            "Comparing {query} with other alternatives. There are pros and cons to both.",
            "Just saw an advertisement for {query}. Will read some reviews later.",
            "I use {query} occasionally. It gets the job done, nothing more, nothing less.",
            "Interesting discussion about {query} in the tech community today."
        ]
        
        neg_templates = [
            "Extremely disappointed with {query}. It keeps crashing on my computer! Useless.",
            "Do NOT buy {query}. The customer support is terrible and it lacks key features.",
            "The new update of {query} completely ruined the user interface. It's so laggy now.",
            "I've been trying to get a refund for {query} for two weeks. They are ignoring my emails.",
            "Honestly, {query} is super overpriced for what it offers. Waste of time and money.",
            "I hate how loud the notification sounds are on {query}. And there is no way to turn them off!",
            "This {query} is the worst product I have bought this year. Broken out of the box."
        ]
        
        users = ["tech_guru", "data_wiz", "market_master", "review_king", "social_butterfly", "sceptic_sam", "early_adopter", "happy_buyer", "angry_user12", "curious_cat"]
        
        query_val = query if query else "product"
        
        posts = []
        for i in range(count):
            # Select random sentiment
            r = random.random()
            if r < 0.45:
                text = random.choice(pos_templates).format(query=query_val)
            elif r < 0.75:
                text = random.choice(neu_templates).format(query=query_val)
            else:
                text = random.choice(neg_templates).format(query=query_val)
                
            # Random date within last 30 days
            delta = timedelta(days=random.randint(0, 30), hours=random.randint(0, 23), minutes=random.randint(0, 59))
            timestamp = datetime.utcnow() - delta
            
            posts.append({
                "text": text,
                "timestamp": timestamp,
                "username": random.choice(users),
                "platform": platform
            })
            
        df = pd.DataFrame(posts)
        df, _ = self.validate_dataset(df)
        return df
