import logging
from collections import Counter
import pandas as pd
from typing import Dict, Any, List, Tuple
from datetime import datetime

logger = logging.getLogger("TrendAnalyzer")

class TrendAnalyzer:
    def __init__(self):
        pass

    def extract_trending_terms(self, cleaned_texts: List[str], top_n: int = 15) -> List[Tuple[str, int]]:
        """
        Extracts the most frequent non-stop words.
        """
        all_words = []
        # Basic common stopwords if nltk isn't loaded yet
        basic_stops = {
            'the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'what', 'which', 'this',
            'that', 'these', 'those', 'then', 'just', 'so', 'than', 'such', 'very', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should', 'now',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'up', 'about', 'into', 'over', 'after', 'is',
            'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'out', 'off', 'over', 'under', 'again', 'further', 'once', 'here', 'there'
        }
        
        for text in cleaned_texts:
            if not isinstance(text, str):
                continue
            words = [w.lower().strip(".,?!;:-_()\"'") for w in text.split()]
            words = [w for w in words if w and w not in basic_stops and len(w) > 2]
            all_words.extend(words)
            
        counter = Counter(all_words)
        return counter.most_common(top_n)

    def extract_trending_hashtags(self, hashtags_lists: List[List[str]], top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Counts top hashtags across all posts.
        """
        all_tags = []
        for tags in hashtags_lists:
            if isinstance(tags, list):
                all_tags.extend([t.lower() for t in tags])
        counter = Counter(all_tags)
        return counter.most_common(top_n)

    def extract_trending_entities(self, entities_lists: List[List[Tuple[str, str]]], top_n: int = 10) -> Dict[str, List[Tuple[str, int]]]:
        """
        Groups NER entities by category (PERSON, ORG, GPE/LOCATION) and returns top items.
        """
        persons = []
        orgs = []
        gpes = []
        
        for entities in entities_lists:
            if not isinstance(entities, list):
                continue
            for ent_text, ent_label in entities:
                ent_text_clean = ent_text.strip()
                if not ent_text_clean or len(ent_text_clean) < 2:
                    continue
                label = ent_label.upper()
                if 'PERSON' in label or label == 'PER':
                    persons.append(ent_text_clean)
                elif 'ORG' in label:
                    orgs.append(ent_text_clean)
                elif 'GPE' in label or 'LOC' in label or 'GEOPOLITICAL' in label:
                    gpes.append(ent_text_clean)
                    
        return {
            "persons": Counter(persons).most_common(top_n),
            "organizations": Counter(orgs).most_common(top_n),
            "locations": Counter(gpes).most_common(top_n)
        }

    def aggregate_sentiment_timeline(self, df: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
        """
        Aggregates average sentiment scores and post volume over a given frequency (e.g. 'D' for daily, 'H' for hourly).
        """
        if 'timestamp' not in df.columns or 'sentiment_score' not in df.columns:
            return pd.DataFrame()
            
        temp_df = df.copy()
        temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
        
        # Set timestamp as index
        temp_df = temp_df.set_index('timestamp')
        
        # Resample
        resampled = temp_df.resample(freq)
        
        timeline = pd.DataFrame()
        timeline['average_polarity'] = resampled['sentiment_score'].mean().fillna(0.0)
        timeline['post_count'] = resampled.size()
        
        # Sentiment counts
        if 'sentiment' in df.columns:
            # Positive count
            timeline['positive_count'] = temp_df[temp_df['sentiment'] == 'Positive'].resample(freq).size().fillna(0)
            # Neutral count
            timeline['neutral_count'] = temp_df[temp_df['sentiment'] == 'Neutral'].resample(freq).size().fillna(0)
            # Negative count
            timeline['negative_count'] = temp_df[temp_df['sentiment'] == 'Negative'].resample(freq).size().fillna(0)
            
        return timeline.reset_index()

    def aggregate_topic_evolution(self, df: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
        """
        Traces topic frequency over time.
        """
        if 'timestamp' not in df.columns or 'topic' not in df.columns:
            return pd.DataFrame()
            
        temp_df = df.copy()
        temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
        
        # Group by timestamp (resampled) and topic
        grouped = temp_df.groupby([pd.Grouper(key='timestamp', freq=freq), 'topic']).size().unstack(fill_value=0)
        
        return grouped.reset_index()
