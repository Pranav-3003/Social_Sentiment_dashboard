import re
from typing import Dict, Any
import pandas as pd

class TextStatsExtractor:
    def __init__(self):
        pass

    def extract_stats(self, text: str) -> Dict[str, Any]:
        """
        Extracts various statistics from a single string.
        """
        if not isinstance(text, str) or not text.strip():
            return {
                "char_count": 0,
                "word_count": 0,
                "sentence_count": 0,
                "avg_word_length": 0.0,
                "punctuation_count": 0,
                "emoji_count": 0,
                "capital_letter_ratio": 0.0
            }
            
        char_count = len(text)
        words = text.split()
        word_count = len(words)
        
        # Sentences - split by . ! ?
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        sentence_count = len(sentences) if len(sentences) > 0 else 1
        
        avg_word_length = (
            sum(len(w) for w in words) / word_count
            if word_count > 0 else 0.0
        )
        
        # Punctuation count
        punctuation = re.findall(r'[^\w\s]', text)
        punctuation_count = len(punctuation)
        
        # Emoji count (Unicode characters representing emoji)
        emoji_pattern = re.compile(
            r'[\U00010000-\U0010ffff]|\u263a|\u2639|[\u2600-\u27BF]|[\u2300-\u23FF]|[\u2b50]'
        )
        emojis = emoji_pattern.findall(text)
        emoji_count = len(emojis)
        
        # Capital letter ratio
        capitals = sum(1 for c in text if c.isupper())
        total_letters = sum(1 for c in text if c.isalpha())
        capital_letter_ratio = (
            capitals / total_letters
            if total_letters > 0 else 0.0
        )
        
        return {
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": round(avg_word_length, 2),
            "punctuation_count": punctuation_count,
            "emoji_count": emoji_count,
            "capital_letter_ratio": round(capital_letter_ratio, 4)
        }

    def extract_features_df(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Applies stats extraction to an entire DataFrame and returns a new DataFrame with the features.
        """
        stats_list = []
        for text in df[text_column]:
            stats_list.append(self.extract_stats(str(text)))
            
        stats_df = pd.DataFrame(stats_list)
        return pd.concat([df.reset_index(drop=True), stats_df.reset_index(drop=True)], axis=1)
