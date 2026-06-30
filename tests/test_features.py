import pytest
import pandas as pd
import numpy as np
from src.features.text_stats import TextStatsExtractor
from src.features.vectorizers import TextVectorizer

def test_text_stats_extraction():
    extractor = TextStatsExtractor()
    stats = extractor.extract_stats("HELLO world! :)")
    assert stats["char_count"] == 15
    assert stats["word_count"] == 3
    # 5 capital letters in 'HELLO' out of 10 letters total
    assert stats["capital_letter_ratio"] == 0.5
    assert stats["punctuation_count"] == 3 # ! : )

def test_stats_extraction_dataframe():
    extractor = TextStatsExtractor()
    df = pd.DataFrame({"text": ["hello", "world!"]})
    df_feat = extractor.extract_features_df(df, text_column="text")
    assert "char_count" in df_feat.columns
    assert df_feat["char_count"].iloc[0] == 5
    assert df_feat["char_count"].iloc[1] == 6

def test_text_vectorizer_tfidf():
    vectorizer = TextVectorizer(method="tfidf", max_features=10)
    corpus = ["apple banana cherry", "banana orange cherry", "apple cherry peach"]
    X = vectorizer.fit_transform(corpus)
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == 3
    assert X.shape[1] <= 10
