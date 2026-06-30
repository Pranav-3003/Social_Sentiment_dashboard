import pytest
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.nlp_pipeline import NLPPipeline

def test_text_cleaner_urls():
    cleaner = TextCleaner({"remove_urls": True, "lowercase": True})
    cleaned, stats = cleaner.clean_text("Check this out http://google.com or visit www.website.org")
    assert "http" not in cleaned
    assert "www" not in cleaned
    assert stats["urls_removed"] == 2

def test_text_cleaner_contractions():
    cleaner = TextCleaner({"expand_contractions": True, "lowercase": True})
    cleaned, stats = cleaner.clean_text("I shouldn't fail, shouldn't I?")
    assert "should not" in cleaned
    assert stats["contractions_expanded"] == 2

def test_text_cleaner_punctuation():
    cleaner = TextCleaner({"remove_punctuation": True, "lowercase": True})
    cleaned, stats = cleaner.clean_text("Hello, world! This is a test...")
    assert "," not in cleaned
    assert "!" not in cleaned
    assert "." not in cleaned
    assert stats["punctuation_removed"] > 0

def test_nlp_pipeline_tokenization():
    pipeline = NLPPipeline(use_spacy=False)
    tokens = pipeline.tokenize("This is a simple sentence.")
    assert len(tokens) == 6
    assert tokens[0] == "This"
    assert tokens[-1] == "."

def test_nlp_pipeline_lemma():
    pipeline = NLPPipeline(use_spacy=False)
    lemmas = pipeline.lemmatize(["running", "cats", "is"])
    # WordNet Lemmatizer handles nouns by default, verbs with tag, but basic checks
    assert "running" in lemmas or "run" in lemmas
    assert "cats" not in lemmas # Should be lemmatized to 'cat'
    assert "cat" in lemmas
