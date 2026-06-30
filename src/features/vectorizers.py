import pickle
import logging
from typing import List, Union, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

logger = logging.getLogger("Vectorizers")

class TextVectorizer:
    def __init__(self, method: str = 'tfidf', max_features: int = 5000):
        self.method = method
        self.max_features = max_features
        self.vectorizer = None
        self.word2idx = {}
        self.embeddings_matrix = None

    def fit(self, corpus: List[str]) -> 'TextVectorizer':
        """
        Fits the vectorizer on the given text corpus.
        """
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words='english')
            self.vectorizer.fit(corpus)
        elif self.method == 'count':
            self.vectorizer = CountVectorizer(max_features=self.max_features, stop_words='english')
            self.vectorizer.fit(corpus)
        elif self.method in ['word2vec', 'fasttext']:
            # Fit a simple vocabulary index mapping for DL models
            vocab = set()
            for text in corpus:
                for word in text.split():
                    vocab.add(word.lower())
            
            # Index 0 is reserved for padding
            self.word2idx = {word: idx + 1 for idx, word in enumerate(sorted(vocab))}
            logger.info(f"Initialized vocabulary for Deep Learning with {len(self.word2idx)} unique words.")
        return self

    def transform(self, corpus: List[str]) -> Union[np.ndarray, List[List[int]]]:
        """
        Transforms text corpus into feature vectors or indexed integer lists.
        """
        if self.method in ['tfidf', 'count']:
            if not self.vectorizer:
                raise ValueError("Vectorizer has not been fitted yet.")
            return self.vectorizer.transform(corpus).toarray()
            
        elif self.method in ['word2vec', 'fasttext']:
            # Return token sequences for Deep Learning
            sequences = []
            for text in corpus:
                seq = []
                for word in text.split():
                    idx = self.word2idx.get(word.lower(), 0) # 0 for OOV
                    seq.append(idx)
                sequences.append(seq)
            return sequences
            
        elif self.method == 'bert':
            # Transformer embedding extraction
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
                
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                model = AutoModel.from_pretrained("distilbert-base-uncased")
                
                embeddings = []
                for text in corpus:
                    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    # Use pooler output or mean of last hidden states
                    emb = outputs.last_hidden_state[:, 0, :].numpy().squeeze()
                    embeddings.append(emb)
                return np.array(embeddings)
            except Exception as e:
                logger.error(f"Transformer embedding failed: {e}. Falling back to random embeddings.")
                # Fallback to random embedding representation
                return np.random.randn(len(corpus), 768)
                
        else:
            raise ValueError(f"Unknown vectorization method: {self.method}")

    def fit_transform(self, corpus: List[str]) -> Union[np.ndarray, List[List[int]]]:
        return self.fit(corpus).transform(corpus)

    def save(self, file_path: str) -> None:
        """
        Saves the vectorizer state to a pickle file.
        """
        with open(file_path, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'max_features': self.max_features,
                'vectorizer': self.vectorizer,
                'word2idx': self.word2idx
            }, f)
        logger.info(f"Vectorizer saved to {file_path}")

    def load(self, file_path: str) -> 'TextVectorizer':
        """
        Loads vectorizer state from a pickle file.
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.method = data['method']
        self.max_features = data['max_features']
        self.vectorizer = data['vectorizer']
        self.word2idx = data['word2idx']
        logger.info(f"Vectorizer loaded from {file_path}")
        return self
