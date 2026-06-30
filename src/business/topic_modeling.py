import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.decomposition import LatentDirichletAllocation, NMF, PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans

logger = logging.getLogger("TopicModeling")

class TopicModeler:
    def __init__(self, n_topics: int = 5, random_state: int = 42):
        self.n_topics = n_topics
        self.random_state = random_state
        self.model = None
        self.vectorizer = None

    def fit_lda(self, corpus: List[str]) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Fits Latent Dirichlet Allocation (LDA) model.
        Returns: (topics_keywords, document_topic_distributions).
        """
        logger.info(f"Fitting LDA model with {self.n_topics} topics...")
        self.vectorizer = CountVectorizer(max_features=2000, stop_words='english')
        dtm = self.vectorizer.fit_transform(corpus)
        
        self.model = LatentDirichletAllocation(
            n_components=self.n_topics, 
            random_state=self.random_state,
            n_jobs=-1
        )
        doc_topics = self.model.fit_transform(dtm)
        
        words = self.vectorizer.get_feature_names_out()
        topics_keywords = []
        for topic_idx, topic in enumerate(self.model.components_):
            top_words_idx = topic.argsort()[:-11:-1]
            top_words = [words[i] for i in top_words_idx]
            weights = [float(topic[i]) for i in top_words_idx]
            # Normalize weights
            sum_weights = sum(weights)
            weights = [w / sum_weights for w in weights] if sum_weights > 0 else weights
            
            topics_keywords.append({
                "topic_id": topic_idx,
                "keywords": top_words,
                "weights": weights
            })
            
        return topics_keywords, doc_topics

    def fit_nmf(self, corpus: List[str]) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Fits Non-Negative Matrix Factorization (NMF) model.
        Returns: (topics_keywords, document_topic_distributions).
        """
        logger.info(f"Fitting NMF model with {self.n_topics} topics...")
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        dtm = self.vectorizer.fit_transform(corpus)
        
        self.model = NMF(
            n_components=self.n_topics, 
            random_state=self.random_state,
            max_iter=500
        )
        doc_topics = self.model.fit_transform(dtm)
        
        words = self.vectorizer.get_feature_names_out()
        topics_keywords = []
        for topic_idx, topic in enumerate(self.model.components_):
            top_words_idx = topic.argsort()[:-11:-1]
            top_words = [words[i] for i in top_words_idx]
            weights = [float(topic[i]) for i in top_words_idx]
            sum_weights = sum(weights)
            weights = [w / sum_weights for w in weights] if sum_weights > 0 else weights
            
            topics_keywords.append({
                "topic_id": topic_idx,
                "keywords": top_words,
                "weights": weights
            })
            
        return topics_keywords, doc_topics

    def fit_bertopic_fallback(self, corpus: List[str], embeddings: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Simulates BERTopic by clustering sentence embeddings (using KMeans) and
        calculating Class-TF-IDF (c-TF-IDF) keywords for each cluster.
        Returns: (topics_keywords, document_topic_distributions).
        """
        logger.info(f"Fitting BERTopic fallback (K-Means + c-TF-IDF) with {self.n_topics} topics...")
        
        # 1. Cluster embeddings using K-Means
        kmeans = KMeans(n_clusters=self.n_topics, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Create document-topic distributions (one-hot or distance-based)
        # We can construct distance-based scores, normalized as probabilities
        distances = kmeans.transform(embeddings)
        # Invert distances to represent similarities
        similarities = 1.0 / (distances + 1e-5)
        doc_topics = similarities / similarities.sum(axis=1, keepdims=True)
        
        # 2. Compute Class-TF-IDF (join all texts in a cluster as a single document)
        cluster_docs = []
        for i in range(self.n_topics):
            docs_in_cluster = [corpus[j] for j in range(len(corpus)) if labels[j] == i]
            joined_docs = " ".join(docs_in_cluster)
            cluster_docs.append(joined_docs if joined_docs.strip() else "empty cluster")
            
        c_tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        c_tfidf_matrix = c_tfidf_vectorizer.fit_transform(cluster_docs)
        
        words = c_tfidf_vectorizer.get_feature_names_out()
        topics_keywords = []
        for topic_idx in range(self.n_topics):
            row = c_tfidf_matrix[topic_idx].toarray().flatten()
            top_words_idx = row.argsort()[:-11:-1]
            top_words = [words[i] for i in top_words_idx]
            weights = [float(row[i]) for i in top_words_idx]
            sum_weights = sum(weights)
            weights = [w / sum_weights for w in weights] if sum_weights > 0 else weights
            
            topics_keywords.append({
                "topic_id": topic_idx,
                "keywords": top_words,
                "weights": weights
            })
            
        return topics_keywords, doc_topics

    def get_topic_coordinates(self, embeddings: np.ndarray, doc_topics: np.ndarray) -> pd.DataFrame:
        """
        Reduces dimensionality of embeddings to 2D using PCA.
        Returns a DataFrame for plotting containing x, y, and dominant topic.
        """
        pca = PCA(n_components=2, random_state=self.random_state)
        coords = pca.fit_transform(embeddings)
        
        dominant_topic = np.argmax(doc_topics, axis=1)
        confidence = np.max(doc_topics, axis=1)
        
        df = pd.DataFrame({
            "x": coords[:, 0],
            "y": coords[:, 1],
            "topic": [f"Topic {t}" for t in dominant_topic],
            "confidence": confidence
        })
        return df
