import logging
from typing import List, Tuple, Dict, Any, Optional
import subprocess
import sys

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Try downloading NLTK packages quietly on import (very fast if already downloaded)
for package in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'maxent_ne_chunker', 'maxent_ne_chunker_tab', 'words']:
    try:
        nltk.download(package, quiet=True)
    except Exception as e:
        logging.warning(f"Failed to download NLTK package {package}: {e}")

logger = logging.getLogger("NLPPipeline")

# Global lazy loaded spaCy model instance
_nlp_model = None

def get_spacy_model() -> Optional[Any]:
    """
    Lazy loads or downloads/loads the spaCy model on demand.
    """
    global _nlp_model
    if _nlp_model is not None:
        return _nlp_model
        
    try:
        import spacy
        try:
            _nlp_model = spacy.load("en_core_web_sm")
            logger.info("Successfully loaded existing spaCy 'en_core_web_sm' model.")
            return _nlp_model
        except OSError:
            logger.warning("spaCy 'en_core_web_sm' model not found. Attempting lazy download...")
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], stdout=subprocess.DEVNULL)
            try:
                _nlp_model = spacy.load("en_core_web_sm")
                logger.info("Successfully downloaded and loaded spaCy 'en_core_web_sm' model.")
                return _nlp_model
            except OSError:
                logger.error("Failed to load spaCy model after download attempt. Falling back to NLTK.")
                return None
    except ImportError:
        logger.warning("spaCy package is not installed. Falling back to NLTK.")
        return None

class NLPPipeline:
    def __init__(self, use_spacy: bool = False):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = set()
            
        # Determine if we should and can use spaCy
        self.use_spacy = False
        self.nlp = None
        
        if use_spacy:
            self.nlp = get_spacy_model()
            if self.nlp is not None:
                self.use_spacy = True
            else:
                logger.warning("use_spacy=True was requested, but spaCy model could not be loaded. Falling back to NLTK.")

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text into words.
        """
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            return [token.text for token in doc]
        else:
            return word_tokenize(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Removes stopwords from tokens list.
        """
        return [t for t in tokens if t.lower() not in self.stop_words]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatizes list of tokens.
        """
        if self.use_spacy and self.nlp:
            doc = self.nlp(" ".join(tokens))
            return [token.lemma_ for token in doc]
        else:
            return [self.lemmatizer.lemmatize(t) for t in tokens]

    def stem(self, tokens: List[str]) -> List[str]:
        """
        Stems list of tokens.
        """
        return [self.stemmer.stem(t) for t in tokens]

    def get_pos_tags(self, text: str) -> List[Tuple[str, str]]:
        """
        Returns POS tags for tokens.
        """
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            return [(token.text, token.pos_) for token in doc]
        else:
            tokens = self.tokenize(text)
            return nltk.pos_tag(tokens)

    def get_named_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extracts Named Entities. Returns list of tuples (entity_text, label).
        """
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            return [(ent.text, ent.label_) for ent in doc.ents]
        else:
            tokens = self.tokenize(text)
            tagged = nltk.pos_tag(tokens)
            chunks = nltk.ne_chunk(tagged)
            entities = []
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    ent_text = " ".join([c[0] for c in chunk])
                    entities.append((ent_text, chunk.label()))
            return entities

    def process_pipeline(self, text: str, 
                         remove_stops: bool = True, 
                         do_lemmatize: bool = True, 
                         do_stem: bool = False) -> Dict[str, Any]:
        """
        Runs the full NLP pipeline on a text string.
        """
        tokens = self.tokenize(text)
        raw_tokens_count = len(tokens)
        
        if remove_stops:
            tokens = self.remove_stopwords(tokens)
            
        processed_tokens = tokens
        if do_lemmatize:
            processed_tokens = self.lemmatize(processed_tokens)
        elif do_stem:
            processed_tokens = self.stem(processed_tokens)
            
        pos_tags = self.get_pos_tags(text)
        entities = self.get_named_entities(text)
        
        return {
            "processed_text": " ".join(processed_tokens),
            "tokens": processed_tokens,
            "token_count": len(processed_tokens),
            "raw_token_count": raw_tokens_count,
            "pos_tags": pos_tags,
            "entities": entities
        }
