import logging
from typing import Dict, Any, Tuple, List
import numpy as np

logger = logging.getLogger("EmotionDetector")

# Emotion Lexicons
EMOTION_LEXICONS = {
    "joy": {
        "happy", "joy", "joyful", "glad", "delight", "wonderful", "excited", "celebrate", "smile", 
        "laughter", "cheer", "success", "pleasure", "thrilled", "great", "excellent", "awesome", "fantastic"
    },
    "sadness": {
        "sad", "sorrow", "cry", "weep", "unhappy", "depressing", "painful", "grief", "disappointed", 
        "mourn", "lonely", "blue", "heartbroken", "hurt", "tragedy", "loss", "gloomy", "defeat"
    },
    "anger": {
        "angry", "mad", "furious", "hate", "rage", "annoyed", "irritated", "resent", "outrage", 
        "offend", "wrath", "bitter", "fuming", "pissed", "scold", "frustrated"
    },
    "fear": {
        "fear", "afraid", "scared", "panic", "terrify", "anxiety", "worry", "dread", "nervous", 
        "horror", "fright", "spooky", "scary", "shaking", "threatened", "coward"
    },
    "surprise": {
        "surprise", "shock", "amazed", "astonished", "unexpected", "wonder", "startle", "sudden", 
        "unbelievable", "omg", "incredible", "stupefied", "miracle"
    },
    "love": {
        "love", "affection", "adore", "passionate", "heart", "sweetheart", "care", "beloved", 
        "romantic", "fondness", "darling", "cherish", "hug", "kiss"
    },
    "disgust": {
        "disgust", "gross", "sick", "repulsive", "loathe", "nasty", "yuck", "despise", 
        "revolting", "offensive", "distasteful", "dislike", "nausea"
    }
}

class EmotionDetector:
    def __init__(self):
        self.emotions = list(EMOTION_LEXICONS.keys())
        self.model = None # Can be loaded/trained for machine learning classification

    def predict_emotion(self, text: str, sentiment_score: float = 0.0) -> Tuple[str, float]:
        """
        Predicts one of the 7 emotions using a lexicon-based matcher weighted by sentiment polarity.
        Returns (emotion_label, confidence_score).
        """
        if not isinstance(text, str) or not text.strip():
            return "neutral", 1.0
            
        words = [w.lower().strip(".,?!;:-_()\"'") for w in text.split()]
        scores = {emotion: 0.0 for emotion in self.emotions}
        
        # Word matching
        for word in words:
            for emotion, lexicon in EMOTION_LEXICONS.items():
                if word in lexicon:
                    scores[emotion] += 1.0
                    
        # Apply sentiment weighting
        # If highly positive sentiment, boost positive emotions
        if sentiment_score > 0.2:
            scores["joy"] += sentiment_score * 1.5
            scores["love"] += sentiment_score * 1.2
            scores["surprise"] += sentiment_score * 0.5
        # If highly negative sentiment, boost negative emotions
        elif sentiment_score < -0.2:
            scores["sadness"] += abs(sentiment_score) * 1.2
            scores["anger"] += abs(sentiment_score) * 1.5
            scores["fear"] += abs(sentiment_score) * 0.8
            scores["disgust"] += abs(sentiment_score) * 1.0

        # Find maximum emotion
        max_emotion = "neutral"
        max_score = 0.0
        
        for emotion, score in scores.items():
            if score > max_score:
                max_score = score
                max_emotion = emotion
                
        # Calculate confidence/strength of emotion
        total_score = sum(scores.values())
        if total_score > 0:
            confidence = max_score / total_score
            # Normalize confidence between 0.4 and 0.99
            confidence = 0.4 + (confidence * 0.59)
        else:
            # If no matches, return neutral (or check sentiment)
            if sentiment_score > 0.2:
                max_emotion = "joy"
                confidence = 0.5
            elif sentiment_score < -0.2:
                max_emotion = "sadness"
                confidence = 0.5
            else:
                max_emotion = "neutral"
                confidence = 1.0
                
        return max_emotion, round(confidence, 2)

    def predict_batch(self, texts: List[str], sentiment_scores: List[float]) -> List[Tuple[str, float]]:
        results = []
        for text, score in zip(texts, sentiment_scores):
            results.append(self.predict_emotion(text, score))
        return results
