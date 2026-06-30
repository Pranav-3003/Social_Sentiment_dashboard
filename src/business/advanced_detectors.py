import re
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("AdvancedDetectors")

# Lexicons for advanced detections
TOXIC_KEYWORDS = {
    "abuse", "abusive", "hate", "idiot", "jerk", "stupid", "moron", "loser", 
    "trash", "dumb", "disgusting", "kill yourself", "die", "screw you"
}

SPAM_KEYWORDS = {
    "make money", "work from home", "click here", "free trial", "buy now", 
    "guaranteed", "earn cash", "special offer", "subscribe", "follow me", 
    "giveaway", "winner", "prize", "discount", "promo code", "visit website"
}

FAKE_NEWS_BUZZWORDS = {
    "unbelievable", "shocking", "will blow your mind", "doctors hate him", 
    "secret trick", "conspiracy", "exposed", "miracle cure", "proof", 
    "they don't want you to know", "alert", "breaking news click"
}

class AdvancedDetectors:
    def __init__(self):
        pass

    def detect_toxic(self, text: str) -> Tuple[bool, float]:
        """
        Detects if a post contains toxic content based on word matching and exclamation density.
        """
        if not text.strip():
            return False, 0.0
            
        words = text.lower().split()
        matches = [w for w in words if any(kw in w for kw in TOXIC_KEYWORDS)]
        
        # Calculate toxicity score
        toxicity_score = len(matches) / max(len(words), 1) * 5.0
        # Boost if lots of capital letters or exclamation marks
        exclamations = text.count('!')
        if exclamations > 2:
            toxicity_score += 0.2
            
        is_toxic = toxicity_score > 0.15 or len(matches) >= 1
        confidence = min(0.5 + (toxicity_score * 2.0), 0.99) if is_toxic else min(0.5 + (1 - toxicity_score), 0.99)
        
        return bool(is_toxic), round(confidence, 2)

    def detect_spam(self, text: str) -> Tuple[bool, float]:
        """
        Detects spam using links, cash keywords, and repetitive words.
        """
        if not text.strip():
            return False, 0.0
            
        text_lower = text.lower()
        
        # Check URLs
        url_count = len(re.findall(r'https?://\S+|www\.\S+', text_lower))
        
        # Keyword matches
        keyword_matches = sum(1 for kw in SPAM_KEYWORDS if kw in text_lower)
        
        spam_score = (url_count * 0.4) + (keyword_matches * 0.3)
        
        is_spam = spam_score > 0.5
        confidence = min(0.5 + spam_score, 0.99) if is_spam else min(0.99, 0.5 + (1.0 - spam_score))
        
        return bool(is_spam), round(confidence, 2)

    def detect_sarcasm(self, text: str, sentiment_score: float = 0.0) -> Tuple[bool, float]:
        """
        Detects sarcasm. Sarcasm often features positive sentiments with negative qualifiers,
        or words like "obviously", "totally", "surely", "great job" in a negative context.
        """
        if not text.strip():
            return False, 0.0
            
        text_lower = text.lower()
        sarcasm_keywords = ["obviously", "totally", "surely", "clearly", "oh great", "so thrilled", "brilliant"]
        
        matches = sum(1 for kw in sarcasm_keywords if kw in text_lower)
        
        # Look for heavy contrast: e.g. "I love it when it crashes!"
        has_contrast = False
        if "love" in text_lower or "great" in text_lower or "best" in text_lower or "wonderful" in text_lower:
            if "crash" in text_lower or "hate" in text_lower or "fail" in text_lower or "broke" in text_lower or "worst" in text_lower:
                has_contrast = True
                
        # Sarcasm indicator: exclamation marks inside quote or trailing
        has_sarcastic_punctuation = text.count('!') > 1 or "?!" in text or "..." in text
        
        sarcasm_score = (matches * 0.3) + (0.5 if has_contrast else 0.0) + (0.2 if has_sarcastic_punctuation else 0.0)
        
        # Sarcasm is highly likely if polarity is opposite to content tone
        is_sarcastic = sarcasm_score >= 0.4
        confidence = min(0.5 + sarcasm_score, 0.99) if is_sarcastic else min(0.99, 0.5 + (1.0 - sarcasm_score))
        
        return bool(is_sarcastic), round(confidence, 2)

    def detect_fake_news(self, text: str) -> Tuple[bool, float]:
        """
        Detects fake news by checking capital ratios and clickbait buzzwords.
        """
        if not text.strip():
            return False, 0.0
            
        text_lower = text.lower()
        
        # Clickbait buzzwords
        buzzword_matches = sum(1 for bw in FAKE_NEWS_BUZZWORDS if bw in text_lower)
        
        # Capital letter ratio (fake news headlines often use ALL CAPS)
        capitals = sum(1 for c in text if c.isupper())
        total_letters = sum(1 for c in text if c.isalpha())
        cap_ratio = capitals / total_letters if total_letters > 10 else 0.0
        
        fake_score = (buzzword_matches * 0.3) + (0.5 if cap_ratio > 0.3 else 0.0)
        
        is_fake = fake_score >= 0.4
        confidence = min(0.5 + fake_score, 0.99) if is_fake else min(0.99, 0.5 + (1.0 - fake_score))
        
        return bool(is_fake), round(confidence, 2)

    def detect_bot(self, username: str, text: str, platform: str) -> Tuple[bool, float]:
        """
        Detects bots based on username patterns (e.g. user123456789), post repetition,
        and high formatting density.
        """
        # Bot username pattern: e.g., screen names with lots of trailing numbers
        has_bot_username = bool(re.search(r'\d{6,}', username)) or username.lower() == "anonymous"
        
        # Bots often post short, repetitive or heavily formatted links
        has_links = "http" in text
        is_short = len(text.split()) < 6
        
        bot_score = (0.4 if has_bot_username else 0.0) + (0.3 if has_links else 0.0) + (0.2 if is_short else 0.0)
        
        is_bot = bot_score >= 0.5
        confidence = min(0.5 + bot_score, 0.99) if is_bot else min(0.99, 0.5 + (1.0 - bot_score))
        
        return bool(is_bot), round(confidence, 2)

    def analyze_post(self, text: str, username: str = "anonymous", platform: str = "manual", sentiment_score: float = 0.0) -> Dict[str, Any]:
        """
        Runs all advanced detectors on a post.
        """
        toxic, toxic_conf = self.detect_toxic(text)
        spam, spam_conf = self.detect_spam(text)
        sarcasm, sarcasm_conf = self.detect_sarcasm(text, sentiment_score)
        fake, fake_conf = self.detect_fake_news(text)
        bot, bot_conf = self.detect_bot(username, text, platform)
        
        return {
            "is_toxic": toxic,
            "toxic_confidence": toxic_conf,
            "is_spam": spam,
            "spam_confidence": spam_conf,
            "is_sarcastic": sarcasm,
            "sarcasm_confidence": sarcasm_conf,
            "is_fake": fake,
            "fake_confidence": fake_conf,
            "is_bot": bot,
            "bot_confidence": bot_conf
        }
