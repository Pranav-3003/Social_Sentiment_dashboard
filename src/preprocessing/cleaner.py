import re
import unicodedata
from typing import Dict, Any, Tuple, List, Optional
import html

# Common English Contractions Map
CONTRACTION_MAP = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot", "cant": "cannot",
    "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "he'd": "he would", "he'll": "he will", "he's": "he is", "i'd": "i would",
    "i'll": "i will", "i'm": "i am", "i've": "i have", "isn't": "is not",
    "it's": "it is", "let's": "let us", "mightn't": "might not", "mustn't": "must not",
    "shan't": "shall not", "she'd": "she would", "she'll": "she will", "she's": "she is",
    "shouldn't": "should not", "that's": "that is", "there's": "there is",
    "they'd": "they would", "they'll": "they will", "they're": "they are",
    "they've": "they have", "we'd": "we would", "we'll": "we will", "we're": "we are",
    "we've": "we have", "weren't": "were not", "what's": "what is", "where's": "where is",
    "who's": "who is", "won't": "will not", "wouldn't": "would not", "you'd": "you would",
    "you'll": "you will", "you're": "you are", "you've": "you have", "y'all": "you all"
}

class TextCleaner:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the text cleaner with configurations.
        """
        cfg = config or {}
        self.remove_urls = cfg.get("remove_urls", True)
        self.remove_emojis = cfg.get("remove_emojis", False)
        self.remove_html = cfg.get("remove_html", True)
        self.remove_mentions = cfg.get("remove_mentions", True)
        self.process_hashtags = cfg.get("process_hashtags", "extract") # 'extract', 'remove', 'keep'
        self.expand_contractions = cfg.get("expand_contractions", True)
        self.remove_punctuation = cfg.get("remove_punctuation", True)
        self.remove_numbers = cfg.get("remove_numbers", False)
        self.lowercase = cfg.get("lowercase", True)
        self.unicode_normalization = cfg.get("unicode_normalization", True)
        self.whitespace_cleanup = cfg.get("whitespace_cleanup", True)

    def clean_text(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Cleans the input text and returns the cleaned text along with statistics.
        """
        if not isinstance(text, str) or not text.strip():
            return "", {"original_length": 0, "cleaned_length": 0, "reduction_ratio": 0.0}
        
        stats = {
            "original_length": len(text),
            "original_word_count": len(text.split()),
            "urls_removed": 0,
            "html_tags_removed": 0,
            "emojis_removed": 0,
            "mentions_removed": 0,
            "hashtags_extracted": [],
            "contractions_expanded": 0,
            "numbers_removed": 0,
            "punctuation_removed": 0
        }
        
        cleaned = text
        
        # 1. Unicode Normalization
        if self.unicode_normalization:
            cleaned = unicodedata.normalize('NFKD', cleaned).encode('ascii', 'ignore').decode('utf-8')
            
        # 2. HTML Entity & Tags removal
        if self.remove_html:
            cleaned = html.unescape(cleaned)
            # Find HTML tags
            html_tags = re.findall(r'<[^>]*>', cleaned)
            stats["html_tags_removed"] = len(html_tags)
            cleaned = re.sub(r'<[^>]*>', ' ', cleaned)
            
        # 3. URL removal
        if self.remove_urls:
            # Match standard http/https/ftp urls and www. domains
            urls = re.findall(r'https?://\S+|www\.\S+|ftp://\S+', cleaned)
            stats["urls_removed"] = len(urls)
            cleaned = re.sub(r'https?://\S+|www\.\S+|ftp://\S+', ' ', cleaned)
            
        # 4. Mention (@username) handling
        if self.remove_mentions:
            mentions = re.findall(r'@\w+', cleaned)
            stats["mentions_removed"] = len(mentions)
            cleaned = re.sub(r'@\w+', ' ', cleaned)
            
        # 5. Hashtag (#tag) processing
        hashtags = re.findall(r'#\w+', cleaned)
        stats["hashtags_extracted"] = [tag[1:] for tag in hashtags]
        if self.process_hashtags == "remove":
            cleaned = re.sub(r'#\w+', ' ', cleaned)
        elif self.process_hashtags == "extract":
            # Replace hashtag symbol with space but keep the word
            cleaned = re.sub(r'#(\w+)', r'\1', cleaned)
            
        # 6. Emoji handling
        # Simple regex matching common emojis and special Unicode range for emoji
        # If we want to remove them or keep them
        emoji_pattern = re.compile(
            r'[\U00010000-\U0010ffff]|\u263a|\u2639|[\u2600-\u27BF]|[\u2300-\u23FF]|[\u2b50]'
        )
        emojis_found = emoji_pattern.findall(cleaned)
        stats["emojis_removed"] = len(emojis_found)
        if self.remove_emojis:
            cleaned = emoji_pattern.sub(' ', cleaned)
            
        # 7. Expand Contractions
        if self.expand_contractions:
            # Lowercase keys search
            words = cleaned.split()
            expanded_words = []
            for word in words:
                clean_word = word.lower().strip(".,?!;:-_()\"'")
                if clean_word in CONTRACTION_MAP:
                    # Maintain case prefix if original word was uppercase
                    replacement = CONTRACTION_MAP[clean_word]
                    if word[0].isupper():
                        replacement = replacement.capitalize()
                    expanded_words.append(replacement)
                    stats["contractions_expanded"] += 1
                else:
                    expanded_words.append(word)
            cleaned = " ".join(expanded_words)
            
        # 8. Number removal
        if self.remove_numbers:
            numbers = re.findall(r'\d+', cleaned)
            stats["numbers_removed"] = len(numbers)
            cleaned = re.sub(r'\d+', ' ', cleaned)
            
        # 9. Punctuation removal
        if self.remove_punctuation:
            # We can keep some punctuation for sentiment if desired, but default is removal
            # Remove all punctuation except maybe apostrophe inside word, but here we strip all standard
            punc_chars = re.findall(r'[^\w\s]', cleaned)
            stats["punctuation_removed"] = len(punc_chars)
            # Replace punctuation with spaces to avoid joining words
            cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
            
        # 10. Lowercasing
        if self.lowercase:
            cleaned = cleaned.lower()
            
        # 11. Whitespace Cleanup
        if self.whitespace_cleanup:
            # Replace tabs, newlines with single space, strip margins
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
        stats["cleaned_length"] = len(cleaned)
        stats["cleaned_word_count"] = len(cleaned.split())
        stats["reduction_ratio"] = (
            round((stats["original_length"] - stats["cleaned_length"]) / stats["original_length"], 4)
            if stats["original_length"] > 0 else 0.0
        )
        
        return cleaned, stats
    
    @staticmethod
    def get_cleaning_pipeline_summary(original_df_column, cleaned_df_column) -> Dict[str, Any]:
        """
        Utility function to generate summary statistics of a cleaned column compared to original.
        """
        orig_lens = [len(str(x)) for x in original_df_column]
        clean_lens = [len(str(x)) for x in cleaned_df_column]
        
        avg_orig = sum(orig_lens) / len(orig_lens) if orig_lens else 0
        avg_clean = sum(clean_lens) / len(clean_lens) if clean_lens else 0
        
        return {
            "total_rows": len(original_df_column),
            "average_original_length": round(avg_orig, 2),
            "average_cleaned_length": round(avg_clean, 2),
            "reduction_percentage": round(((avg_orig - avg_clean) / avg_orig * 100) if avg_orig > 0 else 0, 2)
        }
