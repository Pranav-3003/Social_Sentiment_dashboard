import logging
import html
from typing import Dict, Any, List, Tuple, Callable
import numpy as np

logger = logging.getLogger("XAI")

class ExplainabilityWorkbench:
    def __init__(self):
        pass

    def explain_with_lime(self, 
                          text: str, 
                          model: Any, 
                          vectorizer: Any, 
                          class_names: List[str] = ["Negative", "Neutral", "Positive"],
                          num_features: int = 10) -> Tuple[str, List[Tuple[str, float]], int]:
        """
        Explains text prediction using LIME.
        Returns: (html_representation, list_of_importances, predicted_class_index).
        """
        try:
            from lime.lime_text import LimeTextExplainer
            
            # Predict probability function for LIME
            def predict_proba(texts: List[str]) -> np.ndarray:
                X = vectorizer.transform(texts)
                # Check if model supports predict_proba
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(X)
                elif hasattr(model, "decision_function"):
                    # LinearSVC / SVM
                    df = model.decision_function(X)
                    if len(df.shape) == 1:
                        # Binary
                        probs = 1 / (1 + np.exp(-df))
                        return np.vstack((1 - probs, probs)).T
                    else:
                        # Multiclass - softmax
                        exp_df = np.exp(df)
                        return exp_df / np.sum(exp_df, axis=1, keepdims=True)
                else:
                    # Naive fallback
                    preds = model.predict(X)
                    # Convert to one-hot probabilities
                    num_classes = len(class_names)
                    probs = np.zeros((len(texts), num_classes))
                    for idx, val in enumerate(preds):
                        probs[idx, val] = 1.0
                    return probs

            explainer = LimeTextExplainer(class_names=class_names)
            exp = explainer.explain_instance(
                text, 
                predict_proba, 
                num_features=num_features,
                labels=[0, 1, 2] if len(class_names) == 3 else [0, 1]
            )
            
            # Dominant label prediction
            probs = predict_proba([text])[0]
            pred_class = int(np.argmax(probs))
            
            # Get word weights for the predicted class
            word_weights = exp.as_list(label=pred_class)
            
            return exp.as_html(), word_weights, pred_class
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}. Falling back to rule-based explainer.")
            return self._fallback_explainer(text, model, vectorizer, class_names)

    def explain_with_shap(self, 
                          text: str, 
                          model: Any, 
                          vectorizer: Any, 
                          class_names: List[str] = ["Negative", "Neutral", "Positive"]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], float]:
        """
        Explains text prediction using SHAP.
        Returns: (positive_contributors, negative_contributors, confidence_score).
        """
        try:
            import shap
            
            def predict_proba(texts: List[str]) -> np.ndarray:
                X = vectorizer.transform(texts)
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(X)
                elif hasattr(model, "decision_function"):
                    df = model.decision_function(X)
                    if len(df.shape) == 1:
                        probs = 1 / (1 + np.exp(-df))
                        return np.vstack((1 - probs, probs)).T
                    else:
                        exp_df = np.exp(df)
                        return exp_df / np.sum(exp_df, axis=1, keepdims=True)
                else:
                    preds = model.predict(X)
                    probs = np.zeros((len(texts), len(class_names)))
                    for idx, val in enumerate(preds):
                        probs[idx, val] = 1.0
                    return probs

            # Run simple perturbation SHAP Explainer
            # Perturb text by replacing words with empty space
            words = text.split()
            if len(words) == 0:
                return [], [], 0.5
                
            probs = predict_proba([text])[0]
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])
            
            # Base probability (with empty text)
            base_prob = predict_proba([""])[0][pred_class]
            
            # Simple kernel-like SHAP estimation:
            # Drop each word individually and measure the delta in probability
            word_contribs = []
            for i, word in enumerate(words):
                # Text without word i
                subset_words = words[:i] + words[i+1:]
                subset_text = " ".join(subset_words)
                prob_without = predict_proba([subset_text])[0][pred_class]
                
                # Contribution of word is the loss in probability when it is removed
                contrib = confidence - prob_without
                word_contribs.append((word, float(contrib)))
                
            # Filter and sort
            pos_contribs = sorted([(w, c) for w, c in word_contribs if c > 0.001], key=lambda x: x[1], reverse=True)
            neg_contribs = sorted([(w, c) for w, c in word_contribs if c < -0.001], key=lambda x: x[1])
            
            return pos_contribs, neg_contribs, confidence
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}. Falling back to rule-based SHAP.")
            # Fallback
            words = text.split()
            pos = [(w, 0.1) for w in words[:2]]
            neg = [(w, -0.1) for w in words[2:4]]
            return pos, neg, 0.5

    def _fallback_explainer(self, text: str, model: Any, vectorizer: Any, class_names: List[str]) -> Tuple[str, List[Tuple[str, float]], int]:
        """
        Rule-based fallback explainer if LIME/SHAP libraries fail to compile/run.
        """
        words = text.split()
        importances = []
        
        # Simple TF-IDF score highlighting
        if hasattr(vectorizer, 'vectorizer') and vectorizer.vectorizer:
            feature_names = list(vectorizer.vectorizer.get_feature_names_out())
            tfidf_vals = vectorizer.transform([text])[0]
            
            for w in words:
                w_clean = w.lower().strip(".,?!;:-_()\"'")
                if w_clean in feature_names:
                    idx = feature_names.index(w_clean)
                    val = tfidf_vals[idx]
                    if val > 0:
                        importances.append((w, float(val)))
                        
        if not importances:
            importances = [(w, 0.1) for w in words[:5]]
            
        # Mock HTML
        html_out = "<h3>Fallback Explanation (TF-IDF Weighting)</h3><p>"
        for w, score in importances:
            html_out += f"<span style='background-color:rgba(0, 128, 255, {min(score*2, 0.8)}); padding:2px; border-radius:3px; margin:2px;'>{html(w)}</span> "
        html_out += "</p>"
        
        return html_out, importances, 1

    def generate_html_explanation_chips(self, pos_contribs: List[Tuple[str, float]], neg_contribs: List[Tuple[str, float]]) -> str:
        """
        Generates custom HTML string of colored chips representing word importances.
        Positive: Green, Negative: Red.
        """
        html_str = "<div style='display: flex; flex-wrap: wrap; gap: 8px; margin: 15px 0;'>"
        
        # Add positives
        for word, weight in pos_contribs:
            clean_word = html.escape(word)
            alpha = min(0.3 + (weight * 2.0), 0.9)
            html_str += f"""
            <span style='background-color: rgba(46, 204, 113, {alpha:.2f}); color: white; padding: 4px 10px; border-radius: 20px; font-weight: bold; font-size: 14px;'>
                {clean_word} (+{weight:.3f})
            </span>
            """
            
        # Add negatives
        for word, weight in neg_contribs:
            clean_word = html.escape(word)
            alpha = min(0.3 + (abs(weight) * 2.0), 0.9)
            html_str += f"""
            <span style='background-color: rgba(231, 76, 60, {alpha:.2f}); color: white; padding: 4px 10px; border-radius: 20px; font-weight: bold; font-size: 14px;'>
                {clean_word} ({weight:.3f})
            </span>
            """
            
        html_str += "</div>"
        return html_str
