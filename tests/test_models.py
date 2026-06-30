import pytest
import numpy as np
from src.models.classical_ml import ClassicalMLWorkbench
from src.business.emotion_detector import EmotionDetector
from src.business.advanced_detectors import AdvancedDetectors

def test_classical_ml_training():
    workbench = ClassicalMLWorkbench(random_state=42)
    # Binary classification dummy matrix
    X_train = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1]])
    y_train = np.array([1, 1, 0, 0])
    
    # Train logistic regression
    model, cv_score = workbench.train(
        model_name="Logistic Regression",
        X_train=X_train,
        y_train=y_train,
        cv=2
    )
    assert model is not None
    assert cv_score >= 0.0
    
    # Predict
    preds = model.predict(X_train)
    assert len(preds) == 4

def test_emotion_lexicon_detector():
    detector = EmotionDetector()
    emotion, conf = detector.predict_emotion("This is a wonderful and happy day!")
    assert emotion == "joy"
    assert conf > 0.4
    
    emotion_sad, _ = detector.predict_emotion("I feel very sad and unhappy about the loss.", sentiment_score=-0.6)
    assert emotion_sad == "sadness"

def test_advanced_detectors():
    detectors = AdvancedDetectors()
    toxic, _ = detectors.detect_toxic("You idiot, this is stupid trash!")
    assert toxic is True
    
    spam, _ = detectors.detect_spam("CLICK HERE TO MAKE FREE CASH MONEY NOW!!! http://spam.com")
    assert spam is True
    
    sarcasm, _ = detectors.detect_sarcasm("Oh brilliant, my code crashed again...")
    assert sarcasm is True
