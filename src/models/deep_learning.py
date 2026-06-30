import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional, GRU, Conv1D, GlobalMaxPooling1D, Dropout
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger("DeepLearning")

class KerasDLWorkbench:
    def __init__(self, max_len: int = 100, random_state: int = 42):
        self.max_len = max_len
        self.random_state = random_state
        if TENSORFLOW_AVAILABLE:
            tf.random.set_seed(self.random_state)
        self.models = {}

    def pad_sequences(self, sequences: List[List[int]], vocab_size: int) -> np.ndarray:
        """
        Pads sequences to max_len. Pad character is 0. Truncates if longer.
        """
        padded = np.zeros((len(sequences), self.max_len), dtype=int)
        for i, seq in enumerate(sequences):
            if len(seq) == 0:
                continue
            # Truncate
            if len(seq) > self.max_len:
                padded[i, :] = np.array(seq[:self.max_len])
            else:
                # Pad post
                padded[i, :len(seq)] = np.array(seq)
        return padded

    def _build_model(self, model_name: str, vocab_size: int, num_classes: int, embedding_dim: int = 100) -> Sequential:
        """
        Builds the specified neural network.
        """
        model = Sequential()
        # Add embedding layer
        model.add(Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=self.max_len))
        
        if model_name == "Simple Neural Network":
            model.add(GlobalMaxPooling1D())
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(32, activation='relu'))
            
        elif model_name == "LSTM":
            model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
            model.add(Dense(32, activation='relu'))
            
        elif model_name == "Bi-LSTM":
            model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
            model.add(Dense(32, activation='relu'))
            
        elif model_name == "GRU":
            model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2))
            model.add(Dense(32, activation='relu'))
            
        elif model_name == "CNN for text":
            model.add(Conv1D(128, kernel_size=5, activation='relu'))
            model.add(GlobalMaxPooling1D())
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.3))
            
        else:
            raise ValueError(f"Unknown deep learning model name: {model_name}")
            
        # Output layer
        if num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(Dense(num_classes, activation='softmax'))
            loss = 'categorical_crossentropy'
            
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        return model

    def train(self, model_name: str, 
              X_train_seq: List[List[int]], y_train: np.ndarray, 
              vocab_size: int, num_classes: int,
              epochs: int = 5, batch_size: int = 32, 
              embedding_dim: int = 100,
              validation_split: float = 0.2) -> Tuple[Any, Dict[str, Any]]:
        """
        Pads sequences and trains the deep learning model.
        """
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow/Keras is not installed. Deep Learning models cannot be trained.")
        logger.info(f"Preparing data for Deep Learning model {model_name}...")
        X_padded = self.pad_sequences(X_train_seq, vocab_size)
        
        # Prepare targets
        if num_classes == 2:
            y_train_proc = np.array(y_train, dtype=float)
        else:
            y_train_proc = to_categorical(y_train, num_classes=num_classes)
            
        model = self._build_model(model_name, vocab_size, num_classes, embedding_dim)
        
        logger.info(f"Starting training for {model_name}...")
        # Train
        history = model.fit(
            X_padded, y_train_proc,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        self.models[model_name] = (model, vocab_size, num_classes)
        return model, history.history

    def predict(self, model_name: str, X_seq: List[List[int]]) -> np.ndarray:
        """
        Predicts classes for the input token sequences.
        """
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow/Keras is not installed. Deep Learning predictions cannot run.")
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} has not been trained yet.")
            
        model, vocab_size, num_classes = self.models[model_name]
        X_padded = self.pad_sequences(X_seq, vocab_size)
        
        preds = model.predict(X_padded)
        
        if num_classes == 2:
            return (preds > 0.5).astype(int).flatten()
        else:
            return np.argmax(preds, axis=1)

    def predict_proba(self, model_name: str, X_seq: List[List[int]]) -> np.ndarray:
        """
        Predicts probabilities for the input token sequences.
        """
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow/Keras is not installed. Deep Learning predictions cannot run.")
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} has not been trained yet.")
            
        model, vocab_size, num_classes = self.models[model_name]
        X_padded = self.pad_sequences(X_seq, vocab_size)
        
        preds = model.predict(X_padded)
        if num_classes == 2:
            return np.hstack((1 - preds, preds))
        else:
            return preds
