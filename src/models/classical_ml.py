import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC

logger = logging.getLogger("ClassicalML")

# Optional/conditional imports for gradient boosters
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost is not available in the environment.")

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    logger.warning("LightGBM is not available in the environment.")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost is not available in the environment.")

class ClassicalMLWorkbench:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}

    def get_available_models(self) -> List[str]:
        """
        Returns list of model names that can be trained in the current environment.
        """
        available = [
            "Logistic Regression",
            "Naive Bayes",
            "Decision Tree",
            "Random Forest",
            "SVM",
            "Gradient Boosting"
        ]
        if XGB_AVAILABLE:
            available.append("XGBoost")
        if LGBM_AVAILABLE:
            available.append("LightGBM")
        if CATBOOST_AVAILABLE:
            available.append("CatBoost")
        return available

    def _initialize_model(self, model_name: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Initializes classifier with given parameters.
        """
        p = params or {}
        
        if model_name == "Logistic Regression":
            return LogisticRegression(C=p.get("C", 1.0), max_iter=p.get("max_iter", 1000), random_state=self.random_state)
        elif model_name == "Naive Bayes":
            return MultinomialNB(alpha=p.get("alpha", 1.0))
        elif model_name == "Decision Tree":
            return DecisionTreeClassifier(max_depth=p.get("max_depth", None), random_state=self.random_state)
        elif model_name == "Random Forest":
            return RandomForestClassifier(n_estimators=p.get("n_estimators", 100), max_depth=p.get("max_depth", None), random_state=self.random_state)
        elif model_name == "SVM":
            # LinearSVC is much faster on text classification than standard SVC
            return LinearSVC(C=p.get("C", 1.0), random_state=self.random_state, dual=False)
        elif model_name == "Gradient Boosting":
            return GradientBoostingClassifier(n_estimators=p.get("n_estimators", 100), learning_rate=p.get("learning_rate", 0.1), random_state=self.random_state)
        elif model_name == "XGBoost" and XGB_AVAILABLE:
            return XGBClassifier(n_estimators=p.get("n_estimators", 100), max_depth=p.get("max_depth", 6), learning_rate=p.get("learning_rate", 0.1), random_state=self.random_state, eval_metric="logloss")
        elif model_name == "LightGBM" and LGBM_AVAILABLE:
            return LGBMClassifier(n_estimators=p.get("n_estimators", 100), learning_rate=p.get("learning_rate", 0.1), random_state=self.random_state, verbosity=-1)
        elif model_name == "CatBoost" and CATBOOST_AVAILABLE:
            return CatBoostClassifier(iterations=p.get("iterations", 100), learning_rate=p.get("learning_rate", 0.1), random_seed=self.random_state, verbose=0)
        else:
            raise ValueError(f"Model '{model_name}' is not supported or not available in the current environment.")

    def train(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
              params: Optional[Dict[str, Any]] = None, 
              tune_hyperparameters: bool = False, 
              param_grid: Optional[Dict[str, Any]] = None, 
              cv: int = 3) -> Tuple[Any, float]:
        """
        Trains a model, optionally doing hyperparameter tuning. Returns (trained_model, cv_score).
        """
        model = self._initialize_model(model_name, params)
        
        if tune_hyperparameters and param_grid:
            logger.info(f"Tuning hyperparameters for {model_name} with CV={cv}")
            grid = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            cv_score = grid.best_score_
            logger.info(f"Best parameters: {grid.best_params_} (CV Accuracy: {cv_score:.4f})")
        else:
            logger.info(f"Fitting {model_name} without tuning...")
            model.fit(X_train, y_train)
            # Evaluate using cross validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            cv_score = float(np.mean(cv_scores))
            logger.info(f"Model {model_name} fitted. CV Accuracy: {cv_score:.4f}")
            
        self.models[model_name] = model
        return model, cv_score

    def get_feature_importances(self, model_name: str, feature_names: List[str]) -> pd.DataFrame:
        """
        Extracts feature importances or coefficients and returns as a DataFrame.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} has not been trained yet.")
            
        model = self.models[model_name]
        importances = []
        
        # Check coefficients for linear models
        if hasattr(model, 'coef_'):
            # Binary or multiclass coefficients
            if len(model.coef_.shape) > 1 and model.coef_.shape[0] > 1:
                # Multiclass: take average absolute coefficient weight across classes
                importances = np.mean(np.abs(model.coef_), axis=0)
            else:
                importances = np.abs(model.coef_).flatten()
                
        # Check feature importances for tree-based models
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
        # Catboost specific if it doesn't match above (though it usually does)
        elif CATBOOST_AVAILABLE and isinstance(model, CatBoostClassifier):
            importances = model.get_feature_importance()
            
        if len(importances) == 0:
            # Fallback if no importances found
            logger.warning(f"No coefficients or importances found for {model_name}.")
            return pd.DataFrame()
            
        # Ensure sizes match
        if len(importances) != len(feature_names):
            # Pad or truncate if mismatch occurs (e.g. PCA or SVD representations)
            feature_names = [f"feat_{i}" for i in range(len(importances))]
            
        df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        
        return df

    def get_learning_curve_data(self, model_name: str, X: np.ndarray, y: np.ndarray, cv: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates learning curve data: train_sizes, train_scores, validation_scores.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} has not been trained yet.")
            
        model = self.models[model_name]
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 5), 
            scoring='accuracy', random_state=self.random_state
        )
        return train_sizes, np.mean(train_scores, axis=1), np.mean(val_scores, axis=1)
