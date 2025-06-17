"""Logistic regression model for trading predictions."""

from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel
import logging
from pathlib import Path
import joblib
import json

from .base import SklearnModel

logger = logging.getLogger(__name__)

class LogisticRegressionModel(SklearnModel):
    """Logistic regression model for trading predictions.
    
    This model uses scikit-learn's LogisticRegression with configurable regularization
    and balanced class weights to handle imbalanced datasets. Supports cross-validation,
    hyperparameter tuning, and early stopping.
    """
    
    def __init__(
        self,
        model_name: str = "LogisticRegression",
        target_column: str = "label",
        C: float = 1.0,
        max_iter: int = 50000,
        random_state: int = 42,
        penalty: str = 'l2',
        solver: str = 'lbfgs',
        early_stopping: bool = False,
        n_jobs: int = -1,
        cv_folds: int = 5,
        feature_selection_threshold: Optional[float] = None
    ):
        """Initialize the model.
        
        Args:
            model_name: Name of the model
            target_column: Name of the target column
            C: Inverse of regularization strength
            max_iter: Maximum number of iterations
            random_state: Random seed for reproducibility
            penalty: Regularization penalty ('l1', 'l2', 'elasticnet', None)
            solver: Optimization algorithm ('lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga')
            early_stopping: Whether to use early stopping
            n_jobs: Number of parallel jobs for cross-validation
            cv_folds: Number of cross-validation folds
            feature_selection_threshold: Threshold for feature selection (None to disable)
        """
        # Initialize base model
        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            class_weight='balanced',
            penalty=penalty,
            solver=solver,
            n_jobs=n_jobs
        )
        super().__init__(model_name=model_name, model=model, target_column=target_column)
        
        # Store hyperparameters
        self.metadata.update({
            'C': C,
            'max_iter': max_iter,
            'random_state': random_state,
            'penalty': penalty,
            'solver': solver,
            'early_stopping': early_stopping,
            'cv_folds': cv_folds,
            'feature_selection_threshold': feature_selection_threshold
        })
        
        # Initialize feature selector if threshold is provided
        self.feature_selector = None
        if feature_selection_threshold is not None:
            self.feature_selector = SelectFromModel(
                self.model,
                threshold=feature_selection_threshold,
                prefit=False
            )
    
    def _tune_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Tune model hyperparameters using grid search.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dict containing best parameters and cross-validation scores
        """
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l2'],  # Using only l2 penalty for better convergence
            'solver': ['lbfgs'],  # Using lbfgs solver for better stability
            'max_iter': [100000]  # Increased max iterations
        }
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=self.metadata['cv_folds'],
            scoring='accuracy',
            n_jobs=self.metadata.get('n_jobs', -1)
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        tune_hyperparameters: bool = False,
        **kwargs
    ) -> Dict[str, float]:
        """Train the model.
        
        Args:
            train_data: DataFrame containing training data
            val_data: Optional DataFrame containing validation data
            tune_hyperparameters: Whether to perform hyperparameter tuning
            **kwargs: Additional training arguments
            
        Returns:
            Dict containing training metrics
        """
        # Store feature columns
        self.feature_columns = [
            col for col in train_data.columns
            if col not in ['date', 'ticker', self.target_column]
        ]
        
        # Prepare training data
        X_train = self._prepare_features(train_data, fit=True)
        y_train = train_data[self.target_column].values
        
        # Perform feature selection if enabled
        if self.feature_selector is not None:
            X_train = self.feature_selector.fit_transform(X_train, y_train)
            selected_features = self.feature_selector.get_support()
            self.feature_columns = [f for f, s in zip(self.feature_columns, selected_features) if s]
            logger.info(f"Selected {len(self.feature_columns)} features")
        
        # Perform hyperparameter tuning if requested
        if tune_hyperparameters:
            tuning_results = self._tune_hyperparameters(X_train, y_train)
            self.metadata['tuning_results'] = tuning_results
            logger.info(f"Best parameters: {tuning_results['best_params']}")
            logger.info(f"Best CV score: {tuning_results['best_score']:.4f}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance (average across all classes)
        self.metadata['feature_importance'] = dict(zip(
            self.feature_columns,
            np.mean(np.abs(self.model.coef_), axis=0)
        ))
        
        # Perform cross-validation if enabled
        if self.metadata['cv_folds'] > 1:
            cv_scores = cross_val_score(
                self.model,
                X_train,
                y_train,
                cv=self.metadata['cv_folds'],
                scoring='accuracy',  # Changed from roc_auc to accuracy
                n_jobs=self.metadata.get('n_jobs', -1)
            )
            self.metadata['cv_scores'] = {
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'scores': cv_scores.tolist()
            }
            logger.info(f"Cross-validation scores: {self.metadata['cv_scores']}")
        
        # Evaluate on validation data if provided
        metrics = {}
        if val_data is not None:
            metrics = self._evaluate(val_data)
            logger.info(f"Validation metrics: {metrics}")
        
        return metrics
    
    def _evaluate(self, val_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model on validation data.
        
        Args:
            val_data: DataFrame containing validation data
            
        Returns:
            Dict containing evaluation metrics
        """
        # Prepare validation data
        X_val = self._prepare_features(val_data, fit=False)
        if self.feature_selector is not None:
            X_val = self.feature_selector.transform(X_val)
        y_val = val_data[self.target_column].values
        
        # Get predictions
        predictions = self.model.predict(X_val)
        probabilities = self.model.predict_proba(X_val)
        
        # Calculate metrics for each class
        metrics = {
            'accuracy': accuracy_score(y_val, predictions),
            'precision_macro': precision_score(y_val, predictions, average='macro'),
            'precision_weighted': precision_score(y_val, predictions, average='weighted'),
            'recall_macro': recall_score(y_val, predictions, average='macro'),
            'recall_weighted': recall_score(y_val, predictions, average='weighted')
        }
        
        # Add per-class metrics
        for label in [-1, 0, 1]:
            metrics[f'precision_class_{label}'] = precision_score(
                y_val, predictions, labels=[label], average='micro'
            )
            metrics[f'recall_class_{label}'] = recall_score(
                y_val, predictions, labels=[label], average='micro'
            )
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions on input data.
        
        Args:
            df: DataFrame containing the input features
            
        Returns:
            Dict containing predictions and probabilities
        """
        # Prepare features
        features = self._prepare_features(df, fit=False)
        if self.feature_selector is not None:
            features = self.feature_selector.transform(features)
        
        # Get predictions
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        
        # Create results dictionary
        results = {
            'predictions': predictions,
            'probabilities': probabilities,
            'ticker': df['ticker'].values if 'ticker' in df.columns else None,
            'date': df['date'].values if 'date' in df.columns else None
        }
        
        return results
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model state.
        
        Args:
            path: Path to save the model
        """
        path = Path(path)
        
        # Save model state
        joblib.dump(self.model, path)
        
        # Save feature selector if it exists
        if self.feature_selector is not None:
            joblib.dump(self.feature_selector, path.with_suffix('.selector'))
        
        # Convert NumPy arrays to lists in metadata
        metadata = {
            'model_name': self.model_name,
            'target_column': self.target_column,
            'feature_columns': self.feature_columns,
            'metadata': self.metadata
        }
        
        # Convert any NumPy arrays in metadata to lists
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        metadata = convert_numpy(metadata)
        
        with open(path.with_suffix('.metadata'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self, path: Union[str, Path]) -> None:
        """Load model metadata.
        
        Args:
            path: Path to load metadata from
        """
        path = Path(path)
        metadata_path = path.with_suffix('.metadata')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.model_name = metadata['model_name']
                self.target_column = metadata['target_column']
                self.feature_columns = metadata['feature_columns']
                self.metadata = metadata['metadata']
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'LogisticRegressionModel':
        """Load model state.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model instance
        """
        path = Path(path)
        
        # Create model instance
        model = cls(model_name=cls.__name__)
        
        # Load model state
        model.model = joblib.load(path)
        
        # Load feature selector if it exists
        selector_path = path.with_suffix('.selector')
        if selector_path.exists():
            model.feature_selector = joblib.load(selector_path)
        
        # Load metadata
        model._load_metadata(path)
        
        return model 