"""Base class for scikit-learn models."""

from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from ..base import BaseModel

logger = logging.getLogger(__name__)

class SklearnModel(BaseModel):
    """Base class for scikit-learn models."""
    
    def __init__(
        self,
        model_name: str,
        model: BaseEstimator,
        target_column: str = "label"
    ):
        """Initialize the model.
        
        Args:
            model_name: Name of the model
            model: scikit-learn model instance
            target_column: Name of the target column
        """
        super().__init__(model_name)
        self.model = model
        self.target_column = target_column
        self.scaler = StandardScaler()
        self.feature_means = None
    
    def _prepare_features(
        self,
        df: pd.DataFrame,
        fit: bool = False
    ) -> np.ndarray:
        """Prepare features for model input.
        
        Args:
            df: DataFrame containing features
            fit: Whether to fit the scaler
            
        Returns:
            numpy array of features
        """
        self._validate_features(df)
        
        # Get features in correct order
        X = df[self.feature_columns]
        
        # Handle missing values
        if fit:
            self.feature_means = X.mean()
            X = X.fillna(self.feature_means)
        else:
            X = X.fillna(self.feature_means)
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        return X_scaled
    
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Train the model.
        
        Args:
            train_data: DataFrame containing training data
            val_data: Optional DataFrame containing validation data
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
        
        # Train model
        self.model.fit(X_train, y_train, **kwargs)
        
        # Evaluate on validation data if provided
        metrics = {}
        if val_data is not None:
            X_val = self._prepare_features(val_data, fit=False)
            y_val = val_data[self.target_column].values
            
            # Get predictions
            y_pred = self.model.predict(X_val)
            y_pred_proba = self.model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            from sklearn.metrics import roc_auc_score, precision_score, recall_score
            metrics = {
                'auc': roc_auc_score(y_val, y_pred_proba),
                'precision': precision_score(y_val, y_pred),
                'recall': recall_score(y_val, y_pred)
            }
            
            logger.info(f"Validation metrics: {metrics}")
        
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
        
        # Get predictions
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)[:, 1]
        
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
        model_data = {
            'model': self.model,
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_,
            'feature_means': self.feature_means
        }
        
        # Save model data
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save metadata
        self._save_metadata(path)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'SklearnModel':
        """Load model state.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model instance
        """
        path = Path(path)
        
        # Load model data
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create model instance
        model = cls(
            model_name=model_data['model'].__class__.__name__,
            model=model_data['model']
        )
        
        # Restore state
        model.scaler = StandardScaler()
        model.scaler.mean_ = model_data['scaler_mean']
        model.scaler.scale_ = model_data['scaler_scale']
        model.feature_means = model_data['feature_means']
        
        # Load metadata
        model._load_metadata(path)
        
        return model 