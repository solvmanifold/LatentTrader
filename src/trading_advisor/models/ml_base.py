"""Base class for machine learning trading models."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import pickle
from abc import ABC, abstractmethod
from .base import BaseTradingModel

class MLTradingModel(BaseTradingModel, ABC):
    """Base class for machine learning trading models."""
    
    def __init__(self, model_name: str):
        """Initialize the model.
        
        Args:
            model_name: Name of the model
        """
        super().__init__(model_name)
        self.model = None
        self.feature_columns = None
        self.target_column = None
    
    @abstractmethod
    def train(self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None) -> None:
        """Train the model.
        
        Args:
            train_data: DataFrame containing training data
            val_data: Optional DataFrame containing validation data
        """
        pass
    
    @abstractmethod
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions on input data.
        
        Args:
            df: DataFrame containing the input features
            
        Returns:
            Dict containing predictions and any additional information
        """
        pass
    
    def save(self, path: str) -> None:
        """Save model state.
        
        Args:
            path: Path to save the model
        """
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_name': self.model_name
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, path: str) -> 'MLTradingModel':
        """Load model state.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model instance
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(model_data['model_name'])
        model.model = model_data['model']
        model.feature_columns = model_data['feature_columns']
        model.target_column = model_data['target_column']
        
        return model
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for model input.
        
        Args:
            df: DataFrame containing features
            
        Returns:
            numpy array of features
        """
        if self.feature_columns is None:
            raise ValueError("Model has not been trained yet")
            
        # Ensure all required features are present
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required features: {missing_cols}")
            
        return df[self.feature_columns].values
    
    def _get_prediction_probabilities(self, features: np.ndarray) -> np.ndarray:
        """Get prediction probabilities from the model.
        
        Args:
            features: numpy array of features
            
        Returns:
            numpy array of prediction probabilities
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(features)
        else:
            # For models that don't support predict_proba, return binary predictions
            predictions = self.model.predict(features)
            return np.column_stack([1 - predictions, predictions]) 