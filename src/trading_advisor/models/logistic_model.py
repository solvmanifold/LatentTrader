"""Logistic regression model for trading predictions."""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from .ml_base import MLTradingModel

logger = logging.getLogger(__name__)

class LogisticTradingModel(MLTradingModel):
    """Logistic regression model for trading predictions."""
    
    def __init__(
        self,
        model_name: str = "logistic",
        target_column: str = "label",
        **model_kwargs
    ):
        """Initialize the model.
        
        Args:
            model_name: Name of the model
            target_column: Name of the target column
            **model_kwargs: Additional arguments to pass to LogisticRegression
        """
        super().__init__(model_name)
        self.target_column = target_column
        self.model = LogisticRegression(**model_kwargs)
    
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None
    ) -> None:
        """Train the model.
        
        Args:
            train_data: DataFrame containing training data
            val_data: Optional DataFrame containing validation data
        """
        # Get feature columns (exclude date, ticker, and target)
        self.feature_columns = [
            col for col in train_data.columns
            if col not in ['date', 'ticker', self.target_column]
        ]
        
        # Prepare training data
        X_train = train_data[self.feature_columns].values
        y_train = train_data[self.target_column].values
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation data if provided
        if val_data is not None:
            X_val = val_data[self.feature_columns].values
            y_val = val_data[self.target_column].values
            
            # Get predictions
            y_pred = self.model.predict(X_val)
            y_pred_proba = self.model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            metrics = {
                'auc': roc_auc_score(y_val, y_pred_proba),
                'precision': precision_score(y_val, y_pred),
                'recall': recall_score(y_val, y_pred)
            }
            
            logger.info(f"Validation metrics: {metrics}")
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions on input data.
        
        Args:
            df: DataFrame containing the input features
            
        Returns:
            Dict containing predictions and probabilities
        """
        # Prepare features
        features = self._prepare_features(df)
        
        # Get predictions
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)[:, 1]
        
        # Create results dictionary
        results = {
            'predictions': predictions,
            'probabilities': probabilities,
            'ticker': df['ticker'].values,
            'date': df['date'].values
        }
        
        return results 