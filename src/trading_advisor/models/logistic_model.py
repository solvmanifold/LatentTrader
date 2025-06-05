"""Logistic regression model for trading predictions."""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn

from .base import BaseTradingModel

logger = logging.getLogger(__name__)

class LogisticTradingModel(BaseTradingModel):
    """Logistic regression model for trading predictions."""
    
    def __init__(
        self,
        target_column: str = "label",
        **model_kwargs
    ):
        """Initialize the model.
        
        Args:
            target_column: Name of the target column
            **model_kwargs: Additional arguments to pass to LogisticRegression
        """
        super().__init__()
        self.target_column = target_column
        self.model = LogisticRegression(**model_kwargs)
        self.imputer = SimpleImputer(strategy='constant', fill_value=0)
        self.feature_columns = None
        self.dropped_columns = None  # Store dropped columns
    
    def _align_and_impute(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        # Ensure all required columns are present
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Adding missing columns (all-NaN): {missing_cols}")
            for col in missing_cols:
                df[col] = np.nan
        # Ensure column order
        X = df[self.feature_columns]
        # Log NaN columns
        nan_cols = X.columns[X.isna().any()].tolist()
        if nan_cols:
            logger.warning(f"Found NaN values in columns: {nan_cols}")
            for col in nan_cols:
                nan_count = X[col].isna().sum()
                logger.warning(f"Column {col} has {nan_count} NaN values")
        # Drop columns that are entirely NaN
        all_nan_cols = X.columns[X.isna().all()].tolist()
        if all_nan_cols:
            logger.warning(f"Dropping columns that are entirely NaN: {all_nan_cols}")
            X = X.drop(columns=all_nan_cols)
            if fit:
                self.dropped_columns = all_nan_cols
        # Impute
        if fit:
            X_imputed = self.imputer.fit_transform(X)
        else:
            X_imputed = self.imputer.transform(X)
        return X_imputed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Model predictions
        """
        # Convert tensor to numpy for sklearn model
        x_np = x.detach().cpu().numpy()
        predictions = self.model.predict(x_np)
        return torch.tensor(predictions, dtype=torch.float32)
    
    def prepare_input(self, df: pd.DataFrame) -> torch.Tensor:
        """Prepare input data for the model.
        
        Args:
            df: DataFrame containing the input features
            
        Returns:
            torch.Tensor: Prepared input tensor
        """
        if self.feature_columns is None:
            self.feature_columns = [
                col for col in df.columns
                if col not in ['date', 'ticker', self.target_column]
            ]
        X = self._align_and_impute(df, fit=False)
        return torch.tensor(X, dtype=torch.float32)
    
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
        X_train = self._align_and_impute(train_data, fit=True)
        y_train = train_data[self.target_column].values
        # Robustly cast labels to int and check for non-binary values
        unique_labels = np.unique(y_train)
        if not np.all(np.isin(unique_labels, [0, 1])):
            logger.warning(f"Non-binary label values found: {unique_labels}. Casting to int and clipping to [0, 1].")
            y_train = np.clip(y_train.astype(int), 0, 1)
        else:
            y_train = y_train.astype(int)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation data if provided
        if val_data is not None:
            X_val = self._align_and_impute(val_data, fit=False)
            y_val = val_data[self.target_column].values
            # Robustly cast labels to int and check for non-binary values
            unique_val_labels = np.unique(y_val)
            if not np.all(np.isin(unique_val_labels, [0, 1])):
                logger.warning(f"Non-binary label values found in validation set: {unique_val_labels}. Casting to int and clipping to [0, 1].")
                y_val = np.clip(y_val.astype(int), 0, 1)
            else:
                y_val = y_val.astype(int)
            
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
        # Infer feature columns if not set
        if self.feature_columns is None:
            self.feature_columns = [
                col for col in df.columns
                if col not in ['date', 'ticker', self.target_column]
            ]
        # Drop columns that were dropped during training
        if self.dropped_columns:
            df = df.drop(columns=self.dropped_columns, errors='ignore')
        # Prepare features
        features = self._align_and_impute(df, fit=False)
        
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
    
    def save(self, path: str):
        """Save model state.
        
        Args:
            path: Path to save the model
        """
        import pickle
        model_data = {
            'model': self.model,
            'imputer': self.imputer,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_name': self.model_name,
            'dropped_columns': self.dropped_columns
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, path: str) -> 'LogisticTradingModel':
        """Load model state.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model instance
        """
        import pickle
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(target_column=model_data['target_column'])
        model.model = model_data['model']
        model.imputer = model_data['imputer']
        model.feature_columns = model_data['feature_columns']
        model.dropped_columns = model_data['dropped_columns']
        return model 