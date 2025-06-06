"""Logistic regression model for trading predictions."""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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
        """Initialize model.
        
        Args:
            target_column: Name of the target column
            **model_kwargs: Additional arguments to pass to LogisticRegression
        """
        super().__init__()
        self.target_column = target_column
        self.model = LogisticRegression(**model_kwargs)
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.dropped_columns = None  # Store dropped columns
        self.feature_means = None  # Store training data means
    
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
        # Fill missing values with training data means
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
        # Store feature columns
        self.feature_columns = [col for col in train_data.columns if col != self.target_column]
        
        # Calculate feature means from training data
        X_train = train_data[self.feature_columns]
        self.feature_means = X_train.mean()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.scaler.feature_names_in_ = self.feature_columns
        
        # Train model
        y_train = train_data[self.target_column]
        self.model.fit(X_train_scaled, y_train)
        
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
        
        # Debug logging
        logger.info(f"\nFeature columns: {self.feature_columns}")
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Features dtype: {features.dtype}")
        logger.info(f"Features mean: {np.mean(features, axis=0)}")
        logger.info(f"Features std: {np.std(features, axis=0)}")
        
        # Get predictions
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)[:, 1]
        
        # Debug logging
        logger.info(f"Predictions shape: {predictions.shape}")
        logger.info(f"Predictions unique values: {np.unique(predictions)}")
        logger.info(f"Probabilities shape: {probabilities.shape}")
        logger.info(f"Probabilities mean: {np.mean(probabilities)}")
        logger.info(f"Probabilities std: {np.std(probabilities)}")
        
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
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_name': self.model_name,
            'dropped_columns': self.dropped_columns,
            'feature_means': self.feature_means
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
        model.scaler = StandardScaler()
        model.scaler.mean_ = model_data['scaler_mean']
        model.scaler.scale_ = model_data['scaler_scale']
        model.feature_columns = model_data['feature_columns']
        model.dropped_columns = model_data['dropped_columns']
        model.feature_means = model_data['feature_means']
        return model 