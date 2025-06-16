"""Base class for PyTorch models."""

from typing import Dict, Any, Optional, Union, List
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from abc import abstractmethod

from ..base import BaseModel

logger = logging.getLogger(__name__)

class PyTorchModel(BaseModel, nn.Module):
    """Base class for PyTorch models."""
    
    def __init__(
        self,
        model_name: str,
        target_column: str = "label",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the model.
        
        Args:
            model_name: Name of the model
            target_column: Name of the target column
            device: Device to use for training/inference
        """
        BaseModel.__init__(self, model_name)
        nn.Module.__init__(self)
        self.target_column = target_column
        self.device = device
        self.scaler = None  # Will be set during training
        self.feature_means = None  # Will be set during training
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Model predictions
        """
        pass
    
    def _prepare_features(
        self,
        df: pd.DataFrame,
        fit: bool = False
    ) -> torch.Tensor:
        """Prepare features for model input.
        
        Args:
            df: DataFrame containing features
            fit: Whether to fit the scaler
            
        Returns:
            torch.Tensor: Prepared input tensor
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
        
        # Convert to tensor
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        
        return X_tensor.to(self.device)
    
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        batch_size: int = 32,
        epochs: int = 10,
        learning_rate: float = 0.001,
        **kwargs
    ) -> Dict[str, float]:
        """Train the model.
        
        Args:
            train_data: DataFrame containing training data
            val_data: Optional DataFrame containing validation data
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
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
        y_train = torch.tensor(
            train_data[self.target_column].values,
            dtype=torch.float32
        ).to(self.device)
        
        # Create data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Setup optimizer and loss
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs, y_batch.unsqueeze(1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
        
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
        self.eval()
        with torch.no_grad():
            # Prepare validation data
            X_val = self._prepare_features(val_data, fit=False)
            y_val = torch.tensor(
                val_data[self.target_column].values,
                dtype=torch.float32
            ).to(self.device)
            
            # Get predictions
            outputs = self(X_val)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            predictions = (probabilities > 0.5).astype(int)
            
            # Calculate metrics
            from sklearn.metrics import roc_auc_score, precision_score, recall_score
            metrics = {
                'auc': roc_auc_score(y_val.cpu(), probabilities),
                'precision': precision_score(y_val.cpu(), predictions),
                'recall': recall_score(y_val.cpu(), predictions)
            }
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions on input data.
        
        Args:
            df: DataFrame containing the input features
            
        Returns:
            Dict containing predictions and probabilities
        """
        self.eval()
        with torch.no_grad():
            # Prepare features
            features = self._prepare_features(df, fit=False)
            
            # Get predictions
            outputs = self(features)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            predictions = (probabilities > 0.5).astype(int)
            
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
        torch.save({
            'model_state_dict': self.state_dict(),
            'feature_means': self.feature_means
        }, path)
        
        # Save metadata
        self._save_metadata(path)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'PyTorchModel':
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
        checkpoint = torch.load(path, map_location=model.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.feature_means = checkpoint['feature_means']
        
        # Load metadata
        model._load_metadata(path)
        
        return model 