"""Base model class for trading models."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import pandas as pd

class BaseTradingModel(nn.Module):
    """Base class for all trading models."""
    
    def __init__(self):
        super().__init__()
        self.model_name = self.__class__.__name__
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_features)
            
        Returns:
            torch.Tensor: Model predictions
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def prepare_input(self, df: pd.DataFrame) -> torch.Tensor:
        """Prepare input data for the model.
        
        Args:
            df: DataFrame containing the input features
            
        Returns:
            torch.Tensor: Prepared input tensor
        """
        raise NotImplementedError("Subclasses must implement prepare_input method")
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions on input data.
        
        Args:
            df: DataFrame containing the input features
            
        Returns:
            Dict containing predictions and any additional information
        """
        raise NotImplementedError("Subclasses must implement predict method")
    
    def save(self, path: str):
        """Save model state.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'BaseTradingModel':
        """Load model state.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path)
        model = cls()
        model.load_state_dict(checkpoint['model_state_dict'])
        return model 