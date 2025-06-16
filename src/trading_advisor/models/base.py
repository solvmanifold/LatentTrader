"""Base class for trading models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd

class BaseTradingModel(ABC):
    """Base class for all trading models."""
    
    def __init__(self, model_name: str):
        """Initialize the model.
        
        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.metadata = {}
    
    @abstractmethod
    def train(self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Train the model.
        
        Args:
            train_data: DataFrame containing training data
            val_data: Optional DataFrame containing validation data
            
        Returns:
            Dict containing training metrics
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
        pass
    
    @classmethod
    def load(cls, path: str) -> 'BaseTradingModel':
        """Load model state.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model instance
        """
        pass 