"""Model registry for managing trading models."""

from typing import Dict, Type, Optional, Any
import logging
from pathlib import Path
import importlib
import json

from .base import BaseTradingModel
from .sklearn_models.logistic import LogisticRegressionModel

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Registry for managing trading models."""
    
    def __init__(self):
        """Initialize the registry."""
        self._models: Dict[str, Type[BaseTradingModel]] = {}
        self._model_paths: Dict[str, Path] = {}
        
        # Register default models
        self.register('logistic', LogisticRegressionModel)
    
    def register(self, name: str, model_class: Type[BaseTradingModel], path: Optional[Path] = None) -> None:
        """Register a model class.
        
        Args:
            name: Name to register the model under
            model_class: Model class to register
            path: Optional path to model artifacts
        """
        if name in self._models:
            logger.warning(f"Overwriting existing model registration for '{name}'")
        self._models[name] = model_class
        if path is not None:
            self._model_paths[name] = path
    
    def get_model_class(self, name: str) -> Type[BaseTradingModel]:
        """Get a registered model class.
        
        Args:
            name: Name of the registered model
            
        Returns:
            Registered model class
            
        Raises:
            KeyError: If model is not registered
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' is not registered")
        return self._models[name]
    
    def get_model_path(self, name: str) -> Optional[Path]:
        """Get the path for a registered model.
        
        Args:
            name: Name of the registered model
            
        Returns:
            Path to model artifacts, or None if not set
        """
        return self._model_paths.get(name)
    
    def list_models(self) -> Dict[str, Optional[Path]]:
        """List all registered models and their paths.
        
        Returns:
            Dict mapping model names to their paths
        """
        return {
            name: self._model_paths.get(name)
            for name in self._models
        }
    
    def create_model(self, name: str, **kwargs) -> BaseTradingModel:
        """Create a new model instance.
        
        Args:
            name: Name of the registered model
            **kwargs: Arguments to pass to model constructor
            
        Returns:
            New model instance
            
        Raises:
            KeyError: If model is not registered
        """
        model_class = self.get_model_class(name)
        return model_class(**kwargs)
    
    def load_model(self, name: str, path: Optional[Path] = None) -> BaseTradingModel:
        """Load a saved model.
        
        Args:
            name: Name of the registered model
            path: Optional path to load from (defaults to registered path)
            
        Returns:
            Loaded model instance
            
        Raises:
            KeyError: If model is not registered
            FileNotFoundError: If model file does not exist
        """
        model_class = self.get_model_class(name)
        load_path = path or self.get_model_path(name)
        
        if load_path is None:
            raise ValueError(f"No path specified for model '{name}'")
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
            
        return model_class.load(load_path)

# Create global registry instance
registry = ModelRegistry() 