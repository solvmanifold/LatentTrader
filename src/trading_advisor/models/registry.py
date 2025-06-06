"""Model registry for trading models."""

import logging
from typing import Dict, Type, Any, Optional
from .base import BaseTradingModel
from .technical_scorer import TechnicalScorer
from .logistic_model import LogisticTradingModel

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Registry for trading models."""
    
    def __init__(self):
        """Initialize the model registry."""
        self._models: Dict[str, Type[BaseTradingModel]] = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Register default models."""
        self.register_model('technical', TechnicalScorer)
        self.register_model('logistic2', LogisticTradingModel)
    
    def register_model(self, name: str, model_class: Type[BaseTradingModel]):
        """Register a model class.
        
        Args:
            name: Name of the model
            model_class: Model class to register
        """
        if not issubclass(model_class, BaseTradingModel):
            raise ValueError(f"Model class must inherit from BaseTradingModel")
            
        if name in self._models:
            raise ValueError(f"Model {name} is already registered")
            
        self._models[name] = model_class
        logger.info(f"Registered model: {name}")
    
    def get_model(self, name: str) -> Optional[Type[BaseTradingModel]]:
        """Get a registered model class by name.
        
        Args:
            name: Name of the model to get
            
        Returns:
            The model class if found, None otherwise
        """
        return self._models.get(name)
    
    def list_models(self) -> Dict[str, Type[BaseTradingModel]]:
        """List all registered models.
        
        Returns:
            Dictionary mapping model names to model classes
        """
        return self._models.copy()
    
    def create_model(self, name: str, **kwargs) -> BaseTradingModel:
        """Create a model instance.
        
        Args:
            name: Name of the model to create
            **kwargs: Arguments to pass to the model constructor
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model name is not registered
        """
        model_class = self.get_model(name)
        if model_class is None:
            raise ValueError(f"Unknown model: {name}")
            
        return model_class(**kwargs)

# Create global registry instance
registry = ModelRegistry() 