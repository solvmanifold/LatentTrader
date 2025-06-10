"""Feature normalization module for consistent data scaling across training and inference."""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

class FeatureNormalizer:
    """Handles feature normalization with versioning and reproducibility.
    
    This class manages the normalization of features for both training and inference,
    ensuring that new data is normalized consistently with the training data.
    It supports multiple normalization strategies and stores normalization parameters
    for reproducibility.
    """
    
    # Define normalization strategies
    NORMALIZATION_STRATEGIES = {
        'standard': StandardScaler,  # z-score normalization
        'minmax': MinMaxScaler,     # scale to [0,1] range
        'robust': RobustScaler      # robust to outliers
    }
    
    # Define feature-specific normalization strategies
    FEATURE_STRATEGIES = {
        # Price-based features use robust scaling due to outliers
        'open': 'robust',
        'high': 'robust',
        'low': 'robust',
        'close': 'robust',
        'adj_close': 'robust',
        
        # Volume features use robust scaling
        'volume': 'robust',
        'volume_prev': 'robust',
        
        # Technical indicators use standard scaling
        'rsi': 'standard',
        'macd': 'standard',
        'macd_signal': 'standard',
        'macd_hist': 'standard',
        
        # Bollinger Bands use minmax scaling
        'bb_upper': 'minmax',
        'bb_lower': 'minmax',
        'bb_middle': 'minmax',
        'bb_pband': 'minmax',
        
        # Moving averages use standard scaling
        'sma_20': 'standard',
        'sma_50': 'standard',
        'sma_100': 'standard',
        'sma_200': 'standard',
        'ema_100': 'standard',
        'ema_200': 'standard',
        
        # Returns and momentum use standard scaling
        'returns_1d': 'standard',
        'returns_5d': 'standard',
        'returns_20d': 'standard',
        'momentum_5d': 'standard',
        'momentum_20d': 'standard',
        
        # Relative strength uses standard scaling
        'relative_strength': 'standard',
        'relative_strength_ratio': 'standard'
    }
    
    def __init__(
        self,
        output_dir: Union[str, Path] = "data/normalization",
        version: str = "1.0.0"
    ):
        """Initialize the feature normalizer.
        
        Args:
            output_dir: Directory to store normalization parameters
            version: Version of the normalization strategy
        """
        self.output_dir = Path(output_dir)
        self.version = version
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store normalization parameters
        self.scalers: Dict[str, BaseEstimator] = {}
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        
    def fit(self, df: pd.DataFrame) -> 'FeatureNormalizer':
        """Fit normalization parameters to the data.
        
        Args:
            df: DataFrame containing features to normalize
            
        Returns:
            self for method chaining
        """
        logger.info(f"Fitting normalization parameters for {len(df.columns)} features")
        
        # Initialize storage for feature statistics
        self.feature_stats = {}
        
        # Fit scalers for each feature
        for column in df.columns:
            if column in ['ticker', 'label']:  # Skip non-feature columns
                continue
                
            # Get strategy for this feature
            strategy = self.FEATURE_STRATEGIES.get(column, 'standard')
            scaler_class = self.NORMALIZATION_STRATEGIES[strategy]
            
            # Create and fit scaler
            scaler = scaler_class()
            values = df[column].values.reshape(-1, 1)
            scaler.fit(values)
            
            # Store scaler and statistics
            self.scalers[column] = scaler
            self.feature_stats[column] = {
                'strategy': strategy,
                'mean': float(scaler.mean_[0]) if hasattr(scaler, 'mean_') else None,
                'scale': float(scaler.scale_[0]) if hasattr(scaler, 'scale_') else None,
                'min': float(scaler.min_) if hasattr(scaler, 'min_') else None,
                'max': float(scaler.max_) if hasattr(scaler, 'max_') else None,
                'center': float(scaler.center_) if hasattr(scaler, 'center_') else None,
                'scale_': float(scaler.scale_) if hasattr(scaler, 'scale_') else None
            }
            
        # Save normalization parameters
        self._save_parameters()
        
        return self
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted normalization parameters.
        
        Args:
            df: DataFrame containing features to normalize
            
        Returns:
            DataFrame with normalized features
        """
        if not self.scalers:
            raise ValueError("Normalizer must be fitted before transform")
            
        # Create copy of input DataFrame
        normalized_df = df.copy()
        
        # Transform each feature
        for column in df.columns:
            if column in ['ticker', 'label']:  # Skip non-feature columns
                continue
                
            if column not in self.scalers:
                logger.warning(f"No normalization parameters found for {column}, skipping")
                continue
                
            # Get values and handle NaN
            values = df[column].values.reshape(-1, 1)
            mask = ~np.isnan(values.ravel())
            
            if not mask.any():
                logger.warning(f"All values are NaN for {column}, skipping")
                continue
                
            # Transform non-NaN values
            normalized_values = np.full_like(values, np.nan)
            normalized_values[mask] = self.scalers[column].transform(values[mask])
            
            # Update DataFrame
            normalized_df[column] = normalized_values.ravel()
            
        return normalized_df
        
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform normalized features back to original scale.
        
        Args:
            df: DataFrame containing normalized features
            
        Returns:
            DataFrame with features in original scale
        """
        if not self.scalers:
            raise ValueError("Normalizer must be fitted before inverse_transform")
            
        # Create copy of input DataFrame
        original_df = df.copy()
        
        # Transform each feature
        for column in df.columns:
            if column in ['ticker', 'label']:  # Skip non-feature columns
                continue
                
            if column not in self.scalers:
                logger.warning(f"No normalization parameters found for {column}, skipping")
                continue
                
            # Get values and handle NaN
            values = df[column].values.reshape(-1, 1)
            mask = ~np.isnan(values.ravel())
            
            if not mask.any():
                logger.warning(f"All values are NaN for {column}, skipping")
                continue
                
            # Transform non-NaN values
            original_values = np.full_like(values, np.nan)
            original_values[mask] = self.scalers[column].inverse_transform(values[mask])
            
            # Update DataFrame
            original_df[column] = original_values.ravel()
            
        return original_df
        
    def _save_parameters(self) -> None:
        """Save normalization parameters to disk."""
        # Create version directory
        version_dir = self.output_dir / self.version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save feature statistics
        stats_path = version_dir / "feature_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.feature_stats, f, indent=2)
            
        # Save normalization strategy
        strategy_path = version_dir / "normalization_strategy.json"
        strategy = {
            'version': self.version,
            'feature_strategies': self.FEATURE_STRATEGIES,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        with open(strategy_path, 'w') as f:
            json.dump(strategy, f, indent=2)
            
        logger.info(f"Saved normalization parameters to {version_dir}")
        
    @classmethod
    def load(cls, version: str, output_dir: Union[str, Path] = "data/normalization") -> 'FeatureNormalizer':
        """Load normalization parameters from disk.
        
        Args:
            version: Version of normalization parameters to load
            output_dir: Directory containing normalization parameters
            
        Returns:
            FeatureNormalizer instance with loaded parameters
        """
        output_dir = Path(output_dir)
        version_dir = output_dir / version
        
        if not version_dir.exists():
            raise ValueError(f"No normalization parameters found for version {version}")
            
        # Create normalizer instance
        normalizer = cls(output_dir=output_dir, version=version)
        
        # Load feature statistics
        stats_path = version_dir / "feature_stats.json"
        with open(stats_path) as f:
            normalizer.feature_stats = json.load(f)
            
        # Load normalization strategy
        strategy_path = version_dir / "normalization_strategy.json"
        with open(strategy_path) as f:
            strategy = json.load(f)
            
        # Create scalers from statistics
        for column, stats in normalizer.feature_stats.items():
            strategy_name = stats['strategy']
            scaler_class = normalizer.NORMALIZATION_STRATEGIES[strategy_name]
            scaler = scaler_class()
            
            # Set scaler parameters
            if hasattr(scaler, 'mean_'):
                scaler.mean_ = np.array([stats['mean']])
            if hasattr(scaler, 'scale_'):
                scaler.scale_ = np.array([stats['scale']])
            if hasattr(scaler, 'min_'):
                scaler.min_ = stats['min']
            if hasattr(scaler, 'max_'):
                scaler.max_ = stats['max']
            if hasattr(scaler, 'center_'):
                scaler.center_ = stats['center']
            if hasattr(scaler, 'scale_'):
                scaler.scale_ = stats['scale']
                
            normalizer.scalers[column] = scaler
            
        logger.info(f"Loaded normalization parameters from {version_dir}")
        return normalizer 