"""Technical scoring model implementation."""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import json

from trading_advisor.models.base import BaseTradingModel
from trading_advisor.config import SCORE_WEIGHTS, MAX_RAW_SCORE, MACD_STRONG_DIVERGENCE, MACD_WEAK_DIVERGENCE

class TechnicalScorer(BaseTradingModel):
    """PyTorch implementation of the technical scoring system."""
    
    def __init__(self):
        super().__init__()
        self.model_name = "TechnicalScorer"
        
        # Define the feature names we expect
        self.feature_columns = [
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Upper', 'BB_Lower', 'BB_Middle', 'BB_Pband',
            'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
            'EMA_100', 'EMA_200', 'Close', 'Volume'
        ]
        
        # Define the scoring components
        self.scoring_components = {
            'rsi': self._score_rsi,
            'bollinger': self._score_bollinger,
            'macd': self._score_macd,
            'moving_averages': self._score_moving_averages,
            'volume': self._score_volume,
            'analyst_targets': self._score_analyst_targets
        }
    
    def _score_rsi(self, rsi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score RSI component."""
        score = torch.zeros_like(rsi)
        details = torch.zeros_like(rsi)
        
        # Oversold condition
        oversold_mask = rsi < 30
        score[oversold_mask] = min(SCORE_WEIGHTS['rsi_oversold'], 2.0)
        details[oversold_mask] = min(SCORE_WEIGHTS['rsi_oversold'], 2.0)
        
        # Overbought condition
        overbought_mask = rsi > 70
        score[overbought_mask] = min(SCORE_WEIGHTS['rsi_overbought'], 2.0)
        details[overbought_mask] = min(SCORE_WEIGHTS['rsi_overbought'], 2.0)
        
        return score, details
    
    def _score_bollinger(self, bb_pband: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score Bollinger Bands component."""
        score = torch.zeros_like(bb_pband)
        details = torch.zeros_like(bb_pband)
        
        # Lower band condition
        lower_mask = bb_pband < 0.05
        score[lower_mask] = SCORE_WEIGHTS['bollinger_low']
        details[lower_mask] = SCORE_WEIGHTS['bollinger_low']
        
        # Upper band condition
        upper_mask = bb_pband > 0.95
        score[upper_mask] = SCORE_WEIGHTS['bollinger_high']
        details[upper_mask] = SCORE_WEIGHTS['bollinger_high']
        
        return score, details
    
    def _score_macd(self, macd: torch.Tensor, macd_signal: torch.Tensor, macd_hist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score MACD component."""
        score = torch.zeros_like(macd)
        details = torch.zeros_like(macd)
        
        # Strong divergence
        strong_div_mask = (macd_hist > MACD_STRONG_DIVERGENCE) & (macd > macd_signal)
        score[strong_div_mask] = min(SCORE_WEIGHTS['macd_strong_divergence'], 2.0)
        details[strong_div_mask] = min(SCORE_WEIGHTS['macd_strong_divergence'], 2.0)
        
        # Moderate divergence
        mod_div_mask = (macd_hist > MACD_WEAK_DIVERGENCE) & (macd > macd_signal)
        score[mod_div_mask] = min(SCORE_WEIGHTS['macd_moderate_divergence'], 2.0)
        details[mod_div_mask] = min(SCORE_WEIGHTS['macd_moderate_divergence'], 2.0)
        
        # Crossover
        crossover_mask = (macd_hist < -MACD_STRONG_DIVERGENCE) & (macd < macd_signal)
        score[crossover_mask] = min(SCORE_WEIGHTS['macd_crossover'], 2.0)
        details[crossover_mask] = min(SCORE_WEIGHTS['macd_crossover'], 2.0)
        
        return score, details
    
    def _score_moving_averages(self, price: torch.Tensor, sma_20: torch.Tensor, sma_50: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score Moving Averages component."""
        score = torch.zeros_like(price)
        details = torch.zeros_like(price)
        
        # Strong above SMA20
        strong_above_mask = price > sma_20 * 1.02
        score[strong_above_mask] = SCORE_WEIGHTS.get('sma_strong_above', 2.0)
        details[strong_above_mask] = SCORE_WEIGHTS.get('sma_strong_above', 2.0)
        
        # Strong below SMA20
        strong_below_mask = price < sma_20 * 0.98
        score[strong_below_mask] = -SCORE_WEIGHTS.get('sma_strong_below', 2.0)
        details[strong_below_mask] = -SCORE_WEIGHTS.get('sma_strong_below', 2.0)
        
        # Above SMA20
        above_mask = (price > sma_20) & ~strong_above_mask
        score[above_mask] = SCORE_WEIGHTS.get('sma_above', 1.0)
        details[above_mask] = SCORE_WEIGHTS.get('sma_above', 1.0)
        
        # Above SMA50
        above_50_mask = (price > sma_50) & ~above_mask & ~strong_above_mask
        score[above_50_mask] = SCORE_WEIGHTS.get('sma_above_50', 1.0)
        details[above_50_mask] = SCORE_WEIGHTS.get('sma_above_50', 1.0)
        
        return score, details
    
    def _score_volume(self, volume: torch.Tensor, prev_volume: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score Volume component."""
        score = torch.zeros_like(volume)
        details = torch.zeros_like(volume)
        
        # Calculate volume change
        volume_change = (volume - prev_volume) / prev_volume * 100
        
        # Volume spike
        spike_mask = torch.abs(volume_change) > 20
        score[spike_mask] = min(SCORE_WEIGHTS['volume_spike'], 2.0)
        details[spike_mask] = min(SCORE_WEIGHTS['volume_spike'], 2.0)
        
        return score, details
    
    def _score_analyst_targets(self, current_price: torch.Tensor, median_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score Analyst Targets component."""
        score = torch.zeros_like(current_price)
        details = torch.zeros_like(current_price)
        
        # Calculate upside
        upside = ((median_target - current_price) / current_price) * 100
        
        # Score based on upside
        score = torch.clamp(upside / 10, 0, 2.0)
        details = score.clone()
        
        return score, details
    
    def prepare_input(self, df: pd.DataFrame) -> torch.Tensor:
        """Prepare input data for the model.
        
        Args:
            df: DataFrame containing the input features
            
        Returns:
            torch.Tensor: Prepared input tensor
        """
        # Ensure all required columns are present
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert DataFrame to tensor
        features = df[self.feature_columns].values
        return torch.FloatTensor(features)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, n_features)
            
        Returns:
            Dict containing scores and details for each component
        """
        # Extract features
        rsi = x[:, self.feature_columns.index('RSI')]
        macd = x[:, self.feature_columns.index('MACD')]
        macd_signal = x[:, self.feature_columns.index('MACD_Signal')]
        macd_hist = x[:, self.feature_columns.index('MACD_Hist')]
        bb_pband = x[:, self.feature_columns.index('BB_Pband')]
        price = x[:, self.feature_columns.index('Close')]
        sma_20 = x[:, self.feature_columns.index('SMA_20')]
        sma_50 = x[:, self.feature_columns.index('SMA_50')]
        volume = x[:, self.feature_columns.index('Volume')]
        
        # Calculate scores for each component
        rsi_score, rsi_details = self._score_rsi(rsi)
        bb_score, bb_details = self._score_bollinger(bb_pband)
        macd_score, macd_details = self._score_macd(macd, macd_signal, macd_hist)
        ma_score, ma_details = self._score_moving_averages(price, sma_20, sma_50)
        
        # For volume, we need the previous volume
        prev_volume = torch.roll(volume, 1)
        prev_volume[0] = volume[0]  # Handle first element
        vol_score, vol_details = self._score_volume(volume, prev_volume)
        
        # Calculate total score
        total_score = rsi_score + bb_score + macd_score + ma_score + vol_score
        
        # Normalize score to 0-10 range
        normalized_score = torch.clamp((total_score / MAX_RAW_SCORE) * 10, 0, 10)
        
        return {
            'total_score': normalized_score,
            'details': {
                'rsi': rsi_details,
                'bollinger': bb_details,
                'macd': macd_details,
                'moving_averages': ma_details,
                'volume': vol_details
            }
        }
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions on input data.
        
        Args:
            df: DataFrame containing the input features
            
        Returns:
            Dict containing predictions and score details
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            x = self.prepare_input(df)
            outputs = self.forward(x)
            
            # Convert tensors to numpy arrays
            predictions = {
                'score': outputs['total_score'].numpy(),
                'details': {
                    k: v.numpy() for k, v in outputs['details'].items()
                }
            }
            
            return predictions 