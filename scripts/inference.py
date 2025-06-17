import joblib
import json
from datetime import datetime
from trading_advisor.dataset_v2 import DatasetGeneratorV2
import pandas as pd

def load_model(model_dir):
    """Load the trained model and its metadata."""
    model = joblib.load(f"{model_dir}/model")
    with open(f"{model_dir}/model.metadata", 'r') as f:
        metadata = json.load(f)
    return model, metadata

def predict_single_ticker(ticker, date, model_dir):
    """
    Make predictions for a single ticker and date.
    
    Args:
        ticker (str): Stock ticker symbol
        date (datetime): Date for prediction
        model_dir (str): Directory containing the trained model
    
    Returns:
        dict: Prediction results including probability scores
    """
    # Load model and metadata
    model, metadata = load_model(model_dir)
    
    # Initialize dataset generator
    generator = DatasetGeneratorV2(
        market_features_dir="data/market_features",
        ticker_features_dir="data/ticker_features"
    )
    
    # Prepare inference data
    inference_data = generator.prepare_inference_data(ticker, date)
    
    # Ensure all required features are present
    required_features = metadata['feature_columns']
    missing_features = set(required_features) - set(inference_data.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Select only the required features in the correct order
    X = inference_data[required_features]
    
    # Make prediction
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    # Map prediction to label meaning
    label_map = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}
    
    return {
        "ticker": ticker,
        "date": date.strftime("%Y-%m-%d"),
        "prediction": label_map[prediction],
        "probabilities": {
            "LONG": float(probabilities[1]),
            "SHORT": float(probabilities[2]),
            "NEUTRAL": float(probabilities[0])
        }
    }

if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"
    date = datetime(2024, 3, 20)
    model_dir = "models/2024_all__output"
    
    try:
        result = predict_single_ticker(ticker, date, model_dir)
        print("\nPrediction Results:")
        print(f"Ticker: {result['ticker']}")
        print(f"Date: {result['date']}")
        print(f"Signal: {result['prediction']}")
        print("\nProbabilities:")
        for label, prob in result['probabilities'].items():
            print(f"{label}: {prob:.2%}")
    except Exception as e:
        print(f"Error making prediction: {str(e)}") 