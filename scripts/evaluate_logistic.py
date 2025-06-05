import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from trading_advisor.models.logistic_model import LogisticTradingModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_scaler_features(model_path: Path, scaler_path: Path, features_path: Path):
    model = LogisticTradingModel.load(model_path)
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    else:
        scaler = None
        logger.warning(f"Scaler file not found at {scaler_path}. Proceeding without scaling.")
    if features_path.exists():
        feature_columns = joblib.load(features_path)
    else:
        feature_columns = None
        logger.warning(f"Feature list file not found at {features_path}. Using all numeric columns from test data.")
    return model, scaler, feature_columns

def preprocess_features(df: pd.DataFrame, scaler, feature_columns):
    if feature_columns is not None:
        numeric_cols = feature_columns
    else:
        numeric_cols = sorted(df.select_dtypes(include=[np.number]).columns)
        numeric_cols = [col for col in numeric_cols if col != 'label']
    df_clean = df[numeric_cols].fillna(df[numeric_cols].mean())
    if scaler is not None:
        X = scaler.transform(df_clean)
    else:
        X = df_clean.values
    return X, numeric_cols

def plot_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_feature_importance(model, feature_names, output_path):
    if hasattr(model.model, 'coef_'):
        importance = np.abs(model.model.coef_[0])
        # Use model's feature columns if available, otherwise use provided feature names
        features = model.feature_columns if hasattr(model, 'feature_columns') else feature_names
        importance_df = pd.DataFrame({'feature': features, 'importance': importance})
        importance_df = importance_df.sort_values('importance', ascending=False)
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df.head(20), x='importance', y='feature')
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info("Top 10 most important features:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")

def evaluate_split(split_num, output_dir):
    model_path = Path(f"model_outputs/logistic/logistic_model_split_{split_num}.pkl")
    scaler_path = Path(f"model_outputs/logistic/scaler_split_{split_num}.pkl")
    features_path = Path(f"model_outputs/logistic/features_split_{split_num}.pkl")
    test_path = Path(f"data/ml_datasets/logistic_all/split_{split_num}/test.parquet")
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}")
        return None
    if not test_path.exists():
        logger.error(f"Test data not found at {test_path}")
        return None
    model, scaler, feature_columns = load_model_scaler_features(model_path, scaler_path, features_path)
    test_df = pd.read_parquet(test_path)
    X_test, feature_names = preprocess_features(test_df, scaler, feature_columns)
    y_test = test_df['label'].values
    test_results = model.predict(pd.DataFrame(X_test, columns=feature_names))
    y_pred = test_results['predictions']
    y_proba = test_results['probabilities']
    accuracy = np.mean(y_pred == y_test)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    logger.info(f"Test Set Metrics for split {split_num}:")
    logger.info(f"accuracy: {accuracy:.4f}")
    logger.info(f"precision: {precision:.4f}")
    logger.info(f"recall: {recall:.4f}")
    logger.info(f"auc: {auc:.4f}")
    plot_confusion_matrix(y_test, y_pred, output_dir / f'confusion_matrix_eval_split_{split_num}.png')
    plot_feature_importance(model, feature_names, output_dir / f'feature_importance_eval_split_{split_num}.png')
    return {'split': split_num, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'auc': auc}

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained logistic model on a test split or all splits.")
    parser.add_argument('--split', type=int, help='Split number to evaluate')
    parser.add_argument('--all', action='store_true', help='Evaluate all splits (0-36)')
    args = parser.parse_args()
    output_dir = Path("model_outputs/logistic")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_list = []
    if args.all:
        for split_num in range(37):
            logger.info(f"\n=== Evaluating Split {split_num} ===")
            metrics = evaluate_split(split_num, output_dir)
            if metrics:
                metrics_list.append(metrics)
        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list)
            logger.info("\n=== Aggregated Results Across All Splits ===")
            logger.info("\nMetrics Summary:")
            for metric in ['accuracy', 'precision', 'recall', 'auc']:
                mean_val = metrics_df[metric].mean()
                std_val = metrics_df[metric].std()
                logger.info(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            logger.error("No successful splits to aggregate results from.")
    elif args.split is not None:
        logger.info(f"\n=== Evaluating Split {args.split} ===")
        evaluate_split(args.split, output_dir)
    else:
        logger.error("Please specify either --split N or --all.")

if __name__ == "__main__":
    main() 