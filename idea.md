# LatentTrader: Vision and Architecture

## Project Vision
LatentTrader aims to be a trading assistant that generates actionable, short-term trading playbooks designed to beat the market. The system will leverage both classic technical analysis and machine learning (ML) models to create, test, and iterate on predictive scoring models. The ultimate goal is to support robust experimentation, backtesting, and rapid iteration to discover and deploy the most effective trading strategies.

## Architecture Overview

### 1. Data Layer (Features)
- **Source of Truth:** All raw and engineered features (technical indicators, price/volume, etc.) are stored in Parquet files, one per ticker (e.g., `features/AAPL_features.parquet`).
- **Update Process:** Features are updated daily (or as needed) via a dedicated command (`init-features` or `update-features`).
- **No Model Scores:** These files contain only features, not model outputs.

### 2. Model Layer (Outputs)
- **Model Registry:** Each model (classic, ML, ensemble, etc.) is a Python class/function with a standard interface (e.g., `score(df)` or `generate_playbook(df)`).
- **Model Runner:** A command or function loads features, runs one or more models, and outputs scores/playbooks for each ticker/date.
- **Model Outputs:** Model scores and playbooks are stored in a separate folder (e.g., `model_outputs/`), with one file per model (e.g., `model_outputs/classic.parquet`, `model_outputs/ml_v1.parquet`). These files are keyed by ticker and date.
- **Experimentation:** Easy to add, swap, or update models and compare their outputs.

### 3. Reporting/Analysis Layer
- **Analyze:** Purely reads model outputs, sorts, and generates JSON for reporting. No downloads or recomputation.
- **Report/Prompt:** Consumes the JSON, generates human/LLM-readable outputs.
- **Backtest/Experiment:** CLI or notebook tools to run models on historical data and compare performance.

## Next Steps
- **First Refactor:**
  - Move the current scoring logic (currently `analysis.calculate_score` and the `score` column in features Parquet files) out of the features layer.
  - Implement a model runner that writes model outputs (scores, playbooks) to a dedicated `model_outputs/` folder.
  - Update all downstream analysis and reporting to read from model outputs, not features.
- **Iterate:**
  - Add support for multiple models and easy experimentation.
  - Enable historical backtesting and model comparison.

---

*This architecture will enable rapid iteration, robust experimentation, and a clear separation of concerns between data, modeling, and reporting. The first step is to refactor the scoring model and output storage as described above.*
