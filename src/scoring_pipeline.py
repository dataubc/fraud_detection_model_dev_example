import pandas as pd
import numpy as np
from joblib import load
from pathlib import Path

# Define paths
base_dir = Path(__file__).resolve().parent.parent
model_dir = base_dir / 'model' / 'saved_models'
results_dir = base_dir / 'model'
processed_data_dir = Path(__file__).resolve().parent.parent / 'data/processed'

# Ensure the results directory exists
results_dir.mkdir(exist_ok=True)

def load_model(filename):
    """Load the model from the file."""
    model_path = model_dir / filename
    model = load(model_path)
    return model

def score_dataset(model, dataset_path):
    """Score the dataset using the loaded model."""
    # Load the dataset to be scored
    X_test = pd.read_parquet(dataset_path)
    print(X_test.head())
    # Generate predictions and prediction probabilities
    results = X_test.copy()
    results ['model_prediction'] = model.predict(X_test)
    results ['fraud_probability'] = model.predict_proba(X_test)[:, 1]
    # Map numerical predictions back to labels
    results['model_prediction'] = results['model_prediction'].map({0: 'Legitimate', 1: 'Fraud'})
    return results[['fraud_probability', 'model_prediction']]

def save_results(results_df, filename):
    """Save the results to a Parquet file."""
    results_path = results_dir / filename
    results_df.to_parquet(results_path)
    print(results_df.head())

def scoring_pipeline(model_filename, dataset_path, results_filename='test_results.parquet'):
    """The full scoring pipeline."""
    model = load_model(model_filename)
    scored_results = score_dataset(model, dataset_path)
    save_results(scored_results, results_filename)
    print(f"Scoring completed. Results saved to {results_filename}")

if __name__ == '__main__':
    scoring_pipeline('final_model.joblib', processed_data_dir / 'unseen_data.parquet')
