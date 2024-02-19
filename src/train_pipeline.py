from model_selection import (
    load_or_create_design_matrix,
    split_data,
    train_model,
    train_final_model,
    find_best_threshold_for_recall
)
import pandas as pd
from utils import save_model, log_fbeta_metrics
from metrics import calculate_metrics
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
processed_data_dir = Path(__file__).resolve().parent.parent / 'data/processed'
model_base_dir = Path(__file__).resolve().parent.parent / 'model'
model_dir = model_base_dir / 'saved_models'

# Define the selected features for the model
selected_features = [
    'status', 'amount', 'account_type', 'account_profile', 'is_closed_account',
    'country', 'industry', 'decision', 'user_count',
    'mean_account_age', 'median_account_age', 'min_account_age',
    'max_account_age', 'avg_primary_user', 'unique_email_domains',
    'is_active_org', 'day', 'month', 'hour', 'day_sin',
    'day_cos', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos'
]

def train_pipeline(include_org_id=False, optimize_for_recall=False):
    """
    Executes the model training pipeline, including feature assembly,
    model training, and optionally, recall optimization.
    
    Args:
        include_org_id (bool): Flag to include 'organization_id' in features.
        optimize_for_recall (bool): Flag to optimize the model for recall.
    """
    df = load_or_create_design_matrix(use_existing_file=False)
    features = selected_features.copy() 
    
    # Include 'organization_id' if specified
    if include_org_id:
        features.append('organization_id')

    train_df, validation_df, test_df = split_data(df, features)
    
    X_train, y_train = train_df.drop('decision', axis=1), train_df['decision']
    X_val, y_val = validation_df.drop('decision', axis=1), validation_df['decision']
    X_test, y_test = test_df.drop('decision', axis=1), test_df['decision']
    X_test.to_parquet(processed_data_dir / 'unseen_data.parquet')

    model, best_params = train_model(X_train, y_train, X_val, y_val)
    final_model = train_final_model(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]), X_test, y_test, best_params)
    
    if optimize_for_recall:
        optimize_recall(final_model, X_test, y_test)
        
    save_model(final_model, 'final_model.joblib', model_dir)

def optimize_recall(model, X_test, y_test):
    """
    Find the best threshold for recall and log the F-3( recall's contribution to the F3 score is 3 times greater than that of precision)
    
    Args:
        model: The trained model.
        X_test: Test features.
        y_test: Test labels.
    """
    y_scores = model.predict_proba(X_test)[:, 1]
    best_threshold, best_fbeta_score = find_best_threshold_for_recall(y_test, y_scores)
    y_pred_adjusted = (y_scores >= best_threshold).astype(int)
    
    precision, recall, f1 = calculate_metrics(y_test, y_pred_adjusted)
    logging.info('Following are the metrics when the model is optimized to maximize recall')
    log_fbeta_metrics(precision, recall, f1, best_threshold, best_fbeta_score)

if __name__ == "__main__":
    train_pipeline(include_org_id=False, optimize_for_recall=True)
