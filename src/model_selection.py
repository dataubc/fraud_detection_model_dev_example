import xgboost as xgb
import pandas as pd
import numpy as np
from metrics import evaluate_model, log_metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve
import logging
from design_matrix import create_design_matrix
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
processed_data_dir = Path(__file__).resolve().parent.parent / 'data/processed'
model_base_dir = Path(__file__).resolve().parent.parent / 'model'
model_dir = model_base_dir / 'saved_models'


def load_or_create_design_matrix(use_existing_file=False):
    design_matrix_file = processed_data_dir / 'design_matrix.parquet'

    if use_existing_file and design_matrix_file.exists():
        logging.info("Loading design matrix from Parquet file.")
        design_matrix = pd.read_parquet(design_matrix_file)
    else:
        logging.info("Creating new design matrix.")
        design_matrix = create_design_matrix()
    
    return design_matrix

def split_data(df, features, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2):
    """Splits the data into training, validation, and testing sets."""
    assert train_ratio + validation_ratio + test_ratio == 1, "The sum of ratios must be 1."

    df = df.set_index('txn_id')
    df = df.sort_values(by='txn_created_at').drop(columns='txn_created_at')
    df = df[features]

    train_cutoff = int(len(df) * train_ratio)
    validation_cutoff = train_cutoff + int(len(df) * validation_ratio)

    train_df = df.iloc[:train_cutoff]
    validation_df = df.iloc[train_cutoff:validation_cutoff]
    test_df = df.iloc[validation_cutoff:]
    return train_df, validation_df, test_df


def train_model(X_train, y_train, X_test, y_test, log_to_mlflow=True):
    
    # Calculate scale_pos_weight
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos
    # Define the model
    model = xgb.XGBClassifier(enable_categorical=True, tree_method='hist')
    # Define the parameters for the grid search
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],         
        'learning_rate': [0.01, 0.1, 0.2],
        'scale_pos_weight': [None, scale_pos_weight]

    }

    # Perform Grid Search CV
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring ='f1', verbose=2)
    grid_search.fit(X_train, y_train)

    # Best parameters and model
    model_params = grid_search.best_params_
    model = grid_search.best_estimator_
    precision, recall, f1 = evaluate_model(model, X_test, y_test)
    if log_to_mlflow:    
        log_metrics(model,model_params, model_base_dir, X_test, y_test, precision, recall, f1)
    return model, model_params

def train_final_model(X_train, y_train,X_test, y_test, best_params):
    """Train the final model with the best parameters on the combined train and validation set."""
    model = xgb.XGBClassifier(**best_params, enable_categorical=True, tree_method='hist')
    model.fit(X_train, y_train)
    precision, recall, f1 = evaluate_model(model, X_test, y_test)
    log_metrics(model, best_params, model_base_dir, X_test, y_test, precision, recall, f1, final_model = True)

    return model

def find_best_threshold_for_recall(y_true, y_scores, beta=3):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f2_scores = [(1 + beta**2) * (prec * rec) / (beta**2 * prec + rec) for prec, rec in zip(precision, recall)]
    best_index = np.argmax(f2_scores)
    return thresholds[best_index], f2_scores[best_index]
