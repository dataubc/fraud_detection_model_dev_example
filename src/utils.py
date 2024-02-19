import os
import logging
from joblib import dump, load
import mlflow
from mlflow.exceptions import MlflowException
import matplotlib.pyplot as plt
import datetime

def save_model(model, filename, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    file_path = os.path.join(model_dir, filename)
    dump(model, file_path)
    logging.info(f"Model saved to {file_path}")
    
def load_model(filename, model_base_dir):
    model_dir = model_base_dir /'saved_models'
    file_path = os.path.join(model_dir, filename)
    if os.path.exists(file_path):
        model = load(file_path)
        print(f"Model loaded from {file_path}")
        return model
    print(f"No model file found at {file_path}")
    return None

def save_and_log_plot(fig, filename, base_dir):
    """Save matplotlib plot and log it to MLflow."""
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)  # Create plots directory if it doesn't exist
    plot_path = os.path.join(plots_dir, filename)
    
    fig.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    
def log_to_local_file(params, precision, recall, f1):
    with open("local_logging.txt", "a") as file:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Experiment Time: {current_time}\n")
        file.write(f"Model Parameters: {params}\n")
        file.write(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}\n")
        file.write("------------------------------------------------------\n")

def log_fbeta_metrics(precision, recall, f1, threshold, fbeta_score):
    logging.info(f"Best threshold for F4: {threshold}")
    logging.info(f"Best F4 score: {fbeta_score}")
    logging.info(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    