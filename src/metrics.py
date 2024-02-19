from sklearn.metrics import classification_report
import mlflow
import matplotlib.pyplot as plt
from utils import save_and_log_plot
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
import logging
from mlflow.exceptions import MlflowException
from utils import  log_to_local_file
import numpy as np

def log_metrics(model,model_params,model_base_dir, X_test, y_test, precision, recall, f1,
                final_model = False):
    """Logs metrics either to MLflow or locally."""
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        with mlflow.start_run():
            mlflow.log_params(model_params)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("F1 Score", f1)
            if final_model:
                mlflow.set_tags({"model_type": "final_model"})
            else:
                mlflow.set_tags({"model_type": "validation_model"})
                
            y_pred = model.predict(X_test)
            label_map = {0: 'Legitimate', 1: 'Fraud'}
            y_test_mapped = np.array([label_map[label] for label in y_test])
            y_pred_mapped = np.array([label_map[label] for label in y_pred])
            report = classification_report(y_test_mapped , y_pred_mapped)
            report_path = f"{model_base_dir}/classification_report.txt"
            with open(report_path, "w") as f:
                f.write(report)
            mlflow.log_artifact(report_path)
                    
            # Feature Importance Plot
            importances = model.feature_importances_
            feature_names = X_test.columns
            feature_importance = sorted(zip(importances, feature_names), reverse=True)
            fig, ax = plt.subplots(figsize=(10, 8)) 
            y_pos = range(len(feature_importance))
            ax.barh(y_pos, [i[0] for i in feature_importance], align='center')  
            ax.set_yticks(y_pos)
            ax.set_yticklabels([i[1] for i in feature_importance])
            ax.invert_yaxis()  
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_title('Feature Importance', fontsize=14)
            plt.tight_layout() 
            save_and_log_plot(fig, 'feature_importance.png',  model_base_dir)
            # Precision-Recall Curve
            y_scores = model.predict_proba(X_test)[:, 1]
            precision_values, recall_values, _ = precision_recall_curve(y_test, y_scores)
            fig_pr_curve = plt.figure()
            plt.plot(recall_values, precision_values)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            save_and_log_plot(fig_pr_curve, 'precision_recall_curve.png', model_base_dir)
    except MlflowException as e:
        logging.error(f"Failed to log to MLflow: {e}")
        logging.info("Logging metrics locally.")
        log_to_local_file(model_params, precision, recall, f1)
        
def log_fbeta_metrics(precision, recall, f1, threshold, fbeta_score, beta=2):
    logging.info(f"Best threshold for F{beta}: {threshold}")
    logging.info(f"Best F{beta} score: {fbeta_score}")
    logging.info(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    
def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return precision, recall, and F1 score."""
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return precision, recall, f1


def calculate_metrics(y_test, y_pred):
    """Evaluate the model and return precision, recall, and F1 score."""
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return precision, recall, f1