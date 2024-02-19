# Fraud Detection Model Development
This is an example of a simple ML development and deployment system.
It assumes that you have a db file that has information about fraudulent transactions with different tables, accounts, transactions, etc and train an ML model to detect whether a transactions is fraudulent or legitimate.

# Contents

## Comprehensive Data Processing, Feature Engineering, Training, and Prediction Pipeline

### Setup Instructions

This section includes detailed instructions for setting up the environment and necessary dependencies. It guides you through the end-to-end process of preparing your data, engineering relevant features, training the model, and making predictions. This structured approach ensures a seamless workflow from data ingestion to model deployment
### 1. Create a Virtual Environment:

For Linux/Mac:

```bash
python3.10 -m venv fraud_detection_env
source fraud_detection_env/bin/activate
```

### 2. Install Requirements:
Install the required packages using the requirements file.

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

###  3. Load Data:
Extract data from the database and save it as CSV files:
```bash
python src/data_loading.py
```
This script processes the initial data extraction from the SQLite database and performs schema validation. It converts each table into CSV format and ensures all tables are loaded successfully for further analysis.

### 4. Create Design Matrix:
Assemble and preprocess features from multiple data sources into a design matrix:
```bash
python src/design_matrix.py
```
This script is responsible for constructing the design matrix which serves as the foundation for the model. It assembles features from various data sources, such as transactions, accounts, organizations, and users, after preprocessing. It also provides the option to drop specific features and to save the final design matrix as a Parquet file, ensuring the data is ready for model training. The script logs the process, indicating the successful creation and saving of the design matrix.

### 5. Start MLflow Server
In a separate terminal, initiate the MLflow server:
```bash
mlflow server --backend-store-uri sqlite:///mydb.sqlite
```

### 5. Train Model Pipeline:
Train and evaluate the model:
```bash
python src/train_pipeline.py
```
During the training process, the data was split based on time to create training, validation, and test sets. This temporal split ensures that the model is validated and tested on data that simulates future, unseen transactions, which is critical for a realistic assessment of model performance.

The model's performance was optimized on the validation set to identify the best hyperparameters. This optimization process included experimenting with different compositions of features to understand their impact on model accuracy. Additionally, cross-validation was employed to ensure the robustness and generalizability of the model.

Given the time frame, we directly trained an XGBoost model. However, it is typical in such scenarios to experiment with a variety of models like logistic regression and random forests. Consideration should also be given to appropriate data scaling and transformation techniques to ensure optimal model performance.

Once the best parameters were determined, the final model was trained on the combined training and validation sets. Subsequently, the test set was used to evaluate the model's performance on an unseen dataset, providing a measure of how well the model is expected to perform when deployed in a live environment.

Notably, we chose not to include `organization_id` as a feature, despite its potential to enhance model performance. This decision was made to avoid reliance on data that might not be available for future transactions. Instead, we focused on creating features that describe aspects of organizations, such as the number of users per organization, which are likely to be available and relevant for new organizations as well.

### 6. Scoring Pipeline:
Execute the scoring pipeline to apply the trained model to new data and generate predictions using the following command:
```bash
python src/scoring_pipeline.py
```
The scoring pipeline is designed to automatically load the saved model from the model directory. It then retrieves the unseen datasets from the processed directory, scores the test set, and records the predictions. The results, including the transaction ID, the model's probability scores, and the final predictions, are written to test_results.parquet within the model directory. This output is structured to assist the fraud operations team in their decision-making process.

### Running the Analysis
Ensure Python 3.10 is installed and navigate to the project's root directory. With the virtual environment activated, run the scripts sequentially as listed above.

For a visual walkthrough of these setup instructions, you can watch the [video](watchme.mp4) file. This video demonstrates the steps explained above in sequential order, providing a practical guide to each part of the setup process


### Notes
- data_loading.py: Transforms raw data into a processed format, preparing it for analysis and modeling.

- design_matrix.py: Builds essential features for learning, ensuring the model has relevant and meaningful input data.

- train_pipeline.py: Manages the model training and hyperparameter tuning process, optimizing the model for best performance.

- scoring_pipeline.py: Scores new data, a critical component for operationalizing fraud detection.

- Model Directory: Houses the saved model file final_model.joblib, along with test performance plots and scoring results in test_results.parquet, providing a comprehensive view of the model's performance and output.

- Tests Directory: Contains test scripts that validate data schemas and ensure the quality of the design matrix data. These scripts are executable via pytest from the root directory, ensuring robustness and consistency of the data processing pipeline.

Additionally, there are other helper functions and modules to facilitate various aspects of the workflow:

- schema_definitions.py
- utils.py:
- model_selection.py
- feature_store.py:
- config.py

These additional modules contribute significantly to the modularity, efficiency, and effectiveness of the data processing, model training, and evaluation processes.
