import pandas as pd
from pathlib import Path
import logging
from feature_store import FeatureAssembler
from data_loading import load_data
from data_preprocessing import process_all_dataframes

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the base directory relative to the current script
base_dir = Path(__file__).resolve().parent.parent
processed_data_dir = base_dir / 'data/processed'

def get_base_df(fraud_decisions_df):
    """
    Aggregates fraud decisions and maps decision labels to binary values.
    
    Args:
        fraud_decisions_df (DataFrame): The fraud decisions DataFrame.

    Returns:
        DataFrame: The base DataFrame with decisions aggregated and mapped.
    """
    base_df = fraud_decisions_df.groupby(['txn_id', 'accounts_id'], observed=False)['decision'].agg(pd.Series.mode).reset_index()
    base_df['decision'] = base_df['decision'].map({'Legitimate': 0, 'Fraud': 1})
    logging.info(f'Base df Shape {base_df.shape}')
    return base_df

def create_design_matrix(drop_features=True, drop_features_list=['txn_settled_at'], save_as_parquet=True):
    """
    Creates a design matrix by assembling features from various tables.
    
    Args:
        drop_features (bool): Flag to indicate if features should be dropped.
        drop_features_list (list): List of features to drop if drop_features is True.
        save_as_parquet (bool): Flag to save the design matrix as a Parquet file.

    Returns:
        DataFrame: The design matrix with all features assembled.
    """
    # Load dataframes
    accounts_df, users_df, transactions_df, organizations_df, fraud_decisions_df = load_data()
    base_df = get_base_df(fraud_decisions_df)
    
    # Preprocess and assemble features
    accounts_df, users_df, transactions_df, organizations_df = process_all_dataframes(accounts_df, users_df, transactions_df, organizations_df)
    assembler = FeatureAssembler(transactions_df, accounts_df, organizations_df, users_df)
    design_matrix = assembler.assemble(base_df)
    
    # Drop specified features if needed
    if drop_features:
        design_matrix = design_matrix.drop(columns=drop_features_list)    
    design_matrix[['txn_id', 'accounts_id', 'organization_id', 'account_profile']] = design_matrix[['txn_id', 'accounts_id', 'organization_id', 'account_profile']].astype('category')
    design_matrix = design_matrix.drop(columns="website")

    # Save design matrix as Parquet file if specified
    if save_as_parquet:
        processed_data_dir.mkdir(exist_ok=True)
        design_matrix.to_parquet(processed_data_dir / 'design_matrix.parquet', index=False)
        logging.info("Design matrix saved as Parquet file.")
    if 'object' in list(design_matrix.dtypes):
        print(design_matrix.dtypes)
        logging.error("Design matrix contain at least one column with object data type")

    return design_matrix

if __name__ =='__main__':
    create_design_matrix()
