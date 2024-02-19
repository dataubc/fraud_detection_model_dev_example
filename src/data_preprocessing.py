import pandas as pd
from pathlib import Path
import logging
from data_loading import load_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the base directory relative to the current script
base_dir = Path(__file__).resolve().parent.parent
raw_data_dir = base_dir / 'data/raw'
processed_data_dir = base_dir / 'data/processed'

# Function to preprocess a DataFrame
def preprocess_dataframe(df, rename_columns, categorical_columns, time_columns, boolean_columns=None):
    try:
        df = df.rename(columns=rename_columns)

        for col in time_columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, format='mixed')

        df[categorical_columns] = df[categorical_columns].astype('category')
        if boolean_columns:
            df[boolean_columns] = df[boolean_columns].astype(int)
        return df
    except Exception as e:
        logging.error(f"Error in preprocessing dataframe: {e}")
        raise

def process_all_dataframes(accounts_df, users_df, transactions_df, organizations_df):
    try:
        accounts_df = preprocess_dataframe(
            accounts_df, 
            rename_columns={'ID': 'accounts_id', 'created_at': 'account_created_at', 'is_closed': 'is_closed_account'},
            categorical_columns=['accounts_id', 'organization_id', 'account_type'],
            boolean_columns=['is_closed_account'],
            time_columns=['account_created_at']
        )

        users_df = preprocess_dataframe(
            users_df, 
            rename_columns={'id': 'user_id', 'created_at': 'user_created_at'},
            categorical_columns=['organization_id', 'user_id'],
            boolean_columns=['is_primary_user'],
            time_columns=['user_created_at']
        )

        transactions_df = preprocess_dataframe(
            transactions_df, 
            rename_columns={'id': 'txn_id', 'created_at': 'txn_created_at', 'settled_at': 'txn_settled_at'},
            categorical_columns=['txn_id', 'accounts_id', 'status'],
            time_columns=['txn_created_at', 'txn_settled_at']
        )

        organizations_df = preprocess_dataframe(
            organizations_df, 
            rename_columns={'ID': 'organization_id', 'created_at': 'org_created_at', 'is_active': 'is_active_org'},
            categorical_columns=['organization_id', 'country', 'industry'],
            boolean_columns=['is_active_org'],
            time_columns=['org_created_at']
        )

        logging.info("Data preprocessing completed successfully!")
        return accounts_df, users_df, transactions_df, organizations_df
    except Exception as e:
        logging.error(f"Error in process_all_dataframes: {e}")
        raise

if __name__ == "__main__":
    try:
        accounts_df, users_df, transactions_df, organizations_df, fraud_decisions_df = load_data()
        process_all_dataframes(accounts_df, users_df, transactions_df, organizations_df)
    except Exception as e:
        logging.error(f"Error in main: {e}")
