# design_matrix_test.py
import pandas as pd
import pytest
from pathlib import Path
import pandas as pd
import logging
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent

def check_row_count(df, expected_count):
    actual_count = len(df)
    if actual_count != expected_count:
        raise ValueError(f"Row count mismatch. Expected: {expected_count}, Found: {actual_count}")
    logging.info(f"Row count check passed. Row count: {actual_count}")

def check_null_values(df):
    null_counts = df[['txn_id', 'accounts_id', 'organization_id', 'decision']].isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        raise ValueError(f"NULL values found in columns: \n{null_counts}")
    logging.info("No NULL values found in the DataFrame.")

def check_unique_keys(df, key_columns):
    for col in key_columns:
        if df[col].nunique() != len(df):
            raise ValueError(f"Non-unique values found in key column: {col}")
        logging.info(f"Unique key check passed for column: {col}")

@pytest.fixture
def load_design_matrix():
    processed_data_dir = base_dir / 'data/processed'
    design_matrix_path = processed_data_dir / 'design_matrix.parquet'
    return pd.read_parquet(design_matrix_path)

def test_row_count(load_design_matrix):
    raw_data_dir = base_dir / 'data/raw'
    txn_df_txn_count = pd.read_csv(raw_data_dir /"fraud_decisions.csv")['txn_id'].nunique()
    expected_row_count = txn_df_txn_count
    check_row_count(load_design_matrix, expected_row_count)

def test_null_values(load_design_matrix):
    check_null_values(load_design_matrix)

def test_unique_keys(load_design_matrix):
    key_columns = ['txn_id']  
    check_unique_keys(load_design_matrix, key_columns)