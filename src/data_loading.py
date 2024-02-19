import pandas as pd
import sqlite3
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

base_dir = Path(__file__).resolve().parent.parent
db_file_name = 'interview_database_20231023_204043.db'
db_path = base_dir / 'data' / 'raw' / db_file_name
raw_data_dir = base_dir / 'data' / 'raw'

# Import expected schema configuration
sys.path.append(str(base_dir))
from config import expected_schema
if not db_path.is_file():
    logging.error(f"Database file {db_file_name} does not exist at {db_path}")
    sys.exit(1)  

def validate_table_schema(conn, table_name, expected_schema):
    """Validates the schema of a database table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    actual_schema = {row[1]: row[2] for row in cursor.fetchall()}  # Column name and type

    for column, expected_type in expected_schema[table_name].items():
        if column not in actual_schema:
            raise ValueError(f"Column missing in {table_name}: {column}")
        elif actual_schema[column].upper() != expected_type.upper():
            raise ValueError(f"Type mismatch in {table_name} for column {column}: Expected {expected_type}, found {actual_schema[column]}")
    logging.info(f"Schema validation for table '{table_name}' completed successfully.")

def extract_and_save_tables(db_path, tables, output_dir, expected_schema):
    """Extracts tables from the database and saves them as CSV files."""
    try:
        conn = sqlite3.connect(str(db_path))
        for table in tables:
            validate_table_schema(conn, table, expected_schema)
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            df.to_csv(output_dir / f'{table}.csv', index=False)
            logging.info(f"Data extracted and saved for table '{table}'.")
    except sqlite3.Error as e:
        logging.error(f"An error occurred while connecting to the database: {e}")
        raise e
    finally:
        conn.close()

def load_data():
    """Loads raw data files into DataFrames."""
    accounts_df = pd.read_csv(raw_data_dir / 'accounts.csv')
    users_df = pd.read_csv(raw_data_dir / 'users.csv')
    transactions_df = pd.read_csv(raw_data_dir / 'transactions.csv')
    organizations_df = pd.read_csv(raw_data_dir / 'organizations.csv')
    fraud_decisions_df = pd.read_csv(raw_data_dir / 'fraud_decisions.csv')
    logging.info("Data loaded successfully!")
    return accounts_df, users_df, transactions_df, organizations_df, fraud_decisions_df

if __name__ == "__main__":
    tables = ['accounts', 'users', 'organizations', 'transactions', 'fraud_decisions']
    extract_and_save_tables(db_path, tables, raw_data_dir, expected_schema)
