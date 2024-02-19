import pytest
import sqlite3
from pathlib import Path
import pandas as pd
import logging
import sys
base_dir = Path(__file__).resolve().parent.parent
src_dir = base_dir
sys.path.append(str(src_dir))
from src.data_loading import validate_table_schema  
from config import expected_schema


@pytest.fixture(scope="module")
def db_connection():
    base_dir = Path(__file__).resolve().parent.parent
    db_path = base_dir / 'data/raw/'
    conn = sqlite3.connect(str(db_path))
    yield conn
    conn.close()

def extract_and_save_tables(db_path, tables, output_dir, expected_schema):
    conn = sqlite3.connect(str(db_path))
    for table in tables:
        validate_table_schema(conn, table, expected_schema)
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        df.to_csv(output_dir / f'{table}.csv', index=False)
        logging.info(f"Data extracted and saved for table '{table}'.")
    conn.close()
    conn.close()
def test_users_schema(db_connection):
    validate_table_schema(db_connection, 'users', expected_schema)

def test_organizations_schema(db_connection):
    validate_table_schema(db_connection, 'organizations', expected_schema)

def test_transactions_schema(db_connection):
    validate_table_schema(db_connection, 'transactions', expected_schema)

def test_accounts_schema(db_connection):
    validate_table_schema(db_connection, 'accounts', expected_schema)

def test_fraud_decisions_schema(db_connection):
    validate_table_schema(db_connection, 'fraud_decisions', expected_schema)
