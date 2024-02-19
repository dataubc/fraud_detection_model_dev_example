import numpy as np
import pandas as pd
import logging

def create_time_features(df, time_column):
    """
    Creates cyclical time features from a timestamp column.
    
    Args:
        df (DataFrame): The input DataFrame.
        time_column (str): The name of the timestamp column.

    Returns:
        DataFrame: The DataFrame with added time features.
    """
    df[time_column] = pd.to_datetime(df[time_column])
    df['day'] = df[time_column].dt.day
    df['month'] = df[time_column].dt.month
    df['hour'] = df[time_column].dt.hour
    df['day_sin'] = np.sin(df['day'] * (2. * np.pi / 31))
    df['day_cos'] = np.cos(df['day'] * (2. * np.pi / 31))
    df['month_sin'] = np.sin((df['month'] - 1) * (2. * np.pi / 12))
    df['month_cos'] = np.cos((df['month'] - 1) * (2. * np.pi / 12))
    df['hour_sin'] = np.sin(df['hour'] * (2. * np.pi / 24))
    df['hour_cos'] = np.cos(df['hour'] * (2. * np.pi / 24))
    return df

def create_txn_level_features(df, transactions_df):
    """
    Merges transaction-level features into the main DataFrame.
    
    Args:
        df (DataFrame): The main DataFrame.
        transactions_df (DataFrame): The transactions DataFrame.

    Returns:
        DataFrame: The main DataFrame with transaction-level features merged in.
    """
    return df.merge(transactions_df, on=['txn_id', 'accounts_id'], how="left")

def create_account_level_features(df, accounts_df):
    """
    Merges account-level features into the main DataFrame.
    
    Args:
        df (DataFrame): The main DataFrame.
        accounts_df (DataFrame): The accounts DataFrame.

    Returns:
        DataFrame: The main DataFrame with account-level features merged in.
    """
    accounts_df = accounts_df.drop(columns=['account_created_at'])
    
    org_account_types = pd.pivot_table( accounts_df , index='organization_id', columns='account_type', 
                                   aggfunc='size', fill_value=0)
    org_account_types['Both_Types'] = ((org_account_types['Checking'] > 0) & 
                                    (org_account_types['Savings'] > 0)).astype(int)
    conditions = [
        (org_account_types['Both_Types'] == 1),
        (org_account_types['Checking'] > 0) & (org_account_types['Savings'] == 0),
        (org_account_types['Checking'] == 0) & (org_account_types['Savings'] > 0)
    ]

    choices = ['Both', 'Checking Only', 'Savings Only']
    org_account_types['account_profile'] = np.select(conditions, choices, default='None')

    df_accounts =  accounts_df.merge(org_account_types['account_profile'], on='organization_id', how='left')
    return df.merge(df_accounts , on='accounts_id', how="left")

def create_org_level_features(df, organizations_df):
    """
    Merges organization-level features into the main DataFrame.
    
    Args:
        df (DataFrame): The main DataFrame.
        organizations_df (DataFrame): The organizations DataFrame.

    Returns:
        DataFrame: The main DataFrame with organization-level features merged in.
    """
    return df.merge(organizations_df, on="organization_id", how="left")

def create_user_and_org_level_features(df, users_df):
    """
    Merges user and organization level aggregated features into the main DataFrame.
    
    Args:
        df (DataFrame): The main DataFrame.
        users_df (DataFrame): The users DataFrame.

    Returns:
        DataFrame: The main DataFrame with user and organization level features merged in.
    """
    users_df['account_age'] = (pd.Timestamp.now() - users_df['user_created_at']).dt.days
    users_df['email_domain'] = users_df['email'].str.split('@').str[1]
    user_count_per_org = users_df.groupby('organization_id', observed=False)['user_id'].nunique().reset_index()
    user_count_per_org.rename(columns={'user_id': 'user_count'}, inplace=True)
    org_user_agg = users_df.groupby('organization_id', observed=False).agg({
        'account_age': ['mean', 'median', 'min', 'max'],
        'is_primary_user': 'mean',
        'email_domain': lambda x: x.nunique()
    }).reset_index()
    org_user_agg.columns = ['organization_id', 'mean_account_age', 'median_account_age', 
                            'min_account_age', 'max_account_age', 'avg_primary_user', 'unique_email_domains']
    org_features_combined = pd.merge(user_count_per_org, org_user_agg, on='organization_id')
    return df.merge(org_features_combined, on="organization_id", how="left")

class FeatureAssembler:
    def __init__(self, transactions_df, accounts_df, organizations_df, users_df):
        self.transactions_df = transactions_df
        self.accounts_df = accounts_df
        self.organizations_df = organizations_df
        self.users_df = users_df

    def assemble(self, df, use_txn_features=True, use_account_features=True, use_org_features=True, use_time_features=True, use_org_and_user_features=True):
        if use_txn_features:
            df = create_txn_level_features(df, self.transactions_df)
            logging.info(f'After transaction features: {df.shape}')
        if use_account_features:
            df = create_account_level_features(df, self.accounts_df)
            logging.info(f'After account features: {df.shape}')
        if use_time_features:
            df = create_time_features(df, 'txn_created_at')
            logging.info(f'After time features: {df.shape}')
        if use_org_features:
            df = create_org_level_features(df, self.organizations_df)
            logging.info(f'After org features: {df.shape}')
        if use_org_and_user_features:
            df = create_user_and_org_level_features(df, self.users_df)
            logging.info(f'After user & org features: {df.shape}')
        return df

