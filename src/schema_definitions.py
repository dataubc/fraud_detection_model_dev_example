expected_schema = {
        'users': {
            'id': 'TEXT',
            'created_at': 'TIMESTAMP',
            'first_name': 'TEXT',
            'last_name': 'TEXT',
            'email': 'TEXT',
            'is_primary_user': 'BOOLEAN',
            'organization_id': 'TEXT'
        },
        'organizations': {
            'ID': 'TEXT',
            'website': 'TEXT',
            'created_at': 'TIMESTAMP',
            'is_active': 'BOOLEAN',
            'country': 'TEXT',
            'industry': 'TEXT'
        },
        'transactions': {
            'id': 'TEXT',
            'accounts_id': 'TEXT',
            'created_at': 'TIMESTAMP',
            'settled_at': 'TIMESTAMP',
            'status': 'TEXT',
            'amount': 'FLOAT'
        },
        'accounts': {
            'ID': 'TEXT',
            'organization_id': 'TEXT',
            'account_type': 'TEXT',
            'created_at': 'TIMESTAMP',
            'is_closed': 'BOOLEAN'
        },
        'fraud_decisions': {
            'id': 'TEXT',
            'txn_id': 'TEXT',
            'accounts_id': 'TEXT',
            'created_at': 'TIMESTAMP',
            'reviewer_id': 'TEXT',
            'is_false_positive': 'BOOLEAN',
            'decision': 'TEXT'
        }
    }