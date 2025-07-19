import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import clean_data, handle_missing_values
from src.feature_engineering import engineer_features
import pandas as pd

def test_handle_missing_values():
    df = pd.DataFrame({
        'col1': [1, None, 3],
        'col2': ['A', 'B', None]
    })
    df_imputed = handle_missing_values(df, strategy='impute')
    assert df_imputed['col1'].isna().sum() == 0
    assert df_imputed['col2'].isna().sum() == 0

def test_clean_data():
    df = pd.DataFrame({
        'user_id': [1, 1, 2],
        'signup_time': ['2023-01-01', '2023-01-01', '2023-01-02'],
        'purchase_time': ['2023-01-02', '2023-01-02', '2023-01-03']
    })
    df_cleaned = clean_data(df)
    assert df_cleaned.duplicated().sum() == 0
    assert pd.api.types.is_datetime64_any_dtype(df_cleaned['signup_time'])

def test_engineer_features():
    df = pd.DataFrame({
        'user_id': [1, 2],
        'signup_time': ['2023-01-01 10:00:00', '2023-01-02 12:00:00'],
        'purchase_time': ['2023-01-01 12:00:00', '2023-01-02 14:00:00'],
        'purchase_value': [100, 200],
        'class': [0, 1]
    })
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    df_engineered = engineer_features(df)
    assert 'hour_of_day' in df_engineered.columns
    assert 'time_since_signup' in df_engineered.columns