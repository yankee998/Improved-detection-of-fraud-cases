import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_preprocessing import clean_data, preprocess_timestamps, encode_categorical, handle_missing_values
from src.model_training import load_and_prepare_data, train_test_split_data, train_model
from src.data_transformation import scale_features, apply_smote
from sklearn.linear_model import LogisticRegression

def test_handle_missing_values():
    """Test handling of missing values."""
    df = pd.DataFrame({
        "A": [1, None, 3],
        "B": ["x", "y", None],
        "class": [0, 1, 0]
    })
    df_impute = handle_missing_values(df.copy(), strategy='impute')
    df_drop = handle_missing_values(df.copy(), strategy='drop')
    assert df_impute['A'].isna().sum() == 0, "Numerical NaNs should be imputed"
    assert df_impute['B'].isna().sum() == 0, "Categorical NaNs should be imputed"
    assert len(df_drop) == 1, "Rows with NaNs should be dropped"

def test_preprocess_timestamps():
    """Test timestamp preprocessing."""
    df = pd.DataFrame({
        "signup_time": ["2023-01-01 12:00:00", "2023-01-02 15:30:00", None],
        "class": [0, 1, 0]
    })
    df_processed = preprocess_timestamps(df.copy())
    assert "signup_time_hour" in df_processed.columns, "Timestamp features should be created"
    assert "signup_time" not in df_processed.columns, "Original timestamp should be dropped"
    assert df_processed['signup_time_hour'].isna().sum() == 1, "NaTs should be preserved"

def test_encode_categorical():
    """Test categorical encoding."""
    df = pd.DataFrame({
        "country": ["USA", "Canada", "USA"],
        "class": [0, 1, 0]
    })
    df_encoded = encode_categorical(df.copy(), "class")
    assert df_encoded['country'].dtype == int, "Categorical column should be encoded as integers"

def test_clean_data():
    """Test overall data cleaning."""
    df = pd.DataFrame({
        "signup_time": ["2023-01-01 12:00:00", "2023-01-01 12:00:00"],
        "country": ["USA", "USA"],
        "A": [1, 1],
        "class": [0, 0]
    })
    cleaned_df = clean_data(df, "class")
    assert len(cleaned_df) == 1, "Duplicates should be removed"
    assert "signup_time_hour" in cleaned_df.columns, "Timestamp features should be created"
    assert cleaned_df['country'].dtype == int, "Categorical columns should be encoded"

def test_model_training(tmp_path):
    """Test model training and splitting."""
    # Create a temporary CSV file
    df = pd.DataFrame({
        'feature1': [1, 10, 100, 1000, 2, 20],
        'feature2': [2, 20, 200, 2000, 4, 40],
        'class': [0, 1, 0, 0, 1, 0]
    })
    temp_file = tmp_path / "test_data.csv"
    df.to_csv(temp_file, index=False)

    # Load and prepare data
    X, y = load_and_prepare_data(str(temp_file), 'class')
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split_data(X, y, stratify=y)
    
    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    # Train model
    model = LogisticRegression(random_state=42)
    trained_model = train_model(model, X_train_scaled, y_train)
    
    assert len(X_test) >= 1, "Test split should have at least one row"
    assert len(np.unique(y_train)) == 2, "Training data should have both classes"
    assert trained_model.coef_ is not None, "Model should be trained"

def test_apply_smote():
    """Test SMOTE application."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 0, 0, 1, 1, 1])
    X_res, y_res = apply_smote(X, y, random_state=42)
    assert len(X_res) == 6, "SMOTE should balance classes (3+3 samples)"
    assert sum(y_res) == 3, "SMOTE should create equal number of positive samples"