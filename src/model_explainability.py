import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_transformation import scale_features

def load_and_prepare_data(file_path, target_column):
    """Load data and separate features and target."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {file_path} with shape: {df.shape}")
        # Drop high-cardinality columns
        drop_columns = ['device_id', 'user_id', 'ip_address'] if 'Fraud_Data' in file_path else []
        df = df.drop(columns=[col for col in drop_columns if col in df.columns])
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in {file_path}")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        # Encode categorical features
        categorical_cols = ['source', 'browser', 'sex', 'country'] if 'Fraud_Data' in file_path else []
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        # Ensure all features are numeric
        non_numeric = X.select_dtypes(exclude=['float64', 'int64', 'int32']).columns
        if len(non_numeric) > 0:
            raise ValueError(f"Non-numeric columns found in {file_path}: {non_numeric}")
        # Validate target variable
        unique_y = y.unique()
        print(f"Target values in {file_path}: {unique_y}")
        if len(unique_y) != 2:
            raise ValueError(f"Expected binary target, got {len(unique_y)} unique values: {unique_y}")
        # Map target to 0/1
        y = y.map({unique_y[0]: 0, unique_y[1]: 1}).astype(int)
        print(f"Features after processing: {X.columns.tolist()}")
        return X, y
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise

def train_random_forest(X_train, y_train):
    """Train Random Forest model with reduced complexity."""
    try:
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        print(f"Trained Random Forest with {X_train.shape[1]} features")
        print(f"Model classes: {model.classes_}")
        return model
    except Exception as e:
        print(f"Error training Random Forest: {e}")
        raise

def generate_shap_plots(model, X_train, X_test, y_test, dataset_name, feature_names):
    """Generate SHAP Summary and Force Plots on a sample."""
    try:
        # Dynamically adjust sample size (min of 200 or X_test size)
        sample_size = min(200, X_test.shape[0])
        if sample_size < 1:
            raise ValueError(f"Test set for {dataset_name} is empty")
        # Sample using pandas DataFrame
        X_test_sample = X_test.sample(n=sample_size, random_state=42)
        y_test_sample = y_test.loc[X_test_sample.index]
        if X_test_sample.shape[0] != y_test_sample.shape[0]:
            raise ValueError(f"Shape mismatch: X_test_sample {X_test_sample.shape}, y_test_sample {y_test_sample.shape}")
        print(f"Sampled {sample_size} rows for SHAP calculation in {dataset_name}")

        # Convert sampled data to NumPy for SHAP
        X_test_sample_np = X_test_sample.to_numpy()
        y_test_sample_np = y_test_sample.to_numpy()

        # Initialize SHAP Tree Explainer
        explainer = shap.TreeExplainer(model)
        print("Initialized SHAP TreeExplainer")

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test_sample_np)
        print(f"SHAP values type: {type(shap_values)}, shape: {np.array(shap_values).shape}")

        # Handle unexpected SHAP values shape
        if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            # Reshape from (n_samples, n_features, n_classes) to list of [n_samples, n_features]
            shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        elif isinstance(shap_values, list) and len(shap_values) == 2:
            pass  # Expected format
        else:
            raise ValueError(f"Unexpected SHAP values format: type {type(shap_values)}, shape {np.array(shap_values).shape}")
        print(f"Processed SHAP values shape: {[np.array(sv).shape for sv in shap_values]}")
        if len(shap_values) != 2:
            raise ValueError(f"Expected binary classification SHAP values, got {len(shap_values)} classes")
        print(f"SHAP values[1] shape: {np.array(shap_values[1]).shape}")

        # Summary Plot
        plt.figure()
        shap.summary_plot(shap_values[1], X_test_sample, feature_names=feature_names, show=False)
        plt.title(f'SHAP Summary Plot - Random Forest ({dataset_name})')
        os.makedirs('data/processed/plots', exist_ok=True)
        plt.savefig(f'data/processed/plots/shap_summary_{dataset_name}.png', bbox_inches='tight')
        plt.close()

        # Force Plot for a single fraudulent instance
        fraud_idx = np.where(y_test_sample_np == 1)[0][0] if 1 in y_test_sample_np else 0
        if 1 not in y_test_sample_np:
            print(f"Warning: No fraudulent instances in {dataset_name} sample, using first instance")
        shap.force_plot(
            explainer.expected_value[1],
            shap_values[1][fraud_idx],
            X_test_sample.iloc[fraud_idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.title(f'SHAP Force Plot - Random Forest ({dataset_name})')
        plt.savefig(f'data/processed/plots/shap_force_{dataset_name}.png', bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error generating SHAP plots for {dataset_name}: {e}")
        raise

def main():
    # Define datasets and target columns
    datasets = [
        ('data/processed/Fraud_Data_engineered.csv', 'class', 'Fraud_Data'),
        ('data/processed/cleaned_creditcard.csv', 'Class', 'CreditCard')
    ]

    for file_path, target_column, dataset_name in datasets:
        print(f"Generating SHAP plots for {dataset_name}...")
        try:
            # Load and prepare data
            X, y = load_and_prepare_data(file_path, target_column)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
            
            # Scale features, keeping X_test as DataFrame for sampling
            X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
            # Convert scaled data back to DataFrame to allow sampling
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            
            # Train Random Forest
            model = train_random_forest(X_train_scaled.to_numpy(), y_train.to_numpy())
            
            # Generate SHAP plots
            generate_shap_plots(model, X_train_scaled, X_test_scaled, y_test, dataset_name, X.columns.tolist())
        except FileNotFoundError as e:
            print(f"Error: {e}. Ensure {file_path} exists.")
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")

if __name__ == "__main__":
    main()