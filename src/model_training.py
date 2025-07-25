import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_transformation import scale_features, apply_smote

def load_and_prepare_data(file_path, target_column):
    """Load data and separate features and target."""
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def train_test_split_data(X, y, test_size=0.2, random_state=42, stratify=None):
    """Perform train-test split."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

def train_model(model, X_train, y_train):
    """Train the model."""
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name, dataset_name):
    """Evaluate model using AUC-PR, F1-Score, and Confusion Matrix."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # AUC-PR
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auc_pr = auc(recall, precision)

    # F1-Score
    f1 = f1_score(y_test, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name} ({dataset_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    os.makedirs('data/processed/plots', exist_ok=True)
    plt.savefig(f'data/processed/plots/cm_{model_name}_{dataset_name}.png')
    plt.close()

    return {'AUC-PR': auc_pr, 'F1-Score': f1}

def main():
    # Define datasets and target columns
    datasets = [
        ('data/processed/cleaned_fraud_data.csv', 'class', 'Fraud_Data'),
        ('data/processed/cleaned_creditcard.csv', 'Class', 'CreditCard')
    ]

    # Models to train
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, C=0.1, random_state=42),  # Increased regularization
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # Results storage
    results = []

    for file_path, target_column, dataset_name in datasets:
        # Load and prepare data
        X, y = load_and_prepare_data(file_path, target_column)
        
        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split_data(X, y, stratify=y)

        # Scale features
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

        # Apply SMOTE only if minority class has enough samples
        if sum(y_train == 1) >= 5:  # Ensure at least 5 minority samples for SMOTE
            X_train_res, y_train_res = apply_smote(X_train_scaled, y_train, k_neighbors=3)
        else:
            print(f"Skipping SMOTE for {dataset_name}: not enough minority class samples")
            X_train_res, y_train_res = X_train_scaled, y_train

        for model_name, model in models.items():
            print(f"Training {model_name} on {dataset_name}...")
            # Train model
            trained_model = train_model(model, X_train_res, y_train_res)

            # Evaluate model
            metrics = evaluate_model(trained_model, X_test_scaled, y_test, model_name, dataset_name)
            metrics['Model'] = model_name
            metrics['Dataset'] = dataset_name
            results.append(metrics)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('data/processed/model_results.csv', index=False)
    print("\nModel Evaluation Results:")
    print(results_df)

if __name__ == "__main__":
    main()