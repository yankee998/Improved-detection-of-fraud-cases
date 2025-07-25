from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd

def scale_features(X_train, X_test):
    """Scale numerical features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def apply_smote(X_train, y_train, random_state=42, k_neighbors=5):
    """Apply SMOTE to balance the dataset."""
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    return X_train_res, y_train_res

if __name__ == "__main__":
    # Example usage
    pass