import pandas as pd
import os
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def engineer_features(df, dataset_name):
    """Engineer features for fraud detection."""
    df = df.copy()

    # Convert timestamps to numerical features
    if 'signup_time' in df.columns:
        df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
        df['signup_time_hour'] = df['signup_time'].dt.hour
        df['signup_time_day'] = df['signup_time'].dt.day
        df['signup_time_month'] = df['signup_time'].dt.month
        df['signup_time_weekday'] = df['signup_time'].dt.weekday
        df = df.drop(columns=['signup_time'])

    if 'purchase_time' in df.columns:
        df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')
        df['purchase_time_hour'] = df['purchase_time'].dt.hour
        df['purchase_time_day'] = df['purchase_time'].dt.day
        df['purchase_time_month'] = df['purchase_time'].dt.month
        df['purchase_time_weekday'] = df['purchase_time'].dt.weekday
        df = df.drop(columns=['purchase_time'])

    # Calculate time_since_signup
    if 'signup_time_hour' in df.columns and 'purchase_time_hour' in df.columns:
        df['time_since_signup'] = (df['purchase_time_hour'] - df['signup_time_hour']).abs()

    # Transaction velocity (purchase_value per time_since_signup)
    if 'time_since_signup' in df.columns and 'purchase_value' in df.columns:
        df['transaction_velocity'] = df['purchase_value'] / (df['time_since_signup'] + 1)  # Avoid division by zero

    # Ensure all columns are numeric or categorical for encoding
    categorical_columns = ['device_id', 'source', 'browser', 'sex', 'country']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes

    # Save engineered data
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'Fraud_Data_engineered.csv' if dataset_name == 'fraud_data' else f'cleaned_creditcard.csv')
    df.to_csv(output_path, index=False)
    print(f"Engineered data saved to {output_path}")

    return df

def main():
    # Process Fraud_Data
    input_path = 'data/processed/cleaned_fraud_data.csv'
    print(f"Engineering features for Fraud_Data...")
    df = pd.read_csv(input_path)
    df = engineer_features(df, 'fraud_data')

if __name__ == "__main__":
    main()