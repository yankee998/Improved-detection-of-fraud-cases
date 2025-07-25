import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """Load CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def handle_missing_values(df, strategy='drop'):
    """Handle missing values in the dataset."""
    print("Missing values before handling:\n", df.isnull().sum())
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'impute':
        for column in df.select_dtypes(include=[np.number]).columns:
            df[column] = df[column].fillna(df[column].mean())
        for column in df.select_dtypes(include=['object']).columns:
            df[column] = df[column].fillna(df[column].mode()[0])
    print("Missing values after handling:\n", df.isnull().sum())
    return df

def preprocess_timestamps(df):
    """Convert timestamp columns to numerical features and drop originals."""
    for col in ['signup_time', 'purchase_time']:
        if col in df.columns:
            print(f"Sample {col} before conversion: {df[col].head().tolist()}")
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_hour'] = df[col].dt.hour
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_weekday'] = df[col].dt.weekday
            df = df.drop(columns=[col])
            print(f"{col} dtype after conversion: {df[f'{col}_hour'].dtype}")
            print(f"NaTs in {col}_hour after conversion: {df[f'{col}_hour'].isna().sum()}")
    return df

def encode_categorical(df, target_column):
    """Encode categorical columns using LabelEncoder, excluding target."""
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != target_column]
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def convert_ip_to_int(ip):
    """Convert IP address to integer."""
    try:
        if isinstance(ip, str):
            parts = ip.split('.')
            if len(parts) != 4:
                return np.nan
            return int(parts[0]) * 256**3 + int(parts[1]) * 256**2 + int(parts[2]) * 256 + int(parts[3])
        elif isinstance(ip, (int, float)) and not pd.isna(ip):
            return int(ip)
        return np.nan
    except (ValueError, AttributeError):
        return np.nan

def merge_with_ip_data(fraud_data, ip_data):
    """Merge fraud data with IP-to-country data."""
    print("Converting IP addresses...")
    fraud_data['ip_address_int'] = fraud_data['ip_address'].apply(convert_ip_to_int)
    print(f"NaNs in ip_address_int: {fraud_data['ip_address_int'].isna().sum()}")
    print(f"Sample ip_address_int values: {fraud_data['ip_address_int'].head().tolist()}")
    
    ip_data['lower_bound_ip_address'] = ip_data['lower_bound_ip_address'].apply(lambda x: x if pd.notnull(x) else 0)
    ip_data['upper_bound_ip_address'] = ip_data['upper_bound_ip_address'].apply(lambda x: x if pd.notnull(x) else 0)
    print(f"IP data bounds: min lower={ip_data['lower_bound_ip_address'].min()}, max upper={ip_data['upper_bound_ip_address'].max()}")
    
    merged_data = fraud_data.copy()
    merged_data['country'] = pd.Series(dtype='object')
    matched_rows = 0
    for idx, row in ip_data.iterrows():
        mask = (merged_data['ip_address_int'] >= row['lower_bound_ip_address']) & \
               (merged_data['ip_address_int'] <= row['upper_bound_ip_address']) & \
               (merged_data['ip_address_int'].notna())
        merged_data.loc[mask, 'country'] = row['country']
        matched_rows += mask.sum()
    print(f"Matched {matched_rows} rows out of {len(merged_data)}")
    
    merged_data['country'] = merged_data['country'].fillna('Unknown')
    country_counts = merged_data['country'].value_counts()
    rare_countries = country_counts[country_counts / len(merged_data) < 0.01].index
    merged_data['country'] = merged_data['country'].apply(lambda x: 'Other' if x in rare_countries else x)
    print(f"Unique countries after consolidation: {merged_data['country'].nunique()}")
    print(f"NaNs in country after merge: {merged_data['country'].isna().sum()}")
    
    return merged_data.drop(columns=['ip_address_int', 'ip_address'])

def clean_data(df, target_column):
    """Clean data: handle duplicates, timestamps, and categorical encoding."""
    print(f"Number of duplicates before: {df.duplicated().sum()}")
    df = df.drop_duplicates().copy()
    print(f"Number of duplicates after: {df.duplicated().sum()}")
    
    df = preprocess_timestamps(df)
    df = encode_categorical(df, target_column)
    return df

def preprocess_datasets():
    """Preprocess Fraud_Data and creditcard datasets."""
    datasets = [
        ('data/raw/Fraud_Data.csv', 'data/processed/cleaned_fraud_data.csv', 'class'),
        ('data/raw/creditcard.csv', 'data/processed/cleaned_creditcard.csv', 'Class')
    ]
    
    ip_data = load_data('data/raw/IpAddress_to_Country.csv')
    ip_data = handle_missing_values(ip_data, strategy='impute')
    ip_data = clean_data(ip_data, None)
    ip_data.to_csv('data/processed/IpAddress_to_Country_cleaned.csv', index=False)
    
    for input_path, output_path, target_column in datasets:
        df = load_data(input_path)
        df = handle_missing_values(df, strategy='impute' if 'Fraud_Data' in input_path else 'drop')
        if 'Fraud_Data' in input_path:
            df = merge_with_ip_data(df, ip_data)
        df = clean_data(df, target_column)
        df.to_csv(output_path, index=False)
        print(f"Preprocessed {input_path} and saved to {output_path}")

if __name__ == "__main__":
    preprocess_datasets()