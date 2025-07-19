import pandas as pd
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def handle_missing_values(df, strategy='drop'):
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

def clean_data(df):
    print(f"Number of duplicates before: {df.duplicated().sum()}")
    df = df.drop_duplicates().copy()
    print(f"Number of duplicates after: {df.duplicated().sum()}")
    
    if 'signup_time' in df.columns:
        print(f"Sample signup_time before conversion: {df['signup_time'].head().tolist()}")
        df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce').astype('datetime64[ns]')
        print(f"signup_time dtype after conversion: {df['signup_time'].dtype}")
        print(f"NaTs in signup_time after conversion: {df['signup_time'].isna().sum()}")
    if 'purchase_time' in df.columns:
        print(f"Sample purchase_time before conversion: {df['purchase_time'].head().tolist()}")
        df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce').astype('datetime64[ns]')
        print(f"purchase_time dtype after conversion: {df['purchase_time'].dtype}")
        print(f"NaTs in purchase_time after conversion: {df['purchase_time'].isna().sum()}")
    return df

def convert_ip_to_int(ip):
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
    
    return merged_data.drop(columns=['ip_address_int'])

def main():
    fraud_data = load_data('C:\\Users\\Skyline\\Improved detection of fraud cases\\data\\raw\\Fraud_Data.csv')
    creditcard_data = load_data('C:\\Users\\Skyline\\Improved detection of fraud cases\\data\\raw\\creditcard.csv')
    ip_data = load_data('C:\\Users\\Skyline\\Improved detection of fraud cases\\data\\raw\\IpAddress_to_Country.csv')
    
    fraud_data = handle_missing_values(fraud_data, strategy='impute')
    creditcard_data = handle_missing_values(creditcard_data, strategy='drop')
    ip_data = handle_missing_values(ip_data, strategy='impute')
    
    fraud_data = clean_data(fraud_data)
    creditcard_data = clean_data(creditcard_data)
    ip_data = clean_data(ip_data)
    
    fraud_data = merge_with_ip_data(fraud_data, ip_data)
    
    fraud_data.to_csv('C:\\Users\\Skyline\\Improved detection of fraud cases\\data\\processed\\Fraud_Data_cleaned.csv', index=False)
    creditcard_data.to_csv('C:\\Users\\Skyline\\Improved detection of fraud cases\\data\\processed\\creditcard_cleaned.csv', index=False)
    ip_data.to_csv('C:\\Users\\Skyline\\Improved detection of fraud cases\\data\\processed\\IpAddress_to_Country_cleaned.csv', index=False)

if __name__ == "__main__":
    main()