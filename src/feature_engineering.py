import pandas as pd
import numpy as np

def engineer_features(fraud_data):
    fraud_data = fraud_data.copy()
    
    print("NaNs in timestamps before engineering:\n", fraud_data[['signup_time', 'purchase_time']].isnull().sum())
    
    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'], errors='coerce').astype('datetime64[ns]')
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'], errors='coerce').astype('datetime64[ns]')
    
    fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
    fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
    fraud_data['time_since_signup'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds() / 3600
    fraud_data['transaction_count'] = fraud_data.groupby('user_id')['purchase_time'].transform('count')
    fraud_data['transaction_velocity'] = fraud_data['transaction_count'] / (fraud_data['time_since_signup'] + 1e-6)
    
    fraud_data['hour_of_day'] = fraud_data['hour_of_day'].fillna(-1)
    fraud_data['day_of_week'] = fraud_data['day_of_week'].fillna(-1)
    fraud_data['time_since_signup'] = fraud_data['time_since_signup'].fillna(fraud_data['time_since_signup'].mean() if fraud_data['time_since_signup'].notna().any() else 0)
    fraud_data['transaction_velocity'] = fraud_data['transaction_velocity'].fillna(fraud_data['transaction_velocity'].mean() if fraud_data['transaction_velocity'].notna().any() else 0)
    
    print("NaNs in engineered features:\n", fraud_data[['hour_of_day', 'day_of_week', 'time_since_signup', 'transaction_velocity']].isnull().sum())
    return fraud_data

def main():
    fraud_data = pd.read_csv('C:\\Users\\Skyline\\Improved detection of fraud cases\\data\\processed\\Fraud_Data_cleaned.csv')
    fraud_data = engineer_features(fraud_data)
    fraud_data.to_csv('C:\\Users\\Skyline\\Improved detection of fraud cases\\data\\processed\\Fraud_Data_engineered.csv', index=False)

if __name__ == "__main__":
    main()