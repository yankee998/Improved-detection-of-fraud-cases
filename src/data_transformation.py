import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def transform_data(fraud_data, creditcard_data):
    fraud_cat_cols = ['source', 'browser', 'sex', 'country']
    fraud_num_cols = ['purchase_value', 'age', 'hour_of_day', 'day_of_week', 'time_since_signup', 
                      'transaction_count', 'transaction_velocity']
    creditcard_num_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]

    print("NaNs in fraud_data before preprocessing:\n", fraud_data[fraud_cat_cols + fraud_num_cols].isnull().sum())
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ]), fraud_cat_cols),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), fraud_num_cols)
        ])

    X_fraud = fraud_data.drop(columns=['class', 'user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address'])
    y_fraud = fraud_data['class']
    X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42)

    X_fraud_train = preprocessor.fit_transform(X_fraud_train)
    X_fraud_test = preprocessor.transform(X_fraud_test)

    print("NaNs in X_fraud_train after preprocessing:", np.isnan(X_fraud_train).sum())

    smote = SMOTE(random_state=42)
    X_fraud_train_smote, y_fraud_train_smote = smote.fit_resample(X_fraud_train, y_fraud_train)

    X_creditcard = creditcard_data.drop(columns=['Class'])
    y_creditcard = creditcard_data['Class']
    X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test = train_test_split(X_creditcard, y_creditcard, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_creditcard_train = scaler.fit_transform(X_creditcard_train)
    X_creditcard_test = scaler.transform(X_creditcard_test)

    pd.DataFrame(X_fraud_train_smote).to_csv('C:\\Users\\Skyline\\Improved detection of fraud cases\\data\\processed\\X_fraud_train_smote.csv', index=False)
    pd.DataFrame(y_fraud_train_smote).to_csv('C:\\Users\\Skyline\\Improved detection of fraud cases\\data\\processed\\y_fraud_train_smote.csv', index=False)
    pd.DataFrame(X_fraud_test).to_csv('C:\\Users\\Skyline\\Improved detection of fraud cases\\data\\processed\\X_fraud_test.csv', index=False)
    pd.DataFrame(y_fraud_test).to_csv('C:\\Users\\Skyline\\Improved detection of fraud cases\\data\\processed\\y_fraud_test.csv', index=False)
    pd.DataFrame(X_creditcard_train).to_csv('C:\\Users\\Skyline\\Improved detection of fraud cases\\data\\processed\\X_creditcard_train.csv', index=False)
    pd.DataFrame(y_creditcard_train).to_csv('C:\\Users\\Skyline\\Improved detection of fraud cases\\data\\processed\\y_creditcard_train.csv', index=False)
    pd.DataFrame(X_creditcard_test).to_csv('C:\\Users\\Skyline\\Improved detection of fraud cases\\data\\processed\\X_creditcard_test.csv', index=False)
    pd.DataFrame(y_creditcard_test).to_csv('C:\\Users\\Skyline\\Improved detection of fraud cases\\data\\processed\\y_creditcard_test.csv', index=False)

def main():
    fraud_data = pd.read_csv('C:\\Users\\Skyline\\Improved detection of fraud cases\\data\\processed\\Fraud_Data_engineered.csv')
    creditcard_data = pd.read_csv('C:\\Users\\Skyline\\Improved detection of fraud cases\\data\\processed\\creditcard_cleaned.csv')
    transform_data(fraud_data, creditcard_data)

if __name__ == "__main__":
    main()