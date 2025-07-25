# main.py
from src.data_preprocessing import load_data, merge_ip_data, preprocess_fraud_data, preprocess_creditcard_data, split_data, handle_imbalance
from src.model_training import train_logistic_regression, train_xgboost, evaluate_model, save_model

def main():
    # File paths
    fraud_path = 'data/raw/Fraud_Data.csv'
    creditcard_path = 'data/raw/creditcard.csv'
    ip_path = 'data/raw/IpAddress_to_Country.csv'
    
    # Load data
    fraud_data, creditcard_data, ip_data = load_data(fraud_path, creditcard_path, ip_path)
    
    # Preprocess Fraud_Data
    fraud_data = merge_ip_data(fraud_data, ip_data)
    fraud_data, fraud_preprocessor = preprocess_fraud_data(fraud_data)
    X_fraud = fraud_data.drop(columns=['class', 'signup_time', 'purchase_time', 'user_id', 'device_id', 'ip_address'])
    y_fraud = fraud_data['class']
    X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = split_data(fraud_data, 'class')
    X_fraud_train_resampled, y_fraud_train_resampled = handle_imbalance(X_fraud_train, y_fraud_train)
    
    # Preprocess creditcard data
    creditcard_data, creditcard_preprocessor = preprocess_creditcard_data(creditcard_data)
    X_creditcard = creditcard_data.drop(columns=['Class'])
    y_creditcard = creditcard_data['Class']
    X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test = split_data(creditcard_data, 'Class')
    X_creditcard_train_resampled, y_creditcard_train_resampled = handle_imbalance(X_creditcard_train, y_creditcard_train)
    
    # Apply preprocessing
    X_fraud_train_resampled = fraud_preprocessor.fit_transform(X_fraud_train_resampled)
    X_fraud_test = fraud_preprocessor.transform(X_fraud_test)
    X_creditcard_train_resampled = creditcard_preprocessor.fit_transform(X_creditcard_train_resampled)
    X_creditcard_test = creditcard_preprocessor.transform(X_creditcard_test)
    
    # Train and evaluate models
    lr_fraud = train_logistic_regression(X_fraud_train_resampled, y_fraud_train_resampled)
    xgb_fraud = train_xgboost(X_fraud_train_resampled, y_fraud_train_resampled)
    lr_creditcard = train_logistic_regression(X_creditcard_train_resampled, y_creditcard_train_resampled)
    xgb_creditcard = train_xgboost(X_creditcard_train_resampled, y_creditcard_train_resampled)
    
    # Evaluate models
    evaluate_model(lr_fraud, X_fraud_test, y_fraud_test, "Logistic Regression", "Fraud_Data")
    evaluate_model(xgb_fraud, X_fraud_test, y_fraud_test, "XGBoost", "Fraud_Data")
    evaluate_model(lr_creditcard, X_creditcard_test, y_creditcard_test, "Logistic Regression", "creditcard")
    evaluate_model(xgb_creditcard, X_creditcard_test, y_creditcard_test, "XGBoost", "creditcard")
    
    # Save models
    save_model(lr_fraud, 'models/lr_fraud.pkl')
    save_model(xgb_fraud, 'models/xgb_fraud.pkl')
    save_model(lr_creditcard, 'models/lr_creditcard.pkl')
    save_model(xgb_creditcard, 'models/xgb_creditcard.pkl')

if __name__ == "__main__":
    main()