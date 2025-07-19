# Improved Detection of Fraud Cases
A project to develop machine learning models for fraud detection in e-commerce and bank transactions.

## Project Structure
- `data/raw/`: Raw datasets (`Fraud_Data.csv`, `creditcard.csv`, `IpAddress_to_Country.csv`)
- `data/processed/`: Cleaned and engineered datasets
- `src/`: Python scripts for preprocessing, feature engineering, and modeling
- `notebooks/`: Jupyter notebooks for EDA
- `tests/`: Unit tests for scripts
- `.github/workflows/`: CI/CD pipeline configuration

## Setup Instructions
1. Clone the repository: `git clone https://github.com/yankee998/Improved-detection-of-fraud-cases.git`
2. Create and activate a virtual environment: `python -m venv venv && source venv/Scripts/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run preprocessing: `python src/data_preprocessing.py`
5. Run feature engineering: `python src/feature_engineering.py`
6. Run data transformation: `python src/data_transformation.py`

## Datasets
- `Fraud_Data.csv`: E-commerce transaction data
- `creditcard.csv`: Bank credit transaction data
- `IpAddress_to_Country.csv`: IP to country mapping

## Requirements
See `requirements.txt` for dependencies.