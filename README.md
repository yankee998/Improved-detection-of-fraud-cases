# Improved Detection of Fraud Cases 🚨💳

Welcome to the **Improved Detection of Fraud Cases** project! This repository implements a robust machine learning pipeline to detect fraudulent transactions using three datasets: `Fraud_Data.csv`, `creditcard.csv`, and `IpAddress_to_Country.csv`. Built with Python 3.11.9, this project covers data preprocessing, feature engineering, transformation, and exploratory data analysis (EDA), setting the stage for advanced fraud detection models.

![Fraud Detection](https://img.shields.io/badge/Fraud-Detection-blueviolet?style=for-the-badge) ![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## 🌟 Project Overview

This project processes and analyzes financial transaction data to identify patterns of fraud. Key components include:

- **Data Cleaning**: Removes duplicates and ensures proper datetime formats.
- **Geolocation Mapping**: Links IP addresses to countries using `IpAddress_to_Country.csv`.
- **Feature Engineering**: Creates features like transaction velocity and time since signup.
- **Data Transformation**: Handles class imbalance with SMOTE, scales numerical features, and encodes categorical ones.
- **Exploratory Data Analysis (EDA)**: Visualizes transaction patterns and fraud distributions.

The pipeline is tested with `pytest` and automated via a GitHub Actions CI/CD workflow.

---

## 📂 Project Structure

```
Improved-detection-of-fraud-cases/
├── data/
│   ├── raw/                    # Raw datasets (Fraud_Data.csv, creditcard.csv, IpAddress_to_Country.csv)
│   ├── processed/              # Processed datasets and EDA plots
├── src/
│   ├── data_preprocessing.py   # Data cleaning and IP-to-country mapping
│   ├── feature_engineering.py  # Feature creation
│   ├── data_transformation.py  # Data scaling, encoding, and SMOTE
├── tests/
│   ├── test_preprocessing.py   # Unit tests for preprocessing and feature engineering
├── notebooks/
│   ├── eda_notebook.ipynb      # EDA visualizations
├── .github/
│   ├── workflows/
│   │   ├── ci.yml             # CI/CD pipeline configuration
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
```

---

## 🚀 Getting Started

Follow these steps to set up and run the project on Windows using Visual Studio Code.

### Prerequisites

- **Python 3.11.9**: [Download](https://www.python.org/downloads/release/python-3119/)
- **Git**: [Download](https://git-scm.com/downloads)
- **_vis Studio Code (recommended)
- Datasets: `Fraud_Data.csv`, `creditcard.csv`, `IpAddress_to_Country.csv` (place in `data/raw/`)

### Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yankee998/Improved-detection-of-fraud-cases.git
   cd Improved-detection-of-fraud-cases
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Datasets**:
   Ensure `data/raw/` contains:
   - `Fraud_Data.csv`
   - `creditcard.csv`
   - `IpAddress_to_Country.csv`

### Running the Pipeline

Execute the scripts in order:

1. **Preprocess Data**:
   ```bash
   python src/data_preprocessing.py
   ```
   - Cleans data, removes duplicates, converts timestamps, and merges IP data.

2. **Engineer Features**:
   ```bash
   python src/feature_engineering.py
   ```
   - Creates features like `hour_of_day`, `time_since_signup`, and `transaction_velocity`.

3. **Transform Data**:
   ```bash
   python src/data_transformation.py
   ```
   - Applies SMOTE, scales numerical features, and encodes categorical features.

4. **Run EDA**:
   ```bash
   jupyter notebook notebooks/eda_notebook.ipynb
   ```
   - Generates visualizations (e.g., purchase value distributions, fraud class boxplots).

5. **Run Tests**:
   ```bash
   pytest tests/ -v
   ```
   - Ensures code quality with unit tests.

### Outputs

- **Processed Data**: Saved in `data/processed/` (e.g., `Fraud_Data_cleaned.csv`, `X_fraud_train_smote.csv`).
- **EDA Plots**: Saved in `data/processed/` (e.g., `purchase_value_dist.png`).

---

## 🔍 Key Features

- **Data Cleaning**:
  - Removes duplicates and handles missing values (impute for numerical, mode for categorical).
  - Converts `signup_time` and `purchase_time` to `datetime64[ns]`.

- **Geolocation**:
  - Maps IP addresses to countries, consolidating rare countries (<1% frequency) into 'Other'.

- **Feature Engineering**:
  - Generates features like `hour_of_day`, `day_of_week`, `time_since_signup`, `transaction_count`, and `transaction_velocity`.

- **Data Transformation**:
  - Balances classes with SMOTE.
  - Scales numerical features (`StandardScaler`) and encodes categorical features (`OneHotEncoder`).

- **EDA**:
  - Visualizes transaction amounts, purchase values, and fraud distributions.

---

## 🛠️ Testing and CI/CD

- **Unit Tests**: Located in `tests/test_preprocessing.py`, covering:
  - Missing value handling
  - Data cleaning (duplicates, datetime conversion)
  - Feature engineering
- **CI/CD Pipeline**: GitHub Actions runs tests on push/pull requests to `main`.

Run tests locally:
```bash
pytest tests/ -v
```

---

## 📬 Contact

For questions or contributions, reach out to:

- **Email**: [yaredgenanaw99@gmail.com](mailto:yaredgenanaw99@gmail.com)
- **GitHub**: [yankee998](https://github.com/yankee998)

---

## 📈 Next Steps

- **Model Training**: Build and evaluate fraud detection models (e.g., logistic regression, random forest).
- **Deployment**: Integrate the pipeline into a production environment.
- **Contribute**: Fork the repo, add features, and submit pull requests!

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

⭐ **Star this repository** if you found it helpful! Contributions and feedback are welcome.