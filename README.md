# Improved Detection of Fraud Cases 🚨💳

Welcome to the **Improved Detection of Fraud Cases** project! This repository implements a robust machine learning pipeline to detect fraudulent transactions using three datasets: `Fraud_Data.csv`, `creditcard.csv`, and `IpAddress_to_Country.csv`. Built with Python 3.11.9, the project covers data preprocessing, feature engineering, transformation, model building, training, evaluation, explainability, and exploratory data analysis (EDA), providing a comprehensive solution for fraud detection.

![Fraud Detection](https://img.shields.io/badge/Fraud-Detection-blueviolet?style=for-the-badge) ![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## 🌟 Project Overview

This project processes and analyzes financial transaction data to identify fraudulent patterns using machine learning. Key components include:

- **Data Cleaning**: Removes duplicates (e.g., 1,081 from `creditcard.csv`), handles missing values, and converts timestamps to numerical features (e.g., hour, day, month).
- **Feature Engineering**: Creates features like `signup_time_hour`, `time_since_signup`, and `transaction_velocity`.
- **Geolocation Mapping**: Links IP addresses to countries using `IpAddress_to_Country.csv`, consolidating rare countries into 'Other' (15 unique countries).
- **Data Transformation**: Applies SMOTE to address class imbalance, scales numerical features with `StandardScaler`, and encodes categorical features with `LabelEncoder`.
- **Model Building and Training**: Trains Logistic Regression (`C=0.1`) and Random Forest (`n_estimators=50`, `max_depth=10`) models, evaluated using AUC-PR, F1-Score, and Confusion Matrix.
- **Model Explainability**: Uses SHAP (Shapley Additive exPlanations) to analyze feature importance, generating Summary and Force Plots to identify key fraud drivers.
- **Exploratory Data Analysis (EDA)**: Visualizes transaction patterns, fraud distributions, and model performance metrics.
- **Project Report**: [Detailed Report](Fraud_Detection_Report.md) and [Final Report](Fraud_Detection_Final_Report.md) covering preprocessing, feature engineering, transformation, modeling, explainability, and evaluation.

The pipeline is tested with `pytest` and automated via a GitHub Actions CI/CD workflow.

---

## 📂 Project Structure

```
Improved-detection-of-fraud-cases/
├── data/
│   ├── raw/                    # Raw datasets (Fraud_Data.csv, creditcard.csv, IpAddress_to_Country.csv)
│   ├── processed/              # Processed datasets, EDA plots, model results, and SHAP plots
├── src/
│   ├── data_preprocessing.py   # Data cleaning and IP-to-country mapping
│   ├── feature_engineering.py  # Feature creation
│   ├── data_transformation.py  # Data scaling, encoding, and SMOTE
│   ├── model_training.py       # Model training and evaluation
│   ├── model_explainability.py # SHAP-based model explainability
│   ├── utils.py               # Utility functions
│   ├── __init__.py
├── tests/
│   ├── test_preprocessing.py   # Unit tests for preprocessing and transformation
│   ├── test_models.py          # Unit tests for model training
│   ├── __init__.py
├── notebooks/
│   ├── eda_notebook.ipynb      # EDA and model performance visualizations
├── .github/
│   ├── workflows/
│   │   ├── ci.yml             # CI/CD pipeline configuration
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
├── Fraud_Detection_Report.md   # Detailed project report
├── Fraud_Detection_Final_Report.md # Final report with SHAP analysis
├── README.md                  # This file
├── LICENSE                    # MIT License
```

---

## 🚀 Getting Started

Follow these detailed steps to set up and run the project on Windows using Visual Studio Code (recommended) or any code editor.

### Prerequisites

- **Python 3.11.9**: Download from [python.org](https://www.python.org/downloads/release/python-3119/). Verify with:
  ```powershell
  python --version
  ```
- **Git**: Download from [git-scm.com](https://git-scm.com/downloads). Verify with:
  ```powershell
  git --version
  ```
- **Visual Studio Code** (optional): Install from [code.visualstudio.com](https://code.visualstudio.com/) for a streamlined experience.
- **Datasets**: Obtain `Fraud_Data.csv`, `creditcard.csv`, and `IpAddress_to_Country.csv` (not included in the repository due to size). Contact [yaredgenanaw99@gmail.com](mailto:yaredgenanaw99@gmail.com) if you need assistance sourcing them.

### Setup Instructions

1. **Clone the Repository**:
   ```powershell
   git clone https://github.com/yankee998/Improved-detection-of-fraud-cases.git
   cd Improved-detection-of-fraud-cases
   ```
   - Clones the repository to your local machine (e.g., `C:\Users\Skyline\Improved detection of fraud cases`).

2. **Create and Activate a Virtual Environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # For PowerShell
   # OR
   .\venv\Scripts\activate.bat  # For Command Prompt
   ```
   - Creates an isolated Python environment to manage dependencies.
   - Verify activation: Prompt should show `(venv)`.

3. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
   - Installs required packages (`pandas==2.2.2`, `scikit-learn==1.5.1`, `imbalanced-learn==0.12.3`, `pytest==8.3.2`, `matplotlib==3.9.2`, `seaborn==0.13.2`, `jupyter==1.0.0`, `shap==0.46.0`).
   - Verify installation:
     ```powershell
     pip list
     ```

4. **Set Up Datasets**:
   - Create a `data/raw/` directory if it doesn’t exist:
     ```powershell
     mkdir data\raw
     ```
   - Place `Fraud_Data.csv`, `creditcard.csv`, and `IpAddress_to_Country.csv` in `data/raw/`.
   - **Note**: Ensure file names match exactly, as scripts expect these paths.

5. **Verify Setup**:
   - Confirm Python version:
     ```powershell
     python -c "import pandas; print(pandas.__version__)"
     ```
     - Expected: `2.2.2`
   - Test script execution:
     ```powershell
     python src/data_preprocessing.py
     ```
     - Should create `data/processed/cleaned_fraud_data.csv`, etc., without errors.

### Troubleshooting Setup

- **Python Version Mismatch**: Ensure Python 3.11.9 is used. Update PATH if multiple versions are installed.
- **ModuleNotFoundError**: Verify `requirements.txt` installation and virtual environment activation. Run scripts from the project root.
- **FileNotFoundError**: Check that datasets are in `data/raw/` with correct names.
- **Permission Issues**: Run PowerShell/Command Prompt as Administrator.

### Running the Pipeline

Execute scripts in order to process data, train models, generate explainability plots, and visualize outputs:

1. **Preprocess Data**:
   ```powershell
   python src/data_preprocessing.py
   ```
   - Cleans data, removes duplicates (e.g., 1,081 from `creditcard.csv`), converts timestamps to numerical features (hour, day, month, weekday), maps IP addresses to countries (129,146/151,112 matched), and encodes categorical features.
   - Outputs: `data/processed/cleaned_fraud_data.csv`, `data/processed/cleaned_creditcard.csv`, `data/processed/IpAddress_to_Country_cleaned.csv`.

2. **Engineer Features**:
   ```powershell
   python src/feature_engineering.py
   ```
   - Creates features like `signup_time_hour`, `purchase_time_hour`, `time_since_signup`, and `transaction_velocity`.
   - Outputs: `data/processed/Fraud_Data_engineered.csv`.

3. **Transform Data**:
   ```powershell
   python src/data_transformation.py
   ```
   - Applies `StandardScaler` for feature scaling and SMOTE (`k_neighbors=3`) for class imbalance, skipping SMOTE if minority class has fewer than 5 samples.
   - Outputs: Train/test splits (e.g., `data/processed/X_fraud_train_smote.csv`).

4. **Train and Evaluate Models**:
   ```powershell
   python src/model_training.py
   ```
   - Trains Logistic Regression (`C=0.1`) and Random Forest (`n_estimators=50`, `max_depth=10`) models.
   - Evaluates using AUC-PR, F1-Score, and Confusion Matrix, with Random Forest achieving AUC-PR of 0.815 and F1-Score of 0.834 for `CreditCard`.
   - Outputs: `data/processed/model_results.csv`, confusion matrix plots in `data/processed/plots/`.

5. **Generate SHAP Plots**:
   ```powershell
   python src/model_explainability.py
   ```
   - Generates SHAP Summary and Force Plots for the Random Forest model on a sampled test set (200 rows), highlighting key fraud drivers like `purchase_value`, `time_since_signup` (Fraud_Data), and `V4`, `V14` (CreditCard).
   - Outputs: `data/processed/plots/shap_summary_Fraud_Data.png`, `shap_summary_CreditCard.png`, `shap_force_Fraud_Data.png`, `shap_force_CreditCard.png`.

6. **Run EDA**:
   ```powershell
   jupyter notebook notebooks/eda_notebook.ipynb
   ```
   - Visualizes transaction distributions, fraud patterns, and model performance metrics.
   - Outputs: Plots saved in `data/processed/plots/` (e.g., `purchase_value_dist.png`).

7. **Run Tests**:
   ```powershell
   pytest tests/ -v
   ```
   - Verifies preprocessing, feature engineering, transformation, model training, and explainability with unit tests.
   - Expected: All tests pass.

### Outputs

- **Processed Data**: `data/processed/cleaned_fraud_data.csv`, `data/processed/cleaned_creditcard.csv`, `data/processed/Fraud_Data_engineered.csv`, `data/processed/IpAddress_to_Country_cleaned.csv`, train/test splits.
- **Model Results**: `data/processed/model_results.csv` with AUC-PR, F1-Score, and model details (e.g., Random Forest: AUC-PR 0.815, F1-Score 0.834 for `CreditCard`).
- **Plots**: Confusion matrices, SHAP Summary/Force Plots, and EDA visualizations in `data/processed/plots/` (e.g., `shap_summary_CreditCard.png`, `purchase_value_dist.png`).
- **Reports**: 
  - [Fraud_Detection_Report.md](Fraud_Detection_Report.md): Details preprocessing, feature engineering, transformation, modeling, and evaluation.
  - [Fraud_Detection_Final_Report.md](Fraud_Detection_Final_Report.md): Comprehensive report with SHAP analysis and placeholders for plot screenshots.

---

## 🔍 Key Features

- **Data Cleaning**:
  - Removes duplicates (e.g., 1,081 from `creditcard.csv`) and handles missing values (impute for `Fraud_Data`, drop for `creditcard`).
  - Converts `signup_time` and `purchase_time` to numerical features (hour, day, month, weekday).

- **Feature Engineering**:
  - Creates features like `signup_time_hour`, `purchase_time_hour`, `time_since_signup`, and `transaction_velocity`.

- **Geolocation**:
  - Maps IP addresses to countries (129,146/151,112 matched), consolidating rare countries (<1% frequency) into 'Other' (15 unique countries).

- **Data Transformation**:
  - Balances classes with SMOTE (`k_neighbors=3`), skipped if minority class < 5 samples.
  - Scales numerical features (`StandardScaler`) and encodes categorical features (`LabelEncoder`).

- **Model Building and Training**:
  - Trains Logistic Regression (`C=0.1` for regularization) and Random Forest (`n_estimators=50`, `max_depth=10`).
  - Evaluates using AUC-PR, F1-Score, and Confusion Matrix, with Random Forest outperforming Logistic Regression (AUC-PR: 0.815, F1-Score: 0.834 for `CreditCard`).

- **Model Explainability**:
  - Uses SHAP to generate Summary and Force Plots on a sampled test set (200 rows), identifying key fraud drivers:
    - **Fraud_Data**: `purchase_value`, `time_since_signup`, and `country` are top contributors, indicating large, rapid transactions and regional fraud patterns.
    - **CreditCard**: `V4`, `V14`, and `Amount` drive fraud predictions, highlighting anomalous behavioral patterns and high transaction amounts.

- **Exploratory Data Analysis**:
  - Visualizes transaction distributions, fraud patterns by country and time, and model performance in `notebooks/eda_notebook.ipynb`.

---

## 🛠️ Testing and CI/CD

- **Unit Tests**: Located in `tests/`, covering:
  - Missing value handling
  - Data cleaning (duplicates, datetime conversion)
  - Feature engineering
  - Feature scaling and SMOTE application
  - Model training and explainability
- **CI/CD Pipeline**: GitHub Actions (`ci.yml`) runs tests and scripts on push/pull requests to `main`, ensuring pipeline integrity.

Run tests locally:
```powershell
pytest tests/ -v
```

---

## 📬 Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure code follows PEP 8 style guidelines and includes unit tests in `tests/`.

---

## 📧 Contact

For questions, dataset access, or contributions, reach out to:
- **Email**: [yaredgenanaw99@gmail.com](mailto:yaredgenanaw99@gmail.com)
- **GitHub**: [yankee998](https://github.com/yankee998)

---

## 📈 Next Steps

- **Model Optimization**: Tune Random Forest hyperparameters (e.g., `n_estimators`, `max_depth`) and explore XGBoost or LightGBM.
- **Advanced Features**: Add transaction frequency, user behavior, or network-based features.
- **Class Imbalance**: Experiment with ADASYN or undersampling for `CreditCard` dataset.
- **Cost-Sensitive Learning**: Prioritize false negatives to reduce missed fraud cases.
- **Deployment**: Integrate the pipeline into a production environment.

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

⭐ **Star this repository** if you found it helpful! Contributions and feedback are welcome.