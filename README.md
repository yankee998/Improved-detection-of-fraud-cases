# Improved Detection of Fraud Cases 🚨💳

Welcome to the **Improved Detection of Fraud Cases** project! This repository implements a robust machine learning pipeline to detect fraudulent transactions using three datasets: `Fraud_Data.csv`, `creditcard.csv`, and `IpAddress_to_Country.csv`. Built with Python 3.11.9, this project covers data preprocessing, transformation, model building, training, evaluation, and exploratory data analysis (EDA), providing a comprehensive solution for fraud detection.

![Fraud Detection](https://img.shields.io/badge/Fraud-Detection-blueviolet?style=for-the-badge) ![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## 🌟 Project Overview

This project processes and analyzes financial transaction data to identify fraudulent patterns using machine learning. Key components include:

- **Data Cleaning**: Removes duplicates (e.g., 1081 from `creditcard.csv`), handles missing values, and converts timestamps to numerical features (e.g., hour, day, month).
- **Geolocation Mapping**: Links IP addresses to countries using `IpAddress_to_Country.csv`, consolidating rare countries into 'Other' (15 unique countries).
- **Data Transformation**: Applies SMOTE to address class imbalance, scales numerical features with `StandardScaler`, and encodes categorical features with `LabelEncoder`.
- **Model Building and Training**: Trains Logistic Regression (`C=0.1`) and Random Forest (`n_estimators=100`) models, evaluated using AUC-PR, F1-Score, and Confusion Matrix.
- **Exploratory Data Analysis (EDA)**: Visualizes transaction patterns, fraud distributions, and model performance metrics.
- **Project Report**: [Detailed Report](Fraud_Detection_Report.md) covering preprocessing, transformation, modeling, and evaluation.

The pipeline is tested with `pytest` and automated via a GitHub Actions CI/CD workflow.

---

## 📂 Project Structure

```
Improved-detection-of-fraud-cases/
├── data/
│   ├── raw/                    # Raw datasets (Fraud_Data.csv, creditcard.csv, IpAddress_to_Country.csv)
│   ├── processed/              # Processed datasets, EDA plots, and model results
├── src/
│   ├── data_preprocessing.py   # Data cleaning and IP-to-country mapping
│   ├── data_transformation.py  # Data scaling, encoding, and SMOTE
│   ├── model_training.py       # Model training and evaluation
├── tests/
│   ├── test_preprocessing.py   # Unit tests for preprocessing and transformation
├── notebooks/
│   ├── eda_notebook.ipynb      # EDA and model performance visualizations
├── .github/
│   ├── workflows/
│   │   ├── ci.yml             # CI/CD pipeline configuration
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
├── Fraud_Detection_Report.md   # Detailed project report
├── README.md                  # This file
```

---

## 🚀 Getting Started

Follow these detailed steps to set up and run the project on Windows using Visual Studio Code (recommended) or any code editor.

### Prerequisites

- **Python 3.11.9**: Download from [python.org](https://www.python.org/downloads/release/python-3119/). Verify with:
  ```bash
  python --version
  ```
- **Git**: Download from [git-scm.com](https://git-scm.com/downloads). Verify with:
  ```bash
  git --version
  ```
- **Visual Studio Code** (optional): Install from [code.visualstudio.com](https://code.visualstudio.com/) for a streamlined experience.
- **Datasets**: Obtain `Fraud_Data.csv`, `creditcard.csv`, and `IpAddress_to_Country.csv` (not included in the repository due to size). Contact [yaredgenanaw99@gmail.com](mailto:yaredgenanaw99@gmail.com) if you need assistance sourcing them.

### Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yankee998/Improved-detection-of-fraud-cases.git
   cd Improved-detection-of-fraud-cases
   ```
   - Clones the repository to your local machine (e.g., `C:\Users\Skyline\Improved detection of fraud cases`).

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # For PowerShell
   # OR
   .\venv\Scripts\activate.bat  # For Command Prompt
   ```
   - Creates an isolated Python environment to manage dependencies.
   - Verify activation: Prompt should show `(venv)`.

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   - Installs required packages (`pandas==2.2.2`, `scikit-learn==1.5.1`, `imbalanced-learn==0.12.3`, `pytest==8.3.2`, `matplotlib==3.9.2`, `seaborn==0.13.2`, `jupyter==1.0.0`).
   - Verify installation:
     ```bash
     pip list
     ```

4. **Set Up Datasets**:
   - Create a `data/raw/` directory if it doesn’t exist:
     ```bash
     mkdir data\raw
     ```
   - Place `Fraud_Data.csv`, `creditcard.csv`, and `IpAddress_to_Country.csv` in `data/raw/`.
   - **Note**: Ensure file names match exactly, as the scripts expect these paths.

5. **Verify Setup**:
   - Confirm Python version:
     ```bash
     python -c "import pandas; print(pandas.__version__)"
     ```
     - Expected: `2.2.2`
   - Test script execution:
     ```bash
     python src/data_preprocessing.py
     ```
     - Should create `data/processed/cleaned_fraud_data.csv`, etc., without errors.

### Troubleshooting Setup

- **Python Version Mismatch**: Ensure Python 3.11.9 is used. Update PATH if multiple versions are installed.
- **ModuleNotFoundError**: Verify `requirements.txt` installation and virtual environment activation. Run scripts from the project root or ensure `sys.path` is adjusted in scripts.
- **FileNotFoundError**: Check that datasets are in `data/raw/` with correct names.
- **Permission Issues**: Run PowerShell/Command Prompt as Administrator.

### Running the Pipeline

Execute scripts in order to process data, train models, and generate outputs:

1. **Preprocess Data**:
   ```bash
   python src/data_preprocessing.py
   ```
   - Cleans data, removes duplicates (e.g., 1081 from `creditcard.csv`), converts timestamps to numerical features (hour, day, month, weekday), maps IP addresses to countries (129146/151112 matched), and encodes categorical features.
   - Outputs: `data/processed/cleaned_fraud_data.csv`, `data/processed/cleaned_creditcard.csv`, `data/processed/IpAddress_to_Country_cleaned.csv`.

2. **Transform Data**:
   ```bash
   python src/data_transformation.py
   ```
   - Applies `StandardScaler` for feature scaling and SMOTE (`k_neighbors=3`) for class imbalance, skipping SMOTE if minority class has fewer than 5 samples.
   - Outputs: Transformed datasets used in model training.

3. **Train and Evaluate Models**:
   ```bash
   python src/model_training.py
   ```
   - Trains Logistic Regression (`C=0.1`) and Random Forest (`n_estimators=100`) models.
   - Evaluates using AUC-PR, F1-Score, and Confusion Matrix, with Random Forest achieving AUC-PR of 0.815 and F1-Score of 0.834 for `CreditCard`.
   - Outputs: `data/processed/model_results.csv`, confusion matrix plots in `data/processed/plots/`.

4. **Run EDA**:
   ```bash
   jupyter notebook notebooks/eda_notebook.ipynb
   ```
   - Visualizes transaction distributions and model performance metrics (AUC-PR, F1-Score).
   - Outputs: Plots saved in `data/processed/` (e.g., `cm_RandomForest_CreditCard.png`).

5. **Run Tests**:
   ```bash
   pytest tests/ -v
   ```
   - Verifies preprocessing, transformation, and model training with unit tests.
   - Expected: 6 tests pass.

### Outputs

- **Processed Data**: `data/processed/cleaned_fraud_data.csv`, `data/processed/cleaned_creditcard.csv`, `data/processed/IpAddress_to_Country_cleaned.csv`.
- **Model Results**: `data/processed/model_results.csv` with AUC-PR, F1-Score, and model details (e.g., Random Forest: AUC-PR 0.815, F1-Score 0.834 for `CreditCard`).
- **Plots**: Confusion matrices and EDA visualizations in `data/processed/` (e.g., `cm_RandomForest_CreditCard.png`).
- **Report**: [Fraud_Detection_Report.md](Fraud_Detection_Report.md) details preprocessing, transformation, modeling, and evaluation.

---

## 🔍 Key Features

- **Data Cleaning**:
  - Removes duplicates (e.g., 1081 from `creditcard.csv`) and handles missing values (impute for `Fraud_Data`, drop for `creditcard`).
  - Converts `signup_time` and `purchase_time` to `datetime64[ns]` and extracts numerical features (hour, day, month, weekday).

- **Geolocation**:
  - Maps IP addresses to countries (129146/151112 matched), consolidating rare countries (<1% frequency) into 'Other' (15 unique countries).

- **Data Transformation**:
  - Balances classes with SMOTE (`k_neighbors=3`), skipped if minority class < 5 samples.
  - Scales numerical features (`StandardScaler`) and encodes categorical features (`LabelEncoder`).

- **Model Building and Training**:
  - Trains Logistic Regression (`C=0.1` for regularization) and Random Forest (`n_estimators=100`).
  - Evaluates models using AUC-PR, F1-Score, and Confusion Matrix, with Random Forest outperforming Logistic Regression (AUC-PR: 0.815, F1-Score: 0.834 for `CreditCard`).

- **EDA**:
  - Visualizes transaction distributions and model performance in `notebooks/eda_notebook.ipynb`.

- **Project Report**: [Detailed Report](Fraud_Detection_Report.md) covering preprocessing, transformation, modeling, and evaluation.

---

## 🛠️ Testing and CI/CD

- **Unit Tests**: Located in `tests/test_preprocessing.py`, covering:
  - Missing value handling
  - Data cleaning (duplicates, datetime conversion)
  - Feature scaling and SMOTE application
  - Model training
- **CI/CD Pipeline**: GitHub Actions runs tests and scripts on push/pull requests to `main`.

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

- **Model Optimization**: Tune hyperparameters (e.g., `n_estimators`, `max_depth` for Random Forest) and explore ensemble models (e.g., XGBoost, LightGBM).
- **Advanced Features**: Add transaction frequency, user behavior, or network-based features.
- **Class Imbalance**: Experiment with ADASYN or undersampling for `CreditCard` dataset.
- **Deployment**: Integrate the pipeline into a production environment.
- **Contribute**: Fork the repo, add features, and submit pull requests!

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

⭐ **Star this repository** if you found it helpful! Contributions and feedback are welcome.