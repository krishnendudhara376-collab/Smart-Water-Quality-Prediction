# 💧 Smart Water Quality Prediction (ML Dashboard)

## Project Overview
Smart Water Quality Prediction is a Machine Learning-based web application that predicts water potability using different water quality parameters.

An end-to-end Machine Learning web application built with **Python, Scikit-learn, Flask, SQLite, and Chart.js** to predict whether water is **Potable** or **Not Potable** based on 9 chemical and physical parameters.

## 🚀 Features
- **Data Preprocessing:** Duplicate removal, median imputation, IQR outlier capping, and stratified train-test split.
- **Feature Engineering & Selection:** Domain ratios (`ph_neutral_diff`, `Solids_Conductivity_ratio`, `Chloramines_Sulfate_ratio`, `Total_Chemical_Load`) and Mutual Information feature scoring.
- **Model Benchmarking:** Compares **Random Forest, SVM, KNN, and Logistic Regression** across Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- **Interactive Web Dashboard:** Real-time water potability prediction with confidence score and visual performance comparison charts.
- **Persistent Logging:** Stores all prediction records in an **SQLite** database (`water_quality.db`).

## 🛠️ Tech Stack
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn, Pandas, NumPy, Joblib
- **Frontend:** HTML5, CSS3, JavaScript, Chart.js
- **Database:** SQLite

## ⚙️ How to Run Locally
1. Clone the repository:
   ```bash
   python -m venv venv
   venv\Scripts\activate
``
## Installation
```bash
pip install -r requirements.txt
```
## Train the ML models:
```bash
python train.py
```
## Run Application

```bash
python app.py
```

## Author
Krishnendu Dhara 
   git clone [https://github.com/krishnendudhara/Smart_Water_Quality_Prediction.git](https://github.com/KrishnenduDhara/Smart_Water_Quality_Prediction.git)
   cd Smart_Water_Quality_Prediction
