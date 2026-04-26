# ECON 3916 Final Project — Predicting U.S. County-Level Suicide Rates from Climate Data

## Project Overview
This project uses a Random Forest model to predict U.S. county-level suicide rates from climate variables. It was completed as a final project for ECON 3916 and includes an interactive Streamlit app for exploring model predictions.

---

## Repository Structure
```
├── app.py               # Streamlit web application
├── model.pkl            # Trained Random Forest model
├── requirements.txt     # Python dependencies
└── notebook.ipynb       # Analysis and model training notebook
```

---

## Data Access
The dataset is not included in this repo due to file size. Download it here:

**[SuicideData_US.csv](https://github.com/sheftneal/NCC2018/blob/master/inputs/SuicideData_US.csv)**

Place the file in the root of the repo before running the notebook or app.

---

## Environment Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## How to Run the Notebook
```bash
jupyter notebook notebook.ipynb
```
Ensure the CSV is in the root directory before executing cells.

---

## How to Run the Streamlit App Locally
```bash
streamlit run app.py
```
The app loads `model.pkl` and serves an interactive prediction interface at `http://localhost:8501`.

---

## Dependencies
| Package | Version |
|---|---|
| streamlit | 1.42.0 |
| pandas | 2.2.3 |
| numpy | 2.2.3 |
| joblib | 1.4.2 |
| matplotlib | 3.10.1 |
| scikit-learn | 1.6.1 |
