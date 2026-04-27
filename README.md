# ECON 3916 Final Project — Predicting U.S. County-Level Suicide Rates from Climate Data

## Project Overview
This project trains a Random Forest regression model to predict age-adjusted U.S. county-level suicide rates (per 100,000) from monthly climate and demographic features. Data spans 1968–2004 across 851,088 county-month observations. The final model achieves CV R² = 0.84 and CV RMSE = 2.14. An interactive Streamlit app allows users to explore predictions across climate inputs.

**Features used:** mean monthly temperature (`tmean`), monthly precipitation (`prec`), month, year, county population (`pop`)  
**Target:** `rate_adj` — age-adjusted suicide rate per 100,000  
**Note:** ~74% of `rate_adj` values are structural zeros (CDC privacy suppression); the model is trained on non-zero rows only.

---

## Repository Structure
```
├── app.py                          # Streamlit web application
├── model.pkl                       # Trained Random Forest model (compressed via joblib)
├── requirements.txt                # Python dependencies
└── 3916_Final_Project__1_.ipynb   # Full analysis and model training notebook
```

---

## Data Access
The dataset is not included in this repo due to file size. Download it here:

**[SuicideData_US.csv](https://github.com/sheftneal/NCC2018/blob/master/inputs/SuicideData_US.csv)**  
*(Burke et al. 2018, NCC replication data — 851,088 rows, 13 columns)*

Place the file in the root of the repo before running the notebook.

> **Note:** The notebook was developed in Google Colab and uses `from google.colab import files` in cell 2.1 to upload the CSV. If running locally in Jupyter, replace that cell with `df = pd.read_csv('SuicideData_US.csv')`.

---

## Environment Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install seaborn              # Required for notebook EDA plots (not needed for app.py)
```

---

## How to Run the Notebook
```bash
jupyter notebook 3916_Final_Project__1_.ipynb
```
Ensure `SuicideData_US.csv` is in the root directory. Run cells sequentially — Parts 0–5 must complete before the model is exported to `model.pkl` in Part 5.

---

## How to Run the Streamlit App Locally
```bash
streamlit run app.py
```
Requires `model.pkl` in the same directory. The app serves an interactive prediction interface at `http://localhost:8501` with sidebar sliders for temperature, precipitation, month, year, and population, and a live temperature-sweep chart with uncertainty bounds (±1 RMSE).

---

## Dependencies

| Package | Version | Used in |
|---|---|---|
| streamlit | 1.42.0 | app.py |
| pandas | 2.2.3 | app.py, notebook |
| numpy | 2.2.3 | app.py, notebook |
| joblib | 1.4.2 | app.py, notebook |
| matplotlib | 3.10.1 | app.py, notebook |
| scikit-learn | 1.6.1 | app.py, notebook |
| seaborn | *(latest)* | notebook only |
