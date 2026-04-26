import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title='County Suicide Rate Predictor', layout='wide')

st.title('🌡️ County-Level Suicide Rate Predictor')
st.markdown("""
This app uses a **Random Forest model** (CV R² = 0.84 ± 0.01) trained on U.S. county-level 
climate and demographic data (1968–2004) to predict age-adjusted suicide rates.  
**Predictive tool only — does not imply causal relationships.**
""")

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# ── Sidebar inputs ───────────────────────────────────────────
st.sidebar.header('County Input Parameters')

tmean = st.sidebar.slider(
    'Mean Monthly Temperature (°C)', 
    min_value=-20.0, max_value=35.0, value=12.0, step=0.5
)
prec = st.sidebar.slider(
    'Monthly Precipitation (mm)', 
    min_value=0.0, max_value=500.0, value=80.0, step=5.0
)
month = st.sidebar.selectbox(
    'Month', options=list(range(1, 13)),
    format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun',
                            'Jul','Aug','Sep','Oct','Nov','Dec'][x-1]
)
year = st.sidebar.slider(
    'Year', min_value=1968, max_value=2004, value=1990
)
pop = st.sidebar.number_input(
    'County Population', min_value=1000, max_value=10000000, 
    value=50000, step=1000
)

# ── Prediction ───────────────────────────────────────────────
input_df = pd.DataFrame({
    'tmean': [tmean], 'prec': [prec],
    'month': [month], 'year':  [year], 'pop': [pop]
})

prediction = model.predict(input_df)[0]
prediction = max(0, prediction)  # rates can't be negative

CV_RMSE      = 2.1398   # updated
CV_RMSE_STD  = 0.0987   # keep the same from CVuncertainty  = CV_RMSE + CV_RMSE_STD  # conservative 1-sigma bound

lower = max(0, prediction - uncertainty)
upper = prediction + uncertainty

# ── Metrics row ──────────────────────────────────────────────
st.subheader('Predicted Age-Adjusted Suicide Rate (per 100,000)')
col1, col2, col3 = st.columns(3)
col1.metric('Point Estimate',    f'{prediction:.2f}')
col2.metric('Lower Bound (−1σ)', f'{lower:.2f}')
col3.metric('Upper Bound (+1σ)', f'{upper:.2f}')

st.caption(f'Uncertainty based on CV RMSE = {CV_RMSE:.2f} ± {CV_RMSE_STD:.2f}')

# ── Visualization: sweep temperature ─────────────────────────
st.subheader('How Predicted Rate Changes Across Temperatures')
st.markdown('All other inputs held at sidebar values.')

temp_range = np.linspace(-20, 35, 100)
sweep_df   = pd.DataFrame({
    'tmean': temp_range,
    'prec':  prec,
    'month': month,
    'year':  year,
    'pop':   pop
})
sweep_preds = np.maximum(0, model.predict(sweep_df))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(temp_range, sweep_preds, color='steelblue', linewidth=2)
ax.fill_between(temp_range,
                np.maximum(0, sweep_preds - uncertainty),
                sweep_preds + uncertainty,
                alpha=0.2, color='steelblue', label='±1σ prediction interval')
ax.axvline(tmean, color='red', linestyle='--', label=f'Current input ({tmean}°C)')
ax.set_xlabel('Mean Monthly Temperature (°C)')
ax.set_ylabel('Predicted Rate (per 100,000)')
ax.set_title('Predicted Suicide Rate vs Temperature — Predictive Only, NOT Causal')
ax.legend()
st.pyplot(fig)

# ── Model info footer ─────────────────────────────────────────
with st.expander('Model Details'):
    st.markdown(f"""
    | Metric | Linear Regression | **Random Forest** |
    |---|---|---|
    | CV R² | 0.0685 ± 0.0019 | **0.8409 ± 0.0124** |
    | CV RMSE | 5.1720 ± 0.0843 | **2.1364 ± 0.0987** |
    
    - **Dataset:** U.S. County-Level Suicide Rates with Climate Data (1968–2004)  
    - **N (modeled):** ~220,000 non-zero county-month observations  
    - **Features:** Mean temperature, precipitation, month, year, population  
    - **Limitation:** Model excludes CDC-suppressed counties (small populations)
    """
