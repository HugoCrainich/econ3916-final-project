import streamlit as st
import numpy as np
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Suicide Rate Predictor",
    page_icon="📊",
    layout="wide",
)

# ── Load model (with graceful error handling) ─────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import joblib
        return joblib.load("model.pkl"), None
    except FileNotFoundError:
        return None, "**model.pkl not found.** Place the file in the same directory as app.py and restart."
    except Exception as e:
        return None, f"**Error loading model.pkl:** {e}"

model, load_error = load_model()

# ── Constants ─────────────────────────────────────────────────────────────────
RMSE = 2.14          # cross-validated RMSE used as uncertainty proxy
TEMP_MIN, TEMP_MAX   = -20, 35
PRECIP_MIN, PRECIP_MAX = 0, 500
YEAR_MIN, YEAR_MAX   = 1968, 2004
POP_MIN, POP_MAX     = 1_000, 10_000_000
MONTH_NAMES = {
    1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
    7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec",
}

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("🎛️ Input Parameters")

temperature = st.sidebar.slider(
    "Mean Monthly Temperature (°C)",
    min_value=TEMP_MIN, max_value=TEMP_MAX, value=15, step=1,
)

precipitation = st.sidebar.slider(
    "Monthly Precipitation (mm)",
    min_value=PRECIP_MIN, max_value=PRECIP_MAX, value=80, step=5,
)

month = st.sidebar.selectbox(
    "Month",
    options=list(MONTH_NAMES.keys()),
    format_func=lambda m: f"{m} – {MONTH_NAMES[m]}",
    index=5,  # default: June
)

year = st.sidebar.slider(
    "Year",
    min_value=YEAR_MIN, max_value=YEAR_MAX, value=1986, step=1,
)

population = st.sidebar.slider(
    "County Population",
    min_value=POP_MIN, max_value=POP_MAX, value=500_000, step=10_000,
    format="%d",
)

# ── Helper: single prediction with clamped uncertainty ───────────────────────
def predict(temp, precip, mo, yr, pop):
    """Return (point, lower, upper) — lower is clamped at 0."""
    X = np.array([[temp, precip, mo, yr, pop]], dtype=float)
    point = float(model.predict(X)[0])
    lower = max(0.0, point - RMSE)
    upper = point + RMSE
    return point, lower, upper

# ── Main content ──────────────────────────────────────────────────────────────
st.title("Age-Adjusted Suicide Rate Predictor")
st.markdown(
    "Predicts the **age-adjusted suicide rate per 100,000 people** from climate and "
    "demographic inputs using a Random Forest regression model "
    f"(CV R² = 0.84 · CV RMSE = {RMSE} ± 0.10)."
)

if load_error:
    st.error(load_error)
    st.stop()

# ── Point estimate card ───────────────────────────────────────────────────────
point, lower, upper = predict(temperature, precipitation, month, year, population)

col1, col2, col3 = st.columns(3)
col1.metric(
    label="⬇️ Lower Bound (−1 RMSE)",
    value=f"{lower:.2f}",
    help="Point estimate minus one RMSE; floored at 0",
)
col2.metric(
    label="📍 Predicted Rate",
    value=f"{point:.2f}",
    help="Model point estimate (per 100,000)",
)
col3.metric(
    label="⬆️ Upper Bound (+1 RMSE)",
    value=f"{upper:.2f}",
    help="Point estimate plus one RMSE",
)

st.markdown("---")

# ── Temperature sweep chart ───────────────────────────────────────────────────
# Rebuilds completely on every sidebar change because all inputs feed into it.
st.subheader("📈 Predicted Rate Across Full Temperature Range")
st.caption(
    f"Other inputs held fixed — Precip: **{precipitation} mm** · "
    f"Month: **{MONTH_NAMES[month]}** · Year: **{year}** · "
    f"Population: **{population:,}**"
)

temp_sweep = np.arange(TEMP_MIN, TEMP_MAX + 1, 1)
sweep_results = [predict(t, precipitation, month, year, population) for t in temp_sweep]
points  = [r[0] for r in sweep_results]
lowers  = [r[1] for r in sweep_results]
uppers  = [r[2] for r in sweep_results]

sweep_df = pd.DataFrame({
    "Temperature (°C)": temp_sweep,
    "Lower Bound":       lowers,
    "Predicted Rate":    points,
    "Upper Bound":       uppers,
}).set_index("Temperature (°C)")

st.line_chart(sweep_df, height=380)

# Current temperature marker note
st.info(
    f"Your selected temperature **{temperature}°C** gives a predicted rate of "
    f"**{point:.2f}** [{lower:.2f}, {upper:.2f}] per 100,000."
)

# ── Methodology note ──────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️ Methodology & Uncertainty"):
    st.markdown(f"""
**Model:** Random Forest Regressor  
**Target:** Age-adjusted suicide rate per 100,000 population  
**Features:** Mean monthly temperature (°C), monthly precipitation (mm), month (1–12), year (1968–2004), county population  
**Performance:** CV R² = 0.84 · CV RMSE = {RMSE} ± 0.10  

**Uncertainty bands** are ±1 RMSE ({RMSE} per 100,000) around the point estimate, representing
the model's typical prediction error on held-out data. Lower bounds are clamped at 0 because
negative rates are not meaningful. These bands reflect model uncertainty only and do not
account for population-level or epistemic uncertainty in the underlying data.
""")
