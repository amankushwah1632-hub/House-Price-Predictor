# Agent Instructions

## Project: King County House Price Predictor

### Purpose
This project builds a production-ready machine learning web app for predicting residential house prices in King County, WA.

### Dataset
- **File:** `data/modified_data.csv`
- **Rows:** 4,600 property sale records
- **Target column:** `price` (float — regression problem)
- **Dropped columns (leakage/identifier):** `price_per_sqft`, `street`
- **Date handling:** `date` is parsed to `sale_year` and `sale_month` numeric features

### ML Pipeline
- Numeric features are imputed (median) and scaled (StandardScaler)
- Categorical features are imputed (most_frequent) and one-hot encoded
- Three models are compared: Ridge, Random Forest, Gradient Boosting
- Best model (by Test R²) is saved to `models/model.pkl`
- Feature ranges, city/ZIP lists, and evaluation metrics are saved to `models/metadata.json`

### App
`app.py` reads `models/model.pkl` and `models/metadata.json` at startup.
All input widgets are populated dynamically from metadata — no hard-coded prediction logic.

### How to retrain
```bash
python train_model.py
```

### How to run the app
```bash
streamlit run app.py
```

### Paths
All paths use `os.path.join(os.path.dirname(os.path.abspath(__file__)), ...)` — fully relative, GitHub/cloud compatible.

### No API keys required
This project uses only local data and scikit-learn models. No external API calls are made.
