# 🏠 King County House Price Predictor

A machine-learning web application that predicts house sale prices in King County, Washington, using a real dataset of 4,600 property sales.

## 🗂 Project Structure

```
project/
├── app.py                  # Streamlit web application
├── train_model.py          # Model training + evaluation script
├── requirements.txt        # Python dependencies
├── README.md
├── agent_instructions.md
├── .env.example
├── data/
│   └── modified_data.csv   # King County house sales dataset
└── models/
    ├── model.pkl           # Saved best-performing sklearn Pipeline
    └── metadata.json       # Feature names, metrics, city/ZIP lists
```

## 📊 Dataset

| Property | Value |
|---|---|
| Rows | 4,600 |
| Target | `price` (continuous — regression) |
| Features | bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, sqft_above, sqft_basement, yr_built, yr_renovated, city, statezip, sale year/month |

## 🤖 ML Pipeline

- **Problem type:** Regression
- **Preprocessing:** Median imputation → Standard scaling (numeric); Most-frequent imputation → One-Hot encoding (categorical)
- **Models compared:** Ridge Regression, Random Forest, Gradient Boosting
- **Selection criterion:** Highest Test R² score
- **Evaluation metrics:** R², MAE, RMSE

## 🚀 Running Locally

### 1. Clone & install dependencies

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

This will:
- Load `data/modified_data.csv`
- Compare three regression models
- Save the best pipeline to `models/model.pkl`
- Save feature metadata to `models/metadata.json`

### 3. Launch the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

## ☁️ Deploying to Streamlit Community Cloud

1. Push the entire project to a **public GitHub repository** (including `models/model.pkl`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, and set **Main file path** to `app.py`.
4. Click **Deploy**. Streamlit Cloud reads `requirements.txt` automatically.

> **Note:** If `models/model.pkl` is not committed (e.g. via `.gitignore`), add a startup command `python train_model.py` in the Streamlit Cloud advanced settings, or commit the model file directly.

## 🌐 Deploying to Render

1. Create a **Web Service** on [render.com](https://render.com).
2. Set **Build Command:** `pip install -r requirements.txt && python train_model.py`
3. Set **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## 🔑 Environment Variables

No API keys are required for this project. See `.env.example` for the template.

## 📦 Dependencies

| Package | Purpose |
|---|---|
| streamlit | Web UI |
| pandas | Data loading & manipulation |
| numpy | Numerical operations |
| scikit-learn | ML pipeline, models, evaluation |
| joblib | Model serialisation |
| openpyxl | Excel file support (if needed) |
