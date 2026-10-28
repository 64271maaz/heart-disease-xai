# 🫀 Explainable AI (XAI) in Healthcare — Heart Disease Prediction

A heart disease prediction model with SHAP and LIME explanations, deployed as an interactive Streamlit app.

### 🔗 [Try the Live App][(https://heart-disease-xai.streamlit.app)

## Overview
This project predicts the presence of heart disease from patient clinical data using machine learning (Random Forest), and explains **why** each prediction was made using SHAP and LIME — making the model's decisions transparent and interpretable.

> **Disclaimer:** This is an educational prototype trained on a small public dataset. It is not a medical device and must not be used for real clinical decisions.

## Project Structure
```
heart-disease-xai/
├── data/        # Dataset + processed artifacts (scaler, encoders)
├── model/       # Saved trained model
├── notebook/    # Jupyter notebooks (EDA, preprocessing, training, XAI)
├── app/         # Streamlit web application
└── requirements.txt
```

## Key Features
- Data cleaning: identified and removed 723 duplicate rows (71% of raw data) that caused data leakage
- Compared 3 models: Logistic Regression, Random Forest, XGBoost — **Random Forest selected as final model (81.97% accuracy)**
- Global and local explanations using SHAP (TreeExplainer)
- Local explanations using LIME, cross-validated against SHAP
- Interactive Streamlit app with input validation for physiologically implausible values
- Live SHAP waterfall explanation for every prediction

## Results
Evaluated on a held-out test set of 61 patients (302 unique records after removing duplicates):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 80.33% | 80.00% | 84.85% | 82.35% | 87.88% |
| **Random Forest (final)** | **81.97%** | 84.38% | 81.82% | 83.08% | 87.88% |
| XGBoost | 70.49% | 72.73% | 72.73% | 72.73% | 81.39% |

## How to Run Locally
```bash
# 1. Clone the repo
git clone https://github.com/64271maaz/heart-disease-xai.git
cd heart-disease-xai

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
cd app
streamlit run app.py
```

## Dataset
Heart Disease UCI Dataset (Kaggle)

## Tech Stack
Python, Scikit-learn, XGBoost, SHAP, LIME, Streamlit, Pandas, Matplotlib, Seaborn
