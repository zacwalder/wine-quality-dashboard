
Wine Quality Prediction Dashboard

Interactive Streamlit dashboard that predicts Portuguese Vinho Verde wine quality from physicochemical measurements using XGBoost + SHAP explainability.

## Academic Context
Final project for **Machine Learning and Content Analytics** — MSc Business Analytics @ AUEB.

## Live App
[Try the live dashboard](https://your-username-wine-quality-dashboard.streamlit.app)

## Features
- **Real-time predictor** with 11 chemistry sliders
- **Interactive data explorer** (6,497 wines analyzed)
- **Model comparison** across 5 algorithms
- **SHAP explanations** for every prediction

## Model Performance
| Model | RMSE | R² |
|-------|------|-----|
| Baseline | 0.874 | 0.000 |
| Linear Regression | 0.728 | 0.306 |
| Random Forest | 0.615 | 0.505 |
| **XGBoost** | **0.610** | **0.513** |

## Tech Stack
Python 3 • Streamlit • XGBoost • scikit-learn • SHAP • Plotly

## Dataset
Cortez, P. et al. (2009). *Modeling wine preferences by data mining from physicochemical properties.* UCI Machine Learning Repository.
