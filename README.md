# California House Value Prediction – 10-Fold Cross Validation

**Hafiza Rija Gul – FA24-BSE-044** · Machine Learning Fundamentals (AIC354) · Assignment 3

**Web app:** https://YOUR-APP-NAME.streamlit.app *(add after deploying)*

Predicts the median house value of a California district from the full
[Kaggle California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices) dataset (20,640 districts),
comparing five scikit-learn algorithms with 10-fold cross validation on identical folds, pre-processing and metrics (RMSE, MAE, R²).

| Position | Algorithm | RMSE | MAE | R² |
|---|---|---|---|---|
| 1 | Stacking Regressor (Spline + Nystroem + Random Trees Embedding) | $52,324 | $35,913 | 0.794 |
| 2 | Nystroem Kernel Regression | $54,658 | $37,825 | 0.775 |
| 3 | Random Trees Embedding + SGD | $54,720 | $38,297 | 0.775 |
| 4 | Spline Regression | $62,329 | $44,618 | 0.708 |
| 5 | ElasticNet | $68,191 | $49,099 | 0.650 |

The Stacking Regressor wins all 10 folds (Friedman test p ≈ 1e-7) and is deployed.

## Contents

| File | Description |
|---|---|
| `california_housing_kfold_10fold.ipynb` | Experiment notebook (Google Colab) |
| `housing.csv` | Dataset |
| `california_model.pkl` | Trained Stacking Regressor pipeline (scikit-learn 1.6.1) |
| `app_info.json` | Results and defaults used by the app |
| `app.py`, `requirements.txt` | Streamlit web app |

## Google Colab

1. Upload the notebook at https://colab.research.google.com.
2. **Runtime → Change runtime type → CPU**, then **Run all** (5–10 min). `times_CPU.csv` is downloaded in Step 9.4.
3. Switch to **T4 GPU**, **Run all** again, and upload `times_CPU.csv` when asked to get the CPU-vs-GPU table.
4. Fill in the *Colab execution times* table and download the notebook (**File → Download → .ipynb**).

## Deployment (Streamlit Community Cloud)

1. Push `app.py`, `california_model.pkl`, `app_info.json` and `requirements.txt` to a public GitHub repository.
2. On https://share.streamlit.io choose **Create app**, select the repository, main file `app.py`, Python 3.12 or 3.13.
3. Put the app URL at the top of the notebook and of this README.
