# Pacotes principais
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
import os

# --- Carregando os dados ---
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

# --- Feature Engineering ---
def feature_engineering(df):
    """Cria features de interação entre colunas numéricas"""
    df_new = df.copy()
    numeric_cols = df_new.select_dtypes(include=[np.number]).columns.tolist()

    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            df_new[f"{col1}_x_{col2}"] = df_new[col1] * df_new[col2]

    return df_new

# --- Preparação ---
X = train.drop(columns=["id","accident_risk"])
y = train["accident_risk"]
X_test = test.drop(columns=["id"])

# Aplicar feature engineering
X = feature_engineering(X)
X_test = feature_engineering(X_test)

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Garantir colunas iguais
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# Split validação
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=24)

# --- Função objetivo para Optuna ---
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 5000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "random_state": 24,
        "n_jobs": -1,
        "tree_method": "hist",  # Use "hist" for CPU
        "device": "cuda",             # Use "cpu" for CPU
        "eval_metric": "rmse",
        "early_stopping_rounds": 100
    }

    model = XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    preds = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, preds)
    return rmse

# --- Otimização ---
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print("Melhores hiperparâmetros:", study.best_params)
print("Melhor RMSE:", study.best_value)

# --- Treino final com melhores params ---
best_params = study.best_params
# Adicionar params fixos ao treino final
best_params.update({
    "random_state": 24,
    "n_jobs": -1,
    "tree_method": "hist",  # Use "hist" for CPU
    "device": "cuda",             # Use "cpu" for CPU
    "eval_metric": "rmse",
    "early_stopping_rounds": 100
})

final_model = XGBRegressor(**best_params)
# Treinar com validação para early stopping
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X, y, test_size=0.1, random_state=24
)
final_model.fit(
    X_train_final, y_train_final,
    eval_set=[(X_val_final, y_val_final)],
    verbose=False
)

print(f"Validation RMSE: {root_mean_squared_error(y_val_final, final_model.predict(X_val_final)):.6f}")

# --- Predição e submissão ---
y_pred = final_model.predict(X_test)

submission = pd.DataFrame({
    "id": test["id"],
    "accident_risk": y_pred
})

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Arquivo submission.csv criado em /kaggle/working/")