# Pacotes principais
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# --- Carregando os dados ---
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# --- Feature Engineering ---
def feature_engineering(df):
    """Cria features adicionais que podem melhorar o modelo"""
    df = df.copy()
    
    # Exemplo de features derivadas (adapte conforme suas colunas)
    # Se tiver colunas numéricas, pode criar interações
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Interações entre features numéricas (exemplo)
    if len(numeric_cols) >= 2:
        for i, col1 in enumerate(numeric_cols[:3]):  # Limitar para não explodir
            for col2 in numeric_cols[i+1:4]:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
    
    return df

# --- Preparação ---
X = train.drop(columns=["id", "accident_risk"])
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

# --- Função objetivo com Cross-Validation ---
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "gpu_hist",
        "device": "cuda"  # Use "cpu" se não tiver GPU
    }
    
    # Cross-validation para validação mais robusta
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        model = XGBRegressor(**params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )
        
        preds = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
        rmse_scores.append(rmse)
    
    return np.mean(rmse_scores)

# --- Otimização ---
print("Iniciando otimização de hiperparâmetros...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("\n" + "="*50)
print("Melhores hiperparâmetros:", study.best_params)
print("Melhor RMSE (CV):", study.best_value)
print("="*50 + "\n")

# --- Treino final com melhores params e early stopping ---
best_params = study.best_params.copy()

# Split para early stopping no treino final
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X, y, test_size=0.1, random_state=42
)

final_model = XGBRegressor(**best_params)
final_model.fit(
    X_train_final, y_train_final,
    eval_set=[(X_val_final, y_val_final)],
    verbose=True
)

# Validação final
y_val_pred = final_model.predict(X_val_final)
final_rmse = np.sqrt(mean_squared_error(y_val_final, y_val_pred))
print(f"\nRMSE no conjunto de validação final: {final_rmse:.6f}")

# --- Predição e submissão ---
y_pred = final_model.predict(X_test)

submission = pd.DataFrame({
    "id": test["id"],
    "accident_risk": y_pred
})

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("\nArquivo submission.csv criado em /kaggle/working/")
print(f"Predições - Min: {y_pred.min():.4f}, Max: {y_pred.max():.4f}, Mean: {y_pred.mean():.4f}")