# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Kaggle competition project focused on predicting accident risk using XGBoost regression with Optuna hyperparameter optimization. The codebase is a single-file Python script that handles the complete ML pipeline from data loading to submission generation.

## Development Commands

**Package Management (uv)**
- Install dependencies: `uv sync`
- Add a new package: `uv add <package-name>`
- Run Python script: `uv run python main.py`

**Running the Pipeline**
- Execute the full training and optimization pipeline: `python main.py`
- This will:
  - Load train.csv and test.csv
  - Apply feature engineering
  - Run Optuna hyperparameter optimization (100 trials with 5-fold CV)
  - Train final model with best parameters
  - Generate submission.csv in /kaggle/working/

## Architecture

**Single Script Pipeline (main.py)**

The script follows a linear ML workflow:

1. **Data Loading**: Reads train.csv and test.csv from project root
2. **Feature Engineering** (feature_engineering function):
   - Creates interaction features between numeric columns
   - Applied to both train and test sets
3. **Preprocessing**:
   - Separates features (X) from target (accident_risk)
   - One-hot encoding with drop_first=True
   - Column alignment between train and test
4. **Hyperparameter Optimization** (objective function):
   - Uses Optuna to minimize RMSE
   - 5-fold cross-validation for robust evaluation
   - Searches 11 XGBoost hyperparameters
5. **Final Training**:
   - 90/10 train/validation split for early stopping
   - Uses best parameters from Optuna
6. **Prediction**: Generates submission.csv with id and accident_risk columns

**GPU Configuration**
- Default: `tree_method="gpu_hist"` and `device="cuda"` in main.py:64-65
- For CPU-only environments: Change to `tree_method="hist"` and `device="cpu"`

**Key Variables**
- Target column: "accident_risk"
- ID column: "id"
- All other columns treated as features after engineering

## Dependencies

Core stack (from pyproject.toml):
- Python >=3.12
- pandas, numpy: Data manipulation
- scikit-learn: Preprocessing, metrics, CV
- xgboost: Gradient boosting model
- optuna: Hyperparameter optimization
