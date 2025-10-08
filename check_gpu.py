"""Check if GPU is available for XGBoost"""
import xgboost as xgb
import subprocess

print("XGBoost version:", xgb.__version__)
print("\n" + "="*50)

try:
    # Try to get GPU info
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ NVIDIA GPU detected:")
        print(result.stdout)
    else:
        print("✗ No NVIDIA GPU found")
except FileNotFoundError:
    print("✗ nvidia-smi not found - no GPU available")

print("\n" + "="*50)

# Test GPU with XGBoost
try:
    import pandas as pd
    import numpy as np

    # Create small test data
    X = np.random.rand(1000, 10)
    y = np.random.rand(1000)

    # Try GPU training
    model = xgb.XGBRegressor(tree_method='gpu_hist', device='cuda', n_estimators=10)
    model.fit(X, y)
    print("✓ XGBoost GPU training works!")

except Exception as e:
    print(f"✗ XGBoost GPU training failed: {e}")
    print("\nTo use GPU, change in main.py:")
    print('  tree_method="hist"')
    print('  device="cpu"')
