import numpy as np
from sklearn.linear_model import Ridge
import joblib
import os
from pathlib import Path


X_train = np.loadtxt(Path(Path.cwd(), 'X_train.txt'), delimiter=',')
Y_train = np.loadtxt(Path(Path.cwd(), 'Y_train.txt'), delimiter=',')

# обучим модель с регуляризацией Ridge

alpha = 0.8

model_ridge = Ridge(alpha=alpha, max_iter=10000)
model_ridge.fit(X_train, Y_train)

print(f'Train score = {model_ridge.score(X_train, Y_train):.2f}')

# сохраним обученную модель

model_path = Path(Path.cwd(), 'model.pkl')

if os.path.isfile(model_path):
    os.remove(model_path)

joblib.dump(model_ridge, model_path)
