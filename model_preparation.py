import numpy as np
from sklearn.linear_model import Ridge
import joblib
import os


X_train = np.loadtxt('C:/Users/medov/Desktop/Study/Master/Актуальные предметы/MLops/ДЗ1/X_train.txt', delimiter=',')
# X_test = np.loadtxt('C:/Users/medov/Desktop/Study/Master/Актуальные предметы/MLops/ДЗ1/X_test.txt', delimiter=',')
Y_train = np.loadtxt('C:/Users/medov/Desktop/Study/Master/Актуальные предметы/MLops/ДЗ1/Y_train.txt', delimiter=',')
# Y_test = np.loadtxt('C:/Users/medov/Desktop/Study/Master/Актуальные предметы/MLops/ДЗ1/Y_test.txt', delimiter=',')

# обучим модель с регуляризацией Ridge

alpha = 0.8

model_ridge = Ridge(alpha=alpha, max_iter=10000)
model_ridge.fit(X_train, Y_train)

print(f'Train score = {model_ridge.score(X_train, Y_train):.2f}')

# сохраним обученную модель

model_path = 'C:/Users/medov/Desktop/Study/Master/Актуальные предметы/MLops/ДЗ1/model.pkl'

if os.path.isfile(model_path):
    os.remove(model_path)

joblib.dump(model_ridge, model_path)
