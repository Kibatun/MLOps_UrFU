import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from pathlib import Path


X_test = np.loadtxt(Path(Path.cwd(), 'X_test.txt'), delimiter=',')
Y_test = np.loadtxt(Path(Path.cwd(), 'Y_test.txt'), delimiter=',')

# загрузим обученную модель

model_ridge = joblib.load(Path(Path.cwd(), 'model.pkl'))

print(f'Train score = {model_ridge.score(X_test, Y_test):.2f}')

scoring = {'R2': 'r2',
           '-MSE': 'neg_mean_squared_error',
           '-MAE': 'neg_mean_absolute_error',
           'Max': 'max_error'}

scores = cross_validate(model_ridge, X_test, Y_test,
                        scoring=scoring, cv=ShuffleSplit(n_splits=5, random_state=42))

print('Результаты Кросс-валидации')
DF_cv_linreg = pd.DataFrame(scores)
print('\n')
print(DF_cv_linreg.mean()[2:])
print('\n')
