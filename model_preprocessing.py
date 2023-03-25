import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from pathlib import Path


import_path = Path(Path.cwd(), 'project_data', 'raw_data.csv')
df = pd.read_csv(import_path, delimiter=',')  # импорт

df = df.drop_duplicates()  # удалим дубликаты
df = df.dropna()  # удалим строки с пропущенными значениями

# создадим списки категориальных и числовых данных
cat_columns = []
num_columns = []

for column_name in df.columns:
    if df[column_name].dtypes == object:
        cat_columns += [column_name]
    else:
        num_columns += [column_name]

# удалим экстремальные значения

question_dist = df[(df.Year < 2021) & (df.Distance < 1100)]
df = df.drop(question_dist.index)

question_dist = df[(df.Distance > 1e6)]
df = df.drop(question_dist.index)

question_engine = df[df["Engine_capacity(cm3)"] < 200]
df = df.drop(question_engine.index)

question_engine = df[df["Engine_capacity(cm3)"] > 5000]
df = df.drop(question_engine.index)

question_price = df[(df["Price(euro)"] < 101)]
df = df.drop(question_price.index)

question_price = df[df["Price(euro)"] > 1e5]
df = df.drop(question_price.index)

question_year = df[df.Year < 1971]
df = df.drop(question_year.index)

df = df.reset_index(drop=True)

# проведем нормализацию данных

DF_norm = df.copy()
Xmin = df[num_columns].min()
Xmax = df[num_columns].max()

DF_norm[num_columns] = (df[num_columns] - Xmin)/(Xmax - Xmin)

# сгруппируем редкие значения в общую категорию

counts = DF_norm.Make.value_counts()
rare = counts[(counts.values < 25)]
DF_norm['Make'] = DF_norm['Make'].replace(rare.index.values, 'Rare')

counts = DF_norm.Model.value_counts()
rare = counts[(counts.values < 50)]
DF_norm['Model'] = DF_norm['Model'].replace(rare.index.values, 'Rare')

# преобразуем категориальные данные к числовому виду

DF_norm['Transmission'] = DF_norm['Transmission'].map({'Automatic': 1, 'Manual': 0})

# выполним one-hot encoding

df_one = DF_norm.copy()
df_one = pd.get_dummies(df_one)

# разделим данные на train и test, и сохраним

X = df_one.drop(columns=['Price(euro)']).values
Y = df_one['Price(euro)'].values

X_train, X_test, Y_train, Y_test = \
            train_test_split(X, Y.ravel(), test_size=0.3, random_state=42)

X_train_output_path = Path(Path.cwd(), 'project_data', 'X_train.txt')

if os.path.isfile(X_train_output_path):
    os.remove(X_train_output_path)

np.savetxt(X_train_output_path, X_train, delimiter=',', newline='\n')

X_test_output_path = Path(Path.cwd(), 'project_data', 'X_test.txt')

if os.path.isfile(X_test_output_path):
    os.remove(X_test_output_path)

np.savetxt(X_test_output_path, X_test, delimiter=',', newline='\n')

Y_train_output_path = Path(Path.cwd(), 'project_data', 'Y_train.txt')

if os.path.isfile(Y_train_output_path):
    os.remove(Y_train_output_path)

np.savetxt(Y_train_output_path, Y_train, delimiter=',', newline='\n')

Y_test_output_path = Path(Path.cwd(), 'project_data', 'Y_test.txt')

if os.path.isfile(Y_test_output_path):
    os.remove(Y_test_output_path)

np.savetxt(Y_test_output_path, Y_test, delimiter=',', newline='\n')
