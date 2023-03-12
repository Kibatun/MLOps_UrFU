import pandas as pd
import os


import_path = str(input('import path'))#C:/Users/medov/Desktop/Study/Master/Актуальные предметы/MLops/ДЗ1
import_file_name = str(input('import file name'))#cars_moldova.csv

import_data = import_path + '/' + import_file_name
print(import_data)

df = pd.read_csv(import_data, delimiter=',')  # импортируем данные из источника

output_path = 'C:/Users/medov/Desktop/Study/Master/Актуальные предметы/MLops/ДЗ1/raw_data.csv'

if os.path.isfile(output_path):
    os.remove(output_path)
df.to_csv(output_path, index=True)  # запишем полученные данные в рабочий файл
