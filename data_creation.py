import pandas as pd
import os
from pathlib import Path


import_path = Path(Path.cwd(), 'initial_data_source', 'cars_moldova.csv')

df = pd.read_csv(import_path, delimiter=',')  # импортируем данные из источника

output_path = Path(Path.cwd(), 'project_data', 'raw_data.csv')

if os.path.isfile(output_path):
    os.remove(output_path)
df.to_csv(output_path, index=True)  # запишем полученные данные в рабочий файл
