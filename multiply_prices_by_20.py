import pandas as pd
import sys

# Имя исходного и выходного файла
input_file = 'data/GOOG_2020-2025.csv'
output_file = 'data/GOOG_2020-2025_multiplied.csv'

# Чтение данных

df = pd.read_csv(input_file)

# Умножаем нужные столбцы на 20 (Open, High, Low, Close, Adj Close)
for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
    df[col] = df[col] * 20

# Сохраняем результат

df.to_csv(output_file, index=False)

print(f'Цены успешно умножены на 20 и сохранены в {output_file}')
