import pandas as pd

input_file = 'data/GOOG_2020-2025_multiplied.csv'
output_file = 'data/GOOG_2020-2025_multiplied_rounded.csv'

# Читаем файл

df = pd.read_csv(input_file)

# Округляем все числовые значения до 2 знаков после запятой (кроме даты и объёма)
for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
    df[col] = df[col].round(2)

# Сохраняем результат

df.to_csv(output_file, index=False)

print(f'Округление завершено. Сохранено в {output_file}')
