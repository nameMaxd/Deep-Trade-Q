import re
import csv
from bs4 import BeautifulSoup

HTML_FILE = 'GOOG_2020-2025.csv'
CSV_FILE = 'GOOG_2020-2025_clean.csv'

# Чтение html-кода
with open(HTML_FILE, encoding='utf-8') as f:
    html = f.read()

soup = BeautifulSoup(html, 'html.parser')
rows = soup.find_all('tr')

# Заголовок для csv
header = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

with open(CSV_FILE, 'w', newline='', encoding='utf-8') as fout:
    writer = csv.writer(fout)
    writer.writerow(header)
    for row in rows:
        cols = row.find_all('td')
        if len(cols) == 7:
            # Парсим дату и убираем запятые из чисел, форматируем в YYYY-MM-DD
            import datetime
            date_str = cols[0].text.strip().replace('"','')
            try:
                date_obj = datetime.datetime.strptime(date_str, '%b %d, %Y')
                date = date_obj.strftime('%Y-%m-%d')
            except Exception:
                date = date_str  # fallback, если не парсится
            open_, high, low, close, adj_close = [c.text.strip().replace(',', '') for c in cols[1:6]]
            volume = cols[6].text.strip().replace(',', '')
            writer.writerow([date, open_, high, low, close, adj_close, volume])
print(f"Данные успешно сохранены в {CSV_FILE}")
