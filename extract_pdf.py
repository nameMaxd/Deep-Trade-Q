import os
from PyPDF2 import PdfReader

# Скрипт для извлечения текста из PDF-arXiv
pdf_path = os.path.join(os.path.dirname(__file__), '2208.07165v1.pdf')
reader = PdfReader(pdf_path)
text = []
for page in reader.pages:
    txt = page.extract_text()
    if txt:
        text.append(txt)

with open(os.path.join(os.path.dirname(__file__), 'arxiv.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(text))
print('Extraction complete: arxiv.txt generated')
