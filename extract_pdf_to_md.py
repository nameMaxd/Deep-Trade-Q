import pdfminer.high_level
import pdfminer.layout
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTTextBox, LTAnno, LTTextLine, LTFigure

def extract_pdfminer_text(pdf_path, md_path):
    md_lines = []
    for page_layout in extract_pages(pdf_path):
        for element in page_layout:
            if isinstance(element, (LTTextBox, LTTextLine)):
                text = element.get_text()
                # Простейший костыль для формул: если строка содержит '=', '$', или '\\', помечаем как формулу
                if any(x in text for x in ['$', '=', '\\']):
                    md_lines.append(f'$$\n{text.strip()}\n$$\n')
                else:
                    md_lines.append(text)
            elif isinstance(element, LTFigure):
                # Можно добавить обработку картинок/графиков
                pass
        md_lines.append('\n')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.writelines(md_lines)

if __name__ == '__main__':
    pdf_path = '2208.07165v1.pdf'
    md_path = 'README_EXTRACTED_ARTICLE.md'
    extract_pdfminer_text(pdf_path, md_path)
    print(f'Готово! Сохранил в {md_path}')
