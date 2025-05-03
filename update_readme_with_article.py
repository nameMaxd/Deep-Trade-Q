import shutil
import os

ARTICLE_MD = 'README_EXTRACTED_ARTICLE.md'
PROJECT_README = 'README.md'

# Сохраним старый README.md для отката
if os.path.exists(PROJECT_README):
    shutil.copy(PROJECT_README, PROJECT_README + '.bak')

# Добавим содержимое статьи в конец README.md
with open(ARTICLE_MD, 'r', encoding='utf-8') as src, open(PROJECT_README, 'a', encoding='utf-8') as dst:
    dst.write('\n\n---\n')
    dst.write('# [Автоматически извлечённая статья](2208.07165v1.pdf)\n')
    dst.write(src.read())

print(f'README.md обновлён, статья добавлена в конец!')
