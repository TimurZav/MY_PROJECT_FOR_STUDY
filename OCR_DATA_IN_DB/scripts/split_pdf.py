import sys
from pdf2image import convert_from_path

file = sys.argv[1]
new_file = sys.argv[2]
pages = convert_from_path(file, 500)


for i, page in enumerate(pages):
    page.save(f'{new_file}_{i}.jpg', 'JPEG')
