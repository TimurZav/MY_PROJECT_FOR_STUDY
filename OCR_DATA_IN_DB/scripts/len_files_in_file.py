import os
import sys
import logging
from PyPDF2 import PdfFileReader
from MyJSONFormatter import MyJSONFormatter

main_file = sys.argv[1]
data_json = 'data_json'
if not os.path.exists(data_json):
    os.makedirs(data_json)

formatter = MyJSONFormatter()
console_out = logging.StreamHandler()
console_out.setFormatter(formatter)
logger_out = logging.getLogger('my_json_print')
logger_out.addHandler(console_out)
logger_out.setLevel(logging.INFO)

json_handler = logging.FileHandler(filename='data_json/files_in_file.json')
json_handler.setFormatter(formatter)
logger = logging.getLogger('my_json')
logger.addHandler(json_handler)
logger.setLevel(logging.INFO)


def get_pdf_page_count(path):
  with open(path, 'rb') as fl:
    reader = PdfFileReader(fl)
    return reader.getNumPages()


logger.info(f'{os.path.basename(main_file)}', extra={f'count_files': get_pdf_page_count(f"{main_file}")})
logger_out.info(f"Количество файлов pdf в многостраничном файле: {get_pdf_page_count(main_file)}")
