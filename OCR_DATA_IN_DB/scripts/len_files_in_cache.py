import os
import sys
import logging
from MyJSONFormatter import MyJSONFormatter

formatter = MyJSONFormatter()
console_out = logging.StreamHandler()
console_out.setFormatter(formatter)
logger_out = logging.getLogger('my_json_print')
logger_out.addHandler(console_out)
logger_out.setLevel(logging.INFO)

json_handler = logging.FileHandler(filename='data_json/files_in_cache.json')
json_handler.setFormatter(formatter)
logger = logging.getLogger('my_json')
logger.addHandler(json_handler)
logger.setLevel(logging.INFO)

file = sys.argv[1]


path, dirs, files = next(os.walk(f"{file}"))
file_count = len(files)
logger.info(f'{os.path.basename(file)}', extra={'count_files': file_count})
logger_out.info(f"Количество разбитых файлов pdf в кэше: {file_count}")

