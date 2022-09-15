import json
import re
import sys
from collections import defaultdict
import pandas as pd

filepath = sys.argv[1]
new_filepath = sys.argv[2]
# filepath = '/home/timur/WebstormProjects/MY_PROJECT_FOR_STUDY/OCR_DATA_IN_DB/dir_classific/port/7308 AS ROSALIA от ' \
#            '01.06.2021.pdf_8.jpg.json'
# new_filepath = '/home/timur/WebstormProjects/MY_PROJECT_FOR_STUDY/OCR_DATA_IN_DB/data_json_for_nifi/7308 AS ROSALIA ' \
#                'от 01.06.2021.pdf_8.jpg.json'

with open(filepath, "r") as f:
    data = json.load(f)
    table_for_csv = data["result"][0]["prediction"]


llist_name_for_one_table = ["ship_name", "expeditor", "place_port",
                  "ownership_of_the_vessel", "shipper", "city",  "imo_ship", "consignee",
                  "unloading_port"]

llist_name_for_two_table = ["code_tnved", "goods_name"]


dict_table_for_one_table = dict()
new_dict_table_for_one_table = dict()
arr_value_for_one_table = list()

dict_table_for_two_table = defaultdict(list)
new_dict_table_for_two_table = defaultdict(list)
arr_value_for_two_table = list()

label_dict = defaultdict(list)


def parse_table_from_nanonets():
    for len_table in range(len(table_for_csv)):
        if 'лар' not in table_for_csv[len_table]["cells"][0]['text']:
            print(table_for_csv[len_table]["cells"][0]['text'])
            for j in range(len(table_for_csv[len_table]["cells"])):
                if j < 18:
                    table = table_for_csv[len_table]["cells"][j]["row"], table_for_csv[len_table]["cells"][j]["col"]
                    (row, col) = table
                    dict_table_for_one_table[table] = table_for_csv[len_table]["cells"][j]["text"]
                    new_dict_table_for_one_table[table] = table_for_csv[len_table]["cells"][j]["text"]
                    if col % 2 == 0 and row < 4:
                        data_from_json = re.sub("[|]", "", table_for_csv[len_table]["cells"][j].get("text"))
                        data_from_json = re.sub("[\n]", " ", data_from_json).strip()
                        data_from_json = int(data_from_json) if data_from_json.isdigit() else data_from_json
                        arr_value_for_one_table.append(data_from_json)
                        dict_table_for_one_table[table] = data_from_json
                        new_dict_table_for_one_table[table] = data_from_json
                    else:
                        del new_dict_table_for_one_table[table]
            for (new_key, tuple_key, name_from_table) in zip(llist_name_for_one_table, new_dict_table_for_one_table,
                                                             arr_value_for_one_table):
                new_dict_table_for_one_table[new_key] = new_dict_table_for_one_table.pop(tuple_key)
                new_dict_table_for_one_table[new_key] = name_from_table


parse_table_from_nanonets()
print(new_dict_table_for_one_table)


with open(new_filepath, 'w', encoding='utf-8') as f:
    json.dump([new_dict_table_for_one_table], f, ensure_ascii=False, indent=4)

df = pd.read_json(new_filepath)
df.to_excel(f'{new_filepath}.xlsx')

