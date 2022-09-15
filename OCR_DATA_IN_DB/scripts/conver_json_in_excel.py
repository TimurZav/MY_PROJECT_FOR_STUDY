import os
import sys
import pandas as pd

excl_list_path = sys.argv[1:-1]
full_xlsx = sys.argv[-1:]
excl_list = []

for file in excl_list_path:
    excl_list.append(pd.read_excel(file))

excl_merged = pd.DataFrame()

for excl_file in excl_list:
    excl_merged = excl_merged.append(excl_file, ignore_index=True)

excl_merged.to_excel(f"full_{os.path.basename(full_xlsx[0])}", index=False)

