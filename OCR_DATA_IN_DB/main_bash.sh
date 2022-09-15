#!/bin/bash
activate () {
    . venv/bin/activate
}
activate

pdf_path=$(python3 scripts/read_file_gui.py 2>&1)
path_project=$(pwd "main_bash.sh")

rm -r "${path_project}/dir_classific"
rm -rf "${path_project}/data_json_for_nifi"/*

mkdir "$path_project/cache" # Создаем папку с кэшем
for pdf in $pdf_path

do

    b=$(basename "$pdf")

    python3 scripts/len_files_in_file.py "$pdf" # Заносим в files_in_file.json количество страниц в # многостраничном
    # файле
    python3 scripts/split_pdf.py "$pdf" "$path_project/cache/$b"
#    pdfimages -all "$pdf" "$path_project/cache/$b" # Разделяем на одностаничные файлы из многостраничного файла
    python3 scripts/len_files_in_cache.py "$path_project/cache" # Подсчитываем количество разделенных файлов


    python3 scripts/turn_img.py "$path_project/cache" "$path_project/turned_img" # Переворачиваем файлы, если нужно
    python3 scripts/turn_small_degree.py "$path_project/turned_img" "$path_project/turned_img" # Выравниваем файлы до 0
    # градусов, если нужно

    python3 scripts/load_img_and_classific.py "$path_project/turned_img" "$path_project/dir_classific" #0.9987 0.0002 #
    # Предсказываем к
    # какому типу относится файл (line или port)

    rm -r $path_project/cache/*.jpg # Очищаем кэш

done

rm -r "${path_project}/cache"
rm -r "${path_project}/turned_img"


#mkdir "$path_project/data_json_for_nifi"
array=()
for file_jpg in "$path_project"/dir_classific/port/*.jpg
do
  python3 scripts/send_orc_data_in_nanonets.py "$file_jpg"
  python3 scripts/parse_json_for_nifi.py "$file_jpg.json" "$path_project/data_json_for_nifi/$(basename "$file_jpg.json")"
  array+=("$path_project/data_json_for_nifi/$(basename "$file_jpg.json.xlsx")")
done


python3 scripts/conver_json_in_excel.py "${array[@]}" "$path_project/data_json_for_nifi/$(basename "$file_jpg.json.xlsx")"