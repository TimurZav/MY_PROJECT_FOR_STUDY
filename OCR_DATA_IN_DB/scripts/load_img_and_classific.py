import os
import shutil
import sys
import torch
from transformers import LayoutLMTokenizer
import glob
from PIL import Image
import pytesseract
import numpy as np
import logging
from MyJSONFormatter import MyJSONFormatter


formatter = MyJSONFormatter()

console_out = logging.StreamHandler()
console_out.setFormatter(formatter)
logger_out = logging.getLogger('my_json_print')
logger_out.addHandler(console_out)
logger_out.setLevel(logging.INFO)

json_handler = logging.FileHandler(filename='data_json/predict_img.json')
json_handler.setFormatter(formatter)
logger = logging.getLogger('my_json')
logger.addHandler(json_handler)
logger.setLevel(logging.INFO)

turned_img = sys.argv[1]
dir_classific = sys.argv[2]
dir_line = 'line'
dir_port = 'port'
dir_two_page_port = 'two_page_port'
dir_garbage = 'garbage'

if not os.path.exists(dir_classific):
    os.makedirs(os.path.join(dir_classific, dir_line))
    os.makedirs(os.path.join(dir_classific, dir_port))
    os.makedirs(os.path.join(dir_classific, dir_two_page_port))
    os.makedirs(os.path.join(dir_classific, dir_garbage))

PATH = "training_model/another_model_for_classification_documents_40_files_100_epoch.pth"
model = torch.load(PATH)

tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')


def normalize_box(box, width, height):
    return [
     int(1000 * (box[0] / width)),
     int(1000 * (box[1] / height)),
     int(1000 * (box[2] / width)),
     int(1000 * (box[3] / height)),
    ]


idx2label = {0: "two_page_port", 1: "garbage", 2: "line", 3: "port"}


for file_name in sorted(glob.glob(f"{turned_img}/*.jpg")):

    image = Image.open(file_name)
    width, height = image.size
    # apply ocr to the image
    ocr_df = pytesseract.image_to_data(image, output_type='data.frame', lang='eng+rus')
    float_cols = ocr_df.select_dtypes('float').columns
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
    ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)
    words = list(ocr_df.text)
    words = words[:50]
    coordinates = ocr_df[['left', 'top', 'width', 'height']]
    actual_boxes = []
    for idx, row in coordinates.iterrows():
        x, y, w, h = tuple(row)  # the row comes in (left, top, width, height) format
        actual_box = [x, y, x+w, y+h]  # we turn it into (left, top, left+width, top+height) to get the actual box
        actual_boxes.append(actual_box)

    # normalize the bounding boxes
    boxes = []
    for box in actual_boxes:
        boxes.append(normalize_box(box, width, height))

    token_boxes = []
    for word, box in zip(words, boxes):
        word_tokens = tokenizer.tokenize(word)
        token_boxes.extend([box] * len(word_tokens))
    # add bounding boxes of cls + sep tokens
    token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

    encoding = tokenizer(' '.join(words), return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    token_type_ids = encoding["token_type_ids"]
    bbox = torch.tensor([token_boxes])
    sequence_label = torch.tensor([1])
    outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
                    labels=sequence_label)
    loss = outputs.loss
    logits = outputs.logits
    dict_predict = dict()
    predict_list = [x.item() for x in logits[0]]
    for name, predict in zip(idx2label.values(), predict_list):
        dict_predict[name] = round(predict, 2)
    predicted_class_idx = logits.argmax(-1).item()
    predict = idx2label[predicted_class_idx]
    logger_out.info(f'Filename: {os.path.basename(file_name)}, Predict class: {predict}, Likelihood: {dict_predict}')
    logger.info(f'{os.path.basename(file_name)}',
                extra={'file_name': os.path.basename(file_name), 'predict': predict, **dict_predict, "text": words})

    if predict == 'line':
        shutil.move(file_name, f"{dir_classific}/{dir_line}")
    elif predict == 'port':
        shutil.move(file_name, f"{dir_classific}/{dir_port}")
    elif predict == 'two_page_port':
        shutil.move(file_name, f"{dir_classific}/{dir_two_page_port}")
    elif predict == 'garbage':
        shutil.move(file_name, f"{dir_classific}/{dir_garbage}")

