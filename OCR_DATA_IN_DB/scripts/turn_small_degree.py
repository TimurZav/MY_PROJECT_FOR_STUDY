import glob
import os
import statistics
import sys
from PIL import Image
import cv2
import math
from scipy import ndimage
import logging
from MyJSONFormatter import MyJSONFormatter

formatter = MyJSONFormatter()
console_out = logging.StreamHandler()
console_out.setFormatter(formatter)
logger_out = logging.getLogger('my_json_print')
logger_out.addHandler(console_out)
logger_out.setLevel(logging.INFO)

json_handler = logging.FileHandler(filename='data_json/turn_small_img.json')
json_handler.setFormatter(formatter)
logger = logging.getLogger('my_json')
logger.addHandler(json_handler)
logger.setLevel(logging.INFO)

turned_img = sys.argv[1]
new_dir = sys.argv[2]


def turn_small_degree():
    for file_full_name in sorted(glob.glob(f"{turned_img}/*.jpg")):
        foo = Image.open(file_full_name)
        (width, height) = foo.size
        foo = foo.resize((width // 4, height // 4), Image.ANTIALIAS)
        foo.save("resized_img.jpg", optimize=True, quality=95)

        img_for_define_angle = cv2.imread("resized_img.jpg")
        img_to_save = cv2.imread(file_full_name)
        img_gray = cv2.cvtColor(img_for_define_angle, cv2.COLOR_BGR2GRAY)
        img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
        lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
        angles = []
        for [[x1, y1, x2, y2]] in lines:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)
        median_angle = statistics.median(angles)
        logger.info(
            f'{os.path.basename(file_full_name)}',
            extra={'rotate': median_angle},
        )
        logger_out.info(f'Angle is: {median_angle:.04f}, Filename: {os.path.basename(file_full_name)}')
        if (15 > median_angle > 1) or (-15 < median_angle < -1):
            img_rotated = ndimage.rotate(img_to_save, median_angle)
            file_name = os.path.basename(file_full_name)
            cv2.imwrite(f"{new_dir}/{file_name}", img_rotated)

    os.remove("resized_img.jpg")


turn_small_degree()
