import glob
import os
import re
import math
from typing import Tuple, Union
import numpy as np
import cv2
import pytesseract
import sys
import logging
from MyJSONFormatter import MyJSONFormatter


formatter = MyJSONFormatter()
console_out = logging.StreamHandler()
console_out.setFormatter(formatter)
logger_out = logging.getLogger('my_json_print')
logger_out.addHandler(console_out)
logger_out.setLevel(logging.INFO)

json_handler = logging.FileHandler(filename='data_json/turn_img.json')
json_handler.setFormatter(formatter)
logger = logging.getLogger('my_json')
logger.addHandler(json_handler)
logger.setLevel(logging.INFO)

cache = sys.argv[1]
new_dir = sys.argv[2]


# dir_classific = 'dir_classific'
# dir_line = 'line'
# dir_port = 'port'
# dir_line_or_port = 'line_or_port'
dir_classific = '../../dir_classific'
dir_garbage = 'garbage'
dir_line = 'line'
dir_port = 'port'
dir_two_page_port = 'two_page_port'

if not os.path.exists(dir_classific):
    os.makedirs(os.path.join(dir_classific, dir_garbage))
    os.makedirs(os.path.join(dir_classific, dir_line))
    os.makedirs(os.path.join(dir_classific, dir_port))
    os.makedirs(os.path.join(dir_classific, dir_two_page_port))

if not os.path.exists(new_dir):
    os.mkdir(f'{new_dir}')


def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


def turn_degree():
    for file_name in sorted(glob.glob(f"{cache}/*.jpg")):
        im = cv2.imread(str(file_name))
        rotate_img = pytesseract.image_to_osd(im)
        angle_rotated_image = int(re.search('(?<=Orientation in degrees: )\d+', rotate_img).group(0))
        rotated = rotate(im, angle_rotated_image, (0, 0, 0))
        # if angle_rotated_image > 0 and angle_rotated_image != 180:
        #     shutil.move(file_name, f"{dir_classific}/{dir_line}")
        # else:
        file_name = os.path.basename(file_name)
        cv2.imwrite(f'{new_dir}/{file_name}', rotated)
        logger.info(f'{file_name}', extra={'rotate': angle_rotated_image})
        logger_out.info(f'Rotate: {angle_rotated_image}, Filename: {os.path.basename(file_name)}')


turn_degree()
