import numpy as np
import matplotlib.colors as colors
import cv2

colors_list = [
    [135, 206, 235],  # background 135, 206, 235
    [0, 125, 0],  # grass
    [189, 189, 180],  # pavement
    [255, 255, 255],  # traversable
    [76, 250, 76],  # branches
    [255, 179, 0],  # person
    [255, 0, 0],  # vehicle
    [188, 0, 255],  # robot
    [164, 82, 0],  # tree
    [195, 188, 0]  # dynamic
]


def get_contours_tree(predicted_img):
    class_name = 'Tree'
    color_bbox = (0, 0, 255)
    class_mask = np.all(predicted_img == colors_list[8], axis=-1)
    class_mask = class_mask.astype(np.uint8)
    contours, _ = cv2.findContours(class_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, class_name, color_bbox


def get_contours_branches(predicted_img):
    class_name2 = 'Branches'
    color_bbox2 = (0, 0, 0)
    class_mask2 = np.all(predicted_img == colors_list[4], axis=-1)
    class_mask2 = class_mask2.astype(np.uint8)
    contours2, _ = cv2.findContours(class_mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours2, class_name2, color_bbox2

def get_contours_vehicle(predicted_img):
    class_name3 = 'Vehicle'
    color_bbox3 = (255, 0, 0)
    class_mask3 = np.all(predicted_img == colors_list[6], axis=-1)
    class_mask3 = class_mask3.astype(np.uint8)
    contours3, _ = cv2.findContours(class_mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours3, class_name3, color_bbox3
