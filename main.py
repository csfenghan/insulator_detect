import cv2
import argparse
import numpy as np
from timeit import default_timer as timer

from feature_analysis import findMidLine
from utils import parseInput, preProcess
from exposure_analysis import isOverExposure

def detect(img):
    img_binary = preProcess(img)
    mid_line_detected, mid_line_fitted = findMidLine(img_binary)
    confidence = isOverExposure(img, mid_line_detected, mid_line_fitted)

    if confidence < 0.5:
        print("正常，置信度：{}".format(confidence))
    else:
        print("过曝，置信度：{}".format(confidence))

    for point in mid_line_detected:
        cv2.circle(img, point, 0, (0, 255, 0))
    for point in mid_line_fitted:
        cv2.circle(img, point, 0, (0, 0, 255))
    
    return img

def main(opt):
    data = parseInput(opt)

    for key in data: 
        img = cv2.imread(key)
        img_binary = preProcess(img)
        mid_line_detected, mid_line_fitted = findMidLine(img_binary)


        for value in data[key]:
            img = cv2.imread(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="images", help="images directory")

    opt = parser.parse_args()
    main(opt)
