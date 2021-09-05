import cv2
import argparse
import numpy as np
import sys
import os
import math
from line_fitting import line_fitting
from timeit import default_timer as timer

class Detector:
    def __init__(self):
        pass
    



def main(opt):
    detector = Detector()
    data = detector.parseInput(opt)
    
    for key in data: 
        for value in data[key]:
            img = cv2.imread(value)

            img_binary = detector.preProcess(img)
            mid_line_detected, mid_line_fitted = detector.findMidLine(img_binary)
            confidence = detector.isOverExposure(img, mid_line_detected, mid_line_fitted)

            if confidence < 0.5:
                print("未过曝,置信度:{}".format(confidence))
            else:
                print("过曝,置信度:{}".format(confidence))

            for point in mid_line_detected:
                cv2.circle(img, point, 0, (0, 255, 0))
            for point in mid_line_fitted:
                cv2.circle(img, point, 0, (255, 0, 0))

            cv2.imshow("binary_img", img_binary)
            cv2.imshow("img", img)
            if cv2.waitKey(0) == ord("q"):
                exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="images", help="images directory")

    opt = parser.parse_args()
    main(opt)
