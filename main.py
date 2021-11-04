import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from utils import parseInput, preProcess
from features import findMidLine, filter
from exposure import isOverExposure

def main(opt):
    data = parseInput(opt)
    img_list = []

    for k in data:
        img_list.append(k)
        for v in data[k]:
            img_list.append(v)
    
    for img_path in img_list:
        img = cv2.imread(img_path)
        img_gray, img_binary = preProcess(img)
        mid_line_detected, mid_line_fitted = findMidLine(img_binary)

        x = []
        y = []
        for p in mid_line_detected:
            x.append(p[1]) 
            y.append(img_gray[p[1], p[0]])
        y = filter(y)

        # 显示        
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.plot(x, y)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="images", help="images directory")

    opt = parser.parse_args()
    main(opt)
