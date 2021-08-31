import cv2
import numpy as np
import os
import math

def addExposure(img, center, radius, strength):
    """
    为图像添加曝光效果
    """
    rows, cols, _ = img.shape
    centerX, centerY = center
    r = radius * radius
    img = img.astype(np.int32)

    for i in range(rows):
        for j in range(cols):
            #计算当前点到光照中心距离(平面坐标系中两点之间的距离)
            distance = math.pow((centerY - j), 2) + math.pow((centerX - i), 2)
            if (distance < r):
                inscrement = (int)(strength * (1.0 - math.sqrt(distance) / radius))
                img[i, j] = img[i, j] + inscrement

    img = np.where(img>255, 255, img) 
    img = img.astype(np.uint8)
    return img

def main():
    img = cv2.imread("images/original/001.png")
    img = addExposure(img, (350, 100), 50, 300)
    cv2.imshow("img", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
