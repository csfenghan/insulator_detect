# 此文件包含了与曝光相关的一些函数，包括添加曝光、判断是否过曝

import cv2
import numpy as np
import os
import math
import argparse
from timeit import default_timer as timer

def addExposure(img, center, radius, strength):
    """
    为图像添加曝光效果
    """
    t1 = timer()
    rows, cols, _ = img.shape
    centerX, centerY = center
    r = radius * radius
    img = img.astype(np.int32)

    for i in range(rows):
        for j in range(cols):
            #计算当前点到光照中心距离(平面坐标系中两点之间的距离)
            distance = math.pow((centerY - i), 2) + math.pow((centerX - j), 2)
            if (distance < r):
                inscrement = (int)(strength * (1.0 - math.sqrt(distance) / radius))
                #inscrement = (int)(strength * (1.0 - distance / r))
                img[i, j] = img[i, j] + inscrement

    img = np.where(img>255, 255, img) 
    img = img.astype(np.uint8)
    t2 = timer()
    print("cost time : {}ms".format((t2 - t1) * 1000))
    return img

def isOverExposure(img, mid_line_detected, mid_line_fitted):
    """
    description:给定一张神经网络检测到的绝缘子串的图片，返回过度曝光的置信度
    param:
        img:输入原图片
        mid_line_detected:检测到的中线点
        mid_line_fitted:计算得到的理论中线点
    return:
        返回一个在0-1区间内的置信度，表示收到过度曝光影响的概率
    """
    # 如果len(mid_line_detected) < len(mid_line_fitted)，那么中间一定存在缺失，即存在过曝;
    # 如果len(mid_line_detected) > len(mid_line_fitted)，那么说明没有检测到直线，一定有过曝
    rows, cols, _ = img.shape
    number = len(mid_line_detected)

    if len(mid_line_detected) < len(mid_line_fitted):
        return 1

    # 计算平方和
    k1 = 10000
    square_sum = 0 
    for i in range(number):
        square_sum += math.pow(mid_line_detected[i][0] - mid_line_fitted[i][0], 2)
    square_sum = k1 * square_sum / math.pow(cols, 2) / number

    # 添加缺失惩罚项
    penalty = np.power(np.e, 10 * (rows / number - 1)) - 1
    
    # 归一化
    k2 = 500    
    confidence = square_sum + penalty
    confidence -= 1     # 如果没有光照，一般其值小于1
    if confidence < 0:
        confidence *= k2

    return confidence

def main(opt):
    source, iterator,save_path = opt.source, opt.iterator, opt.save_path
    img = cv2.imread(source)

    np.random.seed(0)
    rows, cols, _ = img.shape

    # 随机添加曝光
    curr = 0
    for i in range(iterator):
        centerY = np.random.randint(0, rows) 
        centerX = np.random.randint(0, cols) 
        radius = np.random.randint(cols // 10, cols)
        strength = np.random.randint(200, 300)
        result = addExposure(img, (centerX, centerY), radius, strength)

        cv2.imshow("img", result)
        key = cv2.waitKey(0)
        if key == ord("w"):
            curr += 1
            save_name = "{:0>3d}.jpg".format(curr)
            save_name = os.path.join(save_path, save_name)
            cv2.imwrite(save_name, result)
            print("picture {} saved successed".format(save_name))
        elif key == ord("q"):
            exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="images/original/001.png", help="输入图片路径")
    parser.add_argument("--iterator", type=int, default=30, help="随机次数")
    parser.add_argument("--save_path", type=str, default="images/test/001", help="保存路径")

    opt = parser.parse_args()
    main(opt)
