# 定义一些辅助函数，包括解析输入、图像预处理 

import os
import cv2
import numpy as np

def parseInput(opt):
    """
    description:解析输入的图片路径，给出路径中可用的图像数据
    param:
        source:输入的图像目录
    return:
        一个字典，key为原始图像的路径，value是一个包含了过度曝光后图像的列表
    example:
        return {"original/001.png":["test/001/right_top.png", "test/001/all_mid.png"]}
    """
    source = opt.source
    original_img_dir = os.path.join(source, "original")
    test_img_dir = os.path.join(source, "test")
    result = {}

    # 获取original目录中的图像
    all_original_imgs = os.listdir(original_img_dir)
    for img in all_original_imgs:
        if not img.endswith(("jpg", "png")):
            all_original_imgs.remove(img)

    # 获取test目录中的图像,并将其添加到result中
    all_test_dirs = os.listdir(test_img_dir)
    for img in all_original_imgs:
        img_prefix_name = img.split('.')[0]

        # 如果test目录中有对应的图像的目录
        if img_prefix_name in all_test_dirs:
            test_img_path = os.path.join(test_img_dir, img_prefix_name)
            key = os.path.join(original_img_dir, img)
            result[key] = []

            value = os.listdir(os.path.join(test_img_dir, img_prefix_name))
            for v in value:
                result[key].append(os.path.join(test_img_path, v))    

    return result 

def preProcess(img):
    """
    description:对图像进行预处理操作
    return:
        二值化以后的图像，绝缘子串为白色区域，背景为黑色
    """
    # 二值化
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.add(img_gray, -30)    # 降低原图的亮度，以消除光照干扰(这一步很重要)
    thresh, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    #thresh, img_binary = cv2.threshold(img_gray, 220, 255, cv2.THRESH_BINARY_INV)

    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    img_binary = cv2.dilate(img_binary, kernel, iterations=2)
    img_binary = cv2.erode(img_binary, kernel, iterations=2)

    # 寻找最大连通区域,将面积较小的连通区域的部分清零，保留面积最大的两块区域
    contours, hiterarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    if len(area) == 0:
        return img_binary

    max_idx = list(np.argsort(area))
    max_idx.reverse()
    for i in range(1, len(max_idx)):
        if i > 1 or area[max_idx[i]] < 0.2 * area[max_idx[0]]:
            cv2.fillPoly(img_binary, [contours[max_idx[i]]], 0)

    """max_idx = -1 if len(area) == 0 else np.argmax(area) 
    for i in range(len(contours)):
        if i != max_idx:
            cv2.fillPoly(img_binary, [contours[i]], 0)
            pass
    """

    return img_binary