import cv2
import argparse
import numpy as np
import sys
import os
from matplotlib import pyplot as plt
import math

class Detector:
    def __init__(self):
        pass

    def plotHistGraph(self, img, color):
        """
        description:绘制直方图
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        histGrapth = np.zeros([256, 256, 3], np.uint8)
        m = max(hist)
        hist = hist * 220 / m
        for h in range(256):
            n = int(hist[h])
            cv2.line(histGrapth, (h, 255), (h, 255 - n), color)
            return histGrapth
 
    def parseInput(self, opt):
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

    def preProcess(self, img):
        """
        description:对图像进行预处理操作
        return:
            二值化以后的图像，绝缘子串为白色区域，背景为黑色
        """
        # 二值化
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thresh, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_OTSU and cv2.THRESH_BINARY_INV)

        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        img_binary = cv2.dilate(img_binary, kernel, iterations=2)
        img_binary = cv2.erode(img_binary, kernel, iterations=2)

        # 寻找最大连通区域,将非最大联通区域的部分清零
        contours, hiterarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        area = []
        for i in range(len(contours)):
            area.append(cv2.contourArea(contours[i]))

        max_idx = -1 if len(area) == 0 else np.argmax(area) 
        for i in range(len(contours)):
            if i != max_idx:
                cv2.fillPoly(img_binary, [contours[i]], 0)

        return img_binary

    def findMidLine(self, img):
        """
        description:寻找并计算绝缘子串的中线
        return：
            绝缘子串的实际中线与理论中线坐标
        """
        left_edge = []
        right_edge = []
        mid_line = []
        mid_line_range = []

        # 寻找绝缘子串的左右轮廓
        rows, cols = img.shape
        temp_img = np.zeros(img.shape, np.uint8)
        for row in range(rows):
            for col1 in range(cols):
                if img[row, col1] == 255:
                    left_edge.append([col1, row])
                    break
            for col2 in range(cols - 1, -1, -1):
                if img[row, col2] == 255:
                    right_edge.append([col2, row])
                    break
            if col1 < col2:
                x = (col1 + col2) // 2
                mid_line.append([x, row])    
                temp_img[row, x - 1 : x + 1] = 255
                if len(mid_line_range) < 2:
                    mid_line_range.append(row)
                if len(mid_line_range) == 2:
                    mid_line_range[1] = row

        # 霍夫变换检测直线,寻找在拟合的直线上的点
        fit_points = []
        lines = cv2.HoughLinesP(temp_img, 1, np.pi / 180, 80, minLineLength=80, maxLineGap=10)
        for point in mid_line:
            for line in lines:
                if (point[1] >= line[0][1]) and (point[1] <= line[0][3]):
                    fit_points.append(point)
                    break

        # 拟合直线
        param = cv2.fitLine(np.array(fit_points), cv2.DIST_L2, 0, 0.01, 0.01)
        k = (param[1] / param[0])[0]
        b = (param[3] - k * param[2])[0]

        # 使用拟合后的直线作为中线的理论值
        mid_line_fitted = []  
        for row in range(mid_line_range[0], mid_line_range[1] + 1):
            mid_line_fitted.append([int((row - b) // k), row]) 
        
        return mid_line, mid_line_fitted, lines

       
    def isOverExposure(self, img, mid_line_detected, mid_line_fitted):
        """
        description:给定一张神经网络检测到的绝缘子串的图片，返回过度曝光的置信度
        param:
            img:输入原图片
            mid_line_detected:检测到的中线点
            mid_line_fitted:计算得到的理论中线点
        return:
            返回一个在0-1区间内的置信度，表示收到过度曝光影响的概率
        """

        # len(mid_line_detected) <= len(mid_line_fitted)一定满足，不存在大于的情况
        # 如果len(mid_line_detected) < len(mid_line_fitted)，那么中间一定存在缺失，即存在过曝;
        if len(mid_line_detected) < len(mid_line_fitted):
            return 1
        
        sum = 0
        for i in range(len(mid_line_detected)):
            sum += math.pow(mid_line_detected[i][0] - mid_line_fitted[i][0], 2)    

        print(sum)
        return 0

def main(opt):
    detector = Detector()
    data = detector.parseInput(opt)
    
    for key in data: 
        img = cv2.imread(key)
        for value in data[key]:
            img = cv2.imread(value)
            img_binary = detector.preProcess(img)

            mid_line_detected, mid_line_fitted, lines = detector.findMidLine(img_binary)
            conn = detector.isOverExposure(img, mid_line_detected, mid_line_fitted)

            for point in mid_line_detected:
                cv2.circle(img, point, 0, (0, 255, 0))
            for point in mid_line_fitted:
                cv2.circle(img, point, 0, (255, 0, 0))
            #for line in lines:
            #    x1, y1, x2, y2 = line[0]
            #    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255))

            cv2.imshow("binary", img_binary)
            cv2.imshow("img", img)
            if cv2.waitKey(0) == ord("q"):
                exit(0)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="images", help="images directory")

    opt = parser.parse_args()
    main(opt)
