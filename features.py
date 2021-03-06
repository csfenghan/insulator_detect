# 此文件包含了对绝缘子串进行特征处理的一些算法，包括中线提取、低通滤波等

import cv2
from ransac import randomSampleConsensus

def findMidLine(img):
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
    # !!!这一段算法需要优化，平均耗时200-300ms，太长了!!!
    rows, cols = img.shape
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
            if len(mid_line_range) < 2:
                mid_line_range.append(row)
            if len(mid_line_range) == 2:
                mid_line_range[1] = row

    # 使用RANSAC拟合直线
    k, b = randomSampleConsensus(mid_line)
    
    # 使用拟合后的直线作为中线的理论值
    mid_line_fitted = []  
    for row in range(mid_line_range[0], mid_line_range[1] + 1):
        mid_line_fitted.append([int((row - b) // k), row]) 
    
    return mid_line, mid_line_fitted

def filter(X):
    # 对灰度曲线进行滤波处理
    # X需要是列表，返回类型与X相同
    ratio = 0.2
    prev_value = X[0]
    for i in range(1, len(X)):
        X[i] = ratio * X[i] + (1 - ratio) * prev_value 
        prev_value = X[i]
    return X

