import math
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
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

    return sigmoid(confidence)