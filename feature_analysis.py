import cv2
from line_fitting import line_fitting

def randomSampleConsensus(points):
    """
    description:拟合points中的直线点，寻找出最接近的直线参数
    param:
        points:离散点的坐标，格式为[[x1, y1], [x2, y2],...]
    return:
        直线的参数，格式为k,b
    """
    sample_size = 30
    if len(points) < sample_size:
        return 1e9, 0
    m = line_fitting(points, threshold=0.01, sample_size=sample_size, 
                    goal_inliers=100, max_iterations=30, stop_at_goal=True, random_seed=0)
    k = - m[0] / m[1]
    b = - m[2] / m[1]

    return k, b

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
    
    # 霍夫变换检测直线,寻找在拟合的直线上的点
    """fit_points = []
    lines = cv2.HoughLinesP(temp_img, 1, np.pi / 180, 80, minLineLength=80, maxLineGap=10)
    for point in mid_line:
        if type(lines) == type(None):
            print("没有检测到直线")
            break
        for line in lines:
            if (point[1] >= line[0][1]) and (point[1] <= line[0][3]):
                fit_points.append(point)
                break

    if len(fit_points) == 0:
        return mid_line, [], [] 

    # 拟合直线
    param = cv2.fitLine(np.array(fit_points), cv2.DIST_L2, 0, 0.01, 0.01)
    k = (param[1] / param[0])[0]
    b = (param[3] - k * param[2])[0]"""

    # 使用拟合后的直线作为中线的理论值
    mid_line_fitted = []  
    for row in range(mid_line_range[0], mid_line_range[1] + 1):
        mid_line_fitted.append([int((row - b) // k), row]) 
    
    return mid_line, mid_line_fitted
