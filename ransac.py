"""
代码引用自https://github.com/falcondai/py-ransac.git
"""
import random

def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
    """
    description:对数据进行ransac变换
    param:
        data：要拟合的数据
        estimate:
        is_inlier:函数，判断是否是inline点
        sample_size:每次随机采样的数目
        goal_inliers:目标直线的最小点数
        max_iterations:最大迭代次数
        stop_at_goal:寻找到符合目标的直线时立即返回
    return:
        ?,cos(x), sin(x)
    """
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    # random.sample cannot deal with "data" being a numpy array
    data = list(data)
    for i in range(max_iterations):
        s = random.sample(data, int(sample_size))
        m = estimate(s)
        ic = 0
        for j in range(len(data)):
            if is_inlier(m, data[j]):
                ic += 1

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    #print('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
    return best_model, best_ic
