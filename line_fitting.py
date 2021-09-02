"""
代码引用自https://github.com/falcondai/py-ransac.git
"""
import numpy as np
from ransac import *

def augment(xys):
    axy = np.ones((len(xys), 3))
    axy[:, :2] = xys
    return axy

def estimate(xys):
    axy = augment(xys[:2])
    return np.linalg.svd(axy)[-1][-1, :]

def is_inlier(coeffs, xy, threshold):
    """
    判断是否是inline点
    """
    return np.abs(coeffs.dot(augment([xy]).T)) < threshold

def line_fitting(data, threshold, sample_size, goal_inliers, 
                max_iterations,stop_at_goal=False, random_seed=0):
    """
    拟合直线
    """
    m, b = run_ransac(data, estimate, lambda x, y: is_inlier(x, y, threshold), sample_size, 
            goal_inliers, max_iterations, stop_at_goal=stop_at_goal, random_seed=random_seed) 
    return m

if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt

    n = 100
    max_iterations = 100
    goal_inliers = n * 0.3

    # test data
    xys = np.random.random((n, 2)) * 10
    xys[:50, 1:] = xys[:50, :1]

    plt.scatter(xys.T[0], xys.T[1])

    # RANSAC
    m, b = run_ransac(xys, estimate, lambda x, y: is_inlier(x, y, 0.01), goal_inliers, max_iterations, 20)
    a, b, c = m
    plt.plot([0, 10], [-c/b, -(c+10*a)/b], color=(0, 1, 0))

    plt.show()
