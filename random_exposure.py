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
