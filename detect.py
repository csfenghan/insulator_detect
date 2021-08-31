import cv2
import argparse
import numpy as np
import sys
import os

def parseArg(opt):
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
        

def main(opt):
    data = parseArg(opt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="images", help="images directory")

    opt = parser.parse_args()
    main(opt)
