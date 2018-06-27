"""
    1.使用align将train切割至train160(有损失)
    2.过滤掉遮挡过多的图片
    3.切分训练集 验证集 测试集
"""
from scipy import misc
import os
import numpy as np
from collections import Counter


def filter(PATH, TARGET_PATH, probe:"遮挡阈值"=0.6):

    if not os.path.exists(TARGET_PATH):
        os.makedirs(TARGET_PATH)

    dirs = os.listdir(PATH)
    results = []
    for clas in dirs:
        if not os.path.isdir(os.path.join(PATH, clas)): continue
        # print(os.path.join(PATH, clas))
        for img_file in os.listdir(os.path.join(PATH, clas)):
            img = misc.imread(os.path.join(PATH, clas, img_file))
            # 像素统计
            statistics = Counter(img.flatten())
            # 黑色遮挡计算
            count = sum([v for k, v in statistics.items() if k <= 50])/len(img.flatten())
            result = ((clas, img_file, int(count*100)/100))
            results.append(result)
            # 阈值判断
            if count > probe:
                print(result)
                # if not os.path.exists(os.path.join(TARGET_PATH, clas)):
                #     os.makedirs(os.path.join(TARGET_PATH, clas))
                # misc.imsave(os.path.join(TARGET_PATH, clas, img_file), img)


if __name__ == "__main__":
    PATH = r'D:\Documents\smartcity\dataset\face\train160'
    TARGET_PATH = r'D:\Documents\smartcity\dataset\face\train160_chen'
    filter(PATH, TARGET_PATH)