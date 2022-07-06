#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：rf_train.py 
@File    ：complex_noise.py
@IDE     ：PyCharm 
@Author  ：王晨昊
@Date    ：2022/6/21 16:36 
'''
import cv2 as cv
import os

# 读取图片
import numpy as np

def main():
    image = cv.imread('./images/girl.jpg')

    # 椒盐噪声

    # 设置添加椒盐噪声的数目比例
    s_p_proportion = 0.5
    # 设置添加噪声图像像素的数目
    amount = 0.02
    noisy_img = np.copy(image)
    # 添加salt噪声
    num_salt = np.ceil(amount * image.size * s_p_proportion)
    # 设置添加噪声的坐标位置
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_img[coords] = 255
    # 添加pepper噪声
    num_pepper = np.ceil(amount * image.size * (1. - s_p_proportion))
    # 设置添加噪声的坐标位置
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_img[coords] = 0

    image = noisy_img

    # 高斯噪声

    # 设置高斯分布的均值和方差
    mean = 0
    # 设置高斯分布的标准差
    sigma = 25

    # 读取图片的属性
    img_width = image.shape[0]
    img_height = image.shape[1]
    img_channels = image.shape[2]

    # print(img_width, img_height, img_channels)
    # 根据均值和标准差生成符合高斯分布的噪声
    gauss = np.random.normal(mean, sigma, (img_width, img_height, img_channels))

    # 给图片添加高斯噪声
    noisy_img = image + gauss
    # 设置图片添加高斯噪声之后的像素值的范围
    noisy_img = np.clip(noisy_img, a_min=0, a_max=255)
    image = noisy_img

    # 泊松噪声
    # 计算图像像素的分布范围
    factor = 1.3
    distribution = len(np.unique(image))

    vals = factor ** np.ceil(np.log2(distribution))
    # 生成泊松噪声
    noisy_img = np.random.poisson(image * vals) / float(vals)
    image = noisy_img

    # 创建保存路径
    save_path = './complex/'
    isExists = os.path.exists(save_path)
    if not isExists:
        # 如果不存在该目录则创建文件夹
        os.makedirs(save_path)
    cv.imwrite('./complex/noisy.jpg', image)


    # 去噪
    # 中值滤波
    denoised_image = cv.medianBlur(np.uint8(image), 3)

    denoised_image = cv.GaussianBlur(denoised_image, (3, 3), 0, 0)
    denoised_image = cv.fastNlMeansDenoisingColored(denoised_image, None, 10, 10, 9, 15)

    cv.imwrite('./complex/denoised.jpg', denoised_image)

if __name__ == '__main__':
    main()