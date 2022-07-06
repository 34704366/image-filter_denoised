#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：rf_train.py 
@File    ：denoising.py
@IDE     ：PyCharm 
@Author  ：王晨昊
@Date    ：2022/6/20 11:21 
'''
import cv2 as cv
import os

# 读取图片
import numpy as np


# 均值滤波
def means_filter(image):
    # 过滤器强度，越大去噪效果越好，但是可能回消除图像细节
    h = [5,10,15]

    # 用于计算权重的模板补丁的像素大小，为奇数，默认7
    templateWindowSize = [5, 7, 9]

    # 搜索窗口的像素大小，用于计算给定像素的加权平均值，作用类似卷积核，必须是奇数，默认21
    searchWindowSize = [15, 21, 27]

    denoised_image_list = []
    # 去噪
    for h in h:
        for templateSize in templateWindowSize:
            for searchSize in searchWindowSize:
                denoised_image = cv.fastNlMeansDenoisingColored(image, None, h, h, templateSize, searchSize)
                name_suffix = f'h{h}tmp{templateSize}srh{searchSize}'
                obj = {
                    'image': denoised_image,
                    'name_suffix' : name_suffix
                }
                denoised_image_list.append(obj)
    return denoised_image_list


# 方框滤波
def box_filter(image):
    # 处理结果图像的图像深度, -1表示于原始图像使用相同的图像深度
    ddepth = -1

    # 滤波核大小
    ksize = [(2, 2), (3,3), (5,5)]

    # 是否归一化， 1归一化 0 不归一化
    normalize = True

    denoised_image_list = []

    for k in ksize:
        denoised_image = cv.boxFilter(image, ddepth, k, normalize=normalize)
        name_suffix = f'kszie{k}'
        obj = {
            'image': denoised_image,
            'name_suffix' : name_suffix
        }
        denoised_image_list.append(obj)

    return denoised_image_list


# 中值滤波，
def median_filter(image):
    # 滤波核大小，必须为奇数
    ksize = [3, 5, 7]

    denoised_image_list = []

    for k in ksize:
        denoised_image = cv.medianBlur(image, k)
        obj = {
            'image': denoised_image,
            'name_suffix': f'k{k}'
        }
        denoised_image_list.append(obj)

    return denoised_image_list


# 高斯滤波
def gauss_filter(image):
    # 滤波核的大小，奇数
    ksize = [(3, 3), (5, 5), (7, 7)]

    # 卷积核在X轴方向的标准差
    sigmaX = [0,]

    # 卷积核在Y轴方向的标准差
    sigmaY = [0,]

    denoised_image_list = []

    for k in ksize:
        for x in sigmaX:
            for y in sigmaY:
                denoised_image = cv.GaussianBlur(image, k, x, y)
                obj = {
                    'image': denoised_image,
                    'name_suffix': f'k{k}x{x}y{y}'
                }
                denoised_image_list.append(obj)

    return denoised_image_list

def get_origin_images(path):
    #  处理目录
    root_dir = path
    # 图像列表
    image_list = []

    # 处理后命名后缀
    suffix_name = 'gauss_noisy'

    for root, dirs, files in os.walk(root_dir):
        for file in files:  # 遍历目录里的所有文件
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(root, file)
                image = cv.imread(image_path)
                object = {
                    'path': image_path,
                    'image': image
                }
                image_list.append(object)
    return image_list


def denoise_process(path, image_list):
    #  处理目录
    root_dir = path

    # image_list = ['./images/girl.jpg']

    # 创建保存路径
    save_path = root_dir + 'denoised/means' + '/'
    isExists = os.path.exists(save_path)
    if not isExists:
        # 如果不存在该目录则创建文件夹
        os.makedirs(save_path)

    # 均值滤波
    for image_path in image_list:
        # 读取图像
        image = cv.imread(image_path)
        # 高斯滤波处理
        denoised_image_list = means_filter(image)

        for obj in denoised_image_list:
            # 图像
            denoised_image = obj['image']
            # 名称后缀
            name_suffix = obj['name_suffix']

            # 生成处理后的文件名
            file_name = image_path.split('/')[-1]
            file_name = file_name.split('.')
            file_name = file_name[0] + '_denoiesd_'+name_suffix+'.' + file_name[1]
            # 保存路径
            result_path = save_path + file_name
            cv.imwrite(result_path, denoised_image)

    # 创建保存路径
    save_path = root_dir + 'denoised/gauss' + '/'
    isExists = os.path.exists(save_path)
    if not isExists:
        # 如果不存在该目录则创建文件夹
        os.makedirs(save_path)
    # 高斯滤波
    for image_path in image_list:
        # 读取图像
        image = cv.imread(image_path)
        # 高斯滤波处理
        denoised_image_list = gauss_filter(image)

        for obj in denoised_image_list:
            denoised_image = obj['image']
            name_suffix = obj['name_suffix']
            # 生成处理后的文件名
            file_name = image_path.split('/')[-1]
            file_name = file_name.split('.')
            file_name = file_name[0] + '_denoiesd_'+name_suffix+'.' + file_name[1]
            # 保存路径
            result_path = save_path + file_name
            cv.imwrite(result_path, denoised_image)

    # 创建保存路径
    save_path = root_dir + 'denoised/box' + '/'
    isExists = os.path.exists(save_path)
    if not isExists:
        # 如果不存在该目录则创建文件夹
        os.makedirs(save_path)
    # 方框滤波
    for image_path in image_list:
        # 读取图像
        image = cv.imread(image_path)
        # 高斯滤波处理
        denoised_image_list = box_filter(image)
        print(denoised_image_list)
        for obj in denoised_image_list:
            name_suffix = obj['name_suffix']
            denoised_image = obj['image']

            # 生成处理后的文件名
            file_name = image_path.split('/')[-1]
            file_name = file_name.split('.')
            file_name = file_name[0] + '_denoiesd_'+name_suffix+'.'+ file_name[1]
            # 保存路径
            result_path = save_path + file_name
            cv.imwrite(result_path, denoised_image)
    #
    # 创建保存路径
    save_path = root_dir + 'denoised/median' + '/'
    isExists = os.path.exists(save_path)
    if not isExists:
        # 如果不存在该目录则创建文件夹
        os.makedirs(save_path)
    # 中值滤波
    for image_path in image_list:
        # 读取图像
        image = cv.imread(image_path)
        # 高斯滤波处理
        denoised_image_list = median_filter(image)

        for obj in denoised_image_list:
            name_suffix = obj['name_suffix']
            denoised_image = obj['image']
            # 生成处理后的文件名
            file_name = image_path.split('/')[-1]
            file_name = file_name.split('.')
            file_name = file_name[0] + '_denoiesd_'+name_suffix+'.' + file_name[1]
            # 保存路径
            result_path = save_path + file_name
            cv.imwrite(result_path, denoised_image)

def get_path():

    dir_list = ['./images/gauss_noisy', './images/poisson_noisy',
                './images/salt_pepper_noisy']

    # 需要处理的图像的列表
    image_list = ['./images/girl.jpg', './images/people.png']
    for dir in dir_list:
        for root, dirs, files in os.walk(dir):
            for file in files:  # 遍历目录里的所有文件
                if file.endswith(".jpg") or file.endswith(".png"):
                    # 加入图片路径
                    image_list.append(os.path.join(root+ '/' + file))

    return image_list

def main():
    path = './images/'
    image_list = get_path()

    print(image_list)
    denoise_process(path, image_list)


if __name__ == '__main__':
    main()