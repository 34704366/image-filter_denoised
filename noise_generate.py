import cv2 as cv
import os

# 读取图片
import numpy as np

# 生成高斯噪声
def gauss_noise_generate(path, image_list):
    # 根目录
    root_dir = path

    # 处理后命名后缀
    suffix_name = 'gauss_noisy'

    # 对路径下的所有图片依次处理
    for object in image_list:
        # 图片内容
        image = object['image']
        # 图片路径
        image_path = object['path']

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

        # # 保存图片
        # cv.imwrite("noisy_img.png",noisy_img)

        # 创建保存路径
        save_path = root_dir + suffix_name + '/'
        isExists = os.path.exists(save_path)
        if not isExists:
            # 如果不存在该目录则创建文件夹
            os.makedirs(save_path)

        # 生成处理后的文件名
        file_name = image_path.split('/')[-1]
        file_name = file_name.split('.')
        file_name = file_name[0] + '_' + suffix_name + '.' + file_name[1]
        # 保存路径
        result_path = save_path + file_name
        cv.imwrite(result_path, noisy_img)

    print(suffix_name + '处理完毕')


# 生成椒盐噪声
def salt_pepper_noise_generate(path, image_list):
    #  处理目录
    root_dir = path

    # 处理后命名后缀
    suffix_name = 'salt_pepper_noisy'


    # 对路径下的所有图片依次处理
    for object in image_list:
        # 图片内容
        image = object['image']
        # 图片路径
        image_path = object['path']

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

        # 创建保存路径
        save_path = root_dir + suffix_name + '/'
        isExists = os.path.exists(save_path)
        if not isExists:
            # 如果不存在该目录则创建文件夹
            os.makedirs(save_path)

        # 生成处理后的文件名
        file_name = image_path.split('/')[-1]
        file_name = file_name.split('.')
        file_name = file_name[0] + '_' + suffix_name + '.' + file_name[1]
        # 保存路径
        result_path = save_path + file_name
        cv.imwrite(result_path, noisy_img)

    print(suffix_name + '处理完毕')


# 生成泊松噪声
def poisson_noisy_generate(path, image_list):
    #  处理目录
    root_dir = path

    # 处理后命名后缀
    suffix_name = 'poisson_noisy'


    factor = 1.

    # 对路径下的所有图片依次处理
    for object in image_list:
        # 图片内容
        image = object['image']
        # 图片路径
        image_path = object['path']

        # 计算图像像素的分布范围
        distribution = len(np.unique(image))

        vals = factor ** np.ceil(np.log2(distribution))
        # 生成泊松噪声
        noisy_img = np.random.poisson(image * vals) / float(vals)

        # 创建保存路径
        save_path = root_dir + suffix_name + '/'
        isExists = os.path.exists(save_path)
        if not isExists:
            # 如果不存在该目录则创建文件夹
            os.makedirs(save_path)

        # 生成处理后的文件名
        file_name = image_path.split('/')[-1]
        file_name = file_name.split('.')
        file_name = file_name[0] + '_' + suffix_name + '.' + file_name[1]
        # 保存路径
        result_path = save_path + file_name

        cv.imwrite(result_path, noisy_img)
    print(suffix_name, '处理完毕')


def get_origin_images(path):
    #  处理目录
    root_dir = path
    # 图像列表
    image_list = []

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



def main():
    path = './images/'
    image = ''
    # 获取图片列表
    image_list = get_origin_images(path)

    # 生成噪声
    gauss_noise_generate(path, image_list)
    salt_pepper_noise_generate(path, image_list)
    poisson_noisy_generate(path, image_list)


if __name__ == '__main__':
    main()
