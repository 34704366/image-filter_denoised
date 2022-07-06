### 运行事项

本次实验中用到code目录下三个python程序，如果需要运行python程序，请先安装相关的依赖（所有用到的依赖在environment.yml中保存）或者根据yml配置文件导入conda环境

### 目录结构说明

##### 代码

1. noise_generate.py为实验中生成各种噪声的程序，将images目录下两个原始图像生成噪声后分别保存在`./images/poisson_noisy` ，`./images/gauss_noisy` ， `./images/salt_pepper_noisy` 目录中。

2. denoising.py为实验中对于生成噪声的图像进行去噪处理的程序，分别对于上一部分提到的三个目录中的所有图像进行去噪处理，结果保存在`./images/denoised` 目录中。
3. complex_noise.py为实验中生成叠加噪声并且组合去噪算法进行去噪的实验，生成的噪声图像和去噪图像都保存在`./complex/`目录中。

##### 图像素材

所有图像素材保存在`./images/`目录中，本次实验中主要用到两张原始图像素材，分别是`./images/people.png`以及`./images/girl.jpg`

