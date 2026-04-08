"""
改进版对比度增强预处理脚本 - 针对YOLOv8-OBB缺陷检测优化
主要改进：
1. 自适应CLAHE参数（根据图像尺寸和内容）
2. 边缘增强（OBB对边缘敏感）
3. 噪声抑制（双边滤波）
4. 自适应对比度增强
5. 优化的多尺度Retinex参数
6. 可选的锐化增强
"""

import cv2
import os
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt


def adaptive_clahe(img, clip_limit=None, tile_size=None):
    """
    自适应CLAHE - 根据图像尺寸和内容动态调整参数

    Args:
        img: 输入图像
        clip_limit: 对比度限制，None时自适应
        tile_size: 瓦片大小，None时自适应

    Returns:
        处理后的图像
    """
    h, w = img.shape[:2]

    # 自适应clipLimit：图像对比度低时增大，高时减小
    if clip_limit is None:
        std_dev = np.std(img)
        if std_dev < 30:  # 低对比度
            clip_limit = 3.0
        elif std_dev < 50:
            clip_limit = 2.5
        else:
            clip_limit = 2.0

    # 自适应tileGridSize：根据图像尺寸调整
    if tile_size is None:
        min_dim = min(h, w)
        if min_dim < 512:
            tile_size = (4, 4)
        elif min_dim < 1024:
            tile_size = (8, 8)
        else:
            tile_size = (16, 16)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(img)


def denoise_bilateral(img, d=5, sigma_color=50, sigma_space=50):
    """
    双边滤波去噪 - 保持边缘的同时去除噪声
    适合缺陷检测，不会模糊边缘细节
    """
    if len(img.shape) == 2:
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    else:
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def adaptive_contrast_enhance(img, alpha_range=(1.0, 1.5), beta=0):
    """
    自适应对比度增强 - 根据图像统计特性调整参数

    Args:
        img: 输入图像
        alpha_range: 对比度增强系数范围 (min, max)
        beta: 亮度偏移

    Returns:
        增强后的图像
    """
    img_float = img.astype(np.float32)
    mean_val = np.mean(img_float)
    std_val = np.std(img_float)

    # 根据标准差自适应调整alpha
    if std_val < 25:  # 低对比度，需要更强增强
        alpha = alpha_range[1]
    elif std_val > 60:  # 高对比度，轻微增强
        alpha = alpha_range[0]
    else:
        # 线性插值
        alpha = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * (25 - std_val) / 35

    enhanced = alpha * (img_float - mean_val) + beta + mean_val
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return enhanced


def edge_enhancement(img, strength=0.3):
    """
    边缘增强 - OBB任务对边缘敏感，增强边缘有助于检测

    Args:
        img: 输入图像
        strength: 增强强度 (0-1)

    Returns:
        边缘增强后的图像
    """
    # 使用拉普拉斯算子进行边缘增强
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = np.clip(laplacian, -255, 255).astype(np.float32)

    # 将边缘信息叠加到原图
    enhanced = img.astype(np.float32) + strength * laplacian
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return enhanced


def unsharp_mask(img, sigma=1.0, strength=0.5):
    """
    非锐化掩模 - 增强细节和边缘

    Args:
        img: 输入图像
        sigma: 高斯模糊标准差
        strength: 锐化强度
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def gaussian_kernel(S, sigma):
    """生成高斯核"""
    K = np.multiply(
        cv2.getGaussianKernel(S.shape[0], sigma, ktype=cv2.CV_32F),
        (cv2.getGaussianKernel(S.shape[1], sigma, ktype=cv2.CV_32F)).T
    )
    return K


def ssr4gry(S, sigma):
    """
    单尺度Retinex - 优化版本，处理数值稳定性
    """
    K = gaussian_kernel(S, sigma)
    fft = np.fft.fft2(K)
    z = np.fft.fftshift(fft)

    # 时域卷积等于频域乘积，估计光照图像L
    L = np.real(np.fft.ifft2(np.fft.fft2(S.astype(np.float32)) * z))
    L = L - np.min(L) + 1e-10  # 避免除零

    # 获取真实图像，去除光照影响
    S_float = S.astype(np.float32) + 1e-10
    R_temp = np.log(S_float) - np.log(L)
    R = cv2.normalize(np.exp(R_temp), None, 0, 255, cv2.NORM_MINMAX)
    return R.astype(np.uint8), R_temp, L


def improved_gamma2d(img_in, gamma=1.5, adaptive=True):
    """
    改进的二维伽马校正 - 针对缺陷检测优化

    Args:
        img_in: 输入图像（灰度图）
        gamma: 伽马值
        adaptive: 是否使用自适应参数

    Returns:
        处理后的图像
    """
    # 转换为HSV进行处理
    bgr_img = cv2.cvtColor(img_in, cv2.COLOR_GRAY2BGR)
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

    # 多尺度光照估计 - 针对小缺陷优化sigma值
    v_list = []
    # 调整sigma值以更好地捕获不同尺寸的缺陷
    sigmas = [10, 50, 150] if adaptive else [15, 80, 250]

    for sigma in sigmas:
        _, _, l = ssr4gry(v, sigma)
        v_list.append(l)

    v_ndarray = np.array(v_list)
    L = np.mean(v_ndarray, axis=0)

    # 自适应或固定m值
    if adaptive:
        m = 255 * np.mean(L) / 255.0  # 自适应校正
        m = max(100, min(150, m))  # 限制在合理范围
    else:
        m = 128

    # 二维伽马校正
    r = gamma ** ((m - L) / m)
    v1 = 255 * ((v.astype(np.float32) / 255.0) ** r)
    v2 = np.clip(v1, 0, 255).astype(np.uint8)

    # 合并通道
    merged = cv2.merge([h, s, v2])
    img_b = cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    return img_b


def process_image_enhanced(img,
                           use_denoise=True,
                           use_edge_enhance=True,
                           use_sharpen=False,
                           clahe_adaptive=True,
                           contrast_adaptive=True,
                           gamma_adaptive=True):
    """
    完整的图像预处理流程 - 针对YOLOv8-OBB优化

    Args:
        img: 输入BGR图像
        use_denoise: 是否使用去噪
        use_edge_enhance: 是否使用边缘增强
        use_sharpen: 是否使用锐化
        clahe_adaptive: CLAHE是否自适应
        contrast_adaptive: 对比度增强是否自适应
        gamma_adaptive: 伽马校正是否自适应

    Returns:
        处理后的灰度图像
    """
    # 1. 转换为灰度图
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 2. 去噪（可选，在增强前进行）
    if use_denoise:
        gray = denoise_bilateral(gray, d=5, sigma_color=50, sigma_space=50)

    # 3. 自适应CLAHE
    if clahe_adaptive:
        gray = adaptive_clahe(gray)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # 4. 自适应对比度增强
    if contrast_adaptive:
        gray = adaptive_contrast_enhance(gray, alpha_range=(1.1, 1.4), beta=0)
    else:
        mean_val = np.mean(gray)
        gray = np.clip(1.2 * (gray.astype(np.float32) - mean_val) + mean_val, 0, 255).astype(np.uint8)

    # 5. 多尺度Retinex + 二维伽马校正
    gray = improved_gamma2d(gray, gamma=1.5, adaptive=gamma_adaptive)

    # 6. 边缘增强（可选，OBB任务推荐开启）
    if use_edge_enhance:
        gray = edge_enhancement(gray, strength=0.2)  # 较小强度，避免过度增强

    # 7. 锐化（可选，通常不需要，除非图像特别模糊）
    if use_sharpen:
        gray = unsharp_mask(gray, sigma=1.0, strength=0.3)

    return gray


def process_batch(input_dir, output_dir,
                  use_denoise=True,
                  use_edge_enhance=True,
                  use_sharpen=False,
                  save_hist=False):
    """
    批量处理图像

    Args:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径
        use_denoise: 是否去噪
        use_edge_enhance: 是否边缘增强
        use_sharpen: 是否锐化
        save_hist: 是否保存直方图
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_list = [f for f in os.listdir(input_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    print(f"找到 {len(file_list)} 张图像，开始处理...")

    for idx, filename in enumerate(file_list):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 读取图像
        img = cv2.imread(input_path)
        if img is None:
            print(f"警告: 无法读取 {filename}")
            continue

        # 处理图像
        processed = process_image_enhanced(
            img,
            use_denoise=use_denoise,
            use_edge_enhance=use_edge_enhance,
            use_sharpen=use_sharpen
        )

        # 保存结果
        cv2.imwrite(output_path, processed)

        # 可选：保存直方图
        if save_hist:
            hist = cv2.calcHist([processed], [0], None, [256], [0, 255])
            plt.figure(figsize=(8, 4))
            plt.plot(hist, color='r')
            plt.title(f'Histogram: {filename}')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            hist_path = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_hist.png')
            plt.savefig(hist_path, dpi=150)
            plt.close()

        if (idx + 1) % 10 == 0:
            print(f"已处理: {idx + 1}/{len(file_list)}")

    print(f"处理完成！结果保存在: {output_dir}")


if __name__ == '__main__':
    # ========== 配置参数 ==========
    # 输入输出路径
    data_base_dir = r'/home/chenhm/PycharmProjects/PythonProject1/ultralytics/dataset_demo/images/train/'  # 修改为你的输入文件夹路径
    outfile_dir = r'/home/chenhm/PycharmProjects/PythonProject1/ultralytics/dataset_pre/images/train/'  # 修改为你的输出文件夹路径

    # 处理选项
    USE_DENOISE = True  # 是否去噪（推荐开启）
    USE_EDGE_ENHANCE = True  # 是否边缘增强（OBB任务推荐开启）
    USE_SHARPEN = False  # 是否锐化（通常不需要）
    SAVE_HIST = False  # 是否保存直方图（调试用）

    # ========== 执行处理 ==========
    process_batch(
        input_dir=data_base_dir,
        output_dir=outfile_dir,
        use_denoise=USE_DENOISE,
        use_edge_enhance=USE_EDGE_ENHANCE,
        use_sharpen=USE_SHARPEN,
        save_hist=SAVE_HIST
    )

    # ========== 单张图像测试示例 ==========
    # 如果需要测试单张图像：
    # test_img = cv2.imread('test_image.jpg')
    # result = process_image_enhanced(test_img,
    #                                use_denoise=True,
    #                                use_edge_enhance=True,
    #                                use_sharpen=False)
    # cv2.imwrite('result.jpg', result)



