import os
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm  # 导入 tqdm 库


def calculate_white_area_ratio(image):
    """计算图像中白色区域占整幅图像的面积比例"""
    if len(image.shape) == 3:  # 如果是彩色图像，转换为单通道
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化图像
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 计算白色区域的面积比例
    total_pixels = binary_image.size  # 图像总像素数
    white_pixels = np.sum(binary_image == 255)  # 白色像素数
    return white_pixels / total_pixels


def remove_directory_if_exists(directory):
    """如果目录存在，则删除它"""
    if os.path.exists(directory):
        shutil.rmtree(directory)


def process_videos(input_folder, N=5):
    """处理输入文件夹下所有子文件夹，并根据占比创建输出目录"""
    # 获取输入文件夹名称
    folder_name = os.path.basename(input_folder)

    # 定义输出目录
    small_dir = os.path.join(os.path.dirname(input_folder), f'{folder_name}_small')
    medium_dir = os.path.join(os.path.dirname(input_folder), f'{folder_name}_medium')
    large_dir = os.path.join(os.path.dirname(input_folder), f'{folder_name}_large')

    # 删除输出目录（如果已存在）
    remove_directory_if_exists(small_dir)
    remove_directory_if_exists(medium_dir)
    remove_directory_if_exists(large_dir)

    # 创建输出目录
    os.makedirs(small_dir)
    os.makedirs(medium_dir)
    os.makedirs(large_dir)

    frame_folder = os.path.join(input_folder, 'frame')
    groundtruth_folder = os.path.join(input_folder, 'groundtruth')

    # 获取所有子文件夹
    sub_folders = [sub for sub in os.listdir(groundtruth_folder) if
                   os.path.isdir(os.path.join(groundtruth_folder, sub))]

    # 使用 tqdm 显示进度条
    for sub_folder in tqdm(sub_folders, desc="Processing subfolders"):
        groundtruth_path = os.path.join(groundtruth_folder, sub_folder)
        frame_path = os.path.join(frame_folder, sub_folder)

        if os.path.isdir(groundtruth_path) and os.path.isdir(frame_path):
            # 随机选择 N 帧
            frames = os.listdir(groundtruth_path)
            selected_frames = random.sample(frames, min(N, len(frames)))

            white_area_ratios = []
            for frame in selected_frames:
                image_path = os.path.join(groundtruth_path, frame)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    ratio = calculate_white_area_ratio(image)
                    white_area_ratios.append(ratio)

            # 计算平均占比
            average_ratio = np.mean(white_area_ratios)

            # 根据占比处理文件夹
            if average_ratio < 0.05:
                target_dir = small_dir
            elif 0.05 <= average_ratio < 0.25:
                target_dir = medium_dir
            else:
                target_dir = large_dir

            # 删除目标目录（如果已存在）
            target_groundtruth_dir = os.path.join(target_dir, 'groundtruth', sub_folder)
            target_frame_dir = os.path.join(target_dir, 'frame', sub_folder)

            remove_directory_if_exists(target_groundtruth_dir)
            remove_directory_if_exists(target_frame_dir)

            # 复制 groundtruth 和 frame 文件夹
            shutil.copytree(groundtruth_path, target_groundtruth_dir)
            shutil.copytree(frame_path, target_frame_dir)



process_videos(r"H:\SPL\MOSE100\E2FGVI", N=10000)
process_videos(r"H:\SPL\MOSE100\FuseFormer", N=10000)
process_videos(r"H:\SPL\MOSE100\STTN", N=10000)


