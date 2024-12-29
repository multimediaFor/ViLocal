import warnings

warnings.filterwarnings("ignore")
import os

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image


class FramesDataset(Dataset):
    def __init__(self, root_dir, n_frames: int, istrain=True):
        self.root_dir = root_dir
        self.n_frames = n_frames
        self.istrain = istrain
        self.resize_frame = transforms.Resize((240, 432), interpolation=Image.BILINEAR)
        self.resize_label = transforms.Resize((60, 108), interpolation=Image.NEAREST)
        self.resize_label2 = transforms.Resize((240, 432), interpolation=Image.NEAREST)
        self.totensor = transforms.ToTensor()

        # 分别指定帧和标签的文件夹路径
        self.frame_dir = os.path.join(root_dir, 'frame')
        self.label_dir = os.path.join(root_dir, 'groundtruth')

        # 获取所有帧和标签的文件名
        self.frame_files = sorted(
            [f for f in os.listdir(self.frame_dir) if os.path.isdir(os.path.join(self.frame_dir, f))])
        self.label_files = sorted(
            [f for f in os.listdir(self.label_dir) if os.path.isfile(os.path.join(self.label_dir, f))])

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        # 获取帧和标签的路径
        frame_folder = self.frame_files[idx]
        frame_path = os.path.join(self.frame_dir, frame_folder)
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        # 获取n帧组里面的图像
        frame_list = []
        for root, dirs, files in os.walk(frame_path):
            for file in files:
                frame_list.append(os.path.basename(file))
        frame_list = sorted(frame_list)

        # 读取n帧图像，存入列表中
        frame_images = []
        for i in range(self.n_frames):
            # 构造每一帧图像的路径
            frame_image_path = os.path.join(frame_path, frame_list[i])

            # 使用PIL库读取图像
            frame_image = Image.open(frame_image_path)
            frame_images.append(frame_image)

        # 读取标签图像
        label_image = Image.open(label_path)

        # 如果是在训练，则进行resize
        if self.istrain:
            # 双线性插值
            frame_images = [self.resize_frame(frame) for frame in frame_images]
            # 最近邻插值
            label_image = self.resize_label(label_image)
        else:
            # 双线性插值
            frame_images = [self.resize_frame(frame) for frame in frame_images]
            # 最近邻插值
            label_image = self.resize_label2(label_image)

        # 将n帧图像和标签图像转换为张量，并返回
        frame_images = [self.totensor(frame) for frame in frame_images]
        label_image = self.totensor(label_image)
        return torch.stack(frame_images, dim=1), label_image
