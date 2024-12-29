import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
from model import ViLocal
from PIL import Image
from torchvision.transforms import transforms
import os
import numpy as np
from tqdm import tqdm

def f1_iou(pred, target):
    # 调整target的大小
    if pred.shape[2:] != target.shape[2:]:
        target = F.interpolate(target, size=pred.shape[2:], mode='nearest')

    pred = torch.sigmoid(pred) > 0.5  # 将预测结果转化为0/1二值图像
    target = target.view(-1)  # 将target展平为一维数组
    pred = pred.view(-1)  # 将pred展平为一维数组
    tp = (target * pred).sum().float()  # 计算TP
    fp = ((1 - target.float()) * pred.float()).sum().float()  # 计算FP
    fn = (target.float() * (1 - pred.float())).sum().float()  # 计算FN
    precision = tp / (tp + fp + 1e-8)  # 计算精确率
    recall = tp / (tp + fn + 1e-8)  # 计算召回率
    f1 = 2 * precision * recall / (precision + recall + 1e-8)  # F1
    iou = tp / (tp + fp + fn + 1e-8)  # iou
    return f1.item(), iou.item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 获取文件夹中的所有文件夹名
def getFlist(file_dir):
    for root, subdirs, files in os.walk(file_dir):
        dir_list = subdirs
        break
    return dir_list
def vis(frame_path, outpath, model):
    # transforms
    resize_frame = transforms.Resize((240, 432), interpolation=Image.BILINEAR)
    totensor = transforms.ToTensor()

    # 获取n帧组里面的图像
    frame_list = []
    for root, dirs, files in os.walk(frame_path):
        for file in files:
            frame_list.append(os.path.basename(file))

    # 读取n帧图像，存入列表中
    frame_images = []
    for i in range(5):
        # 构造每一帧图像的路径
        frame_image_path = os.path.join(frame_path, frame_list[i])

        # 使用PIL库读取图像
        frame_image = Image.open(frame_image_path)
        frame_images.append(frame_image)

    # resize
    frame_images = [resize_frame(frame) for frame in frame_images]

    # 将n帧图像和标签图像转换为张量，并返回
    frame_images = torch.stack([totensor(frame) for frame in frame_images], dim=1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(frame_images)
        pred = pred.squeeze(0)

        # 二值化
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()

        # 转换为numpy数组
        np_img = pred.cpu().numpy().squeeze()

        # 将numpy数组转换为PIL图像
        img = Image.fromarray(np.uint8(np_img * 255.0))
        img.save(os.path.join(outpath, os.path.basename(frame_path) + '.png'))


def batch_vis(file_dir, outpath, model):
    dir_list = getFlist(file_dir)
    for frames_dir in tqdm(dir_list):
        frame_path = os.path.join(file_dir, frames_dir)
        vis(frame_path, outpath, model)


checkpoint_path = './weights/train_VI_OP.pth'
model = ViLocal().to(device)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()
batch_vis('./demo', './output', model)