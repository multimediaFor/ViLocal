import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm
from model import ViLocal
from metric import f1_iou

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resize_frame = transforms.Resize((240, 432), interpolation=Image.BILINEAR)
totensor = transforms.ToTensor()


# 定义加载图像并进行推理的函数
def process_frame_group(frame_paths, model):
    """
    处理每一个5帧组，返回模型输出。
    """
    frames = []
    for frame_path in frame_paths:
        img = Image.open(frame_path)
        frames.append(img)

    frames = [resize_frame(img) for img in frames]
    frames = torch.stack([totensor(frame) for frame in frames], dim=1).unsqueeze(0).to(device)

    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        pred = model(frames)  # 模型推理
        pred = pred.squeeze(0)  # 去掉batch维度
        # 二值化处理：sigmoid 激活函数 + 阈值 0.5
        pred = torch.sigmoid(pred)  # 转换为概率
        pred = (pred > 0.5).float()  # 二值化：大于0.5的值为1，其它为0

    return pred


# 定义主函数
def evaluate_model_on_dataset(txt_file, model, base_dir, output_dir, calculate_metrics=True):
    """
    读取txt文件中的数据，进行推理并保存输出，计算评价指标。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开txt文件并读取
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    f1_scores = []
    iou_scores = []

    # 遍历每行，处理5帧组
    for line in tqdm(lines):
        line = line.strip()

        parts = line.split()
        frame_paths = parts[:5]
        gt_path = parts[5]
        frame_paths = [os.path.join(base_dir, path) for path in frame_paths]  # 拼接相对路径

        # 加载groundtruth
        gt = Image.open(os.path.join(base_dir, gt_path)).convert("1")  # 加载groundtruth，转换为二值图像
        gt = totensor(gt).to(device)  # 转换为Tensor并送到设备

        # 进行推理并二值化
        output = process_frame_group(frame_paths, model)

        # 保存模型输出（二值化后的结果）
        output_path = os.path.join(output_dir, f"{os.path.basename(gt_path)}_output.png")
        output_img = output.squeeze().cpu().numpy()  # 转回numpy数组
        output_img = (output_img * 255).astype(np.uint8)  # 将0-1的值转为0-255，并转换为图片格式
        Image.fromarray(output_img).save(output_path)

        if calculate_metrics:
            # 计算F1和IoU
            f1, iou = f1_iou(output.unsqueeze(0), gt.unsqueeze(0))
            f1_scores.append(f1)
            iou_scores.append(iou)

    # 保存评价指标
    if calculate_metrics:
        with open(os.path.join(output_dir, "metric_result.txt"), 'w') as f:
            f.write(f"Average F1-score: {np.mean(f1_scores)}\n")
            f.write(f"Average IoU: {np.mean(iou_scores)}\n")

    print("Evaluation completed.")


def main():
    # 加载模型
    model = ViLocal()
    model.to(device)
    model_checkpoint = "weights/train_VI_OP.pth"  # 预训练模型的权重文件
    # 加载预训练权重
    if os.path.exists(model_checkpoint):
        model.load_state_dict(torch.load(model_checkpoint))
        print(f"Model weights loaded from {model_checkpoint}")
    else:
        print(f"Error: Model checkpoint not found at {model_checkpoint}")
        return  # 如果模型权重不存在，退出程序


    # MOSE_E2FGVI
    evaluate_model_on_dataset(
        txt_file="/data/myfile/SPL/MOSE100/E2FGVI_large/E2FGVI_large.txt",
        model=model,
        base_dir="/data/myfile/SPL/MOSE100/E2FGVI_large",
        output_dir="/data/myfile/SPL/MOSE100/E2FGVI_large/result",
        calculate_metrics=True
    )
    evaluate_model_on_dataset(
        txt_file="/data/myfile/SPL/MOSE100/E2FGVI_medium/E2FGVI_medium.txt",
        model=model,
        base_dir="/data/myfile/SPL/MOSE100/E2FGVI_medium",
        output_dir="/data/myfile/SPL/MOSE100/E2FGVI_medium/result",
        calculate_metrics=True
    )
    evaluate_model_on_dataset(
        txt_file="/data/myfile/SPL/MOSE100/E2FGVI_small/E2FGVI_small.txt",
        model=model,
        base_dir="/data/myfile/SPL/MOSE100/E2FGVI_small",
        output_dir="/data/myfile/SPL/MOSE100/E2FGVI_small/result",
        calculate_metrics=True
    )

    # MOSE_FuseFormer
    evaluate_model_on_dataset(
        txt_file="/data/myfile/SPL/MOSE100/FuseFormer_large/FuseFormer_large.txt",
        model=model,
        base_dir="/data/myfile/SPL/MOSE100/FuseFormer_large",
        output_dir="/data/myfile/SPL/MOSE100/FuseFormer_large/result",
        calculate_metrics=True
    )
    evaluate_model_on_dataset(
        txt_file="/data/myfile/SPL/MOSE100/FuseFormer_medium/FuseFormer_medium.txt",
        model=model,
        base_dir="/data/myfile/SPL/MOSE100/FuseFormer_medium",
        output_dir="/data/myfile/SPL/MOSE100/FuseFormer_medium/result",
        calculate_metrics=True
    )
    evaluate_model_on_dataset(
        txt_file="/data/myfile/SPL/MOSE100/FuseFormer_small/FuseFormer_small.txt",
        model=model,
        base_dir="/data/myfile/SPL/MOSE100/FuseFormer_small",
        output_dir="/data/myfile/SPL/MOSE100/FuseFormer_small/result",
        calculate_metrics=True
    )

    # MOSE_STTN
    evaluate_model_on_dataset(
        txt_file="/data/myfile/SPL/MOSE100/STTN_large/STTN_large.txt",
        model=model,
        base_dir="/data/myfile/SPL/MOSE100/STTN_large",
        output_dir="/data/myfile/SPL/MOSE100/STTN_large/result",
        calculate_metrics=True
    )
    evaluate_model_on_dataset(
        txt_file="/data/myfile/SPL/MOSE100/STTN_medium/STTN_medium.txt",
        model=model,
        base_dir="/data/myfile/SPL/MOSE100/STTN_medium",
        output_dir="/data/myfile/SPL/MOSE100/STTN_medium/result",
        calculate_metrics=True
    )
    evaluate_model_on_dataset(
        txt_file="/data/myfile/SPL/MOSE100/STTN_small/STTN_small.txt",
        model=model,
        base_dir="/data/myfile/SPL/MOSE100/STTN_small",
        output_dir="/data/myfile/SPL/MOSE100/STTN_small/result",
        calculate_metrics=True
    )


# 运行主函数
if __name__ == "__main__":
    main()
