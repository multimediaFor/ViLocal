import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
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

