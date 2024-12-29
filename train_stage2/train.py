import warnings

warnings.filterwarnings("ignore")
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from metric import f1_iou
from dataset import FramesDataset
from loss import Sigmoid_Focal_Loss
import os
from model import ViLocal

# 定义超参数
LR = 0.0001  # 学习率
EPOCHS = 5  # 训练轮数
BATCH_SIZE = 2  # 批次大小
nw = 8


# 定义训练集和验证集
train_dataset = FramesDataset(root_dir='/train_VI_CP', n_frames=5, istrain=True)
val_dataset1 = FramesDataset(root_dir='/DAVIS2017_test/test_VI',n_frames=5, istrain=True)
val_dataset2 = FramesDataset(root_dir='/DAVIS2017_test/test_OP',n_frames=5, istrain=True)
val_dataset3 = FramesDataset(root_dir='/DAVIS2017_test/test_CP',n_frames=5, istrain=True)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=nw)
val_loader1 = DataLoader(val_dataset1, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw)
val_loader2 = DataLoader(val_dataset2, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw)
val_loader3 = DataLoader(val_dataset3, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw)


val_loaders = [val_loader1, val_loader2, val_loader3]



# 定义训练器
device = torch.device("cuda")
model = ViLocal().to(device)

encoder_weights = torch.load('./epoch_010.pth')
model.encoder.load_state_dict(encoder_weights)
# 冻结编码器的参数
for param in model.encoder.parameters():
    param.requires_grad = False



# 定义损失函数
criterion = Sigmoid_Focal_Loss()

# 定义优化器
optimizer = optim.AdamW(model.decoder.parameters(), lr=LR, weight_decay=1e-4)

# 定义学习率衰减策略
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.000001)

# 权重保存位置
save_dir = './weights'


# 定义训练函数
def train(model, train_loader, val_loaders, criterion, optimizer, scheduler, save_dir):
    # 加载最新的权重文件，如果存在的话
    start_epoch = 0
    latest_weight_file = os.path.join(save_dir, 'latest.pth')
    if os.path.exists(latest_weight_file):
        checkpoint = torch.load(latest_weight_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print('Loaded the latest checkpoint from epoch %d.' % start_epoch)

    for epoch in range(start_epoch, EPOCHS):
        model.train()  # 将模型设置为训练模式
        train_loss = 0.0  # 记录每个epoch的平均loss

        # 使用tqdm显示进度条
        with tqdm(total=len(train_loader)) as pbar:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()  # 梯度清零
                outputs = model(inputs)  # 前向传播
                loss = criterion(outputs, targets)  # 计算loss
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
                train_loss += loss.item() * inputs.size(0)  # 累加loss

                # 更新进度条
                pbar.update(1)
                pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch + 1,
                    batch_idx * len(inputs),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()))

        scheduler.step()  # 更新学习率
        train_loss = train_loss / len(train_loader.dataset)  # 计算平均loss

        print('Epoch: [{}/{}], Train Loss: {:.6f}, nextLR: {:.8f}'.format(epoch + 1,
                                                                          EPOCHS,
                                                                          train_loss,
                                                                          scheduler.get_last_lr()[0]))

        print("\n")

        # 保存每一轮的权重文件
        epoch_weight_file = os.path.join(save_dir, 'epoch_{:0>3}.pth').format(epoch)
        torch.save(model.state_dict(), epoch_weight_file)

        # 保存最新的权重文件
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        latest_weight_file = os.path.join(save_dir, 'latest.pth')
        torch.save(checkpoint, latest_weight_file)

        if epoch % 1 == 0:
            # 验证
            model.eval()  # 将模型设置为验证模式
            val_f1s = [0.0] * len(val_loaders)  # 记录每个epoch的平均F1值
            val_mious = [0.0] * len(val_loaders)  # 记录每个epoch的平均mIoU

            # 不需要计算梯度
            with torch.no_grad():

                for i, val_loader in enumerate(val_loaders):
                    # 使用tqdm显示进度条
                    with tqdm(total=len(val_loader)) as pbar:
                        for batch_idx, (inputs, targets) in enumerate(val_loader):
                            inputs, targets = inputs.to(device), targets.to(device)  # 将inputs和targets转移到GPU上
                            outputs = model(inputs)  # 前向传播  # [B, C, H, W]

                            # 计算metric
                            f1, iou = f1_iou(outputs, targets)  # 计算F1和iou
                            val_f1s[i] += f1 * inputs.size(0)  # 累加F1值
                            val_mious[i] += iou * inputs.size(0)  # 累加mIoU
                            pbar.update(1)  # 更新进度条
                            pbar.set_description('Val Epoch: {} [{}/{} ({:.0f}%)]'.format(
                                epoch + 1,
                                batch_idx * len(inputs),
                                len(val_loader.dataset),
                                100. * batch_idx / len(val_loader)))

                    val_f1s[i] /= len(val_loader.dataset)  # 计算平均F1值
                    val_mious[i] /= len(val_loader.dataset)  # 计算平均mIoU

                    print('Epoch: [{}/{}], Val Set {}:  IoU: {:.4f}, F1: {:.4f}'.format(
                        epoch + 1,
                        EPOCHS,
                        i + 1,
                        val_mious[i],
                        val_f1s[i]))

                    print("\n")

            print("===================================================================================================")


# 训练模型
def main():
    train(model, train_loader, val_loaders, criterion, optimizer, scheduler, save_dir)


if __name__ == "__main__":
    main()
