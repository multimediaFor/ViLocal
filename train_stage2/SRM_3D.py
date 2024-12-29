import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from PIL import Image
import cv2


class SRM3DMoudle(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        q = [4.0, 12.0, 4.0]
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]

        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]

        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, -1, 0, 0],
                   [0, -1, +4, -1, 0],
                   [0, 0, -1, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / q[0]
        filter2 = np.asarray(filter2, dtype=float) / q[1]
        filter3 = np.asarray(filter3, dtype=float) / q[2]

        self.weight = torch.tensor([[filter1, filter2, filter3]],
                                   dtype=torch.float32).unsqueeze(1).repeat(in_channels, in_channels, 1, 1, 1)
        # print(self.weight.shape)

    def forward(self, input):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        result = F.conv3d(input,
                          weight=nn.Parameter(self.weight.to(device), requires_grad=False),
                          stride=(1, 1, 1),
                          padding=(1, 2, 2))
        result = torch.clamp(result, min=0.0, max=4.0)
        return result

# mymodel = SRM3DMoudle2(5).cuda()
# x = torch.zeros(3, 5, 3, 224, 224).cuda()
# print(x.shape)
# out = mymodel(x)
# print(out.shape)
