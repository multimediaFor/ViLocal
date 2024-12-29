import warnings

warnings.filterwarnings("ignore")
import torch
from uniformer import Encoder
from base_model import BaseModel
import torch.nn as nn
from torch.nn.functional import upsample

# 定义解码器
class Decoder(BaseModel):
    def __init__(self):
        super(Decoder, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        out = self.classifier(x)
        return out


class ViLocal(BaseModel):
    def __init__(self):
        super(ViLocal, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}

    def forward(self, inputs):
        x = self.encoder(inputs)
        out = self.decoder(x)
        out = upsample(out, inputs.shape[3:], **self._up_kwargs)

        return out


# mymodel = ViLocal().cuda()
# x = torch.zeros(8, 3, 5, 56, 56).cuda()  # [B, C, T, H, W]
# out = mymodel(x)
# print(out.shape)
