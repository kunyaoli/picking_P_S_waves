import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 若检测到GPU环境则使用GPU，否则使用CPU。

class Lenet_1D(nn.Module):
    def __init__(self, num_ch=32):
        super(Lenet_1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=num_ch, kernel_size=5, stride=2,), #101
            nn.BatchNorm1d(num_ch),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_ch, out_channels=num_ch*2, kernel_size=5, stride=2),#31
            nn.BatchNorm1d(num_ch*2),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_ch*2, out_channels=num_ch*4, kernel_size=5, stride=2),#11
            nn.BatchNorm1d(num_ch*4),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Conv1d(num_ch*4, num_ch*8, kernel_size=1),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(num_ch*8, num_ch*4),#1568
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(num_ch*4, 2),
        )
    def forward(self, x):
        x = x.reshape(-1,1,600)
        x = self.features(x)
        #x = x.view(-1, 2016)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    from ptflops.flops_counter import get_model_complexity_info
    y = torch.randn(16,1,600).cuda()
    model = Lenet_1D().cuda()
    flops, params = get_model_complexity_info(model, (1, 600), as_strings=False,
                                              print_per_layer_stat=True)  # 不用写batch_size大小，默认batch_size=1
    print('Flops:%s '%(flops/1e6))
    print('Params:%s '%(params/1e6))
