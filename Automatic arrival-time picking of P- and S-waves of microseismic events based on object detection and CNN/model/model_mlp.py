import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 若检测到GPU环境则使用GPU，否则使用CPU。

import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden)
        self.hidden2 = nn.Linear(n_hidden,n_hidden//2)
        self.predict = nn.Linear(n_hidden//2,n_output)
    def forward(self,input):
        out = self.hidden1(input)
        out = F.relu(out)
        out = self.hidden2(out)
        out = torch.sigmoid(out)
        out =self.predict(out)

        return out


if __name__ == '__main__':
    from ptflops.flops_counter import get_model_complexity_info
    from torchsummary import summary
    y = torch.randn(16,1,600).cuda()
    model = Net(600,1200,2).cuda()

    flops, params = get_model_complexity_info(model, (1, 600), as_strings=False,
                                              print_per_layer_stat=True)  # 不用写batch_size大小，默认batch_size=1
    print('Flops:%s ' % (flops / 1e6))
    print('Params:%s ' % (params / 1e6))

    model(y)
    #print(model)
    summary(model, (1,600))