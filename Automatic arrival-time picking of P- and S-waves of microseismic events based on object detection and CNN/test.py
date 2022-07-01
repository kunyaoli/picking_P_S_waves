from Dataset.dataset import Mydataset
import torch
#from model.model_10_18_structure_pool import Lenet_1D # for Tabel 2
from model.model_10_18_structure_1 import Lenet_1D  # for Tabel 1
#from model.model_mlp import Net # mlp
from torchvision import transforms
from torch.utils import data
import numpy as np
import pandas as pd


def invers_nrom(data_p , data_s, min_p = 0.0724999979138374 ,max_p = 0.126200005412101,
                min_s =0.0822999998927116 , max_s=0.139899998903274 ):
    range_p = max_p- min_p
    range_s = max_s- min_s
    p = (data_p * range_p)+min_p
    s = (data_s * range_s)+min_s
    return p,s

def MAPE(true, pred):
    diff = np.abs(np.array(true) - np.array(pred))
    print(np.mean(diff), np.std(diff), np.max(diff), np.min(diff))
    # print(diff / true)
    print(len(diff))
    return np.mean(diff / true)


def pct(error, alpha):
    return len(error[abs(error) < alpha]) / len(error)


def MAPE(true, pred):
    diff = np.abs(np.array(true) - np.array(pred))
    print(np.mean(diff),np.std(diff),np.max(diff),np.min(diff))
    #print(diff / true)
    print(len(diff))
    return np.mean(diff / true)
def pct(error,alpha):
    return len(error[abs(error)<alpha])/len(error)
def cal_metrcis(p_real, p_pred, s_real=None, s_pred=None):
    df = pd.DataFrame()
    error_p = p_real - p_pred
    pct1_p = pct(error_p, 0.001)
    pct3_p = pct(error_p, 0.003)
    pct5_p = pct(error_p, 0.005)
    pct10_p = pct(error_p, 0.010)
    pct20_p = pct(error_p, 0.020)
    MAPE_p = MAPE(p_real, p_pred)

    if s_real is not None:
        error_s = s_real - s_pred
        pct1_s = pct(error_s, 0.001)
        pct5_s = pct(error_s, 0.005)
        pct3_s =  pct(error_s, 0.003)
        pct10_s = pct(error_s, 0.010)
        pct20_s = pct(error_s, 0.020)
        MAPE_s = MAPE(s_real, s_pred)
        df["s_metrics"] = [pct1_s,pct3_s, pct5_s, pct10_s,pct20_s, MAPE_s]
    df['col'] = ["pct-1","pct-3","pct-5", "pct-10",'pct-20', "MAPE"]
    df["p_metrics"] = [pct1_p,pct3_p,pct5_p, pct10_p, pct20_p, MAPE_p]

    # df = df.T
    df = df.set_index('col')
    return df.T

transform = transforms.Compose([
    # transforms
    # .Resize((96, 96)),
    transforms.ToTensor(),
])
NAME = 'model_10_18_structure_1' # MLP

EPOCH = "best"


test_Dataset = Mydataset(transform=transform, mode="test")

BATCH_SIZE = 256

test_dl = data.DataLoader(
    test_Dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


#net =Net(600, 1200, 2)
net = Lenet_1D()
print(net)
if torch.cuda.is_available():
    net.to('cuda')
net.load_state_dict(torch.load("./checkpoints/%s/model/%s.pth"%(NAME,EPOCH)))

#net.load_state_dict(torch.load("/home/lky/code/ps_regression/06_1D_cnn/1d_cnn_nromdata/checkpoints/model_invers_fc_relu/128-101-31-11-FC-Relu/model/Epoch149.pth"))
print(net)
if torch.cuda.is_available():
    net.to('cuda')

loss_func = torch.nn.MSELoss()
net.eval()
running_loss=0
p_real = []
s_real = []
p_prediction = []
s_prediction = []
for x, p_value, s_value in test_dl:
    p_real.extend(p_value.numpy())
    s_real.extend(s_value.numpy())
    if torch.cuda.is_available():
        x, p_value, s_value = (x.to('cuda'),
                               p_value.to('cuda'), s_value.to('cuda'))
    value = net(x.to(torch.float32))
    p_pred = value[:, 0]
    s_pred = value[:, 1]
    p_prediction.extend(p_pred.cpu().detach().numpy())
    s_prediction.extend(s_pred.cpu().detach().numpy())

    loss_p = loss_func(p_pred, p_value)
    loss_s = loss_func(s_pred, s_value)
    loss = loss_p +loss_s
    running_loss += loss.item()
print(running_loss)
print(p_real)
print(p_prediction)
#print("p_mse:",mean_squared_error(p_real, p_prediction))
p_prediction,s_prediction = invers_nrom(np.array(p_prediction), np.array(s_prediction))
p_real,s_real = invers_nrom(np.array(p_real), np.array(s_real))
data_all_result = pd.DataFrame()
data_all_result["P_real"] = np.array(p_real)
data_all_result["P_pred"] = np.array(p_prediction)
data_all_result['S_real'] = np.array(s_real)
data_all_result["S_pred"] = np.array(s_prediction)
data_all_result.to_csv("./checkpoints/%s/data_all_result.csv"%(NAME))
data_reslut = cal_metrcis(np.array(p_real),np.array(p_prediction),np.array(s_real),np.array(s_prediction))
print(data_reslut)
#data_reslut.to_csv("./checkpoints/%s/metrics_reslut.csv"%(NAME))