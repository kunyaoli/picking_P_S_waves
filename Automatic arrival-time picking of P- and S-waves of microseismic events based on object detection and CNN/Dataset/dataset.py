from torch.utils import data
import pandas as pd
from torchvision import transforms
import numpy as np
import glob


class Mydataset(data.Dataset):
    def __init__(self, transform=None, mode="train"):
        self.all_data = pd.read_csv(r'/home/lky/code/ps_regression/06_1D_cnn/1d_cnn_nromdata/data/Data_%s_01.csv' % (mode))
        self.transforms = transform
        self.mode = mode

    def __getitem__(self, index):
        # load value
        #value_path = self.all_value_path[index]
        #print(index)
        value_data = np.array(self.all_data.iloc[index,1:601], dtype=np.float32) #.type(torch.float32)
        #print(index)
        if len(value_data) >= 600:
            value_data = value_data[0:600]
        else:
            print("waring")
            value_data = np.pad(value_data, (0, 600 - len(value_data)), 'constant')

        # load label
        P_value =np.array(self.all_data.iloc[index,601], dtype=np.float32)

        S_value = np.array(self.all_data.iloc[index,602], dtype=np.float32)
        return value_data, P_value, S_value

    def __len__(self):
        return len(self.all_data)

if __name__ == '__main__':

    transform = transforms.Compose([
        # transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])

    train_Dataset = Mydataset(transform=transform, mode="val")
    print(len(train_Dataset))
    BATCH_SIZE = 10
    test_dl = data.DataLoader(
        train_Dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    print(len(test_dl))
    #print(test_dl[0].shape)
    #print(next(iter(test_dl))[0].reshape(-1,1,600)[1,:,:])  # batch size, features
    print(next(iter(test_dl)))

