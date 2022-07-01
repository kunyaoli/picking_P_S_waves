from Dataset.dataset import Mydataset
import torch
from torch.optim import lr_scheduler
from model.model_10_18_structure_pool import Lenet_1D
import os
import torch.nn as nn
from torchvision import transforms
from torch.utils import data
import numpy as np

transform = transforms.Compose([
    # transforms.Resize((96, 96)),
    transforms.ToTensor(),
])
alpha = 2.0
NAME = 'discussion_%s'%alpha
if not os.path.exists("./checkpoints/%s"%NAME):
    os.mkdir("./checkpoints/%s"%NAME)

if not os.path.exists("./checkpoints/%s/model"%NAME):
    os.mkdir("./checkpoints/%s/model"%NAME)
print(NAME)
train_Dataset = Mydataset(transform=transform, mode="train")
val_Dataset = Mydataset(transform=transform, mode="val")

BATCH_SIZE = 256

train_dl = data.DataLoader(train_Dataset,batch_size=BATCH_SIZE,shuffle=True,)
#print(next(iter(train_dl)))
val_dl = data.DataLoader(val_Dataset,batch_size=BATCH_SIZE,shuffle=True,)


net = Lenet_1D()
print(net)
if torch.cuda.is_available():
    net.to('cuda')

optimizer = torch.optim.SGD(net.parameters(),lr = 0.01, momentum=0.9, weight_decay = 1e-2)
#optimizer = torch.optim.Adam(net.parameters(),lr = 0.01,  weight_decay = 1e-2)
loss_func = torch.nn.MSELoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


def fit(epoch, model, trainloader, testloader, batchsize = BATCH_SIZE):
    total = 0
    running_loss = 0
    model.train()
    for x, p_value, s_value in trainloader:
         #print(s_value)
         #if np.mean(p_value) and s_value <=
        #print(x.shape)
        if torch.cuda.is_available():
            x, p_value, s_value = (x.to('cuda'),
                                   p_value.to('cuda'), s_value.to('cuda'))
        value = net(x.to(torch.float32))
        #print(value.shape)
        p_pred = value[:, 0]
        #print(p_pred.shape)
        s_pred = value[:, 1]

        loss_p = loss_func(p_pred, p_value)
        loss_s = loss_func(s_pred, s_value)

        loss = loss_p + alpha * loss_s
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            running_loss += loss.item()

    epoch_loss = running_loss
    test_total = 0
    test_running_loss = 0
    scheduler.step()
    model.eval()
    with torch.no_grad():
        for x, p_value, s_value in testloader:
            if torch.cuda.is_available():
                x, p_value, s_value = (x.to('cuda'),p_value.to('cuda'), s_value.to('cuda'))
            value = net(x.to(torch.float32))
            p_pred = value[:, 0]
            s_pred = value[:, 1]

            loss_p = loss_func(p_pred, p_value)
            loss_s = loss_func(s_pred, s_value)
            loss = loss_p + 1.0 * loss_s
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss
    if not epoch % 1:
        torch.save(net.state_dict(), './checkpoints/%s/model/Epoch%d.pth' % (NAME, epoch))
        # torch.save(net.state_dict(), checkpoint_path.format(epoch=epoch))

    print('epoch: ', epoch,
          'loss： ', round(epoch_loss, 5),
          'test_loss： ', round(epoch_test_loss, 5),
          "lr: ", np.float(scheduler.get_last_lr()[0]))
    with open("./checkpoints/%s/train_loss.txt"%(NAME),"a") as f:
        f.write('epoch:%s, train_loss:%s, test_loss:%s,lr:%s\n'%(epoch, round(epoch_loss, 5), round(epoch_test_loss, 5),
                                                                 np.float(scheduler.get_last_lr()[0])))

    return epoch_loss, epoch_test_loss

epochs = 100

train_loss = []
test_loss = []

for epoch in range(epochs):
    epoch_loss, epoch_test_loss = fit(epoch, net, train_dl, val_dl)
    train_loss.append(epoch_loss)
    test_loss.append(epoch_test_loss)