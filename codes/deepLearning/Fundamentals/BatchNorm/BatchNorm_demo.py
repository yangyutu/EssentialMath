
import torch
import matplotlib.pyplot as plt
import numpy as np

import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
import seaborn as sns
from scipy.stats import norm

sns.set()
SMALL_SIZE = 25
MEDIUM_SIZE = 30
BIGGER_SIZE = 35
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=20)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.subplots_adjust(wspace =0.5, hspace =0.5)
plt.close('all')
# ## Dataset and visualisation
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
trainset = torchvision.datasets.MNIST(root='../../data', train=True, 
                                        download=True, 
                                        transform=transforms.ToTensor())

class Net(nn.Module):
    def __init__(self): 
        super(Net, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(784, 48),  # 28 x 28 = 784
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 10)
        )
             
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class NetBN(nn.Module):
    def __init__(self): 
        super(NetBN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(784, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Linear(24, 10)
        )
             
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = Net()
model_bn = NetBN()

print(model)
print(model_bn)

batch_size = 512
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=0.01)
opt_bn = optim.SGD(model_bn.parameters(), lr=0.01)

loss_arr = []
loss_bn_arr = []

max_epochs = 1

x = np.arange(-5, 5, 0.01)
normalp = norm.pdf(x, 0, 1)

for epoch in range(max_epochs):

    for i, data in enumerate(trainloader, 0):

        inputs, labels = data

        # training steps for normal model
        opt.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        opt.step()
        
        # training steps for bn model
        opt_bn.zero_grad()
        outputs_bn = model_bn(inputs)
        loss_bn = loss_fn(outputs_bn, labels)
        loss_bn.backward()
        opt_bn.step()
        
        loss_arr.append(loss.item())
        loss_bn_arr.append(loss_bn.item())
        
        if i % 50 == 0:
        
            inputs = inputs.view(inputs.size(0), -1)
            
            model.eval()
            model_bn.eval()
            
            # access the output of first linear layer
            a = model.classifier[0](inputs)
            # uncomment the following two layers can access the output of the second layer
            a = model.classifier[1](a)
            a = model.classifier[2](a)
            a = a.detach().numpy().ravel()
            plt.figure(figsize=(9, 8))
            sns.distplot(a, kde=True, color='r', label='W.O. BN') 
            
            # access the batchnormed output of first linear layer 
            b = model_bn.classifier[0](inputs)
            b = model_bn.classifier[1](b)
            # uncomment the following three layers can access the output the second layer after batchnorm
            b = model_bn.classifier[2](b)
            b = model_bn.classifier[3](b)
            b = model_bn.classifier[4](b)
            b = b.detach().numpy().ravel()
            
            sns.distplot(b, kde=True, color='g', label='W. BN') 
            #plt.title('%d: Loss = %0.2f, Loss with bn = %0.2f' % (i, loss.item(), loss_bn.item()))
            
            plt.plot(x, normalp, '--b', label='Std. Norm')
            plt.xlabel('x')
            plt.ylabel('Distribution')
            plt.legend()
            plt.savefig('distribution'+str(i) + '.png', dpi=300)
            plt.show()
            
            
            model.train()
            model_bn.train()
        
        
    print('----------------------')

plt.figure(100, figsize=(9, 8))
plt.plot(loss_arr, 'r', label='W.O. BN')
plt.plot(loss_bn_arr, 'g', label='W. BN')
plt.xlabel('Iter')
plt.ylabel('CE Loss')
plt.legend()
plt.show()
plt.savefig('loss_compare.png', dpi=300)

