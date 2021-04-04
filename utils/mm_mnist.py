import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

class MM_MNIST:
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.mnist = torchvision.datasets.MNIST(root, train, transform, target_transform, download)

    def __getitem__(self, *args, **kwargs):
        data, label = self.mnist.__getitem__(*args, **kwargs)
        modalities = [data[:,0:14,0:14],data[:,0:14,14:28],data[:,14:28,0:14],data[:,14:28,14:28]]
        return (modalities, label)

    def __len__(self):
        return len(self.mnist)

class MM_Classifier(nn.Module):
    def __init__(self, num_modalities, modality_shape, head_type='mlp', dropout=0.5, moddrop=0.1, activation=F.relu):
        super().__init__()
        self.dropout=dropout
        self.moddrop=moddrop
        self.activation = activation
        if head_type == 'mlp':
            self.heads = nn.ModuleList([MLP(dropout=dropout, activation=activation) for i in range(num_modalities)])
            self.fc1 = nn.Linear(500, 40)
        else: 
            self.heads = [CNN() for i in range(num_modalities)]
            self.fc1 = nn.Linear(num_modalities*3*3*32, 256)
        self.fc2 = nn.Linear(40, 10)

    def forward(self, x, heads=False, tails=False):
        if heads:
            return self.forward_head(x)
        elif tails:
            return self.forward_tail(x)
        else:
            x = self.forward_head(x)
            return self.forward_tail(x)

    def forward_head(self, x):
        return [model(data) for model, data in zip(self.heads, x)]
        
    def forward_tail(self, x):
        if self.training:
            to_drop = torch.rand((x[0].shape[0],len(x)))<self.moddrop
            newx = [mod.clone() for mod in x]
            for i in range(len(newx)):
                newx[i][to_drop[:,i],:] = 0
            x=newx
        x = torch.cat(x, dim=1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        # self.conv3 = nn.Conv2d(32,64, kernel_size=3)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x.reshape((-1,1,x.shape[-2],x.shape[-1]))))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        # x = F.relu(F.max_pool2d(self.conv3(x),2))
        # x = F.dropout(x, p=0.5, training=self.training)
        return x.view(-1,3*3*32 )

class MLP(nn.Module):
    def __init__(self, dropout=0.5, activation=F.relu):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.fc1 = nn.Linear(14*14, 125)

    def forward(self, x):
        x = F.dropout(x.reshape((-1, 14*14)), p=0.2, training=self.training)
        x = self.activation(self.fc1(x))
        return x