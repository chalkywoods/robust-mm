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
        self.dropout = dropout
        self.moddrop = moddrop
        self.activation = activation
        self.head_type = head_type
        if head_type == 'mlp':
            self.heads = nn.ModuleList([MLP(dropout=dropout, activation=activation) for i in range(num_modalities)])
            self.fc1 = nn.Linear(500, 40)
            self.fc2 = nn.Linear(40, 10)
        elif head_type == 'cnn': 
            self.heads = nn.ModuleList([CNN(dropout=dropout, activation=activation) for i in range(num_modalities)])
            self.conv1 = nn.Conv2d(32, 64, kernel_size=5)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
            self.fc1 = nn.Linear(64, 10)
        else:
            raise ValueError('Invalid head_type {}'.format(head_type))

    def forward(self, x, heads=False, tails=False):
        if heads:
            return [mod.reshape((-1, 32*4*4)) for mod in self.forward_head(x)]   # Flatten for intermediate output
        elif tails:
            return self.forward_tail([mod.reshape((-1, 32, 4, 4)) for mod in x]) # Rebuild original shape 
        else:
            x = self.forward_head(x)
            return self.forward_tail(x)

    def forward_head(self, x):
        return [model(data) for model, data in zip(self.heads, x)]
        
    def forward_tail(self, x):
        if self.training and self.moddrop > 0:
            to_drop = torch.rand((x[0].shape[0],len(x)))<self.moddrop
            newx = [mod.clone() for mod in x]
            for i in range(len(newx)):
                newx[i][to_drop[:,i],:] = 0
            x=newx
        if self.head_type == 'mlp':
            x = torch.cat(x, dim=1)
            x = self.activation(self.fc1(x))
            x = self.fc2(x)
        elif self.head_type == 'cnn':
            moddim = x[0].shape[2]
            recon = torch.zeros((x[0].shape[0], x[0].shape[1], moddim*2, moddim*2))
            recon[:,:,0:moddim,0:moddim] = x[0]
            recon[:,:,moddim:moddim*2,0:moddim] = x[1]
            recon[:,:,0:moddim,moddim:moddim*2] = x[2]
            recon[:,:,moddim:moddim*2,moddim:moddim*2] = x[3]
            x = recon
            x=self.activation(self.conv1(x))
            x=self.activation(F.max_pool2d(self.conv2(x), 2))
            x=self.fc1(x.reshape((-1,1,64)))
        return F.log_softmax(x.reshape((-1, 10)), dim=1)

            

class CNN(nn.Module):
    def __init__(self, dropout=0.2, activation=F.relu):
        super(CNN, self).__init__()
        self.dropout=dropout
        self.activation=activation
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.conv1(x.reshape((-1,1,x.shape[-2],x.shape[-1]))))
        #x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(F.max_pool2d(self.conv2(x), 2))
        #x = F.dropout(x, p=self.dropout, training=self.training)
        # x = F.relu(F.max_pool2d(self.conv3(x),2))
        # x = F.dropout(x, p=0.5, training=self.training)
        return x#.view(-1,4*4*32 )

class MLP(nn.Module):
    def __init__(self, dropout=0.5, activation=F.relu):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.fc1 = nn.Linear(14*14, 125)

    def forward(self, x):
        x = F.dropout(x.reshape((-1, 14*14)), p=self.dropout, training=self.training)
        x = self.activation(self.fc1(x))
        return x