#2020-4-18 at Office
import numpy as np
import pandas as pd
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import scipy as sp
import scipy.ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import time
import gc
import datetime



### for testing
uv_data = pd.read_fwf("/media/joshua/Milano/joshua_mock_uv/new_obs/Sa0dump_00001810_144.967_149.854_40_6.52866e+09_6.18375e+29_-7.11229_7.84132.h5.txt", skiprows=22)


#print("uv_data", uv_data)
#
# print("UV", uv_data[['U (lambda)', 'V (lambda)']])
# #print("V", uv_data['V (lambda)'])
#
# plt.scatter(uv_data['U (lambda)'], uv_data['V (lambda)'])
# plt.title("uv coverage")
# plt.show()

print(uv_data['Iamp (Jy)'], uv_data['Iphase(d)'])
#

class MLP(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(MLP, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h)#, self.fc32(h) # mu, log_var

    def forward(self, x):
        output = self.encoder(x)
        return(output)


#### mlp
mlp = MLP(x_dim=len(uv_data['Iamp (Jy)']), h_dim1= 512, h_dim2=256, z_dim=1)

if torch.cuda.is_available():
    mlp.cuda()


### for training (experimenting)
print(mlp)

#print(len(uv_data['Iamp (Jy)']))

x_data = np.array(uv_data['Iamp (Jy)'])
x_data = torch.from_numpy(x_data).float().cuda().unsqueeze(0)
output = mlp(x_data)

print("output", output)
print("test")
###
