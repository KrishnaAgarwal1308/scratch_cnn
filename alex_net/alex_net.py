import torch
import torch.nn as nn
from torch.nn import functional as F

class AlexNet(nn.Module):
    def __init__(self ):
        super().__init__()
        # hyper parameters 
        leaky_relu_alpha = 0.01
        dropout = 0.3

        #defining some required functions.
        self.flattern = nn.Flatten()
        # defining the layers of the network
        self.layer1 = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11, stride=4, padding = 0),
                               nn.BatchNorm2d(96),
                               nn.LeakyReLU(leaky_relu_alpha),
                                nn.MaxPool2d(kernel_size=3, stride = 2, padding = 0)
                               )
        
        self.layer2 = nn.Sequential(nn.Conv2d(96, 256, kernel_size = 5, padding= 2, stride = 1),
                               nn.BatchNorm2d(256),
                               nn.LeakyReLU(leaky_relu_alpha),
                               nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)
                               )
        
        self.layer3 = nn.Sequential(nn.Conv2d(256, 384, kernel_size = 3, padding= 1, stride = 1),
                               nn.BatchNorm2d(384),
                               nn.LeakyReLU(leaky_relu_alpha)
                               )
        
        self.layer4 = nn.Sequential(nn.Conv2d(384, 384, kernel_size = 3, padding= 1, stride = 1),
                               nn.BatchNorm2d(384),
                               nn.LeakyReLU(leaky_relu_alpha)
                               )
        
        self.layer5 = nn.Sequential(nn.Conv2d(384, 256, kernel_size = 3, padding= 1, stride = 1),
                            #    nn.BatchNorm2d(256),
                               nn.LeakyReLU(leaky_relu_alpha),
                               nn.MaxPool2d(kernel_size= 3, stride = 2, padding = 0),
                               nn.Flatten()
                               )
        self.layer6 = nn.Sequential(nn.Linear(256*5*5, 4096),
                               nn.LeakyReLU(leaky_relu_alpha),
                               nn.Dropout(p = dropout)
                               )
        self.layer7 = nn.Sequential(nn.Linear(4096, 4096),
                               nn.LeakyReLU(leaky_relu_alpha),
                               nn.Dropout(p = dropout)
                               )
        self.layer8 = nn.Sequential(nn.Linear(4096, 10))


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        return x