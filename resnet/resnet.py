import torch
import torch.nn as nn
import torch.nn.functional as F


# defining the basic building block of the resnet.
'''this block shows the residual connection between the layers.
    The input is passed through the first convolutional layer and then through the second convolutional layer.'''
class Block(nn.Module):
    def __init__ (self, in_channels, out_channels, stride = 1, ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias= False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size= 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        out = self.block(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Resnet (nn.Module):
    def __init__(self, num_classes = 10, ):
        super().__init__()

        # defining the first layer of the network.
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size= 7, stride = 2, padding = 3),# note the padding is 3 due to kernel size being 7 thus to make the output size exactly half of the input size we need to add 3 padding to the input.
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        # defining other layers in the network.
        self.layer2 = nn.Sequential(Block(64, 64),
                                    Block(64, 64),)
        self.layer3 = nn.Sequential(Block(64, 128, stride = 2),
                                    Block(128, 128),)
        self.layer4 = nn.Sequential(Block(128, 256, stride = 2),
                                    Block(256, 256),)
        self.layer5 = nn.Sequential(Block(256, 512, stride = 2),
                                    Block(512, 512),)
        self.layer6 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), # this layer is used to make the output size of the network to be 1x1.
                                    nn.Flatten(),
                                    nn.Linear(512, num_classes))
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x