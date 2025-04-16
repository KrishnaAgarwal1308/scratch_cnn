import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from loading_data import trainloader, testloader, classes
# import torch.jit

class Cnn(nn.Module):
    '''making out the constructor for the class 
    the weights is made in teh dimension of the output and the input and kernel size thus is a 4d tensor which we have to work with.'''
    def __init__(self, out_channel, inp_shape, kernel_size, stride=1, padding=0):
        super().__init__()
        self.channesl = inp_shape[0]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_shape = inp_shape
        self.out_channel = out_channel

        self.weight = nn.Parameter(torch.randn(out_channel, self.channesl, kernel_size, kernel_size)* torch.sqrt(torch.tensor(2.0 / (self.channesl * kernel_size * kernel_size))))
        self.bias = nn.Parameter(torch.zeros(out_channel))

    """defining the forward passs for the convolution layer in the CNN"""
    def forward(self, X):
        batch_size = X.shape[0]

        h_input = self.input_shape[1]
        w_input = self.input_shape[2]

        h_out = (h_input + 2 * self.padding - self.kernel_size ) // self.stride + 1
        w_out = (w_input + 2 * self.padding - self.kernel_size ) // self.stride + 1

        out_shape = (batch_size, self.out_channel, h_out, w_out)
        out = torch.zeros(out_shape)

        if self.padding > 0:
            padded_input = F.pad(X, (self.padding, self.padding, self.padding, self.padding))
        else:
            padded_input = X

        for i in range(batch_size):
            for j in range (self.out_channel):
                for k in range(h_out):
                    h_start = k * self.stride
                    h_end = h_start + self.kernel_size
                    for l in range(w_out):
                        w_start = l * self.stride
                        w_end = w_start + self.kernel_size
                        out[i, j, k, l] = torch.sum(padded_input[i, :, h_start:h_end, w_start:w_end] * self.weight[j]) + self.bias[j]
        self.output = out
        return out       
    

# defining the activation function for the cnn layers        
class LeakyRelu(nn.Module):
    def __init__(self, alpha = 0.01):
        super().__init__()
        self.alpha = alpha 
    def forward(self, X):
        return torch.where(X > 0, X, self.alpha * X ) 
    
# defining the pooling layer for the network
class PoolingLayer(nn.Module):
    def __init__(self, kernel_size = 2, stride = 2, padding = [0, 0]):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding


    def forward(self, X):
        padding = self.padding
        kernel_size = self.kernel_size
        stride = self.stride
        batch_size, channels, h_out, w_out = X.shape
        h_pool = (h_out + 2 * padding[0] - kernel_size) // stride + 1
        w_pool = (w_out + 2 * padding[1] - kernel_size) // stride + 1
        out_shape = (batch_size, channels, h_pool, w_pool)

        out = torch.zeros(out_shape)
        if(padding[0] > 0 or padding[1] > 0):
            padded_output = F.pad(X, (padding[1], padding[1], padding[0], padding[0]))
        else:
            padded_output = X

        for i in range (batch_size):
            for j in range (channels):
                for k in range (h_pool):
                    h_start = k*stride
                    h_end = h_start + kernel_size
                    for l in range(w_pool):
                        w_start = l*stride
                        w_end = w_start + kernel_size
                        out[i, j, k, l] = torch.max(padded_output[i, j, h_start:h_end, w_start:w_end])
        self.output = out
        return out

class Ffw(nn.Module):
    def __init__(self, inp_shape, out_shape):
        super().__init__()
        self.inp_shape = inp_shape
        self.out_shape = out_shape
        self.weight = nn.Parameter(torch.randn(inp_shape, out_shape) * torch.sqrt(torch.tensor(2.0 / (inp_shape))))
        self.bias = nn.Parameter(torch.zeros(out_shape))
    def forward(self, x):
        out = x @ self.weight + self.bias
        return out

class SimpleCnn(nn.Module):
    def __init__(self, inp_shape, out_channel, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv1 = Cnn(out_channel, inp_shape, kernel_size, stride, padding)
        self.relu = LeakyRelu()
        self.pool = PoolingLayer(kernel_size=2, stride=2)
        h_in, w_in = inp_shape[1], inp_shape[2]
        h_conv = (h_in + 2 * padding - kernel_size) // stride + 1
        w_conv = (w_in + 2 * padding - kernel_size) // stride + 1

        # Determine spatial size after pooling. Here pooling uses kernel=2, stride=2 with no extra padding.
        h_pool = (h_conv - 2) // 2 + 1
        w_pool = (w_conv - 2) // 2 + 1
        conv2_inp_shape = (out_channel, h_pool, w_pool)
        self.conv2 = Cnn(out_channel, conv2_inp_shape, kernel_size, stride, padding)
        
        # After second conv, relu and pooling:
        # Recalculate sizes:
        h_conv2 = (h_pool + 2 * padding - kernel_size) // stride + 1
        w_conv2 = (w_pool + 2 * padding - kernel_size) // stride + 1
        h_pool2 = (h_conv2 - 2) // 2 + 1
        w_pool2 = (w_conv2 - 2) // 2 + 1

        # Compute flattened size (channels * height * width):
        flattened_dim = out_channel * h_pool2 * w_pool2
        self.fc = Ffw(flattened_dim, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        #giving out the raw logits as the output of the net.
        # x = F.softmax(x, dim=1)
        return x
    
if __name__ == '__main__':
    inp_shape = (3, 32, 32)  
    out_channel = 8        # number of output channels
    kernel_size = 3        # size of the convolution kernel
    stride = 2             # convolution stride
    padding = 1            # padding to preserve spatial dimensions

    # Instantiate the custom CNN layer.
    net = SimpleCnn(inp_shape, out_channel, kernel_size, stride, padding)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.001)

    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0

    print('Finished Training')

# inp_shape = (3, 32, 32)  
# out_channel = 8        # number of output channels
# kernel_size = 3        # size of the convolution kernel
# stride = 2             # convolution stride
# padding = 1            # padding to preserve spatial dimensions

# # Instantiate the custom CNN layer.
# net = SimpleCnn(inp_shape, out_channel, kernel_size, stride, padding)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# for epoch in range(2):  # loop over the dataset multiple times

#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#             running_loss = 0.0

# print('Finished Training')

# # Create a dummy input tensor with a batch size of 10.
# x = torch.randn(10, *inp_shape)

# # Run the forward pass.
# output = model(x)

# # Print the shape of the output tensor.
# print("Output shape:", output.shape)

# model = Cnn(out_channel, inp_shape, kernel_size, stride, padding)

# # Create a dummy input tensor with a batch size of 10.
# x = torch.randn(10, *inp_shape)

# # Run the forward pass.
# output = model(x)
# print("Output shape:", output.shape)