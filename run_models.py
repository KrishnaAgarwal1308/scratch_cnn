from alex_net.alex_net import AlexNet
from loading_data import trainloader, testloader, classes, image_size
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
# from cnn_using_modules import CNNNetwrok
import matplotlib.pyplot as plt
import numpy as np
from simple_vit.scratch_vit import SimpleViT


network_to_use = 'AlexNet' # 'CNNNetwrok' or 'AlexNet' # choose the network to use
network_mapping = {
    'AlexNet': AlexNet,
    'SimpleViT': SimpleViT,
    }
device = 'cuda' if torch.cuda.is_available() else 'cpu'




# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



if __name__ == '__main__':
# defining the model
    '''net = network_mapping[network_to_use]( in_channels=3,
    num_classes=10,
    image_size=image_size,
    patch_size=4,      # Optional: change if you're using a different patch size
    num_heads=4,       # Optional: adjust based on your transformer block
    num_layers=3,      # Optional: number of transformer layers
    dropout=0.1,
    out_channels=64)'''
    net = network_mapping[network_to_use]()

    net.to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.001) #learning rate of .0001 tends to be more stable here
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0
    print('Finished Training')
    PATH = './alex_net/alex_net.pth'
    torch.save(net.state_dict(), PATH)


dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
images = images.to(device)
'''net = network_mapping[network_to_use]( in_channels=3,
    num_classes=10,
    image_size=image_size,
    patch_size=4,      # Optional: change if you're using a different patch size
    num_heads=4,       # Optional: adjust based on your transformer block
    num_layers=3,      # Optional: number of transformer layers
    dropout=0.1,
    out_channels=64)'''
net = network_mapping[network_to_use]()

net.load_state_dict(torch.load(PATH)) 
net.to(device)
net.eval()

# testing how good is the model
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')




