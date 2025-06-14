import torch

# print(torch.linspace(0, 1, 32).unsqueeze(0))
D = 32
dim = torch.arange(D//4).float()
# dim = 10000 ** (2*(dim//2))
dim = 10000 ** 2*(dim//2)/(D//2)
# print(dim)
dim = dim.unsqueeze(1).unsqueeze(0)
print(dim.expand(32, 32, 1))

# x = torch.arange(32).unsqueeze(0).repeat(32,1)/(31)
# print(x[:,:,None].shape)
# x = x * 2 * torch.pi
# x = x[:,:, None] / dim
# print(x)
# print((torch.arange(32).unsqueeze(0)/(32-1)).repeat(32,1))