import torch
import torch.nn as nn
import torch.nn.functional as F

# making out the patches to be used by the transformer block 
class ImagePatch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, image_size = 32):
        super().__init__()
        self.patch_size = kernel_size
        self.number_of_patches = (image_size//self.patch_size)**2 # since h,w are of same size here thus by h*w/ pattch_size**2 we get the number of patches
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size= kernel_size, stride=kernel_size)

    def forward(self,x):
        x = self.projection(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x


# making out the transformer block     
class TransformerBlock(nn.Module):
    
    def __init__(self, in_channels, num_heads, dropout = .1):
        super().__init__()
        ''' the encoder block contains the multihead attention and ffw network all in 
        we have used gelu to make the model more robust and dropout to prevent overfitting '''

        self.transformer_enc = nn.TransformerEncoderLayer(d_model = in_channels, nhead = num_heads, dropout = dropout, activation = 'gelu')

    def forward(self, x):
        x = x.permute(1, 0, 2) # [sequence_len, batch, dim]
        x = self.transformer_enc(x)
        x = x.permute(1, 0, 2) # [batch, sequence_len, dim]
        return x
    
class SimpleViT(nn.Module):
    def __init__(self, in_channels, num_classes, image_size = 32, patch_size = 4, num_heads = 4, num_layers = 6, dropout = .1, out_channels = 64):
        super().__init__()
        self.patch_embedding = ImagePatch(in_channels, kernel_size=patch_size, image_size=image_size, out_channels= out_channels)
        self.position_embedding = nn.Parameter(torch.randn(1,1,out_channels))
        self.transformer_blocks = nn.ModuleList([TransformerBlock(out_channels, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.classification = nn.Parameter(torch.randn(1, 1, out_channels))
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self,x):
        x = self.patch_embedding(x)
        x = x + self.position_embedding
        B, N, C = x.shape
        class_token = self.classification.expand(B, -1, -1) # expand the class token to the batch size
        x = torch.cat((class_token, x), dim=1) 
        for transformer in self.transformer_blocks:
            x = transformer(x)
        x = x[:, 0]
        x = self.classifier(x)
        return x
    

    
