# writing the image sampler or cnn backbone 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

class PositionalEncoding(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        x_embed = (torch.arange(width).unsqueeze(0).repeat(height, 1) / (width - 1)) * 2 * math.pi 
        y_embed = (torch.arange(height).unsqueeze(1).repeat(1, width) / (height - 1)) * 2 * math.pi
        
        dims = torch.arange(channels // 4).float()
        dims = 10000 ** (2 * (dims//2)/(channels//2))

        x_pos = x_embed[:, :, None] / dims
        y_pos = y_embed[:, :, None] / dims

        x_pos = torch.stack((x_pos.sin(), x_pos.cos()), dim = 3).view(height, width, -1) # [height, width, channels//2]
        y_pos = torch.stack((y_pos.sin(), y_pos.cos()), dim = 3).view(height, width, -1) # [height, width, channels//2]

        pos = torch.cat((x_pos, y_pos), dim = 2) # [height, width, channels]
        self.register_buffer('pos_emb', pos.permute(2, 0, 1).unsqueeze(0)) # [1, channels, height, width]
    def forward(self, x):
        return x + self.pos_emb

        

class Backbone(nn.Module): # for efficiency purpose we are going to use the predefined resent model as defined in torch
    def __init__(self, in_channels = 3, out_channels = 2048, image_size = 224, kernel_size = 3, stride = 2):
        super().__init__()
        self.resnet = models.resnet50() # since we are going to train the model we wont be using the previous weights 
        self.out = nn.Sequential(*list(self.resnet.children())[:-2])
         

    def forward(self, x):
        x = self.out(x)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_encoder_layers)
    def forward(self, x):
        return self.transformer_encoder(x)

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_decoder_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_decoder_layers)
    def forward(self, tgt , memory):
        return self.transformer_decoder(tgt, memory)
    

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers = 3):
        super().__init__()
        layer = []
        for i in range(num_layers):
            layer.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim if i<num_layers-1 else out_dim))
        self.layers = nn.ModuleList(layer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


class DETR(nn.Module):
    def __init__(self, num_classes = 10, num_queries = 100, hidden_dim =256, nheads = 8, num_encoder_layer = 6, num_decoder_layer = 6):
        super().__init__()
        self.backbone = Backbone()
        backbone_dim = 2048
        # projecting the output of the bakcbone to dimension taken by transformer
        self.project = nn.Conv2d(backbone_dim, hidden_dim, kernel_size = 1)
        self.transformer_encoder = TransformerEncoder(hidden_dim, nheads, num_encoder_layer, hidden_dim * 4)
        self.transformer_decoder = TransformerDecoder(hidden_dim, nheads, num_decoder_layer, hidden_dim * 4)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, 4, hidden_dim, num_layers = 3)
        self.positional_encoding = nn.ModuleDict()


    def get_pos_enc (self, channel, height, width):
        key = f"{channel}_{height}_{width}"
        if key not in self.positional_encoding:
            self.positional_encoding[key] = PositionalEncoding(channel, height, width)
        return self.positional_encoding[key]

    def forward(self, x):
        # backbone
        conv_proj = self.backbone(x) # [batch, channels, height, width]
        # converting to lower dimension
        conv_proj = self.project(conv_proj) # [batch, hidden_dim, height, width]
        # adding positional encoding
        batch, channels, height, width = conv_proj.shape
        # if self.positional_encoding is None or self.positional_encoding.shape[-2:]!=(height,width):
        positional_encoding = self.get_pos_enc(channels, height, width)
        # self.positional_encoding = PositionalEncoding(channels, height, width)
        conv_proj = positional_encoding(conv_proj)
        # flattening the image
        batch, channels, height, width = conv_proj.shape
        conv_proj = conv_proj.flatten(2).permute(2, 0, 1)
        # [batch, height * width, hidden_dim]
        # creating the queries
        queries = self.query_embed.weight
        tgt = torch.zeros_like(queries).unsqueeze(1).repeat(1, batch, 1)
        # [batch, num_queries, hidden_dim]
        # passing through the transformer
        memory = self.transformer_encoder(conv_proj)
        # [batch, height * width, hidden_dim]
        hs = self.transformer_decoder(tgt, memory) # [batch, num_queries, hidden_dim]
        # class and bbox prediction
        hs = hs.permute(1, 0, 2)
        outputs_class = self.class_embed(hs)
        outputs_bbox = self.bbox_embed(hs).sigmoid() # [batch, num_queries, 4]
        # outputs_bbox = outputs_bbox * torch.tensor([width, height, width, height]).unsqueeze(0).unsqueeze(0).to(outputs_bbox.device)
        outputs = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_bbox[-1]}
        return outputs



if __name__ == "__main__":
    # Simple sanity check
    model = DETR(num_classes=20, num_queries=100)
    
    # Create a dummy batch of 2 RGB images of size 224x224
    dummy_input = torch.randn(2, 3, 224, 224)
    
    outputs = model(dummy_input)
    print("Logits shape:", outputs['pred_logits'].shape)  # Expected: [2, 100, 21]
    print("Boxes shape:", outputs['pred_boxes'].shape)    

    print(outputs['pred_boxes'][-1])