import torch
import torch.nn as nn
from transformers import CLIPModel, AutoProcessor
from torch.nn import MultiheadAttention
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub.*')

    
class ImageEncoder(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ImageEncoder, self).__init__()
        self.CLIP = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
        self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        self.self_attention = MultiheadAttention(embed_dim=768, num_heads=8)
        self.mlp = nn.Sequential(nn.Linear(768, 768),
                                #  nn.Dropout(dropout_rate),
                                 nn.ReLU(),
                                #  nn.Linear(768, 768),
                                #  nn.ReLU(),
                                #  nn.Dropout(dropout_rate),
                                 nn.Linear(768, 512),
                                 nn.ReLU())

        # Freeze CLIP
        for param in self.CLIP.parameters():
            param.requires_grad = False

    def preprocess_image(self, image):
        x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return x

    def forward(self, x):
        x = self.CLIP.get_image_features(pixel_values=x)
        x = x.unsqueeze(1)
        x = x.permute(1, 0, 2)  # change to seq_len, batch, embed_dim for self-attention
        x, _ = self.self_attention(x, x, x)  # self-attention
        x = x.permute(1, 0, 2)  # change back to batch, seq_len, embed_dim
                # 添加后续处理层
        x = F.relu(x)  # ReLU 激活函数
        x = F.dropout(x, p=0.3, training=self.training)  # Dropout 层
        
        x = x.mean(dim=1)
        x = self.mlp(x)
        return x