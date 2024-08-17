import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .image_encoder import ImageEncoder
from .location_encoder import LocationEncoder
# from .transformerDecoder import TransformerDecoder
# from .rotate_encoder import RotateEncoder
from .correction_model import RegressionHead
from .misc import load_gps_data, file_dir, generate_heading_tensor

from PIL import Image
from torchvision.transforms import ToPILImage
from tqdm import tqdm

class GeoCLIP(nn.Module):
    def __init__(self, from_pretrained=False, queue_size=4096,batch_size=32, num_decoder_layers=6, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 /0.07))
        self.image_encoder = ImageEncoder()
        self.location_encoder = LocationEncoder()
        # self.regression_head = RegressionHead()
        # self.rotate_encoder = RotateEncoder()
        # self.transformer_decoder = TransformerDecoder(num_decoder_layers, d_model, nhead, dim_feedforward, dropout)
        # self.fusion_layer = nn.Linear(d_model * 2, d_model)
        # self.output_projection = nn.Linear(d_model, 3)
        self.batch_size = batch_size  # 添加批处理大小属性
        self.gps_gallery = load_gps_data(os.path.join(file_dir, "gps_gallery", "all_coordinates.csv"))
        self.rotate_gallery = generate_heading_tensor()
        self._initialize_gps_queue(queue_size)
        self.gps_features_dim = 512  # 根据实际输入数据的形状设置这个值
        
    
        if from_pretrained:
            self.weights_folder = os.path.join(file_dir, "weights")
            self._load_weights()

        self.device = "cpu"

    def to(self, device):
        self.device = device
        self.image_encoder.to(device)
        self.location_encoder.to(device)
        # self.transformer_decoder.to(device)
        # self.fusion_layer.to(device)
        # self.output_projection.to(device)
        # self.regression_head.to(device)
        # self.rotate_encoder.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
        return super().to(device)

    def _load_weights(self):
        self.image_encoder.mlp.load_state_dict(torch.load(f"{self.weights_folder}/image_encoder_mlp_weights.pth"))
        self.location_encoder.load_state_dict(torch.load(f"{self.weights_folder}/location_encoder_weights.pth"))
        self.logit_scale = nn.Parameter(torch.load(f"{self.weights_folder}/logit_scale_weights.pth"))

    def _initialize_gps_queue(self, queue_size):
        self.queue_size = queue_size
        self.register_buffer("gps_queue", torch.randn(3, self.queue_size))
        self.gps_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.long))
        print(self.queue_size, self.batch_size)
        assert self.queue_size % self.batch_size == 0

    @torch.no_grad()
    def dequeue_and_enqueue(self, gps):
        """ Update GPS queue

        Args:
            gps (torch.Tensor): GPS tensor of shape (batch_size, 2)
        """
        gps_batch_size = gps.shape[0]
        gps_ptr = int(self.gps_queue_ptr)
        
        assert self.queue_size % gps_batch_size == 0, f"Queue size {self.queue_size} should be divisible by batch size {gps_batch_size}"

        # Replace the GPS from ptr to ptr+gps_batch_size (dequeue and enqueue)
        self.gps_queue[:, gps_ptr:gps_ptr + gps_batch_size] = gps.t()
        gps_ptr = (gps_ptr + gps_batch_size) % self.queue_size  # move pointer
        self.gps_queue_ptr[0] = gps_ptr

    def get_gps_queue(self):
        return self.gps_queue.t()
                                             
    def forward(self, image, location):
        """ GeoCLIP's forward pass

        Args:
            image (torch.Tensor): Image tensor of shape (n, 3, 224, 224)
            location (torch.Tensor): GPS location tensor of shape (m, 2)
        """

        # Compute Features
        
        image_features = self.image_encoder(image)
        location_features = self.location_encoder(location)
        # heading_features = self.rotate_encoder(heading)
        logit_scale = self.logit_scale.exp()
        
        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        location_features = F.normalize(location_features, dim=1)

        # Cosine similarity (Image Features & Location Features)
        logits_per_image = logit_scale * (image_features @ location_features.t())
        return logits_per_image
        
        
        # fused_features = torch.cat([image_features, location_features], dim=-1)
        # fused_features = self.fusion_layer(fused_features)
        # print(fused_features.shape)
        # tgt_mask = torch.ones_like(fused_features).unsqueeze(1).unsqueeze(2)
        # print(tgt_mask.shape)
        # memory_mask = torch.zeros_like(tgt_mask)
        # print(memory_mask.shape)
        
        # decoder_output = self.transformer_decoder(fused_features, fused_features, tgt_mask=tgt_mask, memory_mask=memory_mask)
        # regression_output = self.output_projection(decoder_output)
        # return regression_output

        
    
    @torch.no_grad()
    def predict(self, image_path, top_k):
        """ Given an image, predict the top k GPS coordinates

        Args:
            image_path (str): Path to the image
            top_k (int): Number of top predictions to return
        """
        image = Image.open(image_path).convert('RGB')
        
        image = self.image_encoder.preprocess_image(image)
        image = image.to(self.device)
        gps_size = self.gps_gallery.shape[0]
        heading_size = self.rotate_gallery.shape[0]

        # Repeat each row of gps_gallery for heading_size times
        gps_batch = self.gps_gallery.unsqueeze(1).repeat(1, heading_size, 1).view(-1, self.gps_gallery.shape[1])
        heading_batch = self.rotate_gallery.unsqueeze(0).repeat(gps_size, 1).unsqueeze(2).view(-1, 1)

        # Combine gps_batch and rotate_batch along the last dimension
        combined_data = torch.cat((gps_batch, heading_batch), dim=1).to(self.device)
        # print("Length of combined_data:", combined_data.size(0))
        # dataset = TensorDataset(combined_data)
        # dataloader = DataLoader(dataset, batch_size=667440)  # Adjust batch_size according to your memory capacity
        # top_pred_gps = []
        # for batch in dataloader:
        #     batch = batch[0].to(self.device)
        #     top_pred_gps.append(self.forward(image, batch))
        # top_pred_gps = torch.cat(top_pred_gps)

        logits_per_image = self.forward(image, combined_data)
        probs_per_image = logits_per_image.softmax(dim=-1).cpu()
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        top_pred_gps = combined_data[top_pred.indices[0]]
        return top_pred_gps
    
