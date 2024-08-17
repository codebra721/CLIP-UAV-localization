import torch
import torch.nn as nn
from .rff import GaussianEncoding
from .misc import file_dir


class RotateEncoderCapsule(nn.Module):
    def __init__(self, sigma, dropout_rate=0.3):
        super(RotateEncoderCapsule, self).__init__()
        # if you want to chenge input gps , you need to chenge here input_size
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
        self.km = sigma
        self.capsule = nn.Sequential(rff_encoding,
                                     nn.Linear(512, 1024),
                                     nn.ReLU(),
                                    #  nn.Linear(1024, 1024),
                                    #  nn.ReLU(),
                                    
                                     nn.Linear(1024, 1024),
                                     nn.ReLU()
        )
        self.head = nn.Sequential( 
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            # nn.Dropout(dropout_rate),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.capsule(x)
        
        x = self.head(x)
        return x

class RotateEncoder(nn.Module):
    def __init__(self, sigma=[2**0, 2**4, 2**8], from_pretrained=False):
        super(RotateEncoder, self).__init__()
        self.sigma = sigma
        self.n = len(self.sigma)

        for i, s in enumerate(self.sigma):
            self.add_module('RotEnc' + str(i), RotateEncoderCapsule(sigma=s))

    def forward(self, rotation):
        # 轉換為弧度並計算其sin和cos
        rotation_rad = torch.deg2rad(rotation.view(-1, 1))
        rotation_sin_cos = torch.cat((torch.sin(rotation_rad), torch.cos(rotation_rad)), dim=1)
        rotation_features = torch.zeros(rotation_sin_cos.shape[0], 512).to(rotation_sin_cos.device)   
        for i in range(self.n):
            rotation_features += self._modules['RotEnc' + str(i)](rotation_sin_cos)

        return rotation_features