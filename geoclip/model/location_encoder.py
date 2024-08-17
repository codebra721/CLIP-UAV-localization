import torch
import torch.nn as nn
from .rff import GaussianEncoding
from .misc import file_dir

# Constants
A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
SF = 66.50336

def equal_earth_projection(L):
    latitude = L[:, 0]
    longitude = L[:, 1]
    heading = L[:, 2]
    latitude_rad = torch.deg2rad(latitude)
    longitude_rad = torch.deg2rad(longitude)
    sin_theta = (torch.sqrt(torch.tensor(3.0)) / 2) * torch.sin(latitude_rad)
    theta = torch.asin(sin_theta)
    denominator = 3 * (9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1)
    x = (2 * torch.sqrt(torch.tensor(3.0)) * longitude_rad * torch.cos(theta)) / denominator
    y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta
    result = (torch.stack((x, y), dim=1) * SF) / 180
    heading_rad = torch.deg2rad(heading)
    sin_heading = ((torch.sin(heading_rad)/2)+0.5)*0.0001
    cos_heading = ((torch.cos(heading_rad)/2)+0.5)*0.0001
    result = torch.cat((result, sin_heading.unsqueeze(1), cos_heading.unsqueeze(1)), dim=1)
    # print(result[0])
    return result

class Swish(nn.Module):
    def forward(self, input_tensor):
        return input_tensor * torch.sigmoid(input_tensor)
    
class LocationEncoderCapsule(nn.Module):
    def __init__(self, sigma, dropout_rate=0.3):
        super(LocationEncoderCapsule, self).__init__()
        # if you want to chenge input gps , you need to chenge here input_size
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=4, encoded_size=256)
        self.km = sigma
        self.capsule = nn.Sequential(rff_encoding,
                                     nn.Linear(512, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU())
        self.head = nn.Sequential(nn.Linear(1024, 512),
                                  nn.ReLU())

    def forward(self, x):
        x = self.capsule(x)
        
        x = self.head(x)
        return x

class LocationEncoder(nn.Module):
    def __init__(self, sigma=[2**0, 2**4, 2**8], from_pretrained=False):
        super(LocationEncoder, self).__init__()
        self.sigma = sigma
        self.n = len(self.sigma)

        for i, s in enumerate(self.sigma):
            self.add_module('LocEnc' + str(i), LocationEncoderCapsule(sigma=s))

        if from_pretrained:
            self._load_weights()

    def _load_weights(self):
        self.load_state_dict(torch.load(f"{file_dir}/weights/location_encoder_weights.pth"))

    def forward(self, location):
        location = equal_earth_projection(location)
        # print(location[0])
        location_features = torch.zeros(location.shape[0], 512).to(location.device)   
        for i in range(self.n):
            location_features += self._modules['LocEnc' + str(i)](location)

        return location_features
    