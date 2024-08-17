import torch
import torch.nn as nn

class RegressionHead(nn.Module):
    def __init__(self, output_dim=3, dropout_rate=0.3):
        super(RegressionHead, self).__init__()
        # self.conv1 = nn.Conv1d(1, 64, kernel_size=1)
        self.layers = nn.Sequential(
            nn.Linear(512, 128),
            # nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),  # Add a dropout layer after each linear layer
            nn.Linear(128, 32),
            # nn.BatchNorm1d(32),
            nn.Dropout(dropout_rate), 
            nn.Linear(32, output_dim)
        )
    

    def forward(self, x):
        # x = x.view(x.size(0), 1, -1)  # Add an extra dimension for the input channels
        # x = self.conv1(x)
        # x = x.flatten(start_dim=1)
        linear_layer = nn.Linear(x.size(1), 512).to(x.device)  # Ensure the new layer is on the same device as x
        x = linear_layer(x) 
        x = self.layers(x)
        # print(x.shape)
        x = x.view(x.size(0), -1) 
        return x