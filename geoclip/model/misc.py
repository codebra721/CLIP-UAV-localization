import os
import torch
import numpy as np
import pandas as pd

file_dir = os.path.dirname(os.path.realpath(__file__))

def load_gps_data(csv_file):
    data = pd.read_csv(csv_file)
    lat_lon = data[['LAT', 'LON']]
    gps_tensor = torch.tensor(lat_lon.values, dtype=torch.float32)
    return gps_tensor
def generate_heading_tensor():
    heading_tensor = torch.arange(0, 360, 5, dtype=torch.float32)  # Generates a tensor from 0 to 360
    return heading_tensor