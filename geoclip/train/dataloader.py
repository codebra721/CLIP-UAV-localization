import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import exists
from PIL import Image as im
import time
from torchvision import transforms
from torch.utils.data import Dataset
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def img_train_transform():
    train_transform_list = transforms.Compose([
        transforms.RandomChoice([
            # transforms.RandomResizedCrop(224),
            # transforms.Resize((224, 224))
            transforms.RandomResizedCrop(336),
            transforms.Resize((336, 336))
        ]),
        # transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomApply([AddGaussianNoise(0., 1.)], p=0.2)
    ])
    return train_transform_list

def img_val_transform():
    val_transform_list = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    return val_transform_list    

class GeoDataLoader(Dataset):
    """
    DataLoader for image-gps datasets.
    
    The expected CSV file with the dataset information should have columns:
    - 'IMG_FILE' for the image filename,
    - 'LAT' for latitude, and
    - 'LON' for longitude.
    
    Attributes:
        dataset_folder (str): Base folder where images are stored.
        dataset_file (str): CSV file path containing image names and GPS coordinates.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, csv_file, dataset_folder, device='cuda' ):
        self.dataset_folder = dataset_folder
        self.csv_file = csv_file
        self.transform = img_train_transform()
        self.device = device
        self.images, self.coordinates= self.load_dataset()
        # self.images, self.coordinates, self.headings= self.load_dataset()

    def load_dataset(self):
        images = []
        coordinates = []
        # headings = []
        print("Dataset folders:", self.dataset_folder)
        print("CSV folder:", self.csv_file)
        
        # 获取 CSV 文件夹中的所有文件
        csv_files = os.listdir(self.csv_file)
        
        for csv_filename in csv_files:
            # 检查文件扩展名是否为 CSV
            if csv_filename.endswith('.csv'):
                # 构建完整的 CSV 文件路径
                csv_file_path = os.path.join(self.csv_file, csv_filename)
                
                # 获取当前 CSV 文件对应的图像文件夹路径
                
                image_folder = os.path.join(self.dataset_folder)
                
                try:
                    dataset_info = pd.read_csv(csv_file_path)
                except Exception as e:
                    raise IOError(f"Error reading {csv_file_path}: {e}")

                for _, row in tqdm(dataset_info.iterrows(), desc=f"Loading images from {image_folder}"):
                    filename = os.path.join(image_folder, row['IMG_FILE'])
                    if os.path.exists(filename):
                        images.append(filename)
                        latitude = float(row['LAT'])
                        longitude = float(row['LON'])
                        heading = float(row['HEA'])
                        # coordinates.append((latitude, longitude))
                        # headings.append(heading)
                        coordinates.append((latitude, longitude,heading))
                    else:
                        print(f"File {filename} does not exist. Skipping.")

        return images, coordinates#, headings


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        gps = torch.tensor(self.coordinates[idx], device=self.device, dtype=torch.float)
        # heading = torch.tensor(self.headings[idx], device=self.device, dtype=torch.float)  # Get the heading

        image = im.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            # img = transforms.ToPILImage()(image)
            # img.show()
            # time.sleep(360)
        return image, gps#,heading# Return the heading

