from geoclip import GeoCLIP
import torch
import os
import csv
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from math import radians, cos, sin, sqrt, atan2
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.ndimage import gaussian_filter1d
from filterpy.kalman import ExtendedKalmanFilter as EKF
from filterpy.common import Q_discrete_white_noise

logging.basicConfig(filename='prediction_other.log', level=logging.INFO)
# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
directions = np.arange(0, 360, 5)
# Function to get the actual GPS coordinates of an image
def state_transition(x, dt):
    # 狀態轉移函數
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    return F @ x

def measurement_function(x):
    # 測量函數
    return x[:2]

def HJacobian(x):
    # 測量函數的雅可比矩陣
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

# 初始化 EKF
def initialize_ekf():
    ekf = EKF(dim_x=4, dim_z=2)
    ekf.x = np.zeros(4)  # 初始狀態 [lat, lon, v_lat, v_lon]
    ekf.F = np.eye(4)
    ekf.H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
    ekf.R = np.diag([0.1, 0.1])  # 測量噪聲
    ekf.Q = Q_discrete_white_noise(dim=4, dt=1.0, var=0.1)
    return ekf

def get_actual_gps(image_file):
    # Get the name of the subdirectory
    sub_dir_name = os.path.basename(os.path.dirname(image_file))
    
    # Extract the number from the image file name
    file_name = os.path.basename(image_file)
    image_file_number = int(file_name.split('_')[-1].split('.')[0])
    # print(image_file_number)
    # Load the CSV file
    df = pd.read_csv(f'/media/rvl/D/mydataset/gps_csv/{sub_dir_name}_coordinate.csv')

    # Use the image file number as the index to get the row
    row = df.iloc[image_file_number]  # Subtract 1 because pandas uses zero-based indexing

    actual_hea = row['HEA']
    
    diffs = np.abs(actual_hea - directions)

    cloest_direction_index = np.argmin(diffs)
    actual_hea_closest = directions[cloest_direction_index]
    # Print only the image file number
    # print(row['LAT'], row['LON'])

    # Return the lat and lon
    return row['LAT'], row['LON'], actual_hea_closest

# Function to calculate the distance between two GPS coordinates
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # radius of the Earth in meters

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

weights_dir = "/media/rvl/D/geo-clip/result/"
weights_files = os.listdir(weights_dir)
weights_files.sort()
model_num = 0

for weights_file in weights_files:
    # 假設你的模型權重存儲在這個路徑
    # weights_path = "result/0421_onlyone_10_0.pt"
    weights_path = os.path.join(weights_dir, weights_file)
    batch_size = 20 #dataset_len 要考慮進5-flod跟batch_size整除才可以
    num_epochs = 50
    dataset_len = 2600
    logging.info(f'{weights_file} predict results: ')
    # 創建一個GeoCLIP實例
    model = GeoCLIP(queue_size=dataset_len//2, batch_size=batch_size) 
    # print(model.state_dict().keys())
    # 加載模型權重
    model.load_state_dict(torch.load(weights_path))

    # 將模型設置為評估模式
    model.eval()

    # 確保模型在正確的設備上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get a list of all directories in the image directory
    image_dir = '/media/rvl/D/mydataset/image'
    sub_dirs = [os.path.join(image_dir, d) for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    sub_dirs.sort()
    # Initialize the total squared error and the number of predictions
    smooth_total_squared_error = 0
    smooth_total_error = 0
    total_squared_error_hea = 0
    total_circular_error_hea = 0
    total_num_prediciton = 0
    dic_num = 0
    # Loop over all subdirectories
    for i, sub_dir in enumerate(sub_dirs, start=1):
        # Initialize the total squared error and the number of predictions for this subdirectory
        sub_dir_total_squared_error = 0
        sub_dir_total_squared_error_hea = 0  
        sub_dir_total_error = 0
        
        sub_dir_total_circular_error_hea = 0
        
        ekf = initialize_ekf()
        # Get a list of all image files in the subdirectory
        image_files = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if f.endswith('.jpg') or f.endswith('.png')]
        image_files.sort()
        hea_distances = []
        actual_heas = []
        predic_hea = []
        predic_lat = []
        actual_lats = []
        predic_lon = []
        actual_lons = []
        # Open the CSV file for writing
        with open(f'prediction/{weights_file}_predictions_{i}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Image', 'lat', 'lon','hea'])  # Write the header

            # Loop over all image files
            for image_file in tqdm(image_files, desc=f"Predicting for directory {sub_dir}"):
                # Make a prediction
                with torch.no_grad():
                    top_pred_gps = model.predict(image_file, top_k=5)

                # Get the first GPS coordinate and split it into lon and lat
                
                first_gps = top_pred_gps.mean(dim=0)
                # first_gps = top_pred_gps[0]
                lat, lon, hea = first_gps[0].item(), first_gps[1].item(), first_gps[2].item()
                # print("lat: ", lat, "lon: ", lon, "hea: ", hea)
                # ekf.predict()
                # # EKF 更新步驟
                # z = np.array([lat, lon])
                # ekf.update(z, HJacobian, measurement_function)
                
                # # 使用 EKF 估計的結果
                # filtered_lat, filtered_lon = ekf.x[:2]
                predic_hea.append(hea)
                predic_lat.append(lat)
                predic_lon.append(lon)
                # average_pred_gps = top_pred_gps.mean(dim=0, keepdim=True).squeeze() 
                # lat, lon, hea = average_pred_gps[0].item(), average_pred_gps[1].item(), average_pred_gps[2].item()

                # Write the prediction to the CSV file
                writer.writerow([image_file, lat, lon, hea])

                # Get the actual GPS coordinates
                actual_lat, actual_lon, actual_hea= get_actual_gps(image_file)
                actual_lats.append(actual_lat)
                actual_lons.append(actual_lon)
                # print("actual_lat: ", actual_lat, "actual_lon: ", actual_lon, "actual_hea: ", actual_hea)
                actual_hea = 180 - actual_hea - 90
                actual_heass = (actual_hea + 180) % 360 - 180
                actual_heas.append(actual_heass)
                
                # Calculate the difference in heading
                hea_distance = hea - actual_hea
                # print(hea_distance)
                # Adjust the difference to be within -180 to 180
                hea_distance = (hea_distance + 180) % 360 - 180
                # print(hea_distance)
                hea_distances.append(hea_distance)
                # print(hea_distance)
                # Calculate the squared error for heading
                squared_error_hea = hea_distance ** 2

                # Add the squared error to the total squared error for heading
                sub_dir_total_squared_error_hea += squared_error_hea
                total_squared_error_hea += squared_error_hea

                sub_dir_total_circular_error_hea += abs(hea_distance)
                total_circular_error_hea += abs(hea_distance)
            
        # Calculate the root mean square error for this subdirectory
        sub_total_squared_error = 0
        sub_smooth_total_squared_error = 0
        sub_total_error = 0
        sub_smooth_total_error = 0
        smooth_distances = []
        # trajectory = np.column_stack((predic_lon, predic_lat))
        # smoothed_trajectory = gaussian_filter(trajectory, sigma=8)
        smoothed_lats = gaussian_filter1d(np.array(predic_lat), sigma=8)
        smoothed_lons = gaussian_filter1d(np.array(predic_lon), sigma=8)
        
        for i,(pred_lat, pred_lon, act_lat, act_lon) in enumerate(zip(predic_lat, predic_lon, actual_lats, actual_lons)):
            distance = calculate_distance(pred_lat, pred_lon, act_lat, act_lon)
            smooth_distances.append(distance)
            squared_error = distance ** 2
            sub_total_squared_error += squared_error
            sub_total_error += distance
            smooth_distance = calculate_distance(smoothed_lats[i], smoothed_lons[i], act_lat, act_lon)
            # smooth_distances.append(smooth_distance)
            smooth_squared_error = smooth_distance ** 2
            sub_smooth_total_squared_error += smooth_squared_error
            smooth_total_squared_error += smooth_squared_error
            sub_smooth_total_error += smooth_distance
            smooth_total_error += smooth_distance

        num_predictions = len(predic_lat)
        # print(num_predictions)
        total_num_prediciton += num_predictions
        rmse = sqrt(sub_total_squared_error / num_predictions)
        mae = sub_total_error / num_predictions
        
        smooth_rmse = sqrt(sub_smooth_total_squared_error / num_predictions)
        smooth_mae = sub_smooth_total_error / num_predictions

        sub_dir_rmse_hea = sqrt(sub_dir_total_squared_error_hea / num_predictions)
        sub_dir_mae_hea = sub_dir_total_circular_error_hea / num_predictions
        # print(f"The RMSE for heading for directory {sub_dir} is {sub_dir_rmse_hea} degrees.")
        # print(f"The MAE for heading for directory {sub_dir} is {sub_dir_mae_hea} degrees.")
        logging.info(f'orginal RMSE for directory {sub_dir}: {rmse}')
        logging.info(f'orginal MAE for directory {sub_dir}: {mae}')
        logging.info(f'smooth RMSE for directory {sub_dir}: {smooth_rmse}')
        logging.info(f'smooth MAE for directory {sub_dir}: {smooth_mae}')
        logging.info(f'RMSE for heading for directory {sub_dir}: {sub_dir_rmse_hea}')
        logging.info(f'MCE for heading for directory {sub_dir}: {sub_dir_mae_hea}')
        
        plt.figure(figsize=(10, 5))
        # print(smooth_distances)
        plt.plot(range(len(smooth_distances)), smooth_distances, label='displacement error', color='blue')
        plt.xlabel('Frame')
        plt.ylabel('meter')
        plt.title('Distance Error for Each Frame')
        plt.legend()
        plt.savefig(f'/home/rvl/Pictures/dic_figure_{model_num}_{dic_num}.png')
        plt.close()
        print(f"Predictions for directory {sub_dir} finished and saved to predictions_{i}.csv.")
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(hea_distances)), hea_distances, label='Heading Error', color='green')
        plt.plot(range(len(actual_heas)), actual_heas, label='Actual Heading', color='blue')
        plt.plot(range(len(predic_hea)), predic_hea, label='predict Heading', color='red')
        plt.xlabel('Frame')
        plt.ylabel('degrees')
        plt.title('Heading Error for Each Frame')
        plt.legend()
        plt.savefig(f'/home/rvl/Pictures/figure_{model_num}_{dic_num}.png')
        plt.close()
        dic_num = dic_num + 1
        # model_num = model_num+1
        # plt.show()
        # plt.hist(hea_distances, bins=range(0, 360, 10))
        # plt.xlabel('Heading Error (degrees)')
        # plt.ylabel('Frequency')
        # plt.title('Histogram of Heading Errors')
        # plt.show()

    # Calculate the overall root mean square error
    rmse = sqrt(smooth_total_squared_error / total_num_prediciton)
    mae = smooth_total_error / total_num_prediciton
    # print(f"The overall RMSE is {rmse} meters.")
    # print(f"The overall MAE is {mae} meters.")
    rmse_hea = sqrt(total_squared_error_hea / total_num_prediciton)
    mae_hea = total_circular_error_hea / total_num_prediciton
    # print(f"The overall RMSE for heading is {rmse_hea} degrees.")
    # print(f"The overall MAE for heading is {mae_hea} degrees.")
    logging.info(f'Overall RMSE: {rmse}')
    logging.info(f'Overall MAE: {mae}')
    logging.info(f'Overall RMSE for heading: {rmse_hea}')
    logging.info(f'Overall MCE for heading: {mae_hea}')
    logging.info('')
    model_num = model_num+1
    print(f"Finished predictions with weights file: {weights_file}")
