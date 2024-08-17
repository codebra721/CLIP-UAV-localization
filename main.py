import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from geoclip.train.dataloader import GeoDataLoader
from geoclip.model import GeoCLIP
from geoclip.train.train import train,validate
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
import os
import glob


def main():
    # Define hyperparameters
    #這邊調數字會影響到loss
    batch_size = 20 #dataset_len 要考慮進5-flod跟batch_size整除才可以
    num_epochs = 50
    learning_rate = 1e-4
    step_size=20
    gamma =1e-2
    # learning_rate_location = 8e-5
    # learning_rate_rotate = 3e-5
    alpha = 0.85
    beta = 0.15
    dataset_len = 2600

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize your dataset
    dataset = GeoDataLoader(csv_file='/home/rvl122/mydataset/', dataset_folder='/home/rvl122/mydataset/')
    # dataset = new_GeoDataLoader(csv_file='/home/rvl122/mydataset/', dataset_folder='/home/rvl122/mydataset/')
    
    # Split dataset indices into 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    path = '/home/rvl122/geo-clip/logs/'
    file_num = f"0701_queued10_0.0001_{batch_size}_{learning_rate}_{step_size}_{gamma}"
    try:
        os.mkdir( path + file_num)
    except FileExistsError:
        pass
    # TensorBoard writer
    writer = SummaryWriter( path + file_num)
    
    # Initialize your dataset loader
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}:")
        
        # Initialize your model
        print("initialize model")    
        model = GeoCLIP(queue_size=dataset_len//10, batch_size=batch_size)  # Replace YourModelClass with your actual model class
        model = model.to(device)
        
        # Define loss function
        # criterion = nn.CosineEmbeddingLoss()
        # criterion = nn.MSELoss()
        criterion = nn.CrossEntropyLoss()
        # criterion = contrastive_neg_loss
        
        # Define optimizer
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        # optimizers = [
            # optim.AdamW(model.image_encoder.parameters(), lr=learning_rate_image, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False),
            # optim.AdamW(model.location_encoder.parameters(), lr=learning_rate_location, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False),
            # optim.AdamW(model.rotate_encoder.parameters(), lr=learning_rate_rotate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        # ]
        # Define learning rate scheduler
        # shceduler = ReduceLROnPlateau(optimizer, mode='min', factor=1e-1, patience=1, verbose=True)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma = gamma)
        # schedulers = [
        #     optim.lr_scheduler.StepLR(optimizers[0], step_size=5, gamma=0.5),
            # optim.lr_scheduler.StepLR(optimizers[0], step_size=15, gamma=0.5),
            # optim.lr_scheduler.StepLR(optimizers[1], step_size=3, gamma=0.05)
        # ]
        print("initialize model done!") 
        
        # Create data loaders for training and validation
        train_data = [dataset[i] for i in train_idx]
        val_data = [dataset[i] for i in val_idx]
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
        # print("Batch size:", train_dataloader.batch_size)
        print("train_dataloader number of samples:", len(train_dataloader.dataset))
        checkpoint_dir = f"checkpoints/checkpoint_fold_{fold + 1}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        min_val_loss = float('inf')
        # Initialize the counter for the number of epochs without improvement
        epochs_without_improvement = 0
        # Training loop
        for epoch in range(num_epochs):
            # Train for one epoch
            # train_loss = new_train(train_dataloader, model, criterion, optimizer, scheduler, epoch+1, batch_size, device)
            # val_loss = new_validate(val_dataloader, model, criterion, epoch+1 ,batch_size, device)
            train_loss  = train(alpha, beta, train_dataloader, model, criterion, optimizer, epoch+1, batch_size, device)
            writer.add_scalar(f"Fold {fold + 1}/training Loss", train_loss['total_loss'], epoch + 1)
            # writer.add_scalar(f"Fold {fold + 1}/training GPS Loss", train_gps_loss, epoch + 1)
            # writer.add_scalar(f"Fold {fold + 1}/training rotate Loss", train_rotate_loss, epoch + 1)
            # if train_loss['regression_loss'] is not None:
            #     writer.add_scalar('Regression Loss/train', train_loss['regression_loss'], epoch)
            val_loss = validate(alpha, beta,val_dataloader, model, criterion, scheduler, epoch+1 ,batch_size, device)
            # writer.add_scalar(f"Fold {fold + 1}/Validation GPS Loss", val_gps_loss, epoch + 1)
            writer.add_scalar(f"Fold {fold + 1}/Validation Loss", val_loss['total_loss'] ,epoch + 1)
            # if val_loss['regression_loss'] is not None:
            #     writer.add_scalar('Regression Loss/validate', val_loss['regression_loss'], epoch)
            # If the validation loss has not improved for 5 epochs, start training the regression head
            # if val_loss['total_loss'] >= min_val_loss:
            #     epochs_without_improvement += 1
            #     if epochs_without_improvement >= 10:
            #         model.train_regression_head = True
            # else:
            #     min_val_loss = val_loss['total_loss']
            #     epochs_without_improvement = 0
                
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch{epoch + 1}_{fold}.pt")
                torch.save(model.state_dict(), checkpoint_path)

                # Keep only the 3 most recent files
                checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt")), key=os.path.getmtime)
                for checkpoint in checkpoints[:-2]:
                    os.remove(checkpoint)
        
        final_checkpoint_path = os.path.join("result/", f"{file_num}_{fold}.pt")
        torch.save(model.state_dict(), final_checkpoint_path)
    print("Training finished.")
    writer.close()

if __name__ == "__main__":
    main()