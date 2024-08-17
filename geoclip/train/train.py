import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

    
def train(alpha ,beta ,train_dataloader, model, criterion, optimizer, epoch, batch_size, device):
    print("Starting training Epoch", epoch)

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    targets_img_gps = torch.Tensor([i for i in range(batch_size)]).long().to(device)

    total_losses = []

    # for i ,(imgs, gps_list, heading_list) in bar:
    for i ,(imgs, gps_list) in bar:
        imgs = imgs.to(device)
        # gps_tensor = torch.stack([gps.to(device) for gps in gps_list])  # Stack the tensors in the list
        gps_tensor = gps_list.to(device)
        gps_queue = model.get_gps_queue()
        optimizer.zero_grad()
        # print(gps_tensor.shape)
        # print(gps_queue.shape)
        gps_all = torch.cat([gps_tensor, gps_queue], dim=0)
        # print(gps_all.shape)
        model.dequeue_and_enqueue(gps_tensor)
        
        outputs = model.forward(imgs, gps_all)
        # outputs = model.forward(imgs, gps_tensor)
        
        logits_img_gps = outputs
        loss = criterion(logits_img_gps, targets_img_gps)
        # loss = criterion(outputs, gps_tensor)
        loss.backward()
        optimizer.step() 
        bar.set_description("Epoch {} loss: {:.5f}".format(epoch, loss.item()))
        total_losses.append(loss.item())

    avg_total_loss = sum(total_losses) / len(total_losses)
    return {'total_loss': avg_total_loss}
    # avg_regression_loss = sum(regression_losses) / len(regression_losses) if regression_losses else None
    # return {'total_loss': avg_total_loss, 'regression_loss': avg_regression_loss}

def validate(alpha, beta, val_dataloader, model, criterion, scheduler, epoch, batch_size, device):
    print("Starting Validation for Epoch", epoch)
    
    model.eval()  # Set model to evaluation mode

    regression_criterion = nn.MSELoss()
    total_losses = []
    # regression_losses = []
    
    with torch.no_grad():
        bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        targets_img_gps = torch.Tensor([i for i in range(batch_size)]).long().to(device)
        # for i ,(imgs, gps_list, heading_list) in bar:
        for i, (imgs, gps_list) in bar:
            imgs = imgs.to(device)
            gps_tensor = torch.stack([gps.to(device) for gps in gps_list])  # Stack the tensors in the list
            
            
            outputs = model.forward(imgs, gps_tensor)
            
            logits_img_gps = outputs
            loss = criterion(logits_img_gps, targets_img_gps)
            # loss = criterion(outputs, gps_tensor)
            bar.set_description("Epoch {} loss: {:.5f}".format(epoch, loss.item()))
            total_losses.append(loss.item())

    avg_total_loss = sum(total_losses) / len(total_losses) 
    # avg_regression_loss = sum(regression_losses) / len(regression_losses) if regression_losses else None
    scheduler.step()
    model.train()  # Set model back to training mode
    return {'total_loss': avg_total_loss}
    # return {'total_loss': avg_total_loss, 'regression_loss': avg_regression_loss}
