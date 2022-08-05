from logging import root
from typing import OrderedDict
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from data.kitti import KittiOdom
from networks.pose_network import PoseCNN
from networks.depth_network import DepthNetwork
from loss.loss import *
from utils import normalize_image

import random

random.seed(123)
torch.manual_seed(123)
print('seed created')


DISPARITY_SMOOTHNESS = 1e-3
ROOT = r"/dataset/kitti_raw_image"
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
PIN_MEMORY = True
NUM_EPOCHS = 200
CKPT_DIR = "ckpt"
RESUME = False
RESIZE = (192, 640)
STATE_DICT_PATH = 'ckpt\model_epoch_0.ckpt'
TENSORBOARD_FOLDER = 'tensorboard/runs'
LOGGING = True
scales = [0,1,2,3]

USE_MULTISCALE_LOSS = True
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_FOLDER, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

writer = SummaryWriter(TENSORBOARD_FOLDER)
model = {}
model['pose_network'] = PoseCNN(num_input_frames=2).to(device)
model['depth_network'] = DepthNetwork().to(device)

if RESUME:
    state_dict = torch.load(STATE_DICT_PATH)
    model['depth_network'].load_state_dict(state_dict['depth_model_state_dict'])
    model['pose_network'].load_state_dict(state_dict['pose_model_state_dict'])
    epoch = state_dict['epoch']
    loss = state_dict['loss']
    print(f'Resuming from epoch {epoch + 1} the mean was {loss}')

train_data = KittiOdom(csv_path='csv/eigen_zhou_train.csv', root=ROOT, resize=RESIZE)
val_data = KittiOdom(csv_path='csv/eigen_zhou_val.csv', root=ROOT, resize=RESIZE)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
start_epoch = 0


parameters_to_train = list(model['depth_network'].parameters()) + list(model['pose_network'].parameters())
optimizer = torch.optim.Adam(parameters_to_train, lr=LEARNING_RATE)
backproject_depth = OrderedDict()
project_3d = OrderedDict()

for i in scales:
    backproject_depth[i] = BackprojectDepth(BATCH_SIZE, RESIZE[0]//(2**i), RESIZE[1]//(2**i))
    project_3d[i] = Project3D(BATCH_SIZE, RESIZE[0]//(2**i), RESIZE[1]//(2**i))
num_parameters = sum(i.numel() for i in model['depth_network'].parameters()) + sum(i.numel() for i in model['pose_network'].parameters())
num_parameters = num_parameters/1e6
print(f"Number of parameters = {num_parameters} M")
print('Starting training')
step = 0
curr_mean_loss = 10


for epoch in range(start_epoch, NUM_EPOCHS):
    loop = tqdm(train_loader)
    mean_loss = []
    model['depth_network'].train()

    for i, sample in enumerate(loop):
        total_loss = 0
        depth = {}
        inv_depth = {}
        axisangle = {}
        translation = {}

        # Depth prediction 
        target = sample[('target',0)].to(device)
        disp = model['depth_network'](target)
        source_minus_1, source_1 = sample[('source_minus_1', 0)].to(device), sample[('source_1', 0)].to(device)
        # Pose prediction
        axisangle[-1], translation[-1] = model['pose_network'](torch.cat((source_minus_1, target), dim=1))
        axisangle[1], translation[1] = model['pose_network'](torch.cat((target, source_1), dim=1))
        for i in scales:
            
            ## Input to cuda
            source_1, source_minus_1, target = sample[('source_1', i)].to(device), sample[('source_minus_1', i)].to(device), sample[('target', i)].to(device) 
            K, inv_K = sample[('K', i)].to(device), sample[('inv_K', i)].to(device)

            ## disp to depth
            _, depth[i] = disp_to_depth(disp[('disp', i)], 0.1, 100)
            inv_depth[i] = 1 / depth[i]
            mean_inv_depth = inv_depth[i].mean(3, True).mean(2, True)

            ## Pose estimation
            poses = {}
            T = transformation_from_parameters(axisangle[-1][:,0], translation[-1][:,0]* mean_inv_depth[:, 0], invert=True)
            poses['-1'] = T

            T = transformation_from_parameters(axisangle[1][:,0], translation[1][:,0]* mean_inv_depth[:, 0], invert=False)
            poses['1'] = T
            
            ## reprojection
            #  -1
            cam_points = backproject_depth[i](depth[i], inv_K)
            pix_coords = project_3d[i](cam_points, K, poses['-1'])
            pred_recons_image_minus_1 = F.grid_sample(source_minus_1, pix_coords, padding_mode="border")

            #  +1
            cam_points = backproject_depth[i](depth[i], inv_K)
            pix_coords = project_3d[i](cam_points, K, poses['1'])
            pred_recons_image_1 = F.grid_sample(source_1, pix_coords, padding_mode="border")

            ## reprojection loss
            loss = 0
            reprojection_losses = []

            #  -1
            reprojection_losses.append(compute_reprojection_loss(pred_recons_image_minus_1, target))
            
            #  +1
            reprojection_losses.append(compute_reprojection_loss(pred_recons_image_1, target))
            

            reprojection_losses = torch.cat(reprojection_losses, 1)
            combined = reprojection_losses.mean(1, keepdim=True)

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            loss += to_optimise.mean()
            

            ## smooth loss
            mean_disp = disp[('disp', i)].mean(2, True).mean(3, True)
            norm_disp = disp[('disp', i)] / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, target)
            smooth_loss = DISPARITY_SMOOTHNESS * smooth_loss

            ## total loss
            loss = loss + smooth_loss
            total_loss += loss

        ## Training batch
        mean_loss.append(total_loss.item())
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        ## Tensorboard logging
        if LOGGING and step%100==0:
            for j in range(min(4,BATCH_SIZE)):
                writer.add_scalar("Training loss", total_loss, global_step=step)
                writer.add_image("Train/source_minus_1", source_minus_1[j].data, global_step=step)
                writer.add_image("Train/source_1", source_1[j].data, global_step=step)
                writer.add_image("Train/target", target[j].data, global_step=step)
                writer.add_image("Train/reprojected_image from -1", pred_recons_image_minus_1[j].data, global_step=step)
                writer.add_image("Train/reprojected_image from +1", pred_recons_image_1[j].data, global_step=step)
                writer.add_image("Train/disparity", normalize_image(disp[('disp', 0)][j]), global_step=step)
                writer.add_image("Train/depth", normalize_image(depth[j][0]), global_step=step)
                writer.add_image("Train/inv_depth", normalize_image(inv_depth[j][0]), global_step=step)

        step += 1
                



        loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        loop.set_postfix(loss=total_loss.item())
        break



    mean_loss = sum(mean_loss)/len(mean_loss)    
    print('\n===================================')
    print(f"Mean loss is {mean_loss} for epoch {epoch}")
    print('===================================\n')
    if curr_mean_loss>mean_loss or epoch%5==0:
        state_dict = {}
        state_dict['depth_model_state_dict'] = model['depth_network'].state_dict()
        state_dict['pose_model_state_dict'] = model['pose_network'].state_dict()
        state_dict['epoch'] = epoch
        state_dict['loss'] = mean_loss
        torch.save(state_dict, os.path.join(CKPT_DIR, f'model_epoch_{epoch}.ckpt'))
        curr_mean_loss = mean_loss
        
    
    writer.add_scalar("Mean Training loss", total_loss, global_step=epoch)
    num = random.randint(0, len(val_data))
    sample = val_data[num]
    image = sample[('target', 0)].unsqueeze(0).to(device)
    model['depth_network'].eval()

    with torch.no_grad():
        disp = model['depth_network'](image)
        disp = disp[('disp', 0)]      # Full scale disparity image

        ## disp to depth
        _, depth = disp_to_depth(disp, 0.1, 100)
        inv_depth = 1 / depth

        depth = depth.cpu().detach()
        inv_depth = inv_depth.cpu().detach()


    writer.add_image("val/image", image[0].data, global_step=epoch)
    writer.add_image("val/disparity", normalize_image(disp[0].data), global_step=epoch)
    writer.add_image("val/depth", normalize_image(depth[0]), global_step=epoch)
    writer.add_image("val/inv_depth", normalize_image(inv_depth[0]), global_step=epoch)

