from logging import root
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


import random

random.seed(123)
torch.manual_seed(123)
print('seed created')


# APPLY_AUG = True
# PRETRAINED = True
ROOT = r"E:\Deep Learning Projects\datasets\kitti_raw"
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
PIN_MEMORY = True
NUM_EPOCHS = 200
CKPT_DIR = "ckpt"
RESUME = False
RESIZE = (448, 448)
STATE_DICT_PATH = 'ckpt\model_epoch_0.ckpt'
TENSORBOARD_FOLDER = 'tensorboard/runs'
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

train_data = KittiOdom(csv_path='csv/eigen_full_train.csv', root=ROOT, resize=RESIZE)
val_data = KittiOdom(csv_path='csv/eigen_full_val.csv', root=ROOT, resize=RESIZE)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
start_epoch = 0


parameters_to_train = list(model['depth_network'].parameters()) + list(model['pose_network'].parameters())
optimizer = torch.optim.Adam(parameters_to_train, lr=LEARNING_RATE)
backproject_depth = BackprojectDepth(BATCH_SIZE, RESIZE[0], RESIZE[1])
project_3d = Project3D(BATCH_SIZE, RESIZE[0], RESIZE[1])
num_parameters = sum(i.numel() for i in model['depth_network'].parameters()) + sum(i.numel() for i in model['pose_network'].parameters())
num_parameters = num_parameters/1e6
print(f"Number of parameters = {num_parameters} M")
print('Starting training')

for epoch in range(start_epoch, NUM_EPOCHS):
    loop = tqdm(train_loader)
    mean_loss = []
    model['depth_network'].train()
    for i, sample in enumerate(loop):

        source, target = sample['source'].to(device), sample['target'].to(device)
        K, inv_K = sample['K'].to(device), sample['inv_K'].to(device)
        depth = model['depth_network'](target)
        depth = depth[('disp',0)]
        inv_depth = 1 / depth
        mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
        axisangle, translation = model['pose_network'](torch.cat((source, target), dim=1))
        T = transformation_from_parameters(axisangle[:,0], translation[:,0]* mean_inv_depth[:, 0])
        _, depth = disp_to_depth(depth, 0.1, 100)
        cam_points = backproject_depth(depth, inv_K)
        pix_coords = project_3d(cam_points, K, T)

        pred_recons_image = F.grid_sample(source, pix_coords, padding_mode="border")

        loss = compute_reprojection_loss(pred_recons_image, source)
        loss = loss.mean()
        writer.add_scalar("Training loss", loss, global_step=i)

        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        loop.set_postfix(loss=loss.item())

    mean_loss = sum(mean_loss)/len(mean_loss)    
    if epoch%5==0:
        state_dict = {}
        state_dict['depth_model_state_dict'] = model['depth_network'].state_dict()
        state_dict['pose_model_state_dict'] = model['pose_network'].state_dict()
        state_dict['epoch'] = epoch
        state_dict['loss'] = mean_loss
        torch.save(state_dict, os.path.join(CKPT_DIR, f'model_epoch_{epoch}.ckpt'))
        
    
    writer.add_scalar("Mean Training loss", loss, global_step=epoch)
    num = random.randint(0, len(val_data))
    sample = val_data[num]
    image = sample['target'].unsqueeze(0).to(device)
    model['depth_network'].eval()
    depth = model['depth_network'](image)
    depth = depth[('disp',0)]
    inv_depth = 1 / depth

    depth = depth.cpu().detach()
    inv_depth = inv_depth.cpu().detach()

    writer.add_image('depth', depth.squeeze(0))
    writer.add_image('inv_depth', inv_depth.squeeze(0))

