import pandas as pd
import os
from PIL import Image
from torchvision.transforms import transforms
import torch
import numpy as np
from torch.utils.data import Dataset
to_tensor = transforms.Compose([transforms.Resize((384, 1248)), transforms.ToTensor()])
class KittiOdom(Dataset):

    def __init__(self, csv_path, root, resize=(384,1248)) -> None:
        super(KittiOdom, self).__init__()

        self.root = root
        self.img_list = pd.read_csv(csv_path)
        self.K = np.array([[707.0912/1242, 0,            601.8873/1242, 0],
                           [0,             707.0912/375, 183.1104/375,  0],
                           [0,             0,            1,             0],
                           [0,             0,            0,             1]], dtype=np.float32)

        self.K[0, :] *= resize[1]
        self.K[1, :] *= resize[0]

        self.inv_k = np.linalg.pinv(self.K)
        self.to_tensor = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        sample = {}
        source = Image.open(os.path.join(self.root, self.img_list.iloc[index]['source']))
        target = Image.open(os.path.join(self.root, self.img_list.iloc[index]['target']))
        
        source, target = self.to_tensor(source), self.to_tensor(target)
        sample['source'] = source
        sample['target'] = target
        sample['inv_K'] = torch.from_numpy(self.inv_k)
        sample['K'] = torch.from_numpy(self.K)
        return sample

class KittiStereo(Dataset):

    def __init__(self, csv_path, root, resize=(384,1248)) -> None:
        super(KittiStereo, self).__init__()
        self.root = root
        self.img_list = pd.read_csv(csv_path)
        self.K = np.array([[707.0912/1242, 0,            601.8873/1242, 0],
                           [0,             707.0912/375, 183.1104/375,  0],
                           [0,             0,            1,             0],
                           [0,             0,            0,             1]], dtype=np.float32)

        self.K[0, :] *= resize[1]
        self.K[1, :] *= resize[0]

        self.inv_k = np.linalg.pinv(self.K)
        self.to_tensor = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        sample = {}
        source = Image.open(os.path.join(self.root,self.img_list.iloc[index]['source']))
        target = list(self.img_list.iloc[index]['source'])
        target[-21] = '3'
        target = "".join(target)
        target = Image.open(os.path.join(self.root,target))
        source, target = self.to_tensor(source), self.to_tensor(target)
        sample['source'] = source
        sample['target'] = target
        sample['inv_K'] = torch.from_numpy(self.inv_k)
        sample['K'] = torch.from_numpy(self.K)
        return sample