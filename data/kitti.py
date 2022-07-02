import pandas as pd
from PIL import Image
from torchvision.transforms import transforms
import torch
import numpy as np
from torch.utils.data import Dataset
to_tensor = transforms.Compose([transforms.Resize((384, 1248)), transforms.ToTensor()])
class KittiOdom(Dataset):

    def __init__(self, csv_path) -> None:
        super(KittiOdom, self).__init__()

        self.img_list = pd.read_csv(csv_path)
        self.K = np.array([[0.576, 0, 0.4865, 0],
                           [0, 1.872, 0.4823, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.inv_k = np.linalg.pinv(self.K)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        sample = {}
        source = Image.open(self.img_list.iloc[index]['source'])
        target = Image.open(self.img_list.iloc[index]['target'])
        
        source, target = to_tensor(source), to_tensor(target)
        sample['source'] = source
        sample['target'] = target
        sample['inv_K'] = torch.from_numpy(self.inv_k)
        sample['K'] = torch.from_numpy(self.K)
        return sample

class KittiStereo(Dataset):

    def __init__(self, csv_path) -> None:
        super(KittiStereo, self).__init__()

        self.img_list = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        sample = {}
        source = Image.open(self.img_list.iloc[index]['source'])
        target = list(self.img_list.iloc[index]['source'])
        target[-21] = '3'
        target = "".join(target)
        target = Image.open(target)
        # source, target = to_tensor(source), to_tensor(target)
        sample['source'] = source
        sample['target'] = target
        return sample