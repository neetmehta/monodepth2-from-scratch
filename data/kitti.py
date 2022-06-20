import pandas as pd
from PIL import Image
# from torchvision.transforms import transforms
import torch
from torch.utils.data import Dataset
# to_tensor = transforms.Compose([transforms.ToTensor()])
class KittiOdom(Dataset):

    def __init__(self, csv_path) -> None:
        super(KittiOdom, self).__init__()

        self.img_list = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        sample = {}
        source = Image.open(self.img_list.iloc[index]['source'])
        target = Image.open(self.img_list.iloc[index]['target'])
        # source, target = to_tensor(source), to_tensor(target)
        sample['source'] = source
        sample['target'] = target
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