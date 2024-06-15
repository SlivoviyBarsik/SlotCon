import glob
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset

class ImageFolder(Dataset):
    def __init__(
        self,
        dataset,
        data_dir,
        transform
    ):
        super(ImageFolder, self).__init__()
        if dataset == 'ImageNet':
            self.fnames = list(glob.glob(data_dir + '/train/*/*.JPEG'))
        elif dataset == 'COCO':
            self.fnames = list(glob.glob(data_dir + '/train2017/*.jpg'))
        elif dataset == 'COCOplus':
            self.fnames = list(glob.glob(data_dir + '/train2017/*.jpg')) + list(glob.glob(data_dir + '/unlabeled2017/*.jpg'))
        elif dataset == 'COCOval':
            self.fnames = list(glob.glob(data_dir + '/val2017/*.jpg'))
        elif dataset == 'atari':
            self.fnames = list(glob.glob(data_dir + '/*.jpg'))
        elif dataset == 'atari_stacked':
            self.fnames = list(glob.glob(data_dir + '/*.npz'))
        else:
            raise NotImplementedError

        self.fnames = np.array(self.fnames) # to avoid memory leak
        self.transform = transform

        self.dataset = dataset

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fpath = self.fnames[idx]
        if self.dataset == 'atari_stacked':
            data = np.load(fpath)

            frames = data['obs']  # [N_stacked, 3, H, W]
            actions = data['action']  # [1]

            return self.transform(torch.from_numpy(frames)), torch.from_numpy(actions)
        
        image = Image.open(fpath).convert('RGB')
        return self.transform(image), torch.empty(1)
