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
        transform,
        spr_skip=0
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
            self.fnames = list(glob.glob(data_dir + '/*.npz'))
        else:
            raise NotImplementedError

        self.fnames = np.array(sorted(self.fnames)) # to avoid memory leak
        self.transform = transform

        self.dataset = dataset
        self.spr_skip = spr_skip

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fpath = self.fnames[idx]
        if self.dataset == 'atari':
            data = [np.load(fn) for fn in self.fnames[idx:min(idx + self.spr_skip + 1, len(self.fnames))]]
            dones = np.array([d['done'] for d in data])

            frame_1 = data[0]['obs']  # [3, H, W]
            frame_k = data[-1]['obs'] if len(data) == self.spr_skip + 1 and dones.cumsum()[-1] == 0 else np.zeros_like(frame_1)

            actions = data[0]['action']  # [1]
            frames = np.stack([frame_1, frame_k])  # [2, 3, H, W]

            return self.transform(torch.from_numpy(frames)), torch.from_numpy(actions)
        
        image = Image.open(fpath).convert('RGB')
        return self.transform(image), torch.empty(1)
