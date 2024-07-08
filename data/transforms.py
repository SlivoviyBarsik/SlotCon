import math
import random
from typing import List, Tuple
import warnings

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from kornia.filters.gaussian import GaussianBlur2d
from kornia.augmentation import RandomSolarize, ColorJitter


def _get_image_size(img):
    if TF._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

def _compute_intersection(box1, box2):
    i1, j1, h1, w1 = box1
    i2, j2, h2, w2 = box2
    x_overlap = max(0, min(j1+w1, j2+w2) - max(j1, j2))
    y_overlap = max(0, min(i1+h1, i2+h2) - max(i1, i2))
    return x_overlap * y_overlap

def _get_coord(i, j, h, w):
    coord = torch.Tensor([j, i, j + w, i + h])
    return coord

def _clip_coords(coords, params):
    x1_q, y1_q, x2_q, y2_q = coords[0]
    x1_k, y1_k, x2_k, y2_k = coords[1]
    _, _, height_q, width_q = params[0]
    _, _, height_k, width_k = params[1]

    x1_n, y1_n = torch.max(x1_q, x1_k), torch.max(y1_q, y1_k)
    x2_n, y2_n = torch.min(x2_q, x2_k), torch.min(y2_q, y2_k)

    coord_q_clipped = torch.Tensor([float(x1_n - x1_q) / width_q, float(y1_n - y1_q) / height_q,
                                    float(x2_n - x1_q) / width_q, float(y2_n - y1_q) / height_q])
    coord_k_clipped = torch.Tensor([float(x1_n - x1_k) / width_k, float(y1_n - y1_k) / height_k,
                                    float(x2_n - x1_k) / width_k, float(y2_n - y1_k) / height_k])
    return [coord_q_clipped, coord_k_clipped]


class SPRTwoCrop(object):
    def __init__(self, size=128, padding=4):
        if isinstance(size, (tuple, list)):
            self.size = size 
        else:
            self.size = (size, size)

        self.pad = nn.ReplicationPad2d(padding)
        self.pad_size = padding

    def __call__(self, img: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[List, List]]:
        """
        img: torch.Tensor, [N, C, H, W]
        """
        assert img.shape[-2:] == self.size, f'Expected image of size {self.size}, got shape {img.shape[-2:]} instead'

        x0, x1 = torch.randint(0, 2*self.pad_size, img.shape[:-3]), torch.randint(0, 2*self.pad_size, img.shape[:-3])
        y0, y1 = torch.randint(0, 2*self.pad_size, img.shape[:-3]), torch.randint(0, 2*self.pad_size, img.shape[:-3])

        padded_img = self.pad(img)  # [N, C, H+2*padding, W+2*padding]
        
        index0 = torch.tile(x0.unsqueeze(1) + torch.arange(0, self.size[1]).unsqueeze(0), (1, self.size[0])) + \
            torch.repeat_interleave(y0.unsqueeze(1) + torch.arange(0, self.size[0]), self.size[1], 1) * padded_img.shape[-1]
        index0 = index0.unsqueeze(1).repeat((1,img.shape[1],1))  # [N, C, H, W]

        index1 = torch.tile(x1.unsqueeze(1) + torch.arange(0, self.size[1]).unsqueeze(0), (1, self.size[0])) + \
            torch.repeat_interleave(y1.unsqueeze(1) + torch.arange(0, self.size[0]), self.size[1], 1) * padded_img.shape[-1]
        index1 = index1.unsqueeze(1).repeat((1,img.shape[1],1))  # [N, C, H, W]

        crop0 = torch.gather(padded_img.flatten(-2,-1), 2, index0.to(padded_img.device))
        crop0 = crop0.reshape(img.shape)

        crop1 = torch.gather(padded_img.flatten(-2,-1), 2, index1.to(padded_img.device))
        crop1 = crop1.reshape(img.shape)

        # TODO: check that reshapes yield correct results

        x1_n = torch.stack([x0, x1]).max(0).values 
        y1_n = torch.stack([y0, y1]).max(0).values

        x2_n = torch.stack([x0, x1]).min(0).values + self.size[1]
        y2_n = torch.stack([y0, y1]).min(0).values + self.size[0]

        coords0 = torch.stack([(x1_n - x0).float() / self.size[0], (y1_n - y0).float() / self.size[1],
                   (x2_n - x0).float() / self.size[0], (y2_n - y0).float() / self.size[1]]).to(padded_img.device)
        
        coords1 = torch.stack([(x1_n - x1).float() / self.size[0], (y1_n - y1).float() / self.size[1],
                   (x2_n - x1).float() / self.size[0], (y2_n - y1).float() / self.size[1]]).to(padded_img.device)
        
        return [crop0, crop1], [coords0, coords1]

class CustomTwoCrop(object):
    def __init__(self, size=224, scale=(0.2, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=TF.InterpolationMode.BILINEAR,
                condition_overlap=True):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)

        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation

        self.scale = scale
        self.ratio = ratio
        self.condition_overlap = condition_overlap

    @staticmethod
    def get_params(img, scale, ratio, ):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def get_params_conditioned(self, img, scale, ratio, constraint):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
            constraints list(tuple): list of params (i, j, h, w) that should be used to constrain the crop
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width
        for counter in range(10):
            rand_scale = random.uniform(*scale)
            target_area = rand_scale * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                intersection = _compute_intersection((i, j, h, w), constraint)
                if intersection >= 0.01 * target_area: # at least 1 percent of the second crop is part of the first crop.
                    return i, j, h, w
        
        return self.get_params(img, scale, ratio) # Fallback to default option

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            crops (list of lists): result of multi-crop
        """
        crops, coords = [], []
        params1 = self.get_params(img, self.scale, self.ratio)
        coords.append(_get_coord(*params1))
        crops.append(TF.resized_crop(img, *params1, self.size, self.interpolation))

        if not self.condition_overlap:
            params2 = self.get_params(img, self.scale, self.ratio)
        else:
            params2 = self.get_params_conditioned(img, self.scale, self.ratio, params1)
        coords.append(_get_coord(*params2))
        crops.append(TF.resized_crop(img, *params2, self.size, self.interpolation))

        return crops, _clip_coords(coords, [params1, params2])


class CustomRandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, crops, coords):
        crops_flipped, coords_flipped, flags_flipped = [], [], []
        for crop, coord in zip(crops, coords):
            crop_flipped = crop
            coord_flipped = coord
            flag_flipped = False
            if torch.rand(1) < self.p:
                crop_flipped = TF.hflip(crop)
                coord_flipped = coord.clone()
                coord_flipped[0] = 1. - coord[2]
                coord_flipped[2] = 1. - coord[0]
                flag_flipped = True

            crops_flipped.append(crop_flipped)
            coords_flipped.append(coord_flipped)
            flags_flipped.append(flag_flipped)

        return crops_flipped, coords_flipped, flags_flipped
    
class CustomBatchRandomHorFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def __call__(self, crops, coords):
        crops_flipped, coords_flipped, flags = [], [], []
        for crop, coord in zip(crops, coords):
            crop_flipped = torch.zeros_like(crop)
            coord_flipped = torch.zeros_like(coord)

            flag = torch.rand(crop.shape[0]) <= self.p
            mask_to_flip = (flag)[...,None,None,None].to(crop.device)
            
            fl_crop = TF.hflip(crop)

            crop_flipped = fl_crop * mask_to_flip + (~mask_to_flip) * crop  # TODO: check if is correct
            mask_to_flip = mask_to_flip.flatten(0,-1)
            
            coord_flipped[0] = (1. - coord[2]) * mask_to_flip + (~mask_to_flip) * coord[0]
            coord_flipped[1] = coord[1]
            coord_flipped[2] = (1. - coord[0]) * mask_to_flip + (~mask_to_flip) * coord[2]
            coord_flipped[3] = coord[3]

            crops_flipped.append(crop_flipped)
            coords_flipped.append(coord_flipped)
            flags.append(flag)

        return crops_flipped, coords_flipped, flags


class TensorToTensor(nn.Module):
    """
    converts a uint8 tensor with values in [0..255] to a float32 tensor with values in [0..1]
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() / 255.
    
class TensorToFloat(nn.Module):
    """
    converts a uint8 tensor with values in [0..255] to a float32 tensor with values in [0..1]
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float()
    

class CustomColorJitter(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.jitter = ColorJitter(*args, **kwargs)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        input = x.float() / 255. if x.dtype == torch.uint8 else x
        return (self.jitter(input) * 255.).to(torch.uint8)
    

class CustomGaussianBlur(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.blur = GaussianBlur2d(*args, **kwargs)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        input = x.float() / 255. if x.dtype == torch.uint8 else x
        return (self.blur(input) * 255.).to(torch.uint8)
    

class CustomSolarize(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.solarize = RandomSolarize(*args, **kwargs)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        input = x.float() / 255. if x.dtype == torch.uint8 else x
        return (self.solarize(input) * 255.).to(torch.uint8)


class CustomRandomApply(nn.Module):
    def __init__(self, transform: nn.Module, prob: float) -> None:
        super().__init__()

        self.transform = transform
        self.prob = prob 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x.flatten(0,-4)  # [P, 3, H, W]
        mask = torch.rand(input.shape[0]).to(input.device)[..., None, None, None] <= self.prob  # [P, 1, 1, 1]
        output = self.transform(input) * mask + (~mask) * input
        output = output.reshape([*x.shape[:-3], *output.shape[1:]])

        return output


class CustomDataAugmentation(object):
    def __init__(self, size=224, min_scale=0.08, expect_tensors: bool=False):
        color_jitter = transforms.Compose([
            CustomRandomApply(
                CustomColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                prob=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            TensorToTensor() if expect_tensors else transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # self.two_crop = CustomTwoCrop(size, (min_scale, 1), interpolation=TF.InterpolationMode.BICUBIC)
        self.two_crop = SPRTwoCrop(size, padding=4)
        self.hflip = CustomBatchRandomHorFlip(p=0.5)

        # first global crop
        self.global_transfo1 = transforms.Compose([
            color_jitter,
            CustomRandomApply(CustomGaussianBlur(5, (1.5, 1.5)), prob=1.),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            color_jitter,
            CustomRandomApply(CustomGaussianBlur(5, (1.5, 1.5)), prob=0.1),
            CustomSolarize(p=0.2),
            normalize,
        ])

    def __call__(self, image):
        crops, coords = self.two_crop(image)
        crops, coords, flags = self.hflip(crops, coords)
        crops_transformed = []
        crops_transformed.append(self.global_transfo1(crops[0]))
        crops_transformed.append(self.global_transfo2(crops[1]))
        return crops_transformed, coords, flags
    
class DummyAugmentation(object):
    def __init__(self):
        self.normalize = transforms.Compose([
            TensorToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, image):
        image = self.normalize(image)
        crops = [image, image]
        coords = torch.concat([torch.zeros([2, image.shape[0]]), torch.ones([2, image.shape[0]])], 0)
        coords = [coords, coords]
        flags = [torch.zeros(image.shape[0]), torch.zeros(image.shape[0])]

        return crops, coords, flags