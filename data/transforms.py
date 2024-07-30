import math
import random
from typing import List, Tuple
import warnings

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import ImageFilter, ImageOps
from PIL.Image import Image


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



class CustomTwoCrop(object):
    def __init__(self, size=224, scale=(0.2, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=TF.InterpolationMode.BILINEAR,
                condition_overlap=True, global_crop=True):
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
        self.global_crop = global_crop

        if self.global_crop:
            self.padding = 4
            self.pad = transforms.Pad(self.padding, padding_mode="edge")

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
        
        if self.global_crop:
            padded_img = self.pad(img)
            params2 = [random.randint(0, 2*self.padding), random.randint(0, 2*self.padding), *self.size]

            coords[0][0] += self.padding
            coords[0][1] += self.padding

            coords.append(_get_coord(*params2))
            crops.append(TF.crop(padded_img, *params2))
        else:
            if not self.condition_overlap:
                params2 = self.get_params(img, self.scale, self.ratio)
            else:
                params2 = self.get_params_conditioned(img, self.scale, self.ratio, params1)
            coords.append(_get_coord(*params2))
            crops.append(TF.resized_crop(img, *params2, self.size, self.interpolation))

        return crops, _clip_coords(coords, [params1, params2])
    

class CustomTwoCrop_SPR(CustomTwoCrop):
    def __init__(self, size=224, scale=(0.2, 1), ratio=(3 / 4, 4 / 3), interpolation=TF.InterpolationMode.BILINEAR, condition_overlap=True, global_crop=True):
        super().__init__(size, scale, ratio, interpolation, condition_overlap, global_crop)

    def __call__(self, img):
        assert img.shape[0] == 2

        crops, cur_coords = [], []
        params_cur1 = self.get_params(img[0], self.scale, self.ratio)
        cur_coords.append(_get_coord(*params_cur1))
        crops.append(TF.resized_crop(img[0], *params_cur1, self.size, self.interpolation))

        if not self.condition_overlap:
            params_cur2 = self.get_params(img[0], self.scale, self.ratio)
        else:
            params_cur2 = self.get_params_conditioned(img, self.scale, self.ratio, params_cur1)
        cur_coords.append(_get_coord(*params_cur2))  
        crops.append(TF.resized_crop(img[0], *params_cur2, self.size, self.interpolation))  

        crops.append(img[-1])
        cur1_k = _clip_coords([cur_coords[0], torch.Tensor([0., 0., *img.shape[-2:]])], [params_cur1, [0, 0, *img.shape[-2:]]])
        cur2_k = _clip_coords([cur_coords[1], torch.Tensor([0., 0., *img.shape[-2:]])], [params_cur2, [0, 0, *img.shape[-2:]]])

        return crops, _clip_coords(cur_coords, [params_cur1, params_cur2]), (cur1_k, cur2_k)
    

class InterestingTwoCrop(CustomTwoCrop):
    def __init__(self, temp: float=5., gauss_kernel: int=35, gauss_sigma: float=15.5, border: int=18, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.temp = temp 

        self.gauss_kernel = gauss_kernel
        self.gauss_sigma = gauss_sigma

        self.border = border

    def sample_centers(self, img: torch.Tensor) -> Tuple[int]:
        """
        img: torch.Tensor, [3, H, W]
        """
        sobel_f = kornia.filters.sobel(img.float().unsqueeze(0)).max(1).values  # [1, H, W]
        gauss_b = kornia.filters.gaussian_blur2d(sobel_f.unsqueeze(1), kernel_size=self.gauss_kernel, sigma=(self.gauss_sigma, self.gauss_sigma)).flatten(0,2)  # [H, W]

        sm = F.softmax(gauss_b[self.border:img.shape[-2]-self.border, self.border:img.shape[-1]-self.border].flatten() / self.temp, -1)  # [H' * W']
        p = torch.multinomial(sm, 1)

        x = p % (img.shape[-1] - 2 * self.border) + self.border
        y = p // (img.shape[-1] - 2 * self.border) + self.border

        return x.item(), y.item() 

    def get_params(self, img, scale, ratio):
        xc, yc = self.sample_centers(img)
        width, height = _get_image_size(img)
        area = width * height

        xmin, ymin = min(xc, width - xc) * 2, min(yc, height - yc) * 2
        max_scale = xmin * ymin / area

        target_area = random.uniform(scale[0], max_scale) * area 
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect_ratio = math.exp(random.uniform(*log_ratio))

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        return max(yc - h // 2, 0), max(xc - w // 2, 0), h, w 
    
    def get_params_conditioned(self, img, scale, ratio, constraint):
        for _ in range(10):
            i, j, h, w = self.get_params(img, scale, ratio)
            inter = _compute_intersection((i,j,h,w), constraint)
            if inter > 0.01 * h * w:
                return i, j, h, w 
            
        return self.get_params(img, scale, ratio)



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
    

class CustomRandomHorizontalFlip_SPR(CustomRandomHorizontalFlip):
    def __call__(self, crops, coords_cur, coords_next):
        flip1, flip2 = torch.rand(1) < self.p, torch.rand(1) < self.p 
        flags = [flip1.item(), flip2.item()]

        if flip1:
            crops[0] = TF.hflip(crops[0])
            coord = coords_cur[0].clone()

            coord[0] = 1. - coords_cur[0][2]
            coord[2] = 1. - coords_cur[0][0]
            coords_cur[0] = coord  

            coord = coords_next[0][1].clone()
            coord[0] = 1. - coords_next[0][1][2]
            coord[2] = 1. - coords_next[0][1][0]
            coords_next[0][1] = coord

        if flip2:
            crops[1] = TF.hflip(crops[1])
            coord = coords_cur[1].clone()

            coord[0] = 1. - coords_cur[1][2]
            coord[2] = 1. - coords_cur[1][0]
            coords_cur[1] = coord  

            coord = coords_next[1][1].clone()
            coord[0] = 1. - coords_next[1][1][2]
            coord[2] = 1. - coords_next[1][1][0]
            coords_next[1][1] = coord

        return crops, coords_cur, coords_next, flags
    
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


class BatchTwoCrop(object):
    def __init__(self, size: int, scale: Tuple[float, float], ratio: Tuple[float, float]=(3./4.,4./3.), 
                 interpolation=TF.InterpolationMode.BILINEAR, *args, **kwargs):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)

        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation

        self.scale = scale
        self.ratio = ratio
        self.condition_overlap = True 

    def crop(self, img: torch.Tensor):
        width, height = img.shape[-2:]
        area = width * height

        target_area = (torch.rand(1) * (self.scale[1] - self.scale[0]) + self.scale[0]) * area 
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        aspect_ratio = torch.exp(torch.rand(1) * (log_ratio[1] - log_ratio[0]) + log_ratio[0])

        w = torch.round(torch.sqrt(target_area * aspect_ratio)).int()
        h = torch.round(torch.sqrt(target_area / aspect_ratio)).int()

        w = w.clamp(max=width-1)
        h = h.clamp(max=height-1)

        i = torch.randint(0, height - h, [1]) 
        j = torch.randint(0, width - w, [1]) 

        
        img_crop = TF.resized_crop(img, i, j, h, w, img.shape[-2:], interpolation=self.interpolation)
        return img_crop, [i,j,h,w]

    def __call__(self, images: torch.Tensor):
        (crop_q, params0), (crop_k, params1) = self.crop(images), self.crop(images)

        y1_q, x1_q, height_q, width_q = params0 
        y1_k, x1_k, height_k, width_k = params1 

        x2_q = x1_q + width_q
        y2_q = y1_q + height_q
        x2_k = x1_k + width_k
        y2_k = y1_k + height_k

        x1_n, y1_n = torch.max(x1_q, x1_k), torch.max(y1_q, y1_k)
        x2_n, y2_n = torch.min(x2_q, x2_k), torch.min(y2_q, y2_k)

        coord_q = torch.Tensor([(x1_n - x1_q) / width_q, (y1_n - y1_q) / height_q, 
                                (x2_n - x1_q) / width_q, (y2_n - y1_q) / height_q])
        
        
        coord_k = torch.Tensor([(x1_n - x1_k) / width_k, (y1_n - y1_k) / height_k, 
                                (x2_n - x1_k) / width_k, (y2_n - y1_k) / height_k])
        
        return [crop_q, crop_k], (coord_q.unsqueeze(-1).repeat((1,crop_q.shape[0])), coord_k.unsqueeze(-1).repeat((1,crop_k.shape[0])))


class GaussianBlur(nn.Module):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        super().__init__()
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])

        if not isinstance(x, Image):
            pil_imgs = [TF.pil_to_tensor(TF.to_pil_image(i).filter(ImageFilter.GaussianBlur(radius=sigma))) for i in x]
            return torch.stack(pil_imgs)

        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(nn.Module):
    def __init__(self, threshold=128):
        super().__init__()
        self.threshold = threshold

    def __call__(self, sample):
        if not isinstance(sample, Image):
            pil_imgs = [TF.pil_to_tensor(ImageOps.solarize(TF.to_pil_image(i), self.threshold)) for i in sample]
            return torch.stack(pil_imgs)

        return ImageOps.solarize(sample, self.threshold)



class CustomDataAugmentation(object):
    def __init__(self, size=224, min_scale=0.08, interest_crop: bool=False, solarize_p=0.2, global_crop=True, 
                 expect_tensors: bool=False):
        color_jitter = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            TensorToTensor() if expect_tensors else transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        if interest_crop:
            self.two_crop = InterestingTwoCrop(size=size, scale=(min_scale, 1), interpolation=TF.InterpolationMode.BICUBIC,
                                          global_crop=global_crop)
        else:
            self.two_crop = CustomTwoCrop(size, (min_scale, 1), interpolation=TF.InterpolationMode.BICUBIC,
                                          global_crop=global_crop)

        self.hflip = CustomRandomHorizontalFlip(p=0.5)

        self.global_transfo1 = transforms.Compose([
            color_jitter,
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            color_jitter,
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomApply([Solarize()], p=solarize_p),
            normalize,
        ])

    def __call__(self, image):
        crops, coords = self.two_crop(image)
        crops, coords, flags = self.hflip(crops, coords)
        crops_transformed = []
        crops_transformed.append(self.global_transfo1(crops[0].unsqueeze(0)).squeeze(0))
        crops_transformed.append(self.global_transfo2(crops[1].unsqueeze(0)).squeeze(0))
        return crops_transformed, coords, flags
    

class CustomDataAugmentation_SPR(CustomDataAugmentation):
    def __init__(self, size=224, min_scale=0.08, padding=4, slotcon_augm=False, solarize_p=0.2, global_crop=True, 
                 expect_tensors: bool=False):
        color_jitter = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            TensorToTensor() if expect_tensors else transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


        self.two_crop = CustomTwoCrop_SPR(size, (min_scale, 1), interpolation=TF.InterpolationMode.BICUBIC,
                                          global_crop=global_crop)

        self.hflip = CustomRandomHorizontalFlip_SPR(p=0.5)

        self.global_transfo1 = transforms.Compose([
            color_jitter,
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            color_jitter,
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomApply([Solarize()], p=solarize_p),
            normalize,
        ])

    def __call__(self, image):
        crops, cur_coords, next_coords = self.two_crop(image)
        crops, cur_coords, next_coords, flags = self.hflip(crops, cur_coords, next_coords)
        crops_transformed = []
        crops_transformed.append(self.global_transfo1(crops[0].unsqueeze(0)).squeeze(0))
        crops_transformed.append(self.global_transfo2(crops[1].unsqueeze(0)).squeeze(0))

        crops_transformed.append(self.global_transfo1(crops[2].unsqueeze(0)).squeeze(0))
        return crops_transformed, (cur_coords, next_coords), flags 

    
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