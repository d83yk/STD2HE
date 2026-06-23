from functools import partial
from torch import nn
import torch
import torch.nn.functional as F
import random
from torchvision.transforms.functional import InterpolationMode, resize, center_crop

class DiffAugment(nn.Module):
    def __init__(
        self,
        policy='brightness,saturation,contrast,translation,cutout,cutout_th', 
        channels_first=True,
        resize_range=[0.9,1.1],
        translation_ratio=0.125,
        cutout_ratio=0.5,
        cutout_th_pmax_range=[50,100], 
        windowing_range=[0,50], 
        norm_type='01',
    ):
        super().__init__()
        self.policy = policy
        self.channels_first = channels_first
        self.augment_fns = {
            'brightness': [self.rand_brightness], 
            'saturation': [self.rand_saturation],
            'contrast': [self.rand_contrast],
            'resize': [partial(self.rand_resize, range=resize_range)],
            'windowing': [partial(self.rand_windowing, min_range=windowing_range, norm_type=norm_type)],
            'translation': [partial(self.rand_translation, ratio=translation_ratio)],
            'cutout': [partial(self.rand_cutout, ratio=cutout_ratio)],
            'cutout_th_pmax': [partial(self.rand_cutout_th_pmax, max_range=cutout_th_pmax_range, norm_type=norm_type)],
        }

    def forward(self, x):
        if self.policy:
            if not self.channels_first:
                x = x.permute(0, 3, 1, 2)
            for p in self.policy.split(','):
                for f in self.augment_fns[p]:
                    x = f(x)
            if not self.channels_first:
                x = x.permute(0, 2, 3, 1)
            x = x.contiguous()
        return x

    def rand_brightness(self, x):
        x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
        return x

    def rand_saturation(self, x):
        x_mean = x.mean(dim=1, keepdim=True)
        x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
        return x

    def rand_contrast(self, x):
        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
        return x

    def rand_resize(self, x, range=[0.9,1.1]):
        ratio = random.uniform(*range)
        x_size = [x.size(2), x.size(3)]
        crop_size = [int(x_size[0] * ratio + 0.5), int(x_size[1] * ratio + 0.5)]
        x = resize(x, size=crop_size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        x = center_crop(x, x_size)
        return x

    def rand_translation(self, x, ratio):
        shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
        translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
        x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
        return x

    def rand_cutout(self, x, ratio):
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        return x
    
    def rand_cutout_th_pmax(self, x, max_range, norm_type):
        def cutout_th_pmax(img_t:torch.Tensor, mask:torch.Tensor):
            return img_t*(mask >= 0.5) + 1.*(mask < 0.5)
        def min_percentile_mask(img_t:torch.Tensor, img_t_norm_type, max_range=[25,25]):
            img_t = img_t.to(dtype=torch.float32)
            max_p = float(random.randint(max_range[0],max_range[1])) * 0.01
            if img_t_norm_type=='11':
                max_p = (max_p - 0.5) / 0.5
            max_th = torch.quantile(input=img_t,q=torch.tensor([max_p]).to(img_t.device))
            mask = torch.where(img_t<=max_th, 1., 0.)
            return mask
        
        mask = min_percentile_mask(x, norm_type, max_range=max_range)
        return cutout_th_pmax(x, mask)

    def rand_windowing(self, x, min_range=[50,50], norm_type='01'):
        eps = 1.e-8
        min_p = float(random.randint(min_range[0],min_range[1])) * 0.01
        min_th = (min_p - 0.5)*2.0 if norm_type=='11' else min_p
        x = torch.clamp(input=x, min=min_th) - min_th
        x_max = x.max()
        x = x / (x_max+eps)
        if norm_type=='11':
            x = (x - 0.5) *2.0
        return x
