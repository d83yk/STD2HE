import torch
from torchvision.transforms.functional import normalize as norm_tensor
from torchvision.transforms.functional import to_tensor, invert
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import cv2
from typing import List, Union

def tensor2im32b(input_image):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1.0) / 2.0
    else:
        image_numpy = input_image
    return image_numpy.astype(np.float32)

def ssim_compute(real_tensor, fake_tensor):
    real_image = tensor2im32b(real_tensor)
    fake_image = tensor2im32b(fake_tensor)
    value = structural_similarity(real_image[:,:,0], fake_image[:,:,0], win_size=11, gaussian_weights=1.5,
                                  data_range=fake_image.max() - fake_image.min()) 
    return value

def lpips_compute(real_tensor, fake_tensor, net_lpips) -> float:
    value = net_lpips(real_tensor.data, fake_tensor.data).mean().cpu().detach().numpy().copy()
    return np.float64(value)

def dists_compute(real_tensor, fake_tensor, net_dists) -> float:
    def prep_dists(img_tensor):
        img = norm11to01(img_tensor)
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)
        return img
    value = net_dists(prep_dists(fake_tensor), prep_dists(real_tensor)).item()
    return np.float64(value)

def psnr_compute(real_tensor, fake_tensor, data_range=None):
    real_image = tensor2im32b(real_tensor)
    fake_image = tensor2im32b(fake_tensor)
    value = peak_signal_noise_ratio(real_image, fake_image, data_range=data_range)
    return value

def norm11to01(x):
    return norm_tensor(x, mean=[-1.], std=[2.])

def norm01to11(x):
    return norm_tensor(x, mean=[0.5], std=[0.5])

def norm01to11_range(x):
    return norm_tensor(x, mean=[0.], std=[0.5])

def histogram_percentile_width(x: torch.Tensor, y: torch.Tensor, p:float=0.1) -> float:
    min_p, max_p = p, 1.-p
    input = x - y
    min_th, max_th = torch.quantile(input=input, q=torch.tensor([min_p,max_p]).to(x.device))
    return np.float64(abs(max_th.item() - min_th.item()))

def hist_fwtm(x: torch.Tensor, bins=400, p:float=0.1) -> float:
    xx = torch.flatten(x).to('cpu').detach()
    h, b = torch.histogram(xx, bins=bins)
    peak_position = torch.argmax(h)
    peak_value = h[peak_position]
    rt = lt = 0
    for n in range(peak_position, h.shape[-1], 1):
        if h[n] < peak_value*p:
            rt = b[n]
            break
    for n in range(peak_position, 0, -1):
        if h[n] < peak_value*p:
            lt = b[n]
            break
    return np.float64(abs(rt - lt))

def mi_compute(he_t: torch.Tensor, le_t: torch.Tensor, bins=400):
    X = le_t.to('cpu').detach().numpy().flatten()
    Y = he_t.to('cpu').detach().numpy().flatten()
    p_xy, xedges, yedges = np.histogram2d(X, Y, bins=bins, density=True)
    p_x, _ = np.histogram(X, bins=xedges, density=True)
    p_y, _ = np.histogram(Y, bins=yedges, density=True)
    p_x_y = p_x[:, np.newaxis] * p_y
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    elem = p_xy * np.ma.log(p_xy / p_x_y)
    return np.sum(elem * dx * dy)

def des_image(image_a, image_b, rate=1.0, bit=4096, log_tr_for_image_a=True, hist=False):
    if log_tr_for_image_a:
        image_a = log_tr(image_a) 
    image_b = log_tr(image_b)*rate
    des = image_a - image_b
    des = exp_tr(des).mul_(bit)
    des = des_post_processing(des, hist)
    return des

def log_tr(image):
    eps = 1.e-8
    image_min = image.min()
    image = image - image_min
    image_max = image.max()
    image = (image / image_max) + eps
    b8 = 255.
    c = b8 / np.log(b8)
    image = c * torch.log(image*b8)
    return image

def exp_tr(image):
    b8 = 255.
    c = b8 / np.log(b8)
    image = image / c
    image = torch.exp(image) / b8
    return image

def des_post_processing(tensor:torch.Tensor, hist=False)-> torch.Tensor:
    if hist:
        tensor = histogram_equalization(tensor)
    tensor = invert(tensor)
    return tensor

def histogram_equalization(tensor: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
    def apply(t: torch.Tensor) -> torch.Tensor:
        channels, _, _ = t.shape
        img_np = t.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        for c in range(channels):
            img_np[:,:,c] = cv2.equalizeHist(img_np[:,:,c])
        return to_tensor(img_np).to(t.device)

    if tensor.dim()==4:
        batch_size = len(tensor)
        for b in range(batch_size):
            tensor[b] = apply(tensor[b]) 
    elif tensor.dim()==3:
        tensor = apply(tensor) 
    return tensor
