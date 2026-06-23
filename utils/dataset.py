import os
import random
import argparse
import json
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import tifffile as tiff
import pydicom
from functools import partial
import torch.nn as nn

from networks.generators import Single_Generator, Unet8BN, Unet8
from networks.discriminators import Single_Discriminator
from networks.calculator import des_image, norm11to01, norm01to11

def parse_args_options(input_args=None, is_train=True, name_prefix="SmplD1", cross_val_num=1, min_w=5, max_w=45, roi_eval="sum_annot", pth="2000.pth", is_save_image=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross_val_num", type=int, default=cross_val_num)
    parser.add_argument("--min_w", type=int, default=min_w)
    parser.add_argument("--max_w", type=int, default=max_w)
    parser.add_argument("--pth", type=str, default=pth)
    parser.add_argument("--name", default=f"{name_prefix}_{min_w}-{max_w}_{cross_val_num}")
    parser.add_argument("--use_amp", default=True, type=bool)

    # Dataset options
    parser.add_argument("--filename_A", default='CR000003') # Standard/Mix
    parser.add_argument("--filename_B", default='CR000002') # HE
    parser.add_argument("--filename_C", default='CR000001') # LE
    parser.add_argument("--dataset_folder", default=os.path.join("..", "Dataset", "AIDES", "tiff"))
    parser.add_argument("--annot_roi_path", default=[os.path.join("..", "Dataset", "AIDES", "roi", "niu", roi_eval), "CR000005_Annotation.png"])
    parser.add_argument("--json_folder", default=os.path.join(".", "dataset", "cross_validation", str(cross_val_num)))
    
    parser.add_argument("--chans", default=1, type=int)
    parser.add_argument("--log_tr_for_image_a", default=True, type=bool)
    parser.add_argument("--hist_eq_for_des", default=False, type=bool)
    parser.add_argument("--model_he2le_pth", default=f"./weights/fold{cross_val_num}/he2le_gene_a2b.pth", type=str)
    parser.add_argument("--train_model_folder", default=f"./weights/fold{cross_val_num}/", type=str)
    
    parser.add_argument("--output_folder", default=os.path.join(".", "output_cross_valid", parser.get_default('name')))
    parser.add_argument("--ckpt_folder", default=os.path.join(parser.get_default('output_folder'), "checkpoints"))
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow_tf32", default=True)

    # Train options
    if is_train:
        parser.add_argument("--mask_range", type=int, default=[0, 0, parser.get_default('min_w'), parser.get_default('max_w')])
        parser.add_argument("--disc_featurier_train", type=str, default=True)
        parser.add_argument("--diffaugument_policy", type=str, default='brightness,contrast,resize,translation,cutout')
        parser.add_argument("--diffaugument_args", default=dict(resize_range=[0.8, 1.2], translation_ratio=0.125, cutout_th_pmax_range=[min_w, max_w], norm_type='01'))
        parser.add_argument("--gan_loss_type", type=str, default="multilevel_hinge", choices=['sigmoid','sigmoid_s','multilevel_sigmoid','multilevel_sigmoid_s','hinge','multilevel_hinge'])
        parser.add_argument("--lambda_gan", default=[1.0, 0.5, 0.5], type=float)
        parser.add_argument("--lambda_disc", default=0.1, type=float)
        parser.add_argument("--lambda_cr", default=10., type=float)
        parser.add_argument("--lambda_l1", default=10., type=float)
        parser.add_argument("--lambda_cy", default=100., type=float)
        parser.add_argument("--lambda_percept", default=1., type=float)
        parser.add_argument("--train_image_prep", default="resize_1024", type=str)
        parser.add_argument("--val_image_prep", default="resize_1024", type=str)
        parser.add_argument("--eval_folder", default=os.path.join(parser.get_default('output_folder'), "eval"))
        parser.add_argument("--seed", type=int, default=83)
        parser.add_argument("--train_batch_size", type=int, default=2)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
        parser.add_argument("--num_training_epochs", type=int, default=2000)
        parser.add_argument("--num_training_epochs_txt", default=[os.path.join(parser.get_default('ckpt_folder'), "num_training_epochs.txt"), -1])
        parser.add_argument("--save_by_epochs", type=int, default=100)
        parser.add_argument("--ckpt_by_epochs", type=int, default=5)
        parser.add_argument("--eval_by_epochs", type=int, default=5)
        parser.add_argument("--lr_generator", type=float, default=5.e-4)
        parser.add_argument("--lr_discriminator", type=float, default=5.e-5)
        parser.add_argument("--adam_beta1", type=float, default=0.9)
        parser.add_argument("--adam_beta2", type=float, default=0.999)
        parser.add_argument("--adam_weight_decay", type=float, default=0.01)
        parser.add_argument("--adam_epsilon", type=float, default=1.e-08)
        parser.add_argument("--mixed_precision", type=str, default="bfloat16")
        
        parser.add_argument("--discriminator", default=Single_Discriminator(
            model_args=dict(
                diffaug=True,
                policy=parser.get_default('diffaugument_policy'),
                diffaugument_args=parser.get_default('diffaugument_args'),
                device=parser.get_default('device'),
                activation=nn.LeakyReLU(0.2, inplace=True),
            ),
            featurier_train=parser.get_default('disc_featurier_train'),
            is_train=is_train,
            optim_args=dict(
                lr=parser.get_default('lr_discriminator'),
                betas=(parser.get_default('adam_beta1'), parser.get_default('adam_beta2')),
                weight_decay=parser.get_default('adam_weight_decay'),
                eps=parser.get_default('adam_epsilon'),
            ),
            gan_loss_type=parser.get_default('gan_loss_type'),
        ))
    else:
        parser.add_argument("--test_image_prep", default="resize_1024", type=str)
        parser.add_argument("--test_folder", default=os.path.join(parser.get_default('output_folder'), "test", roi_eval))
        parser.add_argument("--is_save_image", default=is_save_image, type=bool)

    parser.add_argument("--generator", default=Single_Generator(
        Unet8BN,
        model_args=dict(
            in_channels=parser.get_default('chans'),
            out_channels=parser.get_default('chans'),
            device=parser.get_default('device'),
        ),
        is_train=is_train,
        optim_args=dict(
            lr=5.e-4 if is_train else 0,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1.e-08,
        )
    ))

    parser.add_argument("--he2le_generator", default=Single_Generator(
        Unet8,
        model_args=dict(
            in_channels=1,
            device=parser.get_default('device')
        ),
        is_train=False
    ))

    args = parser.parse_args(input_args) if input_args is not None else parser.parse_args()
    return args

def set_seed(seed=83):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, json_folder, split, image_prep, mask_range, filename_A, filename_B, filename_C=None, args=None):
        super().__init__()
        self.dataset_folder = dataset_folder
        self.A = filename_A
        self.B = filename_B
        self.C = filename_C
        self.ext = '.tif'
        captions = os.path.join(json_folder, f"{split}_dataset.json")
        self.train = split == 'train'
        if not self.train and args is not None:
            self.log_tr_for_image_a = args.log_tr_for_image_a
            self.hist_eq_for_des = args.hist_eq_for_des

        with open(captions, "r") as f:
            self.captions = json.load(f)
        self.img_ids = list(self.captions.keys())
        self.mask_range = mask_range
        self.T = self.build_transform(image_prep)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_ids = self.img_ids[idx]
        caption = self.captions[img_ids]
        
        img_a, img_size = self.read_aides_image(path=os.path.join(self.dataset_folder, img_ids, self.A+self.ext), norm_type="01")
        img_b = self.read_aides_image(path=os.path.join(self.dataset_folder, img_ids, self.B+self.ext), norm_type="11")[0]

        if self.train:
            mask_a, rate_m = self.percentile_mask(img_a, "01", ranges=self.mask_range, invert=True)
            img_a, img_b, mask_a = torch.split(
                self.T(torch.cat([img_a, img_b, mask_a], dim=0)), 
                [len(img_a), len(img_b), len(mask_a)], dim=0
            )
            mask_a = torch.round(mask_a).to(dtype=torch.float32)
            return img_ids, img_a, img_b, mask_a, rate_m, caption 
        else:
            img_c = self.read_aides_image(path=os.path.join(self.dataset_folder, img_ids, self.C+self.ext), norm_type="11")[0]
            img_a, img_b, img_c = torch.split(
                self.T(torch.cat([img_a, img_b, img_c], dim=0)), 
                [len(img_a), len(img_b), len(img_c)], dim=0
            )
            return img_ids, img_a, img_b, img_c, img_size

    def percentile_mask(self, img_t:torch.Tensor, img_t_norm_type, ranges=[0,25,75,100], invert=True):
        min_p = float(random.randint(ranges[0], ranges[1])) * 0.01
        max_p = float(random.randint(ranges[2], ranges[3])) * 0.01
        rate = max_p - min_p
        if img_t_norm_type == '11':
            min_p = (min_p - 0.5) / 0.5
            max_p = (max_p - 0.5) / 0.5
        
        if invert:
            img_t = transforms.functional.invert(img_t) 

        min_th, max_th = torch.quantile(input=img_t, q=torch.tensor([min_p, max_p]))
        mask = torch.where(img_t <= max_th, 0., 1.)
        return mask, rate

    def read_aides_image(self, path, norm_type):
        def tiff2pil(filepath, win_percentile=[0, 100]):
            img = tiff.imread(filepath)    
            min_w = np.percentile(np.array(img), q=int(win_percentile[0]))
            max_w = np.percentile(np.array(img), q=int(win_percentile[1]))
            img = window_values(img.astype(np.float64), min_w, max_w)
            return Image.fromarray(img.astype(np.float64)).convert('F')    

        def window_values(image, min_w:float, max_w:float):
            ww = max_w - min_w + 1.e-6
            image -= min_w
            image[image < 0.] = 0.
            image[image > ww] = ww
            return image / ww

        def pil2normtensor(img: Image, norm_type='01') -> torch.FloatTensor:
            img = transforms.functional.to_tensor(img)
            if norm_type == "11":
                img = transforms.functional.normalize(img, mean=[0.5], std=[0.5])
            return img

        img = tiff2pil(path)
        img_size = list(reversed(img.size))
        img = pil2normtensor(img, norm_type)
        return img, img_size

    def build_transform(self, image_prep):
        T = transforms.Compose([
            transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR)
        ]) if image_prep in ["resize_1024", "resize_1024x1024"] else transforms.Lambda(lambda x: x)
        return T

@torch.no_grad()
def save_image(tensor, fp, dyn_range=2**8-1, dtype=np.uint8, import_dtype=torch.float32) -> None:
    ndarr = tensor.mul(dyn_range).add_(0.5).clamp_(0, dyn_range).to("cpu", import_dtype).numpy()
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    if os.path.splitext(fp)[1] == '.tif':
        with tiff.TiffWriter(fp, imagej=True) as img_tiff:
            img_tiff.write(ndarr.astype(dtype))     
    else:
        from skimage.io import imsave
        imsave(fp, ndarr.astype(dtype))

def save_args(args, dst):
    if dst is None:
        return
    with open(dst, 'w') as f:
        json.dump(vars(args), f, default=str, indent=4)

def load_args(src):
    if src is not None and os.path.exists(src):
        with open(src, 'r') as f:
            ns = json.load(f)
        return argparse.Namespace(**ns)
