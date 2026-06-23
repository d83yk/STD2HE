import os
import sys
import torch
import torch.nn as nn
from torchvision.transforms.functional import crop, resize
from torchvision.transforms import InterpolationMode
from functools import partial
import lpips
from .dists import DISTS
from .losses import VGGPerceptualLoss

class Twin_Generator(nn.Module):
    def __init__(self, model, model_args=dict(), is_train=True, optim_args=dict()):
        super().__init__()
        self.gene_a2b = model(**model_args)
        self.gene_b2a = model(**model_args)

        if is_train:
            self.L1 = nn.L1Loss(reduction="mean")
            self.percept_loss = VGGPerceptualLoss(chans=3, reduction='mean')
            params = list(self.gene_a2b.parameters()) + list(self.gene_b2a.parameters())
            self.optimizer = torch.optim.AdamW(params, **optim_args)

    def set_eval(self):
        self.gene_a2b.eval()
        self.gene_b2a.eval()
        self.set_requires_grad(False)

    def set_lpips(self, device):
        net_lpips = lpips.LPIPS(net='vgg').to(device)
        net_lpips.requires_grad_(False)
        self.net_lpips = net_lpips
    
    def set_dists(self, device):
        net_dists = DISTS().to(device)
        self.net_dists = net_dists

    def set_train(self):
        self.gene_a2b.train()
        self.gene_b2a.train()
        self.set_requires_grad(True)

    def set_requires_grad(self, requires_grad=False):
        nets = [self.gene_a2b, self.gene_b2a]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def load_networks(self, load_dir, load_optim=False):
        map_location = None if torch.cuda.is_available() else torch.device('cpu')
        checkpoint = torch.load(load_dir, map_location=map_location)
        self.gene_a2b.load_state_dict(checkpoint['gene_a2b_state_dict'])
        self.gene_b2a.load_state_dict(checkpoint['gene_b2a_state_dict'])
        epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch'] if 'best_epoch' in checkpoint else epoch
        loss_gene = checkpoint['loss_gene']
        best_loss_gene = checkpoint['best_loss_gene'] if 'best_loss_gene' in checkpoint else loss_gene
        if load_optim:
            self.optimizer.load_state_dict(checkpoint['optim_gene_state_dict'])
        return epoch, best_epoch, loss_gene, best_loss_gene

    def save_networks(self, save_dir, epoch, best_epoch, loss_gene, best_loss_gene, net_disc, loss_disc):
        torch.save({
            'epoch': epoch,
            'best_epoch': best_epoch,
            'gene_a2b_state_dict': self.gene_a2b.state_dict(),
            'gene_b2a_state_dict': self.gene_b2a.state_dict(),
            'optim_gene_state_dict': self.optimizer.state_dict(),
            'loss_gene': loss_gene,
            'best_loss_gene': best_loss_gene,
            'disc_a_state_dict': net_disc.disc_a.state_dict(),
            'disc_b_state_dict': net_disc.disc_b.state_dict(),
            'optim_disc_state_dict': net_disc.optimizer.state_dict(),
            'loss_disc': loss_disc,
        }, save_dir)

    def load_gene_a2b(self, load_dir):
        map_location = None if torch.cuda.is_available() else torch.device('cpu')
        checkpoint = torch.load(load_dir, map_location=map_location)
        self.gene_a2b.load_state_dict(checkpoint['gene_a2b_state_dict'])
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
        best_epoch = checkpoint['best_epoch'] if 'best_epoch' in checkpoint else epoch
        loss_gene = checkpoint['loss_gene'] if 'loss_gene' in checkpoint else float('inf')
        best_loss_gene = checkpoint['best_loss_gene'] if 'best_loss_gene' in checkpoint else float('inf')
        return epoch, best_epoch, loss_gene, best_loss_gene

    def save_gene_a2b(self, save_dir, epoch, best_epoch, loss_gene=None, best_loss_gene=None):
        torch.save({
            'epoch': epoch,
            'best_epoch': best_epoch,
            'loss_gene': loss_gene,
            'best_loss_gene': best_loss_gene,
            'gene_a2b_state_dict': self.gene_a2b.state_dict(),
        }, save_dir)


class Single_Generator(nn.Module):
    def __init__(self, model, model_args=dict(), is_train=True, loss_chans=1, optim_args=dict()):
        super().__init__()
        self.gene_a2b = model(**model_args)

        if is_train:
            self.L1 = nn.L1Loss(reduction="mean")
            self.percept_loss = VGGPerceptualLoss(chans=loss_chans, reduction='mean')
            params = list(self.gene_a2b.parameters())
            self.optimizer = torch.optim.AdamW(params, **optim_args)

    def set_eval(self):
        self.gene_a2b.eval()
        self.set_requires_grad(False)

    def set_lpips(self, device):
        net_lpips = lpips.LPIPS(net='vgg').to(device)
        net_lpips.requires_grad_(False)
        self.net_lpips = net_lpips

    def set_dists(self, device):
        net_dists = DISTS().to(device)
        self.net_dists = net_dists

    def set_train(self):
        self.gene_a2b.train()
        self.set_requires_grad(True)

    def set_requires_grad(self, requires_grad=False):
        nets = [self.gene_a2b]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def load_networks(self, load_dir, load_optim=False):
        map_location = None if torch.cuda.is_available() else torch.device('cpu')
        checkpoint = torch.load(load_dir, map_location=map_location)
        self.gene_a2b.load_state_dict(checkpoint['gene_a2b_state_dict'])
        epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch'] if 'best_epoch' in checkpoint else epoch
        loss_gene = checkpoint['loss_gene']
        best_loss_gene = checkpoint['best_loss_gene'] if 'best_loss_gene' in checkpoint else loss_gene
        if load_optim:
            self.optimizer.load_state_dict(checkpoint['optim_gene_state_dict'])
        return epoch, best_epoch, loss_gene, best_loss_gene

    def save_networks(self, save_dir, epoch, best_epoch, loss_gene, best_loss_gene, net_disc, loss_disc):
        torch.save({
            'epoch': epoch,
            'best_epoch': best_epoch,
            'gene_a2b_state_dict': self.gene_a2b.state_dict(),
            'optim_gene_state_dict': self.optimizer.state_dict(),
            'loss_gene': loss_gene,
            'best_loss_gene': best_loss_gene,
            'disc_b_state_dict': net_disc.disc_b.state_dict(),
            'optim_disc_state_dict': net_disc.optimizer.state_dict(),
            'loss_disc': loss_disc,
        }, save_dir)

    def load_gene_a2b(self, load_dir):
        map_location = None if torch.cuda.is_available() else torch.device('cpu')
        checkpoint = torch.load(load_dir, map_location=map_location)
        self.gene_a2b.load_state_dict(checkpoint['gene_a2b_state_dict'])
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
        best_epoch = checkpoint['best_epoch'] if 'best_epoch' in checkpoint else epoch
        loss_gene = checkpoint['loss_gene'] if 'loss_gene' in checkpoint else float('inf')
        best_loss_gene = checkpoint['best_loss_gene'] if 'best_loss_gene' in checkpoint else float('inf')
        return epoch, best_epoch, loss_gene, best_loss_gene

    def save_gene_a2b(self, save_dir, epoch, best_epoch, loss_gene=None, best_loss_gene=None):
        torch.save({
            'epoch': epoch,
            'best_epoch': best_epoch,
            'loss_gene': loss_gene,
            'best_loss_gene': best_loss_gene,
            'gene_a2b_state_dict': self.gene_a2b.state_dict(),
        }, save_dir)


class Unet8BN(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, mid_channels=[64,128,256,512,512,512,512], num_drop_layers=3, device='cuda'):
        super().__init__()
        self.channels = [in_channels]
        self.channels.extend(mid_channels)
        self.out_channels = out_channels
        conv_args = dict(kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.encoder = []
        self.down_block = []
        self.up_block = []

        def init_block(in_c, out_c, c_args):
            return nn.Sequential(nn.Conv2d(in_c, out_c, **c_args))
        def down_block(in_c, out_c, c_args):
            return nn.Sequential(
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(in_c, out_c, **c_args),
                nn.BatchNorm2d(out_c),
            )
        
        self.encoder.append(init_block(self.channels[0], self.channels[1], conv_args).to(device))
        for n in range(1, len(self.channels)-1-3):
            self.encoder.append(down_block(self.channels[n], self.channels[n+1], conv_args).to(device))

        for n in range(len(self.channels)-1-3, len(self.channels)-1):
            self.down_block.append(down_block(self.channels[n], self.channels[n+1], conv_args).to(device))

        def bottleneck_block(io_channels, c_args, add_drop):
            return nn.Sequential(
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(io_channels, io_channels, **c_args),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(io_channels, io_channels, **c_args),
                nn.BatchNorm2d(io_channels),
                MyDropout(p=0.5, enabled=True) if add_drop else nn.Identity(),
            )
        self.bottleneck_block = [bottleneck_block(self.channels[-1], conv_args, False).to(device)]

        def up_block(in_c, out_c, c_args, add_drop):
            return nn.Sequential(
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_c*2, out_c, **c_args),
                nn.BatchNorm2d(out_c),
                MyDropout(p=0.5, enabled=True) if add_drop else nn.Identity(),
            )
        def last_block(in_c, out_c, c_args):
            return nn.Sequential(
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_c*2, out_c, **c_args),
                nn.Tanh(),
            )

        self.up_block.insert(0, last_block(self.channels[1], self.out_channels, conv_args).to(device))
        for n in range(1, len(self.channels)-1-num_drop_layers):
            self.up_block.insert(0, up_block(self.channels[n+1], self.channels[n], conv_args, False).to(device))
        for n in range(len(self.channels)-1-num_drop_layers, len(self.channels)-1):
            self.up_block.insert(0, up_block(self.channels[n+1], self.channels[n], conv_args, True).to(device))

        self.model = nn.Sequential(*self.encoder, *self.down_block, *self.bottleneck_block, *self.up_block)

    def forward(self, images):
        x = [images]
        for enc in self.encoder:
            x.append(enc(x[-1]))
        for dw in self.down_block:
            x.append(dw(x[-1]))
        y = self.bottleneck_block[0](x[-1])
        x.reverse()
        for n, up in enumerate(self.up_block):
            y = up(torch.cat([x[n], y], dim=1))
        return y


class Unet8(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, mid_channels=[64,128,256,512,512,512,512], num_drop_layers=3, device='cuda'):
        super().__init__()
        self.channels = [in_channels]
        self.channels.extend(mid_channels)
        self.out_channels = out_channels
        conv_args = dict(kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.encoder = []
        self.down_block = []
        self.up_block = []

        def init_block(in_c, out_c, c_args):
            return nn.Sequential(nn.Conv2d(in_c, out_c, **c_args))
        def down_block(in_c, out_c, c_args):
            return nn.Sequential(
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(in_c, out_c, **c_args),
                nn.InstanceNorm2d(out_c, eps=1e-08, momentum=0.1, affine=False, track_running_stats=False),
            )
        
        self.encoder.append(init_block(self.channels[0], self.channels[1], conv_args).to(device))
        for n in range(1, len(self.channels)-1-3):
            self.encoder.append(down_block(self.channels[n], self.channels[n+1], conv_args).to(device))

        for n in range(len(self.channels)-1-3, len(self.channels)-1):
            self.down_block.append(down_block(self.channels[n], self.channels[n+1], conv_args).to(device))

        def bottleneck_block(io_channels, c_args, add_drop):
            return nn.Sequential(
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(io_channels, io_channels, **c_args),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(io_channels, io_channels, **c_args),
                nn.InstanceNorm2d(io_channels, eps=1e-08, momentum=0.1, affine=False, track_running_stats=False),
                MyDropout(p=0.5, enabled=True) if add_drop else nn.Identity(),
            )
        self.bottleneck_block = [bottleneck_block(self.channels[-1], conv_args, False).to(device)]

        def up_block(in_c, out_c, c_args, add_drop):
            return nn.Sequential(
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_c*2, out_c, **c_args),
                nn.InstanceNorm2d(out_c, eps=1e-08, momentum=0.1, affine=False, track_running_stats=False),
                MyDropout(p=0.5, enabled=True) if add_drop else nn.Identity(),
            )
        def last_block(in_c, out_c, c_args):
            return nn.Sequential(
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_c*2, out_c, **c_args),
                nn.Tanh(),
            )

        self.up_block.insert(0, last_block(self.channels[1], self.out_channels, conv_args).to(device))
        for n in range(1, len(self.channels)-1-num_drop_layers):
            self.up_block.insert(0, up_block(self.channels[n+1], self.channels[n], conv_args, False).to(device))
        for n in range(len(self.channels)-1-num_drop_layers, len(self.channels)-1):
            self.up_block.insert(0, up_block(self.channels[n+1], self.channels[n], conv_args, True).to(device))

        self.model = nn.Sequential(*self.encoder, *self.down_block, *self.bottleneck_block, *self.up_block)

    def forward(self, images):
        x = [images]
        for enc in self.encoder:
            x.append(enc(x[-1]))
        for dw in self.down_block:
            x.append(dw(x[-1]))
        y = self.bottleneck_block[0](x[-1])
        x.reverse()
        for n, up in enumerate(self.up_block):
            y = up(torch.cat([x[n], y], dim=1))
        return y


class MyDropout(nn.Module):
    def __init__(self, p=0.5, enabled=True):
        super().__init__()
        self.p = p
        self.enabled = enabled

    def forward(self, x):
        return nn.functional.dropout(x, p=self.p, training=self.enabled)
