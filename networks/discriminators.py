import os
import torch
import torch.nn as nn
from .diffaugment import DiffAugment
from vision_aided_loss.cv_discriminator import BlurPool
from vision_aided_loss.cv_losses import losses_list

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, diffaug=True, policy='', diffaugument_args=dict(), device='cuda', activation=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()
        self.encoder = BasicD(in_channels=in_channels, channels=[64,128,256,512], diffaug=diffaug, policy=policy, diffaugument_args=diffaugument_args, device=device)
        self.decoder = SimpleD(in_ch=512, out_ch=256, out_size=32, num_classes=0, activation=activation)

    def train(self, featurier=False, decoder=True):
        self.encoder = self.encoder.train(featurier)
        self.decoder = self.decoder.train(decoder)
        return self

    def forward(self, images, detach=False, diffaug=True):
        if detach:
            with torch.no_grad():
                cv_feat = self.encoder(images, diffaug)
        else:
            cv_feat = self.encoder(images, diffaug)
        pred_mask = [self.decoder(cv_feat[-1], c=None)]
        return pred_mask

class Twin_Discriminator(nn.Module):
    def __init__(self, model=Discriminator, model_args=dict(), featurier_train=True, is_train=True, optim_args=dict(), gan_loss_type='multi_level_hinge'):
        super().__init__()
        self.disc_a = model(**model_args)
        self.disc_b = model(**model_args)
        self.featurier_train = featurier_train

        if is_train:
            params = list(self.disc_a.parameters()) + list(self.disc_b.parameters())
            self.optimizer = torch.optim.AdamW(params, **optim_args)
            self.adv_loss = losses_list(loss_type=gan_loss_type)
            self.l1_loss = nn.L1Loss(reduction="mean")
            self.l2_loss = nn.MSELoss(reduction="mean")

    def set_eval(self):
        nets = [self.disc_a, self.disc_b]
        for net in nets:
            net.train(False)
        self.set_requires_grad(False, False)

    def set_train(self):
        nets = [self.disc_a, self.disc_b]
        for net in nets:
            net.train(True)
        self.set_requires_grad(True, self.featurier_train)

    def set_requires_grad(self, requires_grad=False, encoder_requires_grad=True):
        nets = [self.disc_a, self.disc_b]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
                net.encoder.requires_grad_(encoder_requires_grad)

    def print_networks(self, name="net_discriminator", save_dir=None):
        net = [self.disc_a.encoder.model, self.disc_a.decoder]
        print('---------- Networks initialized -------------')
        print(net)
        if not save_dir==None:
            with open(os.path.join(save_dir, name +".txt"),"w") as o:
                print(net, sep=",", file=o) 
        print('-----------------------------------------------')

    def load_encoders(self, load_dir):
        checkpoint = torch.load(load_dir)
        self.disc_a.encoder.load_state_dict(checkpoint['gene_b2a_encoder_state_dict'])
        self.disc_b.encoder.load_state_dict(checkpoint['gene_a2b_encoder_state_dict'])

    def load_decoders(self, load_dir, load_optim=False):
        checkpoint = torch.load(load_dir)
        self.disc_a.decoder.load_state_dict(checkpoint['disc_a_decoder_state_dict'])
        self.disc_b.decoder.load_state_dict(checkpoint['disc_b_decoder_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss_disc']
        if load_optim:
            self.optimizer.load_state_dict(checkpoint['optim_disc_state_dict'])
        return epoch, loss

    def load_networks(self, load_dir, load_optim=False):
        checkpoint = torch.load(load_dir)
        self.disc_a.load_state_dict(checkpoint['disc_a_state_dict'])
        self.disc_b.load_state_dict(checkpoint['disc_b_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss_disc']
        if load_optim:
            self.optimizer.load_state_dict(checkpoint['optim_disc_state_dict'])
        return epoch, loss
    
    def get_loss_for_G(self, model, generate):
        return self._get_loss(model, generate, True, True, False)

    def get_loss_for_D(self, model, target, generate):
        loss_org = self._get_loss(model, target, True, False, False)
        loss_gen = self._get_loss(model, generate, False, False, True)
        return (loss_org + loss_gen) * 0.5

    def get_loss_for_D_org(self, model, image):
        return self._get_loss(model, image, True, False, False), self._get_cr_loss(model, image)

    def get_loss_for_D_gen(self, model, image):
        return self._get_loss(model, image, False, False, True), self._get_cr_loss(model, image)

    def _get_loss(self, model, image, is_real, is_G, is_detach, is_diffaug=True):
        prediction = model(image, detach=is_detach, diffaug=is_diffaug)
        return self.adv_loss(prediction, for_real=is_real, for_G=is_G)

    def _get_cr_loss(self, model, image):
        pred = model(image, detach=True, diffaug=False)
        pred_diffaug = model(image, detach=True, diffaug=True)
        return self.l2_loss(*pred, *pred_diffaug)

    def cutmix_image(self, original:torch.Tensor, generate:torch.Tensor, mask:torch.Tensor):
        return original*(mask < 0.5) + generate*(mask >= 0.5)

    def _initialize_weights(self, net, init_type, init_gain):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                match init_type:
                    case 'normal':
                        nn.init.normal_(m.weight.data, mean=0.0, std=init_gain)
                    case 'xavier_uniform':
                        nn.init.xavier_uniform_(m.weight.data, gain=init_gain)
                    case 'kaiming_uniform':
                        nn.init.kaiming_uniform_(m.weight.data, mode="fan_in", nonlinearity="leaky_relu")
                    case _:
                        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, init_gain)
                nn.init.constant_(m.bias.data, 0.0)
        print('initialize network with %s' % init_type)
        net.apply(init_func)
    
    def initialize_networks(self, init_type='normal', init_gain=0.02, device='cuda'):
        if 'cuda' in str(device):
            assert(torch.cuda.is_available())
            self = self.to(device)
        self._initialize_weights(self.disc_a, init_type, init_gain)
        self._initialize_weights(self.disc_b, init_type, init_gain)


class Single_Discriminator(nn.Module):
    def __init__(self, model=Discriminator, model_args=dict(), featurier_train=True, is_train=True, optim_args=dict(), gan_loss_type='multi_level_hinge'):
        super().__init__()
        self.disc_b = model(**model_args)
        self.featurier_train = featurier_train

        if is_train:
            params = list(self.disc_b.parameters())
            self.optimizer = torch.optim.AdamW(params, **optim_args)
            self.adv_loss = losses_list(loss_type=gan_loss_type)
            self.l1_loss = nn.L1Loss(reduction="mean")
            self.l2_loss = nn.MSELoss(reduction="mean")

    def set_eval(self):
        nets = [self.disc_b]
        for net in nets:
            net.train(False)
        self.set_requires_grad(False, False)

    def set_train(self):
        nets = [self.disc_b]
        for net in nets:
            net.train(True)
        self.set_requires_grad(True, self.featurier_train)

    def set_requires_grad(self, requires_grad=False, encoder_requires_grad=True):
        nets = [self.disc_b]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
                net.encoder.requires_grad_(encoder_requires_grad)

    def print_networks(self, name="net_discriminator", save_dir=None):
        net = [self.disc_b.encoder.model, self.disc_b.decoder]
        print('---------- Networks initialized -------------')
        print(net)
        if not save_dir==None:
            with open(os.path.join(save_dir, name +".txt"),"w") as o:
                print(net, sep=",", file=o) 
        print('-----------------------------------------------')

    def load_encoders(self, load_dir):
        checkpoint = torch.load(load_dir)
        self.disc_b.encoder.load_state_dict(checkpoint['gene_a2b_encoder_state_dict'])

    def load_decoders(self, load_dir, load_optim=False):
        checkpoint = torch.load(load_dir)
        self.disc_b.decoder.load_state_dict(checkpoint['disc_b_decoder_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss_disc']
        if load_optim:
            self.optimizer.load_state_dict(checkpoint['optim_disc_state_dict'])
        return epoch, loss

    def load_networks(self, load_dir, load_optim=False):
        checkpoint = torch.load(load_dir)
        self.disc_b.load_state_dict(checkpoint['disc_b_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss_disc']
        if load_optim:
            self.optimizer.load_state_dict(checkpoint['optim_disc_state_dict'])
        return epoch, loss
    
    def get_loss_for_G(self, model, generate):
        return self._get_loss(model, generate, True, True, False)

    def get_loss_for_D(self, model, target, generate):
        loss_org = self._get_loss(model, target, True, False, False)
        loss_gen = self._get_loss(model, generate, False, False, True)
        return (loss_org + loss_gen) * 0.5

    def get_loss_for_D_org(self, model, image):
        return self._get_loss(model, image, True, False, False), self._get_cr_loss(model, image)

    def get_loss_for_D_gen(self, model, image):
        return self._get_loss(model, image, False, False, True), self._get_cr_loss(model, image)

    def _get_loss(self, model, image, is_real, is_G, is_detach, is_diffaug=True):
        prediction = model(image, detach=is_detach, diffaug=is_diffaug)
        return self.adv_loss(prediction, for_real=is_real, for_G=is_G)

    def _get_cr_loss(self, model, image):
        pred = model(image, detach=True, diffaug=False)
        pred_diffaug = model(image, detach=True, diffaug=True)
        return self.l2_loss(*pred, *pred_diffaug)

    def cutmix_image(self, original:torch.Tensor, generate:torch.Tensor, mask:torch.Tensor):
        return original*(mask < 0.5) + generate*(mask >= 0.5)

    def _initialize_weights(self, net, init_type, init_gain):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                match init_type:
                    case 'normal':
                        nn.init.normal_(m.weight.data, mean=0.0, std=init_gain)
                    case 'xavier_uniform':
                        nn.init.xavier_uniform_(m.weight.data, gain=init_gain)
                    case 'kaiming_uniform':
                        nn.init.kaiming_uniform_(m.weight.data, mode="fan_in", nonlinearity="leaky_relu")
                    case _:
                        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, init_gain)
                nn.init.constant_(m.bias.data, 0.0)
        print('initialize network with %s' % init_type)
        net.apply(init_func)
    
    def initialize_networks(self, init_type='normal', init_gain=0.02, device='cuda'):
        if 'cuda' in str(device):
            assert(torch.cuda.is_available())
            self = self.to(device)
        self._initialize_weights(self.disc_b, init_type, init_gain)

class BasicD(nn.Module):
    def __init__(self, in_channels=1, channels=[64,128,256,512,512,512,512], diffaug=True, policy='', diffaugument_args=dict(), device='cuda'):
        super().__init__()
        conv_args = dict(kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        model = nn.Sequential(nn.Conv2d(in_channels, channels[0], **conv_args))

        def down_block(in_c, out_c):
            return nn.Sequential(
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(in_c, out_c, **conv_args),
                nn.InstanceNorm2d(out_c, eps=1e-08, momentum=0.1, affine=False, track_running_stats=False),
            )
        for n in range(0, len(channels)-1):
            model.add_module(name='down'+str(n), module=down_block(channels[n], channels[n+1]))
        self.model = model.to(device)
        self.diffaugment = DiffAugment(policy=policy if diffaug else '', **diffaugument_args)

    def __call__(self, images, diffaug=True):
        return [self.model(self.diffaugment(images) if diffaug else images)]

class SimpleD(nn.Module):
    def __init__(self, in_ch=768, out_ch=256, out_size=3, num_classes=0, activation=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()
        self.decoder = nn.Sequential(
            BlurPool(in_ch, pad_type='zero', stride=1, pad_off=1),
            nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2)),
            activation,
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(out_ch*out_size*out_size, out_ch)),
            activation,
        )
        self.out = nn.utils.spectral_norm(nn.Linear(out_ch, 1))
        self.embed = None
        if num_classes > 0:
            self.embed = nn.Embedding(num_classes, out_ch)    

    def forward(self, x, c):
        h = self.decoder(x)
        out = self.out(h)
        if self.embed is not None:
            out += torch.sum(self.embed(c) * h, 1, keepdim=True)
        return out
