import torch
from torch import nn, tensor
from torch.nn import functional as F
from torchvision.models import vgg16
from torchist import histogram
from torch.nn.functional import normalize
import lpips

class lpipsLoss(nn.Module):
    def __init__(self, net='vgg', device='cuda', reduction='mean') -> None:
        super().__init__()
        self.reduction = reduction
        self.net_lpips = lpips.LPIPS(net=net)
        self.net_lpips.to(device)
        self.net_lpips.requires_grad_(False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = self.net_lpips(x, y)
        return loss.mean() if self.reduction=='mean' else loss

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss() 
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, pred, target_is_real):
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(pred)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss.mean()

class mixGANLoss(nn.Module):
    def __init__(self, gan_mode):
        super(mixGANLoss, self).__init__()
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss() 
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, pred, label):
        if type(label)==torch.Tensor:
            return label.to(pred.device)
        elif type(label)==float:
            return torch.FloatTensor([label]).to(pred.device).expand_as(pred)
        else:
            raise NotImplementedError('[%s] is not supported; available "float" or "torch.Tensor"' % type(label))

    def __call__(self, prediction, label):
        target_tensor = self.get_target_tensor(prediction, label)
        loss = self.loss(prediction, target_tensor)
        return loss.mean()

class SSIMLoss(nn.Module):
    def __init__(self, kernel_size: int=11, sigma: float=1.5, k1:float=0.01, k2:float=0.03, use_loss:bool=True, input_nc:int=1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.k1 = k1
        self.k2 = k2
        self.use_loss = use_loss
        self.input_nc = input_nc
        self.gaussian_kernel = self._create_gaussian_kernel(self.kernel_size, self.sigma, self.input_nc)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self.gaussian_kernel.is_cuda:
            self.gaussian_kernel = self.gaussian_kernel.to(x.device)
        ssim_map = self._ssim(x, y)
        if self.use_loss:
            return abs(1. - ssim_map.mean())
        else:
            return ssim_map.mean()
        
    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ux = F.conv2d(x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.input_nc)
        uy = F.conv2d(y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.input_nc)
        uxx = F.conv2d(x * x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.input_nc)
        uyy = F.conv2d(y * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.input_nc)
        uxy = F.conv2d(x * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.input_nc)
        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy
        c1 = self.k1 ** 2
        c2 = self.k2 ** 2
        numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
        denominator = (ux ** 2 + uy ** 2 + c1) * (vx + vy + c2)
        return numerator / (denominator + 1e-12)

    def _create_gaussian_kernel(self, kernel_size:int, sigma:float, input_nc:int=1) -> torch.Tensor:
        start = (1 - kernel_size) / 2
        end = (1 + kernel_size) / 2
        kernel_1d = torch.arange(start, end, step=1, dtype=torch.float)
        kernel_1d = torch.exp(-torch.pow(kernel_1d / sigma, 2) / 2)
        kernel_1d = (kernel_1d / kernel_1d.sum()).unsqueeze(dim=0)
        kernel_2d = torch.matmul(kernel_1d.t(), kernel_1d)
        kernel_2d = kernel_2d.expand(input_nc, 1, kernel_size, kernel_size).contiguous()
        return kernel_2d

class HistoSimLoss(nn.Module):
    def __init__(self, bins=512, low=-1.0, upp=1.0, method='BHATTA', reduction='mean') -> None:
        super().__init__()
        self.bins = bins
        self.low = low
        self.upp = upp
        self.reduction = reduction
        self.method = method

    def bhatta_distance(self, x1: torch.Tensor, x2: torch.Tensor, histogram_normalized=True) -> torch.Tensor:
        assert(x1.shape[-1]==x2.shape[-1])
        bins = 1. if histogram_normalized else torch.sqrt(x1.mean()*x2.mean()*x1.shape[-1]*x2.shape[-1])
        score = 0.
        for _x1, _x2 in zip(x1, x2):
            score += torch.sqrt(_x1*_x2)
        score = torch.sqrt(1. - score / bins)
        return score

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_num, *_ = x.shape
        loss = torch.empty(batch_num)
        for b in range(batch_num):
            x_histo = normalize(histogram(torch.flatten(x, start_dim=1)[b,:], bins=self.bins, low=self.low, upp=self.upp))[0]
            y_histo = normalize(histogram(torch.flatten(y, start_dim=1)[b,:], bins=self.bins, low=self.low, upp=self.upp))[0]
            loss[b] = 1.-F.cosine_similarity(x_histo, y_histo, dim=-1) if self.method=='SIM' else \
                      F.l1_loss(x_histo, y_histo, reduction='sum') if self.method=='L1' else \
                      F.mse_loss(x_histo, y_histo, reduction='sum') if self.method=='MSE' else \
                      self.bhatta_distance(x_histo, y_histo) if self.method=='BHATTA' else \
                      0
        return loss.mean() if self.reduction=='mean' else loss.sum() if self.reduction=='sum' else loss

class VGGPerceptualLoss(nn.Module):
    def __init__(self, chans=1, device="cuda", reduction='mean'):
        super(VGGPerceptualLoss, self).__init__()
        self.reduction = reduction
        vgg = vgg16(pretrained=True)
        if chans==1:
            vgg_ = nn.Sequential(*list(vgg.children())[0])
            vgg_[0].weight = nn.Parameter(vgg_[0].weight.sum(dim=1).unsqueeze(1))
        blocks = []
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:16].eval())
        blocks.append(vgg.features[16:23].eval())
        blocks.append(vgg.features[23:30].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = nn.ModuleList(blocks).to(device)
        self.mean = nn.Parameter(tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad=False).to(device) if chans==3 else \
                    nn.Parameter(tensor([0.5,]).view(1, 1, 1, 1), requires_grad=False).to(device)
        self.std = nn.Parameter(tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad=False).to(device) if chans==3 else \
                   nn.Parameter(tensor([0.5,]).view(1, 1, 1, 1), requires_grad=False).to(device)

    def forward(self, fake, real):
        x = (real - self.mean) / self.std
        y = (fake - self.mean) / self.std
        loss = 0.0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += nn.functional.l1_loss(x, y, reduction=self.reduction)
        return loss
