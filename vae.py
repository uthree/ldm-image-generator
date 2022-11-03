import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.c2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.1)
    def forward(self, x):
        res = x
        x = self.c1(x)
        x = self.act(x)
        x = self.c2(x)
        return x + res

class Conv2dMod(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, eps=1e-8, groups=1, demodulation=True):
        super(Conv2dMod, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_channels, input_channels // groups, kernel_size, kernel_size, dtype=torch.float32))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu') # initialize weight
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.demodulation = demodulation
        self.groups = groups

    def forward(self, x, y):
        # x: (batch_size, input_channels, H, W)
        # y: (batch_size, output_channels)
        # self.weight: (output_channels, input_channels, kernel_size, kernel_size)
        N, C, H, W = x.shape

        # reshape weight
        w1 = y[:, None, :, None, None]
        w1 = w1.swapaxes(1, 2)
        w2 = self.weight[None, :, :, :, :]
        # modulate
        weight = w2 * (w1 + 1)

        # demodulate
        if self.demodulation:
            d = torch.rsqrt((weight ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weight = weight * d
        # weight: (batch_size, output_channels, input_channels, kernel_size, kernel_size)

        # reshape
        x = x.reshape(1, -1, H, W)
        _, _, *ws = weight.shape
        weight = weight.reshape(self.output_channels * N * self.groups, *ws)

        # padding
        x = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2), mode='replicate')

        # convolution
        x = F.conv2d(x, weight, stride=1, padding=0, groups=N)
        x = x.reshape(N, self.output_channels, H, W)
        return x

class StyleBlock(nn.Module):
    def __init__(self, channels, style_dim=512):
        super().__init__()
        self.conv = Conv2dMod(channels, channels, kernel_size=3)
        self.l1 = nn.Linear(style_dim, channels)
        self.l2 = nn.Linear(style_dim, channels)

    def forward(self, x, y):
        a = self.l1(y)
        b = self.l2(y).unsqueeze(2).unsqueeze(3)
        x = self.conv(x, a) + b
        return x

class DecoderLayer(nn.Module):
    def __init__(self, channels, num_blocks=2, style_dim=512, output_channels=3):
        super().__init__()
        self.blocks = nn.ModuleList([StyleBlock(channels, style_dim) for _ in range(num_blocks)])
        self.l1 = nn.Linear(style_dim, output_channels)
        self.l2 = nn.Linear(style_dim, output_channels)
        self.torgb = Conv2dMod(channels, output_channels, kernel_size=1, demodulation=False)

    def forward(self, x, y):
        for blk in self.blocks:
            x = blk(x, y)
        a = self.l1(y)
        b = self.l2(y).unsqueeze(2).unsqueeze(3)
        rgb = self.torgb(x, a) + b
        return x, rgb

class Encoder(nn.Module):
    def __init__(self, input_channels=3, latent_channels=3, channels=[64, 128, 256], stages=[4, 4, 4], style_dim=512):
        super().__init__()
        self.stacks = nn.ModuleList([])
        self.downsamples = nn.ModuleList([])
        self.input_conv = nn.Conv2d(input_channels, channels[0], 1, 1, 0)
        self.output_conv = nn.Conv2d(channels[-1], latent_channels*2, 1, 1) # to mean, logvar
        self.to_style = nn.Linear(channels[-1], style_dim*2)
        for i, (c, s) in enumerate(zip(channels, stages)):
            self.stacks.append(nn.Sequential(*[ResBlock(c) for _ in range(s)]))
            if i != len(channels)-1:
                self.downsamples.append(nn.Conv2d(channels[i], channels[i+1], 2, 2, 0))
            else:
                self.downsamples.append(nn.Identity())

    def forward(self, x):
        # encode
        x = self.input_conv(x)
        for s, d in zip(self.stacks, self.downsamples):
            x = s(x)
            x = d(x)
        style_mean, style_logvar = torch.chunk(self.to_style(x.mean(dim=(2,3))), 2, dim=1)
        x = self.output_conv(x)
        mean, logvar = torch.chunk(x, 2, dim=1)

        return mean, logvar, style_mean, style_logvar

class Mapper(nn.Module):
    def __init__(self, style_dim, num_layers=7):
        super().__init__()
        seq = []
        for _ in range(num_layers):
            seq.append(nn.Linear(style_dim, style_dim))
            seq.append(nn.LeakyReLU(0.1))
        seq.append(nn.Linear(style_dim, style_dim))
        self.seq = nn.Sequential(*seq)

    def forward(self, style):
        return self.seq(style)

class Decoder(nn.Module):
    def __init__(self, output_channels=3, latent_channels=3, channels=[256, 128, 64], stages=[4, 4, 4], style_dim=512):
        super().__init__()
        self.stacks = nn.ModuleList([])
        self.upsamples = nn.ModuleList([])
        self.input_conv = nn.Conv2d(latent_channels, channels[0], 1, 1, 0)
        self.mapper = Mapper(style_dim)
        for i, (c, s) in enumerate(zip(channels, stages)):
            self.stacks.append(DecoderLayer(c, s, style_dim=style_dim, output_channels=output_channels))
            if i != len(channels)-1:
                self.upsamples.append(nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2, 0))
            else:
                self.upsamples.append(nn.Identity())
    
    def forward(self, x, style):
        x = self.input_conv(x)
        rgb_out = None
        style = self.mapper(style)
        for s, u in zip(self.stacks, self.upsamples):
            x, rgb = s(x, style)
            if rgb_out == None:
                rgb_out = rgb
            else:
                rgb_out = rgb + F.interpolate(rgb_out, scale_factor=2)
            x = u(x)
        return rgb_out

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def caluclate_loss(self, x):
        mean, logvar, style_mean, style_logvar = self.encoder(x)
        gaussian_style = torch.randn(*style_mean.shape, device=x.device)
        gaussian_feat = torch.randn(*mean.shape, device=x.device)
        z = gaussian_feat * torch.exp(logvar) + mean
        style = gaussian_style * torch.exp(style_logvar) + style_mean
        y = self.decoder(z, style)
        reconstruction_loss = torch.abs(x - y).mean()
        kl_feat = -1 - logvar.mean() + torch.exp(logvar).mean() + (mean**2).mean()
        kl_style = -1 - style_logvar.mean() + torch.exp(style_logvar).mean() + (style_mean**2).mean()
        kl_divergence_loss = kl_feat + kl_style
        return reconstruction_loss, kl_divergence_loss, y

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, channels=[64, 128, 256], stages=[2, 2, 4]):
        super().__init__()
        self.stacks = nn.ModuleList([])
        self.downsamples = nn.ModuleList([])
        self.input_conv = nn.Conv2d(input_channels, channels[0], 1, 1, 0)
        self.to_logit = nn.Conv2d(channels[-1], 1, 1, 1, 0)
        for i, (c, s) in enumerate(zip(channels, stages)):
            self.stacks.append(nn.Sequential(*[ResBlock(c) for _ in range(s)]))
            if i != len(channels)-1:
                self.downsamples.append(nn.Conv2d(channels[i], channels[i+1], 2, 2, 0))
            else:
                self.downsamples.append(nn.Identity())

    def forward(self, x):
        x = self.input_conv(x)
        for s, d in zip(self.stacks, self.downsamples):
            x = s(x)
            x = d(x)
        x = self.to_logit(x)
        return x

