import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

class Conv2dMod(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, eps=1e-8, groups=1, demodulation=True):
        super().__init__()
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
    def __init__(self, channels, style_dim=256):
        super().__init__()
        self.conv = Conv2dMod(channels, channels, kernel_size=3)
        self.l1 = nn.Linear(style_dim, channels)
        self.l2 = nn.Linear(style_dim, channels)

    def forward(self, x, y):
        a = self.l1(y)
        b = self.l2(y).unsqueeze(2).unsqueeze(3)
        x = self.conv(x, a) + b
        return x

class ResBlock(nn.Module):
    def __init__(self, channels, head_dim=32):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels//head_dim)
        self.c2 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.act = nn.LeakyReLU()
    
    def forward(self, x):
        return self.c2(self.act(self.c1(x))) + x

class Encoder(nn.Module):
    def __init__(self, input_channels, latent_channels, channels=[48, 96, 192, 384], stages=[3, 3, 4, 4], style_dim=256):
        super().__init__()
        self.input_conv = nn.Conv2d(input_channels, channels[0], 1, 1, 0)
        self.output_conv = nn.Conv2d(channels[-1], latent_channels*2, 1, 1, 0)
        self.to_style = nn.Linear(channels[-1], style_dim)
        self.stages = nn.ModuleList([])
        self.downsamples = nn.ModuleList([])
        for i, (c, n) in enumerate(zip(channels, stages)):
            if i == len(stages)-1:
                self.downsamples.append(nn.Identity())
            else:
                self.downsamples.append(nn.Conv2d(channels[i], channels[i+1], 2, 2, 0))
            self.stages.append(nn.ModuleList([ResBlock(c) for _ in range(n)]))

    def forward(self, x):
        x = self.input_conv(x)
        for st, ds in zip(self.stages, self.downsamples):
            for l in st:
                x = l(x)
            x = ds(x)
        mean, logvar = torch.chunk(self.output_conv(x), 2, dim=1)
        style = self.to_style(x.mean(dim=(2, 3)))
        return mean, logvar, style

class DecoderBlock(nn.Module):
    def __init__(self, channels, style_dim):
        super().__init__()
        self.c1 = StyleBlock(channels, style_dim)
        self.c2 = StyleBlock(channels, style_dim)
        self.act = nn.LeakyReLU()

    def forward(self, x, style):
        res = x
        x = self.c1(x, style)
        x = self.act(x)
        x = self.c2(x, style)
        x = x + res
        return x

class Mapper(nn.Module):
    def __init__(self, style_dim, num_layers=4):
        super().__init__()
        seq = []
        for _ in range(num_layers):
            seq.append(nn.Linear(style_dim, style_dim))
            seq.append(nn.LeakyReLU())
        seq.append(nn.Linear(style_dim, style_dim))
        self.seq = nn.Sequential(*seq)

    def forward(self, style):
        return self.seq(style)

class Decoder(nn.Module):
    def __init__(self, output_channels, latent_channels, channels=[384, 192, 96, 48], stages=[4, 4, 3, 3], style_dim=256):
        super().__init__()
        self.mapper = Mapper(style_dim)
        self.input_conv = nn.Conv2d(latent_channels, channels[0], 1, 1, 0)
        self.output_conv = Conv2dMod(channels[-1], output_channels, kernel_size=1, demodulation=False)
        self.output_affine = nn.Linear(style_dim, output_channels)
        self.to_style = nn.Linear(channels[-1], style_dim)
        self.stages = nn.ModuleList([])
        self.upsamples = nn.ModuleList([])
        for i, (c, n) in enumerate(zip(channels, stages)):
            if i == len(stages)-1:
                self.upsamples.append(nn.Identity())
            else:
                self.upsamples.append(nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2, 0))
            self.stages.append(nn.ModuleList([DecoderBlock(c, style_dim=style_dim) for _ in range(n)]))

    def forward(self, x, style):
        style = self.mapper(style)
        x = self.input_conv(x)
        for st, us in zip(self.stages, self.upsamples):
            for l in st:
                x = l(x, style)
            x = us(x)
        x = self.output_conv(x, self.output_affine(style))
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, channels=[32, 64, 128, 256], stages=[2, 2, 2, 2]):
        super().__init__()
        self.input_conv = nn.Conv2d(input_channels, channels[0], 1, 1, 0)
        self.output_conv = nn.Conv2d(channels[-1], 1, 1, 1, 0)
        self.stages = nn.ModuleList([])
        self.downsamples = nn.ModuleList([])
        for i, (c, n) in enumerate(zip(channels, stages)):
            if i == len(stages)-1:
                self.downsamples.append(nn.Identity())
            else:
                self.downsamples.append(nn.Conv2d(channels[i], channels[i+1], 2, 2, 0))
            self.stages.append(nn.ModuleList([ResBlock(c) for _ in range(n)]))

    def forward(self, x, r):
        r.requires_grad = False
        x = self.input_conv(x)
        r = self.input_conv(r)
        feat_match_loss = 0
        for st, ds in zip(self.stages, self.downsamples):
            for l in st:
                x = l(x)
                r = l(r)
                feat_match_loss += (r-x).abs().mean()
            x = ds(x)
            r = ds(r)
        logits = self.output_conv(x)
        return logits, feat_match_loss

    def discriminate(self, x):
        x = self.input_conv(x)
        for st, ds in zip(self.stages, self.downsamples):
            for l in st:
                x = l(x)
            x = ds(x)
        logits = self.output_conv(x)
        return logits

# VAE (with style)
class VAE(nn.Module):
    def __init__(self, channels=3, latent_channels=4, style_dim=256):
        super().__init__()
        self.style_dim = style_dim
        self.encoder = Encoder(channels, latent_channels, style_dim=style_dim)
        self.decoder = Decoder(channels, latent_channels, style_dim=style_dim)

    def train_encdec(self, dataset, batch_size=1, num_epoch=100, weight_kl=1, weight_recon=1, vae_save_path="./vae.pt", use_autocast=True, lr=1e-4):
        dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.RAdam(self.parameters(), lr=lr)
        scaler = torch.cuda.amp.GradScaler(enabled=use_autocast)
        device = self.parameters().__next__().device
        print(f"device: {device}")
        for epoch in range(num_epoch):
            print(f"epoch {epoch}")
            bar = tqdm(total=len(dataset))
            for batch, x in enumerate(dl):
                x = x.to(device)
                N = x.shape[0]
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_autocast):
                    mean, logvar, style = self.encoder(x)
                    z = torch.randn(*mean.shape, device=device) * torch.sqrt(torch.exp(logvar)) + mean
                    y = self.decoder(z, style)
                    loss_recon = (x-y).abs().mean() * weight_recon
                    loss_kl = (-1 - logvar + torch.exp(logvar)).mean() * weight_kl
                    loss = loss_recon + loss_kl
                loss = scaler.scale(loss)
                loss.backward()
                scaler.step(optimizer)
                scaler.update()
                bar.set_description(f"Recon: {loss_recon.item():.6f}, KL: {loss_kl.item():.6f}")
                bar.update(N)
                if batch % 200 == 0:
                    torch.save(self.state_dict(), vae_save_path)

    def train_decoder(self, dataset, batch_size=1, lr=1e-4, num_epoch=100, weight_adv=1, weight_recon=1, weight_feat=1, discriminator=None, vae_save_path="./vae.pt", discriminator_save_path="./discriminator.pt", use_autocast=True):
        dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer_g = optim.RAdam(self.parameters(), lr=lr)
        optimizer_d = optim.RAdam(discriminator.parameters(), lr=lr)
        scaler = torch.cuda.amp.GradScaler(enabled=use_autocast)
        device = self.parameters().__next__().device
        discriminator = discriminator.to(device)
        print(f"device: {device}")
        for epoch in range(num_epoch):
            print(f"epoch {epoch}")
            bar = tqdm(total=len(dataset))
            for batch, x in enumerate(dl):
                x = x.to(device)
                N = x.shape[0]

                # Train G.
                optimizer_g.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_autocast):
                    z, style = self.encode(x, sigma=0)
                    y = self.decoder(z, style)
                    loss_recon = (x-y).abs().mean() * weight_recon
                    logits, loss_feat = discriminator(y, x)
                    loss_adv = F.relu(-logits).mean() * weight_adv
                    loss_feat = loss_feat * weight_feat
                    loss = loss_recon + loss_feat + loss_adv
                loss = scaler.scale(loss)
                loss.backward()
                scaler.step(optimizer_g)

                # Train D.
                y = y.detach()
                optimizer_d.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_autocast):
                    loss_fake = F.relu(discriminator.discriminate(y)).mean()
                    loss_real = F.relu(discriminator.discriminate(x)).mean()
                    loss_d = loss_fake + loss_real
                loss_d = scaler.scale(loss_d)
                loss_d.backward()

                scaler.update()
                bar.set_description(f"Recon: {loss_recon.item():.6f}, Adv: {loss_adv.item():.6f}, Feat: {loss_feat.item():.6f}, Disc: {loss_fake.item()+loss_real.item():.6f}")
                bar.update(N)
                if batch % 200 == 0:
                    torch.save(self.state_dict(), vae_save_path)
                    torch.save(discriminator.state_dict(), discriminator_save_path)

    @torch.no_grad()
    def encode(self, x, sigma=0):
        mean, logvar, style = self.encoder(x)
        z = torch.randn(*mean.shape, device=x.device) * torch.sqrt(torch.exp(logvar)) * sigma + mean
        return mean, style

    @torch.no_grad()
    def decode(self, z, style=None):
        if style == None:
            style = torch.randn(1, self.style_dim, device=z.device)
        return self.decoder(z, style)


