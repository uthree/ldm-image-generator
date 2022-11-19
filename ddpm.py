import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm
from unet import UNet

# Class of DDPM/DDIM.
# This class is able to apply any dimentional Conditional U-Net.
class DDPM(nn.Module):
    # model: model(x: Torch.tensor [Batch_size, *], condition: Torch.tensor, time: Torch.tensor [Batch_size]) -> Torch.tensor
    # x: Gaussian noise
    # condition: Condition vectors(Tokens) or None
    # time: Timestep
    def __init__(self, model=UNet(), beta_min=1e-4, beta_max=0.02, num_timesteps=1000, loss_function = nn.L1Loss()):
        super().__init__()
        self.model = model
        self.beta = torch.linspace(beta_min, beta_max, num_timesteps)
        self.alpha = 1 - self.beta
        self.num_timesteps = num_timesteps
        self.loss_function = loss_function

        # caluclate alpha_bar
        self.alpha_bar = []
        for t in range(1, num_timesteps+1):
            self.alpha_bar.append(torch.prod(self.alpha[:t]))
        self.alpha_bar = torch.Tensor(self.alpha_bar)

        # calculate beta_tilde ( for caluclate sigma[t] )
        self.beta_tilde = [1]
        for t in range(1, num_timesteps):
            self.beta_tilde.append((1-self.alpha_bar[t-1])/(1-self.alpha_bar[t]) * self.beta[t])
        self.beta_tilde = torch.Tensor(self.beta_tilde)

    def calculate_loss(self, x, condition=None):
        t = torch.randint(low=1, high=self.num_timesteps, size=(x.shape[0],))
        alpha_bar_t = torch.index_select(self.alpha_bar, 0, t).to(x.device)
        while alpha_bar_t.ndim < x.ndim:
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
        e = torch.randn(*x.shape, device=x.device)
        t = t.to(x.device)
        e_theta = self.model(x=torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * e, time=t, condition=condition)
        loss = self.loss_function(e_theta, e)
        return loss

    @torch.no_grad()
    def sample(self, x_shape=(1, 3, 64, 64), condition=None, seed=None, use_autocast=True):
        # device
        device = self.model.parameters().__next__().device
        
        # seed
        if seed != None:
            # Python random
            random.seed(seed)
            # Pytorch
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        # Initialize
        x = torch.randn(*x_shape, device=device)
        
        bar = tqdm(total=self.num_timesteps)
        with torch.cuda.amp.autocast(enabled=use_autocast):
            for t in reversed(range(self.num_timesteps)):
                z = torch.randn(*x_shape, device=device)
                sigma = torch.sqrt(self.beta_tilde[t])
                if t == 0:
                    sigma = sigma * 0
                t_tensor = torch.full((x_shape[0],), t, device=device)
                x = (1/torch.sqrt(self.alpha[t])) * (x - ((1-self.alpha[t])/torch.sqrt(1-self.alpha_bar[t])) * self.model(x=x, time=t_tensor, condition=condition)) + sigma * z
                bar.set_description(f"sigma: {sigma.item():.6f}")
                bar.update(1)
        return x
    
    # sample as DDIM (http://arxiv.org/abs/2010.02502)
    @torch.no_grad()
    def sample_implicitly(self, x_shape=(1, 3, 64, 64), condition=None, seed=None, num_steps=25, use_autocast=True, schedule='linear', eta=0):
        # device
        device = self.model.parameters().__next__().device
        
        if seed != None:
            # Python random
            random.seed(seed)
            # Pytorch
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        # Initialize
        x = torch.randn(*x_shape, device=device)

        if schedule == 'linear':
            steps = list(torch.linspace(0, self.num_timesteps-1, num_steps).int().numpy())
        elif type(schedule) == list:
            steps = schedule
        else:
            raise f"schedule \"{schedule}\" is not implemented."
        steps_next = [0] + steps[:-1]
        alpha = torch.cumprod((1-self.beta), dim=0)
        bar = tqdm(total=len(steps))
        with torch.cuda.amp.autocast(enabled=use_autocast):
            for t, t_next in zip(reversed(steps), reversed(steps_next)):
                t_tensor = torch.full((x_shape[0],), t, device=device)
                e_theta = self.model(x=x, time=t_tensor, condition=condition)
                e = torch.randn(*x_shape, device=device)
                sigma = eta * torch.sqrt((1 - alpha[t_next])/(1 - alpha[t])) * torch.sqrt(1 - alpha[t] / alpha[t_next])
                x_t0 = (x - torch.sqrt(1 - alpha[t]) * e_theta) / torch.sqrt(alpha[t])
                term_1 = torch.sqrt(alpha[t_next]) * x_t0
                term_2 = torch.sqrt(1 - alpha[t_next] - sigma**2) * e_theta 
                term_3 = sigma * e
                bar.set_description(f"t: {t}, sigma: {sigma}")
                if t == 0:
                    x = x_t0
                else:
                    x = term_1 + term_2 + term_3
                bar.update(1)
        return x
