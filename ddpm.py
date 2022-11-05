import torch
import torch.nn as nn
import math
import random
from tqdm import tqdm

class DDPM(nn.Module):
    # model: model(x: Torch.tensor [Batch_size, *], condition: Torch.tensor, time: Torch.tensor [Batch_size]) -> Torch.tensor
    # x: Gaussian noise
    # condition: Condition vectors(Tokens) or None
    # time: Timestep
    def __init__(self, model, beta_min=1e-4, beta_max=0.02, num_timesteps=1000, loss_function = nn.L1Loss()):
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

    def caluclate_loss(self, x, condition=None):
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
    def sample(self, x_shape=(1, 3, 64, 64), condition=None, seed=1):
        # device
        device = self.model.parameters().__next__().device

        # Python random
        random.seed(seed)
        # Pytorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Initialize
        x = torch.randn(*x_shape, device=device)
        
        bar = tqdm(total=self.num_timesteps)
        for t in reversed(range(self.num_timesteps)):
            z = torch.randn(*x_shape, device=device)
            sigma = torch.sqrt(self.beta_tilde[t])
            if t == 0:
                sigma = sigma * 0
            t_tensor = torch.full((x_shape[0],), t, device=device)
            x = (1/torch.sqrt(self.alpha[t])) * (x - ((1-self.alpha[t])/torch.sqrt(1-self.alpha_bar[t])) * self.model(x=x, time=t_tensor, condition=condition)) + sigma * z
            bar.set_description(f"sugma: {sigma.item():.6f}")
            bar.update(1)
        return x
