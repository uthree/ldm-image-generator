import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowAttention(nn.Module):
    def __init__(self, channels=512, n_heads=8, window_size=4, shift=0):
        super().__init__()
        self.attention = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.window_size = window_size
        self.shift = shift
    
    # x: [N, C, H, W]
    def forward(self, x):
        # Apply attention without window if image size is smaller than window size
        if x.shape[2] <= self.window_size and x.shape[3] <= self.window_size:
            return self._apply_attention_without_window(x)

        # Padding
        N, C, H, W = x.shape
        ws = self.window_size
        pad_h, pad_w = ws - (H % ws), ws - (W % ws)

        x = torch.cat([x, torch.zeros(1, 1, pad_h, 1, device=x.device).expand(N, C, pad_h, x.shape[3])], dim=2)
        x = torch.cat([x, torch.zeros(1, 1, 1, pad_w, device=x.device).expand(N, C, x.shape[2], pad_w)], dim=3)
        
        # Generate Padding mask
        mask = torch.zeros(1, 1, x.shape[2], x.shape[3], dtype=bool, device=x.device).expand(N, C, x.shape[2], x.shape[3])
        mask[:, :, H:, :] = 1
        mask[:, :, :, W:] = 1

        # Shift
        x = torch.roll(x, (self.shift, self.shift), (2, 3))
        mask = torch.roll(x, (self.shift, self.shift), (2, 3))

        # Split
        nwin_h, nwin_w = x.shape[2] // ws, x.shape[3] // ws
        x = self._split_window(x)
        mask = self._split_window(mask)

        # Attention
        x = self._apply_attention(x, mask)

        # Concat
        x = self._concat_window(x, nwin_h, nwin_w)

        # Unshift
        x = torch.roll(x, (-self.shift, -self.shift), (2, 3))

        # Remove pad
        x = x[:, :, :H, :W]

        return x
    
    def _split_window(self, x):
        ws = self.window_size
        x = torch.cat(torch.split(x, ws, dim=3), dim=2)
        x = torch.cat(torch.split(x, ws, dim=2), dim=0)
        return x

    def _concat_window(self, x, nwin_h, nwin_w):
        x = torch.cat(torch.chunk(x, nwin_h*nwin_w, dim=0), dim=2)
        x = torch.cat(torch.chunk(x, nwin_w, dim=2), dim=3)
        return x
    
    def _apply_attention(self, x, mask):
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1) # N, C, L
        x = x.transpose(1, 2) # N, L, C
        mask = mask.reshape(shape[0], shape[1], -1) # N, C, L
        mask = mask.transpose(1, 2) # N, L, C
        mask = mask[:, :, 0] # N, L
        x, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = x.transpose(1, 2) # N, C, L
        x = x.reshape(*shape)
        return x
    
    def _apply_attention_without_window(self, x):
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1) # N, C, L
        x = x.transpose(1, 2) # N, L, C
        x, _ = self.attention(x, x, x)
        x = x.transpose(1, 2) # N, C, L
        x = x.reshape(*shape)
        return x

