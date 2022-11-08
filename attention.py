import torch
import torch.nn as nn
import torch.nn.functional as F

# LSH Attention ( without masking )
class LSHAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, bucket_size=16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.proj_qk = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.proj_o = nn.Linear(d_model, d_model)
        self.bucket_size = bucket_size
    
    # x: [BatchSize, Length, D_Model]
    # keys: [BatchSize, Length, D_Model]
    def forward(self, x, keys=None, mask=None):
        L = x.shape[1]

        # concat x and keys if keys is not None
        if keys != None:
            x = torch.cat([x, keys], dim=1)

        # padding
        pad_len = self.bucket_size - (x.shape[1] % self.bucket_size)
        pads = torch.zeros(x.shape[0], pad_len, self.d_model, device=x.device)
        x = torch.cat([x, pads], dim=1)
        
        # projections
        qk = self.proj_qk(x)
        v = self.proj_v(x)

        # split by heads [N, L, dModel]
        qk_list = torch.chunk(qk, self.n_heads, dim=2)
        v_list = torch.chunk(v, self.n_heads, dim=2)

        x = []
        for qk, v in zip(qk_list, v_list):
            with torch.no_grad():
                # hash projection
                hout = torch.matmul(qk, torch.randn(self.d_model // self.n_heads, 2, device=qk.device))
                hx, hy = hout[:, :, 0], hout[:, :, 1]
                # sort by hash
                _, indices = torch.sort(torch.atan(hx / hy), dim=1)
                indices = indices.unsqueeze(2).expand(qk.shape)
            # sort qk and v
            qk = torch.scatter(qk, 1, indices, qk)
            v = torch.scatter(v, 1, indices, v)

            # split by buckets and concat as batch
            num_buckets = qk.shape[1] // self.bucket_size
            qk = torch.cat(torch.chunk(qk, num_buckets, dim=1), dim=0)
            v = torch.cat(torch.chunk(v, num_buckets, dim=1), dim=0)

            # caluclate attention scores
            scores = torch.bmm(qk, qk.transpose(1, 2))
                        
            # diag-mask
            mask = torch.diag(torch.ones(self.bucket_size, device=qk.device)).bool()
            scores.masked_fill_(mask.unsqueeze(0).expand(scores.shape), -float('Inf'))
            scores = F.softmax(scores, -1)
            
            # weighted-sum
            o = torch.bmm(scores, v)

            # split batch and concatenate
            o = torch.cat(torch.chunk(o, num_buckets, dim=0), dim=1)

            # unsort
            o = o.gather(1, indices)
            x.append(o)

        # concatenate heads
        x = torch.cat(x, dim=2)

        # projection
        x = self.proj_o(x)
        
        # remove keys and paddings
        x = x[:, :L]

        return x
