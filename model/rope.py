import torch
from torch import nn

class RoPE(nn.Module):
    def __init__(
        self,
        nhead,
        d_model,
        max_seq_len
    ):
        super().__init__()
        head_dim = d_model // nhead
        assert head_dim % 2 == 0
        d = head_dim

        k = torch.arange(head_dim // 2, dtype=torch.float32)
        inv_freq = 10000 ** (-2 * k / head_dim)

        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)

        cos_cache = torch.cos(angles).repeat_interleave(repeats=2, dim=-1)[None, None, :, :]
        sin_cache = torch.sin(angles).repeat_interleave(repeats=2, dim=-1)[None, None, :, :]

        self.register_buffer('cos_cache', cos_cache)
        self.register_buffer('sin_cache', sin_cache)


    def rotate(self, x):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotate_even = -x_odd
        rotate_odd = x_even

        rotate = torch.stack([rotate_even, rotate_odd], dim=-1).reshape(*x.shape)

        return rotate

    def forward(self, x, offset=None):
        B, H, S, D = x.shape

        if offset == None:
            cos = self.cos_cache[:,:,:S,:]
            sin = self.sin_cache[:,:,:S,:]

        else:
            cos = self.cos_cache[:,:, offset:offset+S,:]
            sin = self.sin_cache[:,:, offset:offset+S,:]

        x = x * cos + self.rotate(x) * sin

        return x