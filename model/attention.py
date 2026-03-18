import torch
from torch import nn

class MHA(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        rope,
        bias=True,
        dropout=0.0,
    ):
        super().__init__()

        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        assert self.head_dim % 2 == 0

        self.rope = rope

        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids,
        causal_mask=None,
        attention_mask=None,
        kv_cache=None,
        return_cache=False
    ):
        
        B, S, E = input_ids.shape
        query = self.w_q(input_ids).view(B, S, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        key = self.w_k(input_ids).view(B, S, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        value = self.w_v(input_ids).view(B, S, self.nhead, self.head_dim).permute(0, 2, 1, 3)

        if kv_cache != None:
            offset = kv_cache[0].shape[2]
            
            key = self.rope(key, offset=offset)
            key = torch.cat([kv_cache[0], key], dim=2)
            
            value = torch.cat([kv_cache[1], value], dim=2)
            
            kv_cache = [key, value]

            query = self.rope(query, offset=offset)

        elif return_cache:
            kv_cache = [key, value]

        else:
            query = self.rope(query)
            key = self.rope(key)

        attention = query @ key.transpose(-1, -2)
        attention = attention / (self.head_dim ** 0.5)

        if causal_mask is not None:
            causal_mask = causal_mask[None, None, :, :]
            attention = attention.masked_fill(causal_mask, float('-inf'))

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention = attention.masked_fill(attention_mask, float('-inf'))

        attention = torch.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        output = attention @ value
        output = output.transpose(1, 2).contiguous().view(B, S, self.d_model)
        output = self.w_o(output)
        output = self.dropout(output)

        if kv_cache == None and not return_cache:
            return output

        elif kv_cache != None or return_cache:
            return output, kv_cache