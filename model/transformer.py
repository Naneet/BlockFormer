import torch
import copy
from torch import nn

from .rope import RoPE
from .attention import MHA

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        rope,
        d_model=512,
        nhead=8,
        dim_ff=2048,
        activation=nn.GELU(),
        bias=True,
        dropout=0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = MHA(
            d_model=d_model,
            nhead=nhead,
            rope=rope,
            bias=bias,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.dff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        causal_mask,
        kv_cache=None,
        return_cache=False
    ):
        
        x = input_ids
        residual = x

        if kv_cache != None or return_cache:
            x, kv_cache = self.mha(
                input_ids=self.norm1(x),
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                kv_cache=kv_cache,
                return_cache=return_cache
            )
        else:
            x = self.mha(
                input_ids=self.norm1(x),
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                kv_cache=kv_cache,
                return_cache=return_cache
            )
            
        x = residual + x
        
        residual = x
        x = self.dff(self.norm2(x))
        x = residual + x

        if kv_cache != None or return_cache:
            return x, kv_cache
        else:
            return x




class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoderlayer,
        num_layers
    ):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoderlayer) for _ in range(num_layers)])

    def forward(
        self,
        input_ids,
        attention_mask,
        causal_mask,
        kv_cache=None,
        return_cache=False
    ):
        if return_cache and kv_cache == None:
            use_cache = True
            kv_cache = [None for _ in range(len(self.layers))]
        elif kv_cache is not None:
            use_cache = True
        else:
            use_cache = False
        x = input_ids

        for n, decoderlayer in enumerate(self.layers):
            if use_cache:
                x, kv_cache[n] = decoderlayer(
                    input_ids=x,
                    attention_mask=attention_mask,
                    causal_mask=causal_mask,
                    kv_cache=kv_cache[n],
                    return_cache=return_cache
                )

            elif return_cache:
                x, kv_cache[n] = decoderlayer(
                    input_ids=x,
                    attention_mask=attention_mask,
                    causal_mask=causal_mask,
                    kv_cache=None,
                    return_cache=return_cache
                )

            else:
                x = decoderlayer(
                    input_ids=x,
                    attention_mask=attention_mask,
                    causal_mask=causal_mask,
                    kv_cache=None,
                    return_cache=return_cache
                )

        if kv_cache != None or return_cache:
            return x, kv_cache
        else:
            return x



class BlockFormer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_ff=2048,
        dropout=0.15,
        vocab=16000,
        max_deq_len=2048
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab,
            embedding_dim=d_model,
            padding_idx=0
        )

        rope = RoPE(nhead=nhead,
                   d_model=d_model,
                   max_seq_len=max_deq_len)

        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(
                rope=rope,
                d_model=512,
                nhead=8,
                dim_ff=2048,
                activation=nn.GELU(),
                bias=True,
                dropout=dropout
            ),
            num_layers=num_layers
        )

        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab)
        self.fc.weight = self.embedding.weight

    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        causal_mask=None, 
        kv_cache=None, 
        return_cache=False
    ):

        b, s = input_ids.shape
    
        causal_mask = torch.triu(
            torch.ones(s, s), diagonal=1
        ).bool().to('cuda')
        
        x = self.embedding(input_ids)

        if kv_cache != None or return_cache:
            x, kv_cache = self.decoder(
                input_ids=x,
                causal_mask=causal_mask,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                return_cache=return_cache
            )

            x = self.ln(x)
            logits = self.fc(x)
            return logits, kv_cache

        else:
            x = self.decoder(
                input_ids=x,
                causal_mask=causal_mask,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                return_cache=return_cache
            )

        x = self.ln(x)
        logits = self.fc(x)

        if kv_cache != None or return_cache:
            return logits, kv_cache
            
        else:
            return logits

    @torch.inference_mode()
    def generate(
        self, 
        input_ids, 
        max_new_tokens, 
        use_cache=True
    ):

        max_len = input_ids.size(1) + max_new_tokens
        full_mask = torch.triu(
            torch.ones(max_len, max_len), diagonal=1
        ).bool().to(input_ids.device)
        
        b, s = input_ids.shape
        preds = []
        
        for _ in range(max_new_tokens):
            if _ == 0 and use_cache:
                kv_cache = None
                return_cache=True
                preds = [input_ids]
                
            elif _ == 0 and not use_cache:
                kv_cache = None
                return_cache=False
            
            causal_mask = full_mask[:s, :s]

            if use_cache:
                logits, kv_cache = self.forward(
                    input_ids=input_ids,
                    attention_mask=None,
                    causal_mask=causal_mask,
                    kv_cache=kv_cache,
                    return_cache=return_cache
                )

            else:
                logits = self.forward(
                    input_ids=input_ids,
                    attention_mask=None,
                    causal_mask=causal_mask
                )

            temperature = 1
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            if use_cache:
                preds.append(next_token)
                input_ids = next_token

            else:
                input_ids = torch.cat([input_ids, next_token], dim=1)

            s+=1
        if use_cache:
            preds = torch.cat(preds, dim=1)
            return preds

        else:
            return input_ids