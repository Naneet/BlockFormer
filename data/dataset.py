import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class TinyStories(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df['text'].iloc[idx]
        tokens = self.tokenizer.encode(text).ids
        input_ids = torch.tensor(tokens[:-1])
        target = torch.tensor(tokens[1:])
        return input_ids, target

def collate_fn(batch):
    
    input_ids, target = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    target = pad_sequence(target, batch_first=True, padding_value=0)

    b, s = input_ids.shape

    attention_mask = (input_ids == 0)

    causal_mask = torch.triu(
        torch.ones(s, s), diagonal=1
    ).bool()

    return input_ids, attention_mask, causal_mask, target