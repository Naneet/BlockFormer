from data import *
from model import BlockFormer

import pandas as pd
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
import torch
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


train_df = pd.read_csv('/kaggle/input/datasets/thedevastator/tinystories-narrative-classification/train.csv').dropna()
test_df = pd.read_csv('/kaggle/input/datasets/thedevastator/tinystories-narrative-classification/validation.csv')

tokenizer = Tokenizer.from_file('data/BlockFormer.json')
bos_id = tokenizer.token_to_id("<bos>")
eos_id = tokenizer.token_to_id("<eos>")

tokenizer.post_processor = TemplateProcessing(
    single="<bos> $A <eos>",
    special_tokens=[
        ("<bos>", bos_id),
        ("<eos>", eos_id),
    ],
)

train_dataset = TinyStories(df=train_df, tokenizer=tokenizer)
test_dataset = TinyStories(df=test_df, tokenizer=tokenizer)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    prefetch_factor=10
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=4,
    prefetch_factor=8
)

scaler = GradScaler('cuda')

def train_step(model, optimizer, dataloader, epoch, device="cuda"):
    model.train()
    train_loss = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True)
    
    for batch_idx, (input_ids, attention_mask, causal_mask, target) in enumerate(progress_bar):
        input_ids, attention_mask, causal_mask, target = input_ids.to(device), attention_mask.to(device), causal_mask.to(device), target.to(device)

        with autocast('cuda'):
            logits = model(input_ids, attention_mask, causal_mask)
            loss = F.cross_entropy(logits.permute(0,2,1), target, ignore_index=0, label_smoothing=0.1)

            batch_loss = loss.item()
            train_loss += batch_loss

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        avg_loss = train_loss / (batch_idx + 1)
        progress_bar.set_postfix(batch_loss=f"{batch_loss:.4f}", avg_loss=f"{avg_loss:.4f}")

        del input_ids, attention_mask, causal_mask, loss, logits

    avg_loss = train_loss / len(dataloader)

    print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f}")
    return avg_loss



def test_step(model, epoch, dataloader, device="cuda"):
    model.eval()
    test_loss, total_tokens = 0, 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} (Eval)", dynamic_ncols=True)

    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, causal_mask, target) in enumerate(progress_bar):
            input_ids, attention_mask, causal_mask, target = input_ids.to(device), attention_mask.to(device), causal_mask.to(device), target.to(device)

            with autocast('cuda'):
                logits = model(input_ids, attention_mask, causal_mask)
                loss = F.cross_entropy(logits.permute(0,2,1), target, ignore_index=0, label_smoothing=0.1, reduction='sum')
    
                batch_loss = loss.item()
                test_loss += batch_loss
                batch_tokens = (~attention_mask).sum().item()
                total_tokens += batch_tokens

            batch_ppl = torch.exp(torch.tensor(batch_loss / batch_tokens))
            progress_bar.set_postfix(batch_ppl=f"{batch_ppl:.4f}")
    
            del input_ids, attention_mask, causal_mask, loss, logits

    avg_loss = test_loss / total_tokens
    avg_nll = test_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_nll))

    print(f"Epoch {epoch} | Test Loss: {avg_loss:.4f} | Test Preplexity: {ppl:.4f}")
    return avg_loss, ppl

torch.manual_seed(42)
torch.cuda.manual_seed(42)

model = BlockFormer(
    d_model=512,
    nhead=8,
    num_layers=6,
    dim_ff=2048,
    dropout=0.15,
    vocab=16000,
    max_deq_len=2048
).to('cuda')

model = nn.DataParallel(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=5,   
    eta_min=3e-5       
)

# checkpoint = torch.load("/kaggle/input/notebooks/oliveseed/blockformer-trainer/BlockFormer_13_epochs.pth", map_location="cuda")

# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# print("Checkpoint loaded successfully.")


current = 0
num_epochs = 30

start = 1 + current
epochs = num_epochs + current + 1

print(f"Running from {start} to {epochs-1}")

torch.manual_seed(42)
torch.cuda.manual_seed(42)

print("Ready to train!!")

for epoch in range(start, epochs):
    train_step(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        dataloader=train_dataloader
    )

    test_step(
        model=model,
        epoch=epoch,
        dataloader=test_dataloader,
    )
    # scheduler.step()
    checkpoint = {
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(obj=checkpoint, f=f"BlockFormer_{epoch}_epochs.pth")