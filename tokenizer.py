from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import pandas as pd

df = pd.read_csv('/kaggle/input/datasets/thedevastator/tinystories-narrative-classification/train.csv')

df["text"].to_csv(
    "data.txt",
    index=False,
    header=False
)

tokenizer = Tokenizer(BPE(unk_token="<unk>"))

tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=16000,
    special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]
)

tokenizer.train(["data.txt"], trainer)

tokenizer.save("BlockFormer.json")

encoding = tokenizer.encode("The capital of India is Delhi")

print(encoding.ids)
print(tokenizer.decode(encoding.ids))
