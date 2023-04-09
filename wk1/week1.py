### Imports

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers.optimization import AdamW

### Model Specs

model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)

model_name = "EleutherAI/pythia-70m-deduped"
model_revision = "step3000"
model_cache_dir = "./pythia-70m-deduped/step3000"
bs = 16

### Loading the Data

def collate_fn(batch):
    tokens = [tokenizer.encode(example["text"], return_tensors="pt", truncation=True) for example in batch]
    max_length = max([t.size(1) for t in tokens])
    tokens_padded = [torch.cat([t, t.new_zeros(t.size(0), max_length - t.size(1))], dim=1) for t in tokens]
    tokens_padded = torch.cat(tokens_padded, dim=0)
    return tokens_padded

val_dataset = load_dataset("the_pile_val.py", split="validation") # should possibly be only one subset of the data (e.g. enron_emails)
dataloader = DataLoader(val_dataset, batch_size = bs, collate_fn=collate_fn)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Sanity Testing
first_batch = next(iter(dataloader))
x = first_batch
y = [val_dataset[i]["meta"] for i in range(4)]
print("Text (x):", x)
print("Metadata (y):", y)

### Training Shadow Learners - not really necessary, since we're just loss thresholding and not fine-tuning on any new data

N = 2

# Initialize the model and optimizer
models = [GPTNeoXForCausalLM.from_pretrained(model_name, revision=model_revision, cache_dir=model_cache_dir).to(device) for i in range(N)]

### Membership Inference Attacks

# In-distribution data (split dataset into chunks of size 1/(N))
dataset = torch.utils.data.random_split(val_dataset, [1/(N) for i in range(N)])
dataloaders = [DataLoader(dataset[i], batch_size = bs, collate_fn=collate_fn) for i in range(N)]

# Get data that is out of distribution - see data sheet (https://arxiv.org/abs/2201.07311)
# TODO

# Get loss on these tokens fragments vs in-dist data
# TODO

# Loss Thresholding
threshold = 0.1

# Compute loss on N shadow learners vs. true dataset on each of N dataset parts (train/test)
# TODO

# Compute ROC curve for different loss thresholds
# TODO