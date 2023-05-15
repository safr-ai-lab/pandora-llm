# %%
# !pip install datasets
# !pip install zstandard
# !pip install transformers

import sys
import torch
from torch.utils.data import DataLoader
import json
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers.optimization import AdamW
from common_code import *

# %%
model_title = "pythia-6.9B-deduped"
bs = 8 ## batch size
nbatches = 1000 ## Number of batches to gather data on. Number of data points is bs * nbatches
samplelength = 50 ## How long are the sequences we take from the training and validation sets.
print(model_title, bs, nbatches, samplelength)
####################################################################################

# %%
## Model and tokenizer
model_name = "EleutherAI/" + model_title
model_revision = "step143000"
model_cache_dir = "./"+ model_title +"/"+model_revision

model = GPTNeoXForCausalLM.from_pretrained(
  model_name,
  revision=model_revision,
  cache_dir=model_cache_dir,
)

tokenizer = AutoTokenizer.from_pretrained(
  model_name,
  revision=model_revision,
  cache_dir=model_cache_dir,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.half()
model.eval()
model.to(device)
mem_stats()


# %%
## Collate functions for loading dataset
def collate_fn(batch):
    tokens = [tokenizer.encode(example["text"], return_tensors="pt", truncation=True) for example in batch]
    max_length = max([t.size(1) for t in tokens])
    tokens_padded = [torch.cat([t, t.new_zeros(t.size(0), max_length - t.size(1))], dim=1) for t in tokens]
    tokens_padded = torch.cat(tokens_padded, dim=0)
    return tokens_padded

def collate_already_encoded(batch):
    tokens = batch
    max_length = max([len(t['tokens']) for t in tokens])
    tokens_padded = torch.zeros((len(tokens),max_length),dtype=torch.int)
    for i in range(len(tokens)):
        tokens_padded[i,:] = torch.Tensor(tokens[i]['tokens'])
    return tokens_padded

# %%
## Training and validation datasets
training_dataset = load_dataset("EleutherAI/pile-deduped-pythia-random-sampled", split="train")
validation_dataset = load_dataset("the_pile_val.py", split="validation") 
mem_stats()

# %%
## Dataloader 
training_dataloader = DataLoader(training_dataset, batch_size = bs, collate_fn=collate_already_encoded)
validation_dataloader = DataLoader(validation_dataset, batch_size = bs, collate_fn=collate_fn)
mem_stats()

# %%
train_cross_entropy = compute_dataloader_cross_entropy(training_dataloader, nbatches, bs, device, model, samplelength)
val_cross_entropy = compute_dataloader_cross_entropy(validation_dataloader, nbatches, bs, device, model, samplelength)
mem_stats()

## Save outputs
with torch.no_grad():
    valuestraining   = torch.flatten(train_cross_entropy) 
    valuesvalidation = torch.flatten(val_cross_entropy)

notnan = torch.logical_and(~valuestraining.isnan(), ~valuesvalidation.isnan())
valuestraining = valuestraining[notnan]
valuesvalidation = valuesvalidation[notnan]


## save as pt file
torch.save(torch.vstack((valuestraining, valuesvalidation)), 
model_title + " (Training, Validation) data: bs=" + str(bs)+", nbatches="+str(nbatches)+", length="+str(samplelength)+").pt")

plot_hist(train_cross_entropy, val_cross_entropy, show_plot = True, 
            save_plot=True, 
            plot_title = model_title + " Histogram (bs=" + str(bs)+", nbatches="+str(nbatches)+", length="+str(samplelength)+")",
            plot_name=model_title + " Histogram (bs=" + str(bs)+", nbatches="+str(nbatches)+", length="+str(samplelength)+").png")

plot_ROC(train_cross_entropy, val_cross_entropy, show_plot = True, save_plot = True, log_scale = False, 
        plot_title =model_title + " ROC (bs=" + str(bs)+", nbatches="+str(nbatches)+", length="+str(samplelength)+")", 
        plot_name = model_title + " ROC (bs=" + str(bs)+", nbatches="+str(nbatches)+", length="+str(samplelength)+").png")


