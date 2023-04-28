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

# %%
model_title = "pythia-6.9B-deduped"
bs = 8 ## batch size
nbatches = 500 ## Number of batches to gather data on. Number of data points is bs * nbatches
samplelength = 50 ## How long are the sequences we take from the training and validation sets.
print(model_title, bs, nbatches, samplelength)
####################################################################################
# %%
## Memory statistics. I had to be careful with cuda memory 
def mem_stats():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    print("total memory: ", t, "reserve mem: ", r, "total-reserved:", t-r, "allocated:", a,"\n (total-reserved)/total:", (t-r)/t,"(reserved-allocated)/reserved:", (r-a)/r)
    return 


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
training_dataloader = DataLoader(training_dataset, batch_size = bs, collate_fn=collate_already_encoded, shuffle = True)
validation_dataloader = DataLoader(validation_dataset, batch_size = bs, collate_fn=collate_fn, shuffle = True)
mem_stats()

# %%
## Training dataset information
from torch.nn import CrossEntropyLoss
def compute_input_ids_cross_entropy(model, input_ids):
  mask  = (input_ids > 0).detach()                                     

  model.train(False)

  with torch.no_grad():
    outputs = model(input_ids=input_ids.to(torch.long), attention_mask = mask)
    logits = outputs.logits

  loss_fn = CrossEntropyLoss()
  input_len = input_ids.shape[-1] - 1
  input_ids_without_first_token = input_ids[:, 1:].long()
  logits_without_last_token = logits[:, :-1, :]

  # print(input_ids_without_first_token)
  ans = []
  for i in range(len(logits_without_last_token)):
    length = len(input_ids_without_first_token[i,:])
    if len(torch.where(input_ids_without_first_token[i,:] == 0)[0]) > 0:
      length = torch.where(input_ids_without_first_token[i,:] == 0)[0].min()
    # print(logits_without_last_token[i, :length].shape)
    # print(input_ids_without_first_token[i, :length].shape)
    # print(loss_fn(logits_without_last_token[i, :length], input_ids_without_first_token[i, :length]))
    ce_loss = loss_fn(logits_without_last_token[i, :length], input_ids_without_first_token[i, :length])
    ans.append(ce_loss/length)

  ## Clean up 
  del outputs, logits, input_ids_without_first_token, logits_without_last_token
  torch.cuda.empty_cache()
  torch.cuda.synchronize()

  return torch.Tensor(ans)

def compute_cross_entropy(dataloader, nbatches, bs, device, model, samplelength):    
    cross_entropy = torch.zeros((nbatches, bs))
    for batchno, data_x in enumerate(dataloader):
        if batchno >= nbatches:
            break
        with torch.no_grad():       
            ## Get predictions on training data                       
            data_x = data_x[:,:samplelength].to(device).detach()
   
            ## Compute average log likelihood
            cross_entropy[batchno, :] = compute_input_ids_cross_entropy(model, data_x)

        if batchno % 50 == 0:
            print("batch no. ", batchno)  
            print("Memory after training")
            mem_stats()
            print()
    return cross_entropy

train_cross_entropy = compute_cross_entropy(training_dataloader, nbatches, bs, device, model, samplelength)
val_cross_entropy = compute_cross_entropy(validation_dataloader, nbatches, bs, device, model, samplelength)

# %%
import matplotlib.pyplot as plt
import numpy as np

# generate two sets of random values
with torch.no_grad():
    valuestraining   = torch.flatten(train_cross_entropy) 
    valuesvalidation = torch.flatten(val_cross_entropy)

notnan = torch.logical_and(~valuestraining.isnan(), ~valuesvalidation.isnan())
valuestraining = valuestraining[notnan]
valuesvalidation = valuesvalidation[notnan]

    
torch.save(torch.vstack((valuestraining, valuesvalidation)), 
model_title + " (Training, Validation) data: bs=" + str(bs)+", nbatches="+str(nbatches)+", length="+str(samplelength)+").pt")

# create a figure and axis object
plt.figure()
fig, ax = plt.subplots()

# plot a histogram of the first set of values with 20 bins
ax.hist(valuestraining, bins=20, alpha=0.5, label='training')

# plot a histogram of the second set of values with 20 bins
ax.hist(valuesvalidation, bins=20, alpha=0.5, label='validation')

# add a legend to the plot
ax.legend(loc='upper right')

# add labels and a title to the plot
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.set_title(model_title + " Histogram (bs=" + str(bs)+", nbatches="+str(nbatches)+", length="+str(samplelength)+")")

# show the plot

plt.savefig(model_title + " Histogram (bs=" + str(bs)+", nbatches="+str(nbatches)+", length="+str(samplelength)+").png")


# %%
from sklearn.metrics import roc_curve, auc

## Scale all values to be between 0 and 1
st = min(valuestraining.min(),valuesvalidation.min())
end = max(valuestraining.max(),valuesvalidation.max())
print(st,end)


y_scores =  torch.cat((valuestraining, valuesvalidation))
y_scores = y_scores-min(y_scores)
y_scores = y_scores/max(y_scores)
y_true   = [1 for _ in range(len(valuestraining))] + [0 for _ in range(len(valuesvalidation))]

fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Calculate the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(model_title + " ROC (bs=" + str(bs)+", nbatches="+str(nbatches)+", length="+str(samplelength)+").png")
plt.legend(loc="lower right")
plt.savefig(model_title + " ROC (bs=" + str(bs)+", nbatches="+str(nbatches)+", length="+str(samplelength)+").png")



# %%



