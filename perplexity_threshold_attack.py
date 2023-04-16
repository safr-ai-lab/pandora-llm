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
samplelength = 25 ## How long are the sequences we take from the training and validation sets.
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
training_sum_perplexity = torch.zeros((nbatches, bs))
for batchno, training_x in enumerate(training_dataloader):
    if batchno >= nbatches:
        break
    with torch.no_grad():       
        ## Get predictions on training data                       
        training_x = training_x[:,:samplelength].to(device).detach()
        mask  = (training_x>0).detach()                                     

        model.train(False)
        logits_training = model(input_ids=training_x.int(), attention_mask = mask)

        ## Find sum of log likelihood of each sequence
        for batch in range(bs):
            seq_logits = torch.zeros((len(training_x[0])))
            
            ## Find sum of log likelihood of each sequence
            for idx, w in enumerate(training_x[batch,]):
                if training_x[batch,idx] == 0:
                    break
                seq_logits[idx] = logits_training.logits[batch,idx,w]
            
            ## Case for when we reach the end of the sequence
            if training_x[batch,idx] != 0:
                idx += 1

            ## Compute perplexity
            training_sum_perplexity[batchno, batch]= torch.sum(seq_logits[:idx]-torch.log(1+torch.exp(seq_logits[:idx])))/idx

            ## Clean up 
            del seq_logits
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        ## Cleaning up
        del training_x, mask, logits_training
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if batchno % 100 == 0:
        print("Training batch no. ", batchno)  
        print("Memory")
        mem_stats()
        print()

# %%
validation_sum_perplexity = torch.zeros((nbatches, bs))
for batchno, validation_x in enumerate(validation_dataloader):
    if batchno >= nbatches:
        break
    with torch.no_grad():               
        ## Get predictions on validation data                 
        validation_x = validation_x[:,:samplelength].to(device).detach()
        mask  = (validation_x>0).detach()                                     

        model.train(False)
        logits_validation = model(input_ids=validation_x.int(), attention_mask = mask)

        ## Find sum of log likelihood of each sequence
        for batch in range(bs):
            seq_logits = torch.zeros((len(validation_x[0])))

            ## Find logits of each word in the sequence
            for idx, w in enumerate(validation_x[batch,]):
                if validation_x[batch,idx] == 0:
                    break
                seq_logits[idx] = logits_validation.logits[batch,idx,torch.Tensor.int(w)]
            if validation_x[batch,idx] != 0:
                idx += 1
            validation_sum_perplexity[batchno, batch]= torch.sum(seq_logits[:idx]-torch.log(1+torch.exp(seq_logits[:idx])))/idx

            ## Clean up 
            del seq_logits
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


        ## Cleaning up
        del validation_x, mask, logits_validation
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if batchno % 100 == 0:
        print("Validation batch no. ", batchno)  
        print("Memory")
        mem_stats()
        print()

# %%
import matplotlib.pyplot as plt
import numpy as np

# generate two sets of random values
with torch.no_grad():
    valuestraining   = torch.flatten(training_sum_perplexity) 
    valuesvalidation = torch.flatten(validation_sum_perplexity)

torch.save(torch.vstack((valuestraining, valuesvalidation)), 
model_title + " (Training, Validation) data: bs=" + str(bs)+", nbatches="+str(nbatches)+", length="+str(samplelength)+").pt")


valuestraining[valuestraining <= -1000] = -1000
valuesvalidation[valuesvalidation <= -1000] = -1000
not_nan = torch.logical_not(torch.logical_or( torch.isnan(valuestraining),torch.isnan(valuesvalidation)))

valuestraining = valuestraining[not_nan]
valuesvalidation = valuesvalidation[not_nan]

# create a figure and axis object
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



