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
def collate_already_encoded(batch):
    tokens = batch
    max_length = max([len(t['tokens']) for t in tokens])
    tokens_padded = torch.zeros((len(tokens),max_length),dtype=torch.int)
    for i in range(len(tokens)):
      tokens_padded[i,:] = torch.Tensor(tokens[i]['tokens'])
    return tokens_padded



def collate_fn(batch):
    tokens = [tokenizer.encode(example["text"], return_tensors="pt", truncation=True) for example in batch]
    max_length = max([t.size(1) for t in tokens])
    tokens_padded = [torch.cat([t, t.new_zeros(t.size(0), max_length - t.size(1))], dim=1) for t in tokens]
    tokens_padded = torch.cat(tokens_padded, dim=0)
    return tokens_padded

training_dataset = load_dataset("EleutherAI/pile-deduped-pythia-random-sampled", split="train")
validation_dataset = load_dataset("the_pile_val.py", split="validation") 

training_dataloader = DataLoader(training_dataset, batch_size = bs, collate_fn=collate_already_encoded)
validation_dataloader = DataLoader(validation_dataset, batch_size = bs, collate_fn=collate_fn)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Sanity Testing
batchsz = 50
length  = 1000

training_x = next(iter(training_dataloader))
training_x = training_x[:batchsz,:length].to(device)
mask  = training_x>0                                        
print(training_x.size())

logits_training = model(input_ids=training_x, attention_mask = mask)

validation_x = next(iter(validation_dataloader))
validation_x = validation_x[:batchsz,:length].to(device)
mask  = validation_x>0                                        
print(validation_x.size())

##########################################################################################

import matplotlib.pyplot as plt
import numpy as np

# generate two sets of random values
with torch.no_grad():
  valuestraining = logits_training[0][:,-1,:].sum(dim=1).cpu()
  valuesvalidation = logits_validation[0][:, -1, :].sum(dim=1).cpu()

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
ax.set_title('Histogram of Two Classes of Values')

# show the plot
plt.savefig("Training and Validation Histogram.png")

##########################################################################################

from sklearn.metrics import roc_curve, auc

st = min(valuestraining.min(),valuesvalidation.min())
end = max(valuestraining.max(),valuesvalidation.max())
print(st,end)

y_scores =  -torch.cat((valuestraining,valuesvalidation))
y_scores = y_scores-min(y_scores)
y_scores = y_scores/max(y_scores)
y_true   = [1 for _ in range(batchsz)] + [0 for _ in range(batchsz)]

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
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig("ROC Curve.png")

##########################################################################################

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
# https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm
# TODO

# Get loss on these tokens fragments vs in-dist data
# TODO

# Loss Thresholding
threshold = 0.1

# Compute loss on N shadow learners vs. true dataset on each of N dataset parts (train/test)
# TODO

# Compute ROC curve for different loss thresholds
# TODO