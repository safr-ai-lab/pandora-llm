import sys
import torch
from torch.utils.data import DataLoader
import json
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.optimization import AdamW
from tqdm import tqdm
from attack_utils import *
from dataset_utils import *
from Attack import *
import time

"""
Sample command line prompt:
python run_mope.py 70m 10 0.1 1000
"""

## Command line prompts
mod_size = sys.argv[1]
n_new_models = int(sys.argv[2])
noise_variance = float(sys.argv[3])
nsamples = int(sys.argv[4])

## Other parameters
model_revision = "step98000"
seed = 229

## Load model and training and validation dataset
device = "cuda" if torch.cuda.is_available() else "cpu"

## Stopwatch for testing timing
start = time.time()

model_title = f"pythia-{mod_size}-deduped"
model_name = "EleutherAI/" + model_title
model_cache_dir = "./"+ model_title +"/"+model_revision

model = GPTNeoXForCausalLM.from_pretrained(model_name,revision=model_revision,cache_dir=model_cache_dir)
max_length = model.config.max_position_embeddings
del model
tokenizer = AutoTokenizer.from_pretrained(model_name)

training_dataset = load_train_pile_random(number=nsamples,seed=seed,num_splits=1)[0] # TODO - replace w/ sequence at some point
validation_dataset = load_val_pile(number=nsamples, seed=seed, num_splits=1)[0]

training_dataloader = DataLoader(training_dataset, batch_size = 1, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, length=max_length))
validation_dataloader = DataLoader(validation_dataset, batch_size = 1, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, length=max_length))

## Run MoPe attack

config_mope = {
    "training_dl": training_dataloader,
    "validation_dl": validation_dataloader,
    "n_new_models": n_new_models,
    "noise_variance": noise_variance,
    "bs" : 1,
    "nbatches": None,
    "samplelength": None,
    "device": device
}

## Stopwatch for testing MoPe runtime
end = time.time()
print(f"- Code initialization time was {end-start} seconds.")

start = time.time()

MoPer = MoPe(model_name, model_revision=model_revision, cache_dir=model_cache_dir)
MoPer.inference(config_mope)

MoPer.attack_plot_ROC(mod_size + " " +str(noise_variance), show_plot = True, save_name = None, log_scale = False)
MoPer.save(None) # TODO - specify name of experiment

end = time.time()
print(f"- MoPe at {mod_size} and {n_new_models} new models took {end-start} seconds.")