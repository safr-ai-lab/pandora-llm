import sys
import torch
from torch.utils.data import DataLoader
import json
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.optimization import AdamW
from attack_utils import *
from dataset_utils import *
from Attack import *
device = "cuda" if torch.cuda.is_available() else "cpu"

# LOSS

# Model Init 

mod_size = "70m"
model_title = f"pythia-{mod_size}-deduped"
model_name = "EleutherAI/" + model_title
model_revision = "step143000"
model_cache_dir = "./"+ model_title +"/"+model_revision

# Data Init
training_dataset = load_dataset("EleutherAI/pile-deduped-pythia-random-sampled", split="train")
validation_dataset = load_val_pile(percentage=0.025, seed=229, num_splits=1)

training_dataloader = DataLoader(training_dataset, batch_size = bs, collate_fn=collate_already_encoded_marvin)
validation_dataloader = DataLoader(validation_dataset, batch_size = bs, collate_fn=collate_fn_marvin)
mem_stats()

LOSSer = LOSS(model_path=save_path, model_revision=model_revision, cache_dir=model_cache_dir)
configs = {
    "training_dl": training_dataloader,
    "validation_dl": validation_dataloader,
    "bs" : 8,
    "nbatches": 1000,
    "samplelength": 50,
    "device": device
}

LOSSer.inference(configs)

plot_ROC(LOSSer.train_cross_entropy, LOSSer.val_cross_entropy, show_plot = True, save_plot = True, log_scale = False, 
        plot_title = "LOSS Test", plot_name = "LOSS_test", plot_dir = "./", plot_format = "png")
