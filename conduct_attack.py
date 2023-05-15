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


mod_size = "70m"

model_title = f"pythia-{mod_size}-deduped"
model_name = "EleutherAI/" + model_title
model_revision = "step143000"
model_cache_dir = "./"+ model_title +"/"+model_revision

model = GPTNeoXForCausalLM.from_pretrained(
  model_name,
  revision=model_revision,
  cache_dir=model_cache_dir,
)
model.half()
model.eval()
model.to(device)

model.save_pretrained()

LoRa()


tokenizer = AutoTokenizer.from_pretrained(
  model_name,
  revision=model_revision,
  cache_dir=model_cache_dir,
)

train_dataset, val_dataset = load_val_pile(percentage=0.025,seed=229,num_splits=2)

## Collate functions for loading dataset
def collate_fn(batch):
    tokens = [tokenizer.encode(example, return_tensors="pt", truncation=True,max_length=model.config.max_position_embeddings) for example in batch]
    max_length = max([t.size(1) for t in tokens])
    tokens_padded = [torch.cat([t, t.new_zeros(t.size(0), max_length - t.size(1))], dim=1) for t in tokens]
    tokens_padded = torch.cat(tokens_padded, dim=0)
    return {
        "input_ids":tokens_padded,
        "labels":tokens_padded,
        "attention_mask": torch.tensor(tokens_padded>0,dtype=int)
    }
def train(model, train_dataset, val_dataset, collate_fn, batch_size, epochs):
    model.config.use_cache = False
    training_args = TrainingArguments(output_dir="fine-tuning",
                                        do_train=True,
                                        do_eval=True,
                                        num_train_epochs=epochs,
                                        per_device_train_batch_size=batch_size,
                                        per_device_eval_batch_size=batch_size,
                                        evaluation_strategy="epoch",
                                        logging_strategy="epoch",
                                        save_strategy="epoch",
                                        gradient_accumulation_steps=1,
                                        gradient_checkpointing=False,
                                        load_best_model_at_end = True,
                                        optim="adafactor",
                                        )
    trainer = Trainer(model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    tokenizer=tokenizer,
                    data_collator=collate_fn,
                    )
    trainer.train()
    return model
