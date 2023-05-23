import torch
from torch.utils.data import DataLoader
from transformers import GPTNeoXForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from tqdm import tqdm
from attack_utils import *
from dataset_utils import *
from LoRa import LoRa
import time
import argparse
from accelerate import Accelerator

"""
Sample command line prompt (no acceleration)
python run_lora.py --mod_size 70m --n_samples 1000
Sample command line prompt (with acceleration)
python run_lora.py --mod_size 70m --n_samples 1000 --accelerate
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mod_size', action="store", type=str, required=True, help='Pythia Model Size')
    parser.add_argument('--bs', action="store", type=int, required=False, default=1, help='Batch size')
    parser.add_argument('--n_samples', action="store", type=int, required=True, help='Number of batches')
    parser.add_argument('--sample_length', action="store", type=int, required=False, help='Number of tokens for entropy calculation')
    parser.add_argument('--accelerate', action="store_true", required=False, help='Use accelerate')
    args = parser.parse_args()

    ## Other parameters
    model_revision = "step98000"
    seed = 229

    ## Load model and training and validation dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## Stopwatch for testing timing
    start = time.time()

    model_title = f"pythia-{args.mod_size}-deduped"
    model_name = "EleutherAI/" + model_title
    model_cache_dir = "./"+ model_title +"/"+model_revision

    print("Initializing Base Model")
    model = GPTNeoXForCausalLM.from_pretrained(model_name,revision=model_revision,cache_dir=model_cache_dir)
    max_length = model.config.max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading Data")
    # training_dataset = load_train_pile_random_deduped(number=args.n_samples,seed=seed,num_splits=1)[0]
    dataset = load_val_pile(number=args.n_samples, seed=seed, num_splits=2)
    training_dataset, validation_dataset = dataset[0], dataset[1]

    if args.accelerate:
        torch.save(training_dataset,"train_data.pt")
        torch.save(validation_dataset,"val_data.pt")

    training_dataloader = DataLoader(training_dataset, batch_size = 1, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, length=max_length))
    validation_dataloader = DataLoader(validation_dataset, batch_size = 1, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, length=max_length))

    ## Run LoRa attack

    training_args = TrainingArguments(output_dir="fine-tuning",
                                        do_train=True,
                                        do_eval=True,
                                        num_train_epochs=1,
                                        per_device_train_batch_size=args.bs,
                                        per_device_eval_batch_size=args.bs,
                                        evaluation_strategy="epoch",
                                        logging_strategy="epoch",
                                        save_strategy="epoch",
                                        gradient_accumulation_steps=1,
                                        gradient_checkpointing=False,
                                        load_best_model_at_end = True,
                                        deepspeed='ds_config_zero3.json' if args.accelerate else None
                                        )
    trainer = Trainer(model=model,
                    args=training_args,
                    train_dataset=training_dataset,
                    eval_dataset=validation_dataset,
                    tokenizer=tokenizer,
                    data_collator=lambda batch: collate_fn(batch, tokenizer=tokenizer, length=max_length),
                    )

    config_lora = {
        "trainer": trainer,
        "training_dl": training_dataloader,
        "validation_dl": validation_dataloader,
        "tokenizer": tokenizer,
        "device": device,
        "n_batches": args.n_samples,
        "n_samples": args.sample_length,
        "bs": args.bs,
        "accelerate": args.accelerate
    }

    ## Stopwatch for testing MoPe runtime
    end = time.time()
    print(f"- Code initialization time was {end-start} seconds.")

    start = time.time()

    LoRaer = LoRa(model_name, model_revision=model_revision, cache_dir=model_cache_dir)
    LoRaer.inference(config_lora)

    LoRaer.attack_plot_ROC(log_scale = False, show_plot=False)
    LoRaer.attack_plot_ROC(log_scale = True, show_plot=False)
    LoRaer.save()

    end = time.time()
    print(f"- LoRa at {args.mod_size} took {end-start} seconds.")

if __name__ == "__main__":
    main()