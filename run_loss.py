import torch
from torch.utils.data import DataLoader
import transformers
import datasets
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from tqdm import tqdm
from attack_utils import *
from dataset_utils import *
from LOSS import LOSS
import time
import argparse
from accelerate import Accelerator

"""
Sample command line prompt (no acceleration)
python run_loss.py --mod_size 70m --n_samples 1000
Sample command line prompt (with acceleration)
accelerate launch run_loss.py --mod_size 70m --n_samples 1000 --accelerate
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mod_size', action="store", type=str, required=True, help='Pythia Model Size')
    parser.add_argument('--n_samples', action="store", type=int, required=True, help='Number of samples')
    parser.add_argument('--sample_length', action="store", type=int, required=False, help='Number of tokens to calculate loss for')
    parser.add_argument('--accelerate', action="store_true", required=False, help='Use accelerate')
    parser.add_argument('--train_dl', action="store", required=False, help='.pt file of train dataloader')
    parser.add_argument('--val_dl', action="store", required=False, help='.pt file of val dataloader')
    args = parser.parse_args()

    accelerator = Accelerator() if args.accelerate else None

    ## Other parameters
    model_revision = "step98000"
    seed = 229

    ## Load model and training and validation dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## Stopwatch for testing timing
    start = time.time()

    model_title = f"pythia-{args.mod_size}"
    model_name = "EleutherAI/" + model_title
    model_cache_dir = "./"+ model_title +"/"+model_revision

    if accelerator is None or accelerator.is_main_process:
        print("Initializing Base Model")
    model = GPTNeoXForCausalLM.from_pretrained(model_name,revision=model_revision,cache_dir=model_cache_dir)
    max_length = model.config.max_position_embeddings
    del model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if accelerator is None or accelerator.is_main_process:
        print("Loading Data")
    
    if args.val_dl:
        fixed_input = args.val_dl + ".pt" if not args.val_dl.endswith(".pt") else args.val_dl
        print("You are using a self-specified validation dataloader. Verify that it has the batching/window lengths you intend.")
        validation_dataloader = torch.load(fixed_input)
        print("Val Data Loaded!")
    else:
        validation_dataset = load_val_pile(number=args.n_samples, seed=seed, num_splits=1)[0]
        validation_dataloader = DataLoader(validation_dataset, batch_size = 1, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, length=max_length))

    if args.train_dl:
        fixed_input = args.train_dl + ".pt" if not args.train_dl.endswith(".pt") else args.train_dl
        print("You are using a self-specified training dataloader. Verify that it has the batching/window lengths you intend.")
        training_dataloader = torch.load(fixed_input)
        print("Train data loaded!")
    else:
        training_dataset = load_train_pile_random_duped(number=args.n_samples,seed=seed,num_splits=1)[0] # TODO - replace w/ sequence at some point
        training_dataloader = DataLoader(training_dataset, batch_size = 1, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, length=max_length))
    
    ## Run LOSS attack

    config_loss = {
        "training_dl": training_dataloader,
        "validation_dl": validation_dataloader,
        "bs" : 1,
        "nbatches": args.n_samples,
        "samplelength": args.sample_length,
        "device": device,
        "accelerator": accelerator
    }

    ## Stopwatch for testing LOSS runtime
    end = time.time()

    if accelerator is None or accelerator.is_main_process:
        print(f"- Code initialization time was {end-start} seconds.")

    start = time.time()

    LOSSer = LOSS(model_name, model_revision=model_revision, cache_dir=model_cache_dir)

    LOSSer.inference(config_loss)

    LOSSer.attack_plot_ROC(log_scale = False, show_plot=False)
    LOSSer.attack_plot_ROC(log_scale = True, show_plot=False)
    LOSSer.save()

    end = time.time()

    if accelerator is None or accelerator.is_main_process:
        print(f"- LOSS at {args.mod_size} and {args.n_samples} samples took {end-start} seconds.")

if __name__ == "__main__":
    main()