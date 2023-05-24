import torch
from torch.utils.data import DataLoader
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from tqdm import tqdm
from attack_utils import *
from dataset_utils import *
from MoPe import MoPe
import time
import argparse
from accelerate import Accelerator

"""
Sample command line prompt (no acceleration)
python run_mope.py --mod_size 70m --n_samples 1000 --n_models 15 --sigma 0.001
Sample command line prompt (with acceleration)
python run_mope.py --mod_size 70m --n_samples 1000 --n_models 15 --sigma 0.001 --accelerate
Sample command line prompt (with validation dataset pt file)
python run_mope.py --mod_size 1B --n_samples 1000 --n_models 15 --sigma 0.001 --val_pt val_data.pt 
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mod_size', action="store", type=str, required=True, help='Pythia Model Size')
    parser.add_argument('--n_models', action="store", type=int, required=True, help='Number of new models')
    parser.add_argument('--n_samples', action="store", type=int, required=True, help='Number of samples')
    parser.add_argument('--sigma', action="store", type=float, required=True, help='Noise standard deviation')
    parser.add_argument('--accelerate', action="store_true", required=False, help='Use accelerate')
    parser.add_argument('--train_pt', action="store", required=False, help='.pt file of train dataset')
    parser.add_argument('--val_pt', action="store", required=False, help='.pt file of val dataset')
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
    del model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading Data")

    if args.val_pt:
        fixed_input = args.val_pt + ".pt" if not args.val_pt.endswith(".pt") else args.val_pt
        print("You are using a self-specified validation dataset...")
        validation_dataset = torch.load(fixed_input)[:args.n_samples]
        validation_dataloader = DataLoader(validation_dataset, batch_size = 1, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, length=max_length))
        print("Val Dataset Setup Done!")
    else:
        validation_dataset = load_val_pile(number=args.n_samples, seed=seed, num_splits=1)[0]
        validation_dataloader = DataLoader(validation_dataset, batch_size = 1, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, length=max_length))

    if args.train_pt:
        print("You are using a self-specified training dataset...")
        fixed_input = args.train_pt + ".pt" if not args.train_pt.endswith(".pt") else args.train_pt
        training_dataset = torch.load(fixed_input)[:args.n_samples]
        training_dataloader = DataLoader(training_dataset, batch_size = 1, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, length=max_length))
        print("Train Dataset Setup Done!")
    else:
        training_dataset = load_train_pile_random_deduped(number=args.n_samples,seed=seed,num_splits=1)[0] # TODO - replace w/ sequence at some point
        training_dataloader = DataLoader(training_dataset, batch_size = 1, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, length=max_length))

    train_pt = args.train_pt if args.accelerate and args.train_pt else "train_data.pt"
    val_pt = args.val_pt if args.accelerate and args.val_pt else "val_data.pt"

    ## Run MoPe attack

    config_mope = {
        "training_dl": training_dataloader,
        "validation_dl": validation_dataloader,
        "n_new_models": args.n_models,
        "noise_stdev": args.sigma,
        "bs" : 1,
        "nbatches": args.n_samples,
        "samplelength": None,
        "device": device,
        "accelerate": args.accelerate, # if this is false, then train_pt and val_pt are not used
        "tokenizer": tokenizer, 
        "train_pt": train_pt,
        "val_pt": val_pt
    }

    ## Stopwatch for testing MoPe runtime
    end = time.time()
    print(f"- Code initialization time was {end-start} seconds.")

    start = time.time()

    MoPer = MoPe(model_name, model_revision=model_revision, cache_dir=model_cache_dir)
    MoPer.inference(config_mope)

    MoPer.attack_plot_ROC(log_scale = False, show_plot=False)
    MoPer.attack_plot_ROC(log_scale = True, show_plot=False)
    MoPer.save()

    end = time.time()
    print(f"- MoPe at {args.mod_size} and {args.n_models} new models took {end-start} seconds.")

if __name__ == "__main__":
    main()