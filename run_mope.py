import torch
from torch.utils.data import DataLoader
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoConfig
from attack_utils import *
from dataset_utils import *
from MoPe import MoPe
import time
import argparse

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
    parser.add_argument('--sigma', action="store", type=float, required=True, help='Noise standard deviation')
    parser.add_argument('--deduped', action="store_true", required=False, help='Use deduped models')
    parser.add_argument('--checkpoint', action="store", type=str, required=False, help='Model revision. If not specified, use last checkpoint.')
    parser.add_argument('--pack', action="store_true", required=False, help='Pack validation set')
    parser.add_argument('--unpack', action="store_true", required=False, help='Unpack training set')
    parser.add_argument('--n_samples', action="store", type=int, required=True, help='Dataset size')
    parser.add_argument('--sample_length', action="store", type=int, required=False, help='Truncate number of tokens')
    parser.add_argument('--min_length', action="store", type=int, required=False, default=20, help='Min number of tokens')
    parser.add_argument('--seed', action="store", type=int, required=False, default=229, help='Seed')
    parser.add_argument('--bs', action="store", type=int, required=False, default=1, help='Batch size')
    parser.add_argument('--accelerate', action="store_true", required=False, help='Use accelerate')
    parser.add_argument('--train_pt', action="store", required=False, help='.pt file of train dataloader')
    parser.add_argument('--val_pt', action="store", required=False, help='.pt file of val dataloader')
    parser.add_argument('--model_half', action="store_true", required=False, help='Use half precision (fp16). 1 for use; 0 for not.')
    args = parser.parse_args()

    if not (args.pack ^ args.unpack):
        print(f"WARNING: for an apples-to-apples comparison, we recommend setting exactly one of pack ({args.pack}) and unpack ({args.unpack})")

    ## Other parameters
    model_revision = args.checkpoint
    seed = args.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_title = f"pythia-{args.mod_size}" + ("-deduped" if args.deduped else "")
    model_name = "EleutherAI/" + model_title
    model_cache_dir = "./"+ model_title + ("/"+model_revision if args.checkpoint else "")

    ## Load model and training and validation dataset
    start = time.perf_counter()

    max_length = AutoConfig.from_pretrained(model_name).max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading Data")

    if args.train_pt:
        print("You are using a self-specified training dataset...")
        fixed_input = args.train_pt + ".pt" if not args.train_pt.endswith(".pt") else args.train_pt
        training_dataset = torch.load(fixed_input)[:args.n_samples]
        training_dataloader = DataLoader(training_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, length=max_length))
    else:
        if args.deduped and args.unpack:
            training_dataset = load_train_pile_random_deduped_unpacked(number=args.n_samples,seed=seed,num_splits=1,min_length=args.min_length)[0]
        elif args.deduped and not args.unpack:
            training_dataset = load_train_pile_random_deduped(number=args.n_samples,seed=seed,num_splits=1)[0]
        elif not args.deduped and args.unpack:
            training_dataset = load_train_pile_random_duped_unpacked(number=args.n_samples,seed=seed,num_splits=1,min_length=args.min_length)[0]
        else:
            training_dataset = load_train_pile_random_duped(number=args.n_samples,seed=seed,num_splits=1)[0]
        training_dataloader = DataLoader(training_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, length=max_length))

    if args.val_pt:
        fixed_input = args.val_pt + ".pt" if not args.val_pt.endswith(".pt") else args.val_pt
        print("You are using a self-specified validation dataset...")
        validation_dataset = torch.load(fixed_input)[:args.n_samples]
        validation_dataloader = DataLoader(validation_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, length=max_length))
    else:
        if args.pack:
            validation_dataset = load_val_pile_packed(number=args.n_samples, seed=seed, num_splits=1)[0]
        else:
            validation_dataset = load_val_pile(number=args.n_samples, seed=seed, num_splits=1)[0]
        validation_dataloader = DataLoader(validation_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, length=max_length))
    
    train_pt = args.train_pt if args.accelerate and args.train_pt else "train_data.pt"
    val_pt = args.val_pt if args.accelerate and args.val_pt else "val_data.pt"

    ## Run MoPe attack

    config_mope = {
        "training_dl": training_dataloader,
        "validation_dl": validation_dataloader,
        "n_new_models": args.n_models,
        "noise_stdev": args.sigma,
        "bs" : args.bs,
        "nbatches": args.n_samples,
        "samplelength": args.sample_length,
        "device": device,
        "accelerate": args.accelerate, # if this is false, then train_pt and val_pt are not used
        "tokenizer": tokenizer, 
        "train_pt": train_pt,
        "val_pt": val_pt,
        "model_half": args.model_half,
    }

    ## Stopwatch for testing MoPe runtime
    end = time.perf_counter()
    print(f"- Code initialization time was {end-start} seconds.")

    start = time.perf_counter()

    MoPer = MoPe(model_name, model_revision=model_revision, cache_dir=model_cache_dir)
    MoPer.inference(config_mope)

    MoPer.attack_plot_ROC(log_scale = False, show_plot=True)
    MoPer.attack_plot_ROC(log_scale = True, show_plot=True)
    print("Plotting...")
    MoPer.plot_stat_hists(args.n_models, show_plot=True)
    MoPer.plot_mope_loss_linear_ROC(show_plot=True, log_scale=False)
    MoPer.plot_mope_loss_linear_ROC(show_plot=True, log_scale=True)
    MoPer.plot_mope_loss_plot(show_plot=True, log_scale=False)
    MoPer.plot_mope_loss_plot(show_plot=True, log_scale=True)
    MoPer.save()

    end = time.perf_counter()
    print(f"- MoPe at {args.mod_size} and {args.n_models} new models took {end-start} seconds.")

if __name__ == "__main__":
    main()