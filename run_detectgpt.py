import torch
from torch.utils.data import DataLoader
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoConfig
from attack_utils import *
from dataset_utils import *
from DetectGPT import DetectGPT
import time
import argparse
from accelerate import Accelerator

"""
Sample command line prompt (no acceleration)
python run_loss.py --mod_size 70m --deduped --checkpoint step98000 --n_samples 1000
Sample command line prompt (with acceleration)
accelerate launch run_loss.py --accelerate --mod_size 70m --deduped --checkpoint step98000 --n_samples 1000
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mod_size', action="store", type=str, required=True, help='Pythia Model Size')
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
    parser.add_argument('--train_pt', action="store", required=False, help='.pt file of train dataset (not dataloader)')
    parser.add_argument('--val_pt', action="store", required=False, help='.pt file of val dataset (not dataloader)')
    parser.add_argument('--model_half', action="store_true", required=False, help='Use half precision (fp16). 1 for use; 0 for not.')
    parser.add_argument('--n_perts', action="store", type=int, required=False, default=5, help='num of perturbations for detectgpt')
    args = parser.parse_args()

    accelerator = Accelerator() if args.accelerate else None

    if not (args.pack ^ args.unpack):
        if accelerator is None or accelerator.is_main_process:
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

    if accelerator is None or accelerator.is_main_process:
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

    ## Run LOSS attack

    config_loss = {
        "training_dl": training_dataloader,
        "validation_dl": validation_dataloader,
        "bs": args.bs,
        "nbatches": args.n_samples,
        "samplelength": args.sample_length,
        "device": device,
        "accelerator": accelerator,
        "model_half": args.model_half,
        "detect_args": {'buffer_size':1, 
        'mask_top_p': 10, 
        'pct_words_masked':.2, 
        'span_length':2,
        'num_perts': args.n_perts, 
        'device': device, 
        "model_max_length": max_length}
    }

    end = time.perf_counter()
    if accelerator is None or accelerator.is_main_process:
        print(f"- Code initialization time was {end-start} seconds.")

    start = time.perf_counter()

    LOSSer = DetectGPT(model_name, model_revision=model_revision, cache_dir=model_cache_dir)

    LOSSer.inference(config_loss)
    LOSSer.save()

    LOSSer.attack_plot_ROC(log_scale = False, show_plot=False)
    LOSSer.attack_plot_ROC(log_scale = True, show_plot=False)

    end = time.perf_counter()

    if accelerator is None or accelerator.is_main_process:
        print(f"- DetectGPT at {args.mod_size} and {args.n_samples} samples took {end-start} seconds.")

if __name__ == "__main__":
    main()