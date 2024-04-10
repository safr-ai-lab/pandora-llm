import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments, Trainer
from src.utils.attack_utils import *
from src.utils.dataset_utils import *
from src.attacks.approx_LoRa import approx_LoRa
import time
import os
import argparse

"""
Sample command line prompt (no acceleration): 
python run_approx_lora.py --mod_size 70m --deduped --checkpoint step98000 --pack --n_samples 100 --lr 0.00005

Note that this routine requires input pt's. 
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mod_size', action="store", type=str, required=True, help='Pythia Model Size')
    parser.add_argument('--checkpoint', action="store", type=str, required=False, help='Model revision. If not specified, use last checkpoint.')
    parser.add_argument('--deduped', action="store_true", required=False, help='Use deduped models')
    parser.add_argument('--pack', action="store_true", required=False, help='Pack validation set')
    parser.add_argument('--bs', action="store", type=int, required=False, default=1, help='Batch size')
    parser.add_argument('--seed', action="store", type=int, required=False, default=229, help='Seed')
    parser.add_argument('--n_samples', action="store", type=int, required=True, help='Dataset size')
    parser.add_argument('--sample_length', action="store", type=int, required=False, help='Truncate number of tokens')
    parser.add_argument('--model_half', action="store_true", required=False, help='Use half precision (fp16). 1 for use; 0 for not.')
    parser.add_argument('--train_pt', action="store", required=False, help='.pt file of train dataset (not dataloader)')
    parser.add_argument('--val_pt', action="store", required=False, help='.pt file of val dataset (not dataloader)')
    parser.add_argument('--unpack', action="store_true", required=False, help='Unpack training set')
    parser.add_argument('--lr', action="store", type=float, required=True, default=5e-05, help='Learning rate. Deafult is 5e-05.')
    # parser.add_argument('--unlearning', action="store", required=False, default=False, help="Use this flag for the unlearning version.")
    args = parser.parse_args()

    ## Other parameters
    model_revision = args.checkpoint
    seed = args.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_title = f"pythia-{args.mod_size}" + ("-deduped" if args.deduped else "")
    model_name = "EleutherAI/" + model_title
    model_cache_dir = "./"+ model_title + ("/"+model_revision if args.checkpoint else "")

    max_length = AutoConfig.from_pretrained(model_name).max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ## Load model and training and validation dataset
    start = time.perf_counter()

    if args.train_pt:
        print("You are using a self-specified training dataset...")
        fixed_input = args.train_pt + ".pt" if not args.train_pt.endswith(".pt") else args.train_pt
        training_dataset = torch.load(fixed_input)[:args.n_samples]
    else:
        print("Requires input pt!")
    if args.val_pt:
        fixed_input = args.val_pt + ".pt" if not args.val_pt.endswith(".pt") else args.val_pt
        print("You are using a self-specified validation dataset...")
        validation_dataset = torch.load(fixed_input)[:args.n_samples]
    else:
        print("Requires input pt!")

    end = time.perf_counter()
    print(f"- Data initialization time was {end-start} seconds.")

    # Make `approx_LoRa` directory
    directory_path = "approx_LoRa"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists. Using it!")

    start = time.perf_counter()

    train_ids = [tokenizer.encode(t, return_tensors='pt') for t in training_dataset]
    val_ids = [tokenizer.encode(t, return_tensors='pt') for t in validation_dataset]

    config_lora = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "tokenizer": tokenizer,
        "device": device,
        "lr": args.lr,
        "n_samples": args.n_samples,
        "bs": 1, # purely for naming
        "nbatches": args.n_samples # purely for naming 
    }

    aLoRaer = approx_LoRa(model_name, model_revision=model_revision, cache_dir=model_cache_dir)
    aLoRaer.inference(config_lora)

    aLoRaer.attack_plot_ROC(log_scale = False, show_plot=False)
    aLoRaer.attack_plot_ROC(log_scale = True, show_plot=False)
    aLoRaer.save()

    end = time.perf_counter()
    print(f"- LoRa at {args.mod_size} took {end-start} seconds.")

if __name__ == "__main__":
    main()