import os
import time
import json
import math
import argparse
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from accelerate import Accelerator
from accelerate.utils import set_seed
from llmprivacy.utils.dataset_utils import collate_fn, load_train_pile_random, load_val_pile, process_domain_specific_data
from llmprivacy.utils.log_utils import get_my_logger
from llmprivacy.attacks.GradNorm import GradNorm
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Sample command line prompt (no acceleration)
python run_gradnorm.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 1000 --pack --seed 229
Sample command line prompt (with acceleration)
accelerate launch run_gradnorm.py --accelerate --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 1000 --pack --seed 229
"""

def main():
    ####################################################################################################
    # SETUP
    ####################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', action="store", type=str, required=False, help='Experiment name. Used to determine save location.')
    parser.add_argument('--tag', action="store", type=str, required=False, help='Use default experiment name but add more information of your choice.')
    # Model Arguments
    parser.add_argument('--model_name', action="store", type=str, required=True, help='Huggingface model name')
    parser.add_argument('--model_revision', action="store", type=str, required=False, help='Model revision. If not specified, uses main.')
    parser.add_argument('--model_cache_dir', action="store", type=str, required=False, help='Model cache directory. If not specified, uses main.')
    # Dataset Arguments
    parser.add_argument('--num_samples', action="store", type=int, required=True, help='Dataset size')
    parser.add_argument('--start_index', action="store", type=int, required=False, default=0, help='Slice dataset starting from this index')
    parser.add_argument('--bs', action="store", type=int, required=False, default=1, help='Batch size')
    parser.add_argument('--min_length', action="store", type=int, required=False, default=20, help='Min number of tokens (filters)')
    parser.add_argument('--max_length', action="store", type=int, required=False, help='Max number of tokens (truncates)')
    parser.add_argument('--pack', action="store_true", required=False, help='Pack validation set')
    parser.add_argument('--unpack', action="store_true", required=False, help='Unpack training set')
    parser.add_argument('--train_pt', action="store", required=False, help='.pt file of train dataset (not dataloader)')
    parser.add_argument('--val_pt', action="store", required=False, help='.pt file of val dataset (not dataloader)')
    parser.add_argument('--data_subset', action="store", choices=["arxiv", "dm_mathematics", "github", 
                                                                  "hackernews", "pile_cc", "pubmed_central", 
                                                                  "wikipedia_(en)", "full_pile", "c4", 
                                                                  "temporal_arxiv", "temporal_wiki"],
                        required=False, help='Subest of pile to choose')

    # Attack Arguments
    parser.add_argument('--norms', action="store", nargs="+", required=False, help='Norm orders to compute')
    # Device Arguments
    parser.add_argument('--seed', action="store", type=int, required=False, default=229, help='Seed')
    parser.add_argument('--accelerate', action="store_true", required=False, help='Use accelerate')
    parser.add_argument('--model_half', action="store_true", required=False, help='Use half precision (fp16). 1 for use; 0 for not.')
    args = parser.parse_args()
    
    accelerator = Accelerator() if args.accelerate else None
    set_seed(args.seed)

    args.model_cache_dir = args.model_cache_dir if args.model_cache_dir is not None else f"models/{args.model_name.replace('/','-')}"
    if args.experiment_name is None:
        args.experiment_name = (
            (f"GradNorm_{args.model_name.replace('/','-')}") +
            (f"_{args.model_revision.replace('/','-')}" if args.model_revision is not None else "") +
            (f"_N={args.num_samples}_S={args.start_index}_seed={args.seed}") +
            (f"_tag={args.tag}" if args.tag is not None else "")
        )
        args.experiment_name = f"results/GradNorm/{args.experiment_name}/{args.experiment_name}"
    os.makedirs(os.path.dirname(args.experiment_name), exist_ok=True)
    logger = get_my_logger(log_file=f"{args.experiment_name}.log")
    with open(f"{args.experiment_name}_args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    ####################################################################################################
    # LOAD DATA
    ####################################################################################################
    start = time.perf_counter()

    max_length = AutoConfig.from_pretrained(args.model_name).max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    logger.info("Loading Data")    

    if not (args.pack ^ args.unpack):
        logger.info(f"WARNING: for an apples-to-apples comparison, we recommend setting exactly one of pack ({args.pack}) and unpack ({args.unpack})")
    
    ## Load training and validation data 
    if args.data_subset:
        training_dataset      = load_dataset("iamgroot42/mimir", args.data_subset, 
                                        split="ngram_13_0.8").map(lambda x: {"text": x["member"]},
                                        remove_columns=["member","nonmember","member_neighbors","nonmember_neighbors"])["text"]
        validation_dataset    = load_dataset("iamgroot42/mimir", args.data_subset, 
                                        split="ngram_13_0.8").map(lambda x: {"text": x["nonmember"]},
                                        remove_columns=["member","nonmember","member_neighbors","nonmember_neighbors"])["text"]
        training_dataset      = process_domain_specific_data(training_dataset)[0]
        validation_dataset    = process_domain_specific_data(validation_dataset)[0]

        training_dataloader   = DataLoader(training_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))
        validation_dataloader = DataLoader(validation_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))

    else:
        if args.train_pt:
            logger.info("You are using a self-specified training dataset...")
            fixed_input = args.train_pt + ".pt" if not args.train_pt.endswith(".pt") else args.train_pt
            training_dataset = torch.load(fixed_input)[args.start_index:args.start_index+args.num_samples]
            training_dataloader = DataLoader(training_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))
        else:
            training_dataset = load_train_pile_random(number=args.num_samples,start_index=args.start_index,seed=args.seed,num_splits=1,min_length=args.min_length,deduped="deduped" in args.model_name,unpack=args.unpack)[0]
            training_dataloader = DataLoader(training_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))

        # Load validation data
        if args.val_pt:
            fixed_input = args.val_pt + ".pt" if not args.val_pt.endswith(".pt") else args.val_pt
            logger.info("You are using a self-specified validation dataset...")
            validation_dataset = torch.load(fixed_input)[args.start_index:args.start_index+args.num_samples]
            validation_dataloader = DataLoader(validation_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))
        else:
            validation_dataset = load_val_pile(number=args.num_samples,start_index=args.start_index,seed=args.seed,num_splits=1,tokenizer=tokenizer,window=2048 if args.pack else 0)[0]
            validation_dataloader = DataLoader(validation_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))

    end = time.perf_counter()
    logger.info(f"- Dataset loading took {end-start} seconds.")
    ####################################################################################################
    # RUN ATTACK
    ####################################################################################################
    start = time.perf_counter()
    logger.info("Running Attack")

    # Initialize attack
    GradNormer = GradNorm(args.model_name, model_revision=args.model_revision, model_cache_dir=args.model_cache_dir)
    
    # Compute statistics
    GradNormer.load_model()
    train_gradients = GradNormer.compute_gradients(training_dataloader,norms=args.norms,num_batches=math.ceil(args.num_samples/args.bs),device=device,model_half=args.model_half,accelerator=accelerator,max_length=args.max_length)
    torch.save(train_gradients,f"{args.experiment_name}_train.pt")
    val_gradients = GradNormer.compute_gradients(validation_dataloader,norms=args.norms,num_batches=math.ceil(args.num_samples/args.bs),device=device,model_half=args.model_half,accelerator=accelerator,max_length=args.max_length)
    torch.save(val_gradients,f"{args.experiment_name}_val.pt")
    GradNormer.unload_model()

    # Plot ROCs
    GradNormer.attack_plot_ROC(train_gradients, val_gradients, title=args.experiment_name, log_scale=False, show_plot=False)
    GradNormer.attack_plot_ROC(train_gradients, val_gradients, title=args.experiment_name, log_scale=True, show_plot=False)
    GradNormer.attack_plot_histogram(train_gradients, val_gradients, title=args.experiment_name, normalize=False, show_plot=False)
    GradNormer.attack_plot_histogram(train_gradients, val_gradients, title=args.experiment_name, normalize=True, show_plot=False)

    end = time.perf_counter()

    logger.info(f"- Experiment {args.experiment_name} took {end-start} seconds.")

if __name__ == "__main__":
    main()