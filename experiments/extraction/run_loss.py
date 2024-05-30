import os
import time
import json
import math
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from accelerate import Accelerator
from accelerate.utils import set_seed
from llmprivacy.utils.dataset_utils import collate_fn
from llmprivacy.utils.log_utils import get_my_logger
from llmprivacy.attacks.LOSS import LOSS
from llmprivacy.utils.extraction_utils import compute_extraction_metrics
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Sample command line prompt (no acceleration)
python run_loss.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --n_samples 1000 --pack --seed 229
Sample command laine prompt (with acceleration)
accelerate launch run_loss.py --accelerate --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --n_samples 1000 --pack --seed 229
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
    parser.add_argument('--ground_truth', action="store", type=str, required=False, help='.pt file of ground truth input_ids')
    parser.add_argument('--generations', action="store", type=str, required=False, help='.pt file of generated input_ids')
    parser.add_argument('--prefix_length', action="store", type=int, required=False, help='Prefix length')
    parser.add_argument('--suffix_length', action="store", type=int, required=False, help='Suffix length')
    parser.add_argument('--num_samples', action="store", type=int, required=True, help='Dataset size')
    parser.add_argument('--start_index', action="store", type=int, required=False, default=0, help='Slice dataset starting from this index')
    parser.add_argument('--bs', action="store", type=int, required=False, default=1, help='Batch size')
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
            (f"LOSS_{args.model_name.replace('/','-')}") +
            (f"_{args.model_revision.replace('/','-')}" if args.model_revision is not None else "") +
            # (f"_{args.ground_truth.replace('/','-')}_{args.generations.replace('/','-')}") +
            (f"_N={args.num_samples}_S={args.start_index}_seed={args.seed}") +
            (f"_tag={args.tag}" if args.tag is not None else "") +
            (f"_extract")
        )
        args.experiment_name = f"results/LOSS/{args.experiment_name}/{args.experiment_name}"
    os.makedirs(os.path.dirname(args.experiment_name), exist_ok=True)
    logger = get_my_logger(log_file=f"{args.experiment_name}.log")
    with open(f"{args.experiment_name}_args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    ####################################################################################################
    # LOAD DATA
    ####################################################################################################
    start = time.perf_counter()
    
    logger.info("Loading Data")    
    ground_truth = torch.load(args.ground_truth)[args.start_index:args.start_index+args.num_samples]
    generations = torch.load(args.generations)[args.start_index:args.start_index+args.num_samples]
    # dataset = torch.cat((ground_truth[:,None,:],generations),dim=1).flatten(end_dim=1) # [true x N, gen 1 x N, ... gen M x N]
    ground_truth_dataloader = DataLoader(ground_truth, batch_size = args.bs)
    generations_dataloader = DataLoader(generations.flatten(end_dim=1), batch_size = args.bs)

    end = time.perf_counter()
    logger.info(f"- Dataset loading took {end-start} seconds.")
    ####################################################################################################
    # RUN ATTACK
    ####################################################################################################
    start = time.perf_counter()
    logger.info("Running Attack")

    # Initialize attack
    LOSSer = LOSS(args.model_name, model_revision=args.model_revision, model_cache_dir=args.model_cache_dir)
    
    # Compute statistics
    LOSSer.load_model()
    ground_truth_statistics = LOSSer.compute_statistic(ground_truth_dataloader,num_batches=math.ceil(args.num_samples/args.bs),device=device,model_half=args.model_half,accelerator=accelerator)
    torch.save(ground_truth_statistics,f"{args.experiment_name}_true_statistics.pt")
    generations_statistics = LOSSer.compute_statistic(generations_dataloader,device=device,model_half=args.model_half,accelerator=accelerator)
    generations_statistics = generations_statistics.reshape(generations.shape[0],generations.shape[1])
    torch.save(generations_statistics,f"{args.experiment_name}_gen_statistics.pt")
    LOSSer.unload_model()

    # Compute metrics
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    compute_extraction_metrics(
        ground_truth=ground_truth,
        generations=generations,
        ground_truth_statistics=ground_truth_statistics,
        generations_statistics=generations_statistics,
        prefix_length=args.prefix_length,
        suffix_length=args.suffix_length,
        tokenizer=tokenizer,
        title=args.experiment_name,
    )

    end = time.perf_counter()

    logger.info(f"- Experiment {args.experiment_name} took {end-start} seconds.")

if __name__ == "__main__":
    main()