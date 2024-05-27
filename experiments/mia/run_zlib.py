import os
import time
import argparse
import torch
from accelerate.utils import set_seed
from llmprivacy.utils.dataset_utils import load_train_pile_random, load_val_pile
from llmprivacy.utils.log_utils import get_my_logger
from llmprivacy.attacks.ZLIB import ZLIB
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Sample command line prompt (no acceleration)
python run_zlib.py --n_samples 1000 --pack --seed 229
Sample command line prompt (with acceleration)
accelerate launch run_zlib.py --accelerate --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --n_samples 1000 --pack --seed 229
"""

def main():
    ####################################################################################################
    # SETUP
    ####################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', action="store", type=str, required=False, help='Experiment name. Used to determine save location.')
    parser.add_argument('--tag', action="store", type=str, required=False, help='Use default experiment name but add more information of your choice.')
    # Dataset Arguments
    parser.add_argument('--dataset_name', action="store", type=str, required=True, help='Dataset name')
    parser.add_argument('--num_samples', action="store", type=int, required=True, help='Dataset size')
    parser.add_argument('--start_index', action="store", type=int, required=False, default=0, help='Slice dataset starting from this index')
    parser.add_argument('--min_length', action="store", type=int, required=False, default=20, help='Min number of tokens (filters)')
    parser.add_argument('--max_length', action="store", type=int, required=False, help='Max number of tokens (truncates)')
    parser.add_argument('--pack', action="store_true", required=False, help='Pack validation set')
    parser.add_argument('--unpack', action="store_true", required=False, help='Unpack training set')
    parser.add_argument('--train_pt', action="store", required=False, help='.pt file of train dataset (not dataloader)')
    parser.add_argument('--val_pt', action="store", required=False, help='.pt file of val dataset (not dataloader)')
    # Device Arguments
    parser.add_argument('--seed', action="store", type=int, required=False, default=229, help='Seed')
    args = parser.parse_args()
    
    set_seed(args.seed)
    if args.experiment_name is None:
        args.experiment_name = (
            (f"ZLIB_{args.dataset_name.replace('/','-')}") +
            (f"_N={args.num_samples}_S={args.start_index}_seed={args.seed}") +
            (f"_tag={args.tag}" if args.tag is not None else "")
        )
        args.experiment_name = f"results/ZLIB/{args.experiment_name}/{args.experiment_name}"
    os.makedirs(os.path.dirname(args.experiment_name), exist_ok=True)
    logger = get_my_logger(log_file=f"{args.experiment_name}.log")
    ####################################################################################################
    # LOAD DATA
    ####################################################################################################
    start = time.perf_counter()
    
    logger.info("Loading Data")    

    if not (args.pack ^ args.unpack):
        logger.info(f"WARNING: for an apples-to-apples comparison, we recommend setting exactly one of pack ({args.pack}) and unpack ({args.unpack})")
    
    # Load training data
    if args.train_pt:
        logger.info("You are using a self-specified training dataset...")
        fixed_input = args.train_pt + ".pt" if not args.train_pt.endswith(".pt") else args.train_pt
        training_dataset = torch.load(fixed_input)[args.start_index:args.start_index+args.num_samples]
    else:
        if args.dataset_name=="pile":
            training_dataset = load_train_pile_random(number=args.num_samples,start_index=args.start_index,seed=args.seed,num_splits=1,min_length=args.min_length,deduped=False,unpack=args.unpack)[0]
        elif args.dataset_name=="pile-deduped":
            training_dataset = load_train_pile_random(number=args.num_samples,start_index=args.start_index,seed=args.seed,num_splits=1,min_length=args.min_length,deduped=True,unpack=args.unpack)[0]
        else:
            raise NotImplementedError(f"Dataset unsupported: {args.dataset_name}")

    # Load validation data
    if args.val_pt:
        fixed_input = args.val_pt + ".pt" if not args.val_pt.endswith(".pt") else args.val_pt
        logger.info("You are using a self-specified validation dataset...")
        validation_dataset = torch.load(fixed_input)[args.start_index:args.start_index+args.num_samples]
    else:
        if args.dataset_name=="pile" or args.dataset_name=="pile-deduped":
            validation_dataset = load_val_pile(number=args.num_samples,start_index=args.start_index,seed=args.seed,num_splits=1,window=2048 if args.pack else 0)[0]
        else:
            raise NotImplementedError(f"Dataset unsupported: {args.dataset_name}")

    end = time.perf_counter()
    logger.info(f"- Dataset loading took {end-start} seconds.")
    ####################################################################################################
    # RUN ATTACK
    ####################################################################################################
    start = time.perf_counter()
    logger.info("Running Attack")

    # Initialize attack
    ZLIBer = ZLIB("pile")
    
    # Compute statistics
    train_statistics = ZLIBer.compute_statistic(training_dataset,num_samples=args.num_samples)
    torch.save(train_statistics,f"{args.experiment_name}_train.pt")
    val_statistics = ZLIBer.compute_statistic(validation_dataset,num_samples=args.num_samples)
    torch.save(val_statistics,f"{args.experiment_name}_val.pt")

    # Plot ROCs
    ZLIBer.attack_plot_ROC(train_statistics, val_statistics, title=args.experiment_name, log_scale=False, show_plot=False)
    ZLIBer.attack_plot_ROC(train_statistics, val_statistics, title=args.experiment_name, log_scale=True, show_plot=False)
    ZLIBer.attack_plot_histogram(train_statistics, val_statistics, title=args.experiment_name, normalize=False, show_plot=False)
    ZLIBer.attack_plot_histogram(train_statistics, val_statistics, title=args.experiment_name, normalize=True, show_plot=False)

    end = time.perf_counter()

    logger.info(f"- Experiment {args.experiment_name} took {end-start} seconds.")

if __name__ == "__main__":
    main()