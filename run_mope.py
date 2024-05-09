import time
import math
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from src.utils.attack_utils import *
from src.utils.dataset_utils import *
from src.utils.log_utils import get_my_logger
from src.attacks.MoPe import MoPe
from accelerate import Accelerator
from accelerate.utils import set_seed
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Sample command line prompt (no acceleration)
python run_mope.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --n_samples 1000 --pack --seed 229
Sample command laine prompt (with acceleration)
accelerate launch run_mope.py --accelerate --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --n_samples 1000 --pack --seed 229
"""

def main():
    ####################################################################################################
    # SETUP
    ####################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', action="store", type=str, required=False, help='Experiment name. Used to determine save location.')
    # Model Arguments
    parser.add_argument('--model_name', action="store", type=str, required=True, help='Huggingface model name')
    parser.add_argument('--model_revision', action="store", type=str, required=False, help='Model revision. If not specified, uses main.')
    parser.add_argument('--model_cache_dir', action="store", type=str, required=False, help='Model cache directory. If not specified, uses main.')
    # Dataset Arguments
    parser.add_argument('--num_samples', action="store", type=int, required=True, help='Dataset size')
    parser.add_argument('--bs', action="store", type=int, required=False, default=1, help='Batch size')
    parser.add_argument('--min_length', action="store", type=int, required=False, default=20, help='Min number of tokens (filters)')
    parser.add_argument('--max_length', action="store", type=int, required=False, help='Max number of tokens (truncates)')
    parser.add_argument('--pack', action="store_true", required=False, help='Pack validation set')
    parser.add_argument('--unpack', action="store_true", required=False, help='Unpack training set')
    parser.add_argument('--train_pt', action="store", required=False, help='.pt file of train dataset (not dataloader)')
    parser.add_argument('--val_pt', action="store", required=False, help='.pt file of val dataset (not dataloader)')
    # Attack Arguments
    parser.add_argument('--num_models', action="store", type=int, required=True, help='Number of new models')
    parser.add_argument('--noise_stdev', action="store", type=float, required=True, help='Noise standard deviation')
    parser.add_argument('--noise_type', action="store", type=str, default="gaussian", required=False, help='Noise to add to model. Either `gaussian` or `rademacher`')
    parser.add_argument('--use_old', action="store_true", required=False, help='Use previously generated models')
    # Device Arguments
    parser.add_argument('--seed', action="store", type=int, required=False, default=229, help='Seed')
    parser.add_argument('--accelerate', action="store_true", required=False, help='Use accelerate')
    parser.add_argument('--model_half', action="store_true", required=False, help='Use half precision (fp16). 1 for use; 0 for not.')
    args = parser.parse_args()
    
    accelerator = Accelerator() if args.accelerate else None
    set_seed(args.seed)
    args.model_cache_dir = args.model_cache_dir if args.model_cache_dir is not None else f"models/{args.model_name.replace('/','-')}"
    args.experiment_name = args.experiment_name if args.experiment_name is not None else MoPe.get_default_name(args.model_name,args.model_revision,args.num_samples,args.seed,args.num_models,args.noise_stdev,args.noise_type)
    logger = get_my_logger(log_file=f"{args.experiment_name}.log")
    ####################################################################################################
    # LOAD DATA
    ####################################################################################################
    start = time.perf_counter()

    max_length = AutoConfig.from_pretrained(args.model_name).max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    logger.info("Loading Data")    

    if not (args.pack ^ args.unpack):
        logger.info(f"WARNING: for an apples-to-apples comparison, we recommend setting exactly one of pack ({args.pack}) and unpack ({args.unpack})")
    
    # Load training data
    if args.train_pt:
        logger.info("You are using a self-specified training dataset...")
        fixed_input = args.train_pt + ".pt" if not args.train_pt.endswith(".pt") else args.train_pt
        training_dataset = torch.load(fixed_input)[:args.num_samples]
        training_dataloader = DataLoader(training_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))
    else:
        training_dataset = load_train_pile_random(number=args.num_samples,seed=args.seed,num_splits=1,min_length=args.min_length,deduped="deduped" in args.model_name,unpack=args.unpack)[0]
        training_dataloader = DataLoader(training_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))
        if accelerator is not None: # for subprocess call
            args.train_pt = "results/MoPe/train_dataset.pt"
            torch.save(training_dataset,args.train_pt)

    # Load validation data
    if args.val_pt:
        fixed_input = args.val_pt + ".pt" if not args.val_pt.endswith(".pt") else args.val_pt
        logger.info("You are using a self-specified validation dataset...")
        validation_dataset = torch.load(fixed_input)[:args.num_samples]
        validation_dataloader = DataLoader(validation_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))
    else:
        validation_dataset = load_val_pile(number=args.num_samples, seed=args.seed, num_splits=1, window=2048 if args.pack else 0)[0]
        validation_dataloader = DataLoader(validation_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))
        if accelerator is not None: # for subprocess call
            args.val_pt = "results/MoPe/val_dataset.pt"
            torch.save(validation_dataset,args.val_pt)

    end = time.perf_counter()
    logger.info(f"- Dataset loading took {end-start} seconds.")
    ####################################################################################################
    # RUN ATTACK
    ####################################################################################################
    start = time.perf_counter()
    logger.info("Running Attack")

    # Initialize attack
    MoPer = MoPe(args.model_name, model_revision=args.model_revision, model_cache_dir=args.model_cache_dir)
    start_gen = time.perf_counter()
    if not args.use_old:
        MoPer.generate_new_models(tokenizer=tokenizer, num_models=args.num_models, noise_stdev=args.noise_stdev, noise_type=args.noise_type)
    end_gen = time.perf_counter()
    logger.info(f"- Perturbing Models took {end_gen-start_gen} seconds!")
    
    # Compute statistics
    train_losses = torch.zeros((args.num_models+1, args.num_samples, args.bs))  
    val_losses = torch.zeros((args.num_models+1, args.num_samples, args.bs))  
    for model_index in range(args.num_models+1):
        logger.info(f"- Computing on Model {model_index+1}/{args.num_models+1}")
        MoPer.load_model(model_index)
        train_losses[model_index,:,:] = MoPer.compute_model_statistics(model_index,training_dataloader,num_batches=math.ceil(args.num_samples/args.bs),device=device,model_half=args.model_half,accelerator=accelerator,dataset_pt=args.train_pt).reshape(-1,1)
        torch.save(train_losses,f"{args.experiment_name}_train.pt")
        val_losses[model_index,:,:] = MoPer.compute_model_statistics(model_index,validation_dataloader,num_batches=math.ceil(args.num_samples/args.bs),device=device,model_half=args.model_half,accelerator=accelerator,dataset_pt=args.val_pt).reshape(-1,1)
        torch.save(val_losses,f"{args.experiment_name}_val.pt")
        MoPer.unload_model()
    train_losses = train_losses.flatten(start_dim=1)
    val_losses = val_losses.flatten(start_dim=1)
    train_statistics = train_losses[0,:]-train_losses[1:,:].mean(dim=0)
    val_statistics = val_losses[0,:]-val_losses[1:,:].mean(dim=0)

    # Plot ROCs
    MoPer.attack_plot_ROC(train_statistics, val_statistics, title=args.experiment_name, log_scale=False, show_plot=False)
    MoPer.attack_plot_ROC(train_statistics, val_statistics, title=args.experiment_name, log_scale=False, show_plot=False)
    # MoPer.plot_loss_ROC(log_scale = False)
    # MoPer.plot_loss_ROC(log_scale = True)
    # MoPer.plot_mope_loss_linear_ROC(log_scale=False)
    # MoPer.plot_mope_loss_linear_ROC(log_scale=True)
    # MoPer.plot_mope_loss_LR_ROC(log_scale = False)
    # MoPer.plot_mope_loss_LR_ROC(log_scale = True)
    # MoPer.plot_mope_loss(log_scale=False)
    # MoPer.plot_mope_loss(log_scale=True)
    # MoPer.plot_stat_hists(args.n_models)

    end = time.perf_counter()

    logger.info(f"- Experiment {args.experiment_name} took {end-start} seconds.")

if __name__ == "__main__":
    main()