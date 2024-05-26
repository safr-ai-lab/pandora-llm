import time
import math
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from llmprivacy.utils.attack_utils import *
from llmprivacy.utils.dataset_utils import *
from llmprivacy.utils.log_utils import get_my_logger
from llmprivacy.attacks.ModelStealing import ModelStealing
from accelerate import Accelerator
from accelerate.utils import set_seed
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Sample command line prompt (no acceleration)
python run_modelstealing.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --n_samples 1000 --pack --seed 229
Sample command line prompt (with acceleration)
accelerate launch run_modelstealing.py --accelerate --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --n_samples 1000 --pack --seed 229
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
    # Dataset arguments 
    parser.add_argument('--num_samples', action="store", type=int, required=True, help='Dataset size')
    parser.add_argument('--start_index', action="store", type=int, required=False, default=0, help='Slice dataset starting from this index')
    parser.add_argument('--bs', action="store", type=int, required=False, default=1, help='Batch size')
    parser.add_argument('--min_length', action="store", type=int, required=False, default=20, help='Min number of tokens (filters)')
    parser.add_argument('--max_length', action="store", type=int, required=False, help='Max number of tokens (truncates)')
    parser.add_argument('--pack', action="store_true", required=False, help='Pack validation set')
    parser.add_argument('--unpack', action="store_true", required=False, help='Unpack training set')
    parser.add_argument('--train_pt', action="store", required=False, help='.pt file of train dataset (not dataloader)')
    parser.add_argument('--val_pt', action="store", required=False, help='.pt file of val dataset (not dataloader)')
    # Attack Arguments
    parser.add_argument('--modelstealing_method', action="store", type=str, required=False, default="one_sided_projection", help="Model stealing method")
    ### Projection arguments
    parser.add_argument('--proj_type', action="store", type=str, required=False, default="rademacher", help='type of projection (rademacher or normal)')
    parser.add_argument('--proj_seed', action="store", type=int, required=False, default=229, help='Seed for random projection')
    parser.add_argument('--proj_dim_last', action="store", type=int, required=False,default=4096, help='Dimension of projection of last layer gradients for model stealing. Default = 4096.')
    # Device Arguments
    parser.add_argument('--seed', action="store", type=int, required=False, default=229, help='Seed')
    parser.add_argument('--accelerate', action="store_true", required=False, help='Use accelerate. Not supported.')
    parser.add_argument('--model_half', action="store_true", required=False, help='Use half precision (fp16). 1 for use; 0 for not.')
    args = parser.parse_args()
    
    accelerator = Accelerator() if args.accelerate else None
    set_seed(args.seed)
    args.model_cache_dir = args.model_cache_dir if args.model_cache_dir is not None else f"models/{args.model_name.replace('/','-')}"
    args.experiment_name = args.experiment_name if args.experiment_name is not None else ModelStealing.get_default_name(args.model_name,args.model_revision,args.num_samples,args.start_index,args.seed)
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
        validation_dataset = load_val_pile(number=args.num_samples,start_index=args.start_index,seed=args.seed,num_splits=1,window=2048 if args.pack else 0)[0]
        validation_dataloader = DataLoader(validation_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))

    end = time.perf_counter()
    logger.info(f"- Dataset loading took {end-start} seconds.")
    ####################################################################################################
    # RUN ATTACK
    ####################################################################################################
    start = time.perf_counter()
    logger.info("Running Attack")

    # Initialize attack
    ModelStealer = ModelStealer(args.model_name, model_revision=args.model_revision, model_cache_dir=args.model_cache_dir)
    ModelStealer.load_model()

    # Load some random internet text
    svd_dataset = load_val_pile(number=next(ModelStealer.model.parameters()).shape[1], seed=314159, num_splits=1, window=2048 if args.pack else 0)[0]
    svd_dataloader = DataLoader(svd_dataset, batch_size = 1, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))
    
    # Compute statistics
    train_features = {}
    val_features = {}
    train_features["model_stealing"] = ModelStealer.compute_model_stealing(
        dataloader=training_dataloader,svd_dataloader=svd_dataloader,
        method=args.modelstealing_method,proj_dim=args.proj_dim_last,proj_type=args.proj_type,proj_seed=args.proj_seed,
        num_batches=math.ceil(args.num_samples/args.bs),device=device,model_half=args.model_half,accelerator=accelerator
    )
    val_features["model_stealing"] = ModelStealer.compute_model_stealing(
        dataloader=validation_dataloader,svd_dataloader=svd_dataloader,
        method=args.modelstealing_method,proj_dim=args.proj_dim_last,proj_type=args.proj_type,proj_seed=args.proj_seed,
        num_batches=math.ceil(args.num_samples/args.bs),device=device,model_half=args.model_half,accelerator=accelerator
    )
    ModelStealer.unload_model()

    torch.save(train_features,f"{args.experiment_name}_train.pt")
    torch.save(val_features,f"{args.experiment_name}_val.pt")

    end = time.perf_counter()

    logger.info(f"- Experiment {args.experiment_name} took {end-start} seconds.")

if __name__ == "__main__":
    main()