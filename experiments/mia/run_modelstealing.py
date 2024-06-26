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
from pandora_llm.utils.dataset_utils import collate_fn, load_train_pile_random, load_val_pile
from pandora_llm.utils.log_utils import get_my_logger
from pandora_llm.attacks.ModelStealing import ModelStealing
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Sample command line prompt (with acceleration)
accelerate launch run_modelstealing.py --accelerate --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --n_samples 1000 --pack --seed 229
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
    parser.add_argument('--embedding_projection_file', action="store", type=str, required=False, default=None, help="Location of embedding projection file")
    ### Projection arguments
    parser.add_argument('--project_type', action="store", type=str, required=False, default="rademacher", help='type of projection (rademacher or normal)')
    parser.add_argument('--proj_seed', action="store", type=int, required=False, default=229, help='Seed for random projection')
    parser.add_argument('--proj_dim_last', action="store", type=int, required=False,default=512, help='Dimension of projection of last layer gradients for model stealing. Default = 512.')
    
    # Device Arguments
    parser.add_argument('--seed', action="store", type=int, required=False, default=229, help='Seed')
    parser.add_argument('--accelerate', action="store_true", required=False, help='Use accelerate. Not supported.')
    parser.add_argument('--model_half', action="store_true", required=False, help='Use half precision (fp16). 1 for use; 0 for not.')
    args = parser.parse_args()
    
    set_seed(args.seed)

    args.model_cache_dir = args.model_cache_dir if args.model_cache_dir is not None else f"models/{args.model_name.replace('/','-')}"
    if args.experiment_name is None:
        args.experiment_name = (
            (f"ModelStealing_{args.model_name.replace('/','-')}") +
            (f"_{args.model_revision.replace('/','-')}" if args.model_revision is not None else "") +
            (f"_N={args.num_samples}_S={args.start_index}_seed={args.seed}") +
            (f"_tag={args.tag}" if args.tag is not None else "")
        )
        args.experiment_name = f"results/ModelStealing/{args.experiment_name}/{args.experiment_name}"
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
        validation_dataset = load_val_pile(number=args.num_samples,start_index=args.start_index,seed=args.seed,tokenizer=tokenizer,num_splits=1,window=2048 if args.pack else 0)[0]
        validation_dataloader = DataLoader(validation_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))

    end = time.perf_counter()
    logger.info(f"- Dataset loading took {end-start} seconds.")
    ####################################################################################################
    # RUN ATTACK
    ####################################################################################################
    start = time.perf_counter()
    logger.info("Running Attack")

    # Initialize attack
    ModelStealer = ModelStealing(args.model_name, model_revision=args.model_revision, model_cache_dir=args.model_cache_dir)
    ModelStealer.load_model()

    # Load some random internet text
    svd_dataset = load_val_pile(number=next(ModelStealer.model.parameters()).shape[1], seed=314159, num_splits=1, tokenizer=tokenizer, window=2048 if args.pack else 0)[0]
    svd_dataloader = DataLoader(svd_dataset, batch_size = 1, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))
    
    # Approximate embedding projection layer
    projector = ModelStealer.compute_model_stealing(
        svd_dataloader = svd_dataloader, 
        project_file = args.embedding_projection_file,
        proj_type = args.project_type,
        proj_dim = args.proj_dim_last,
        proj_seed = args.proj_seed,
        device=device,
        saveas = f"{args.experiment_name}_projector.pt"
    )

    # Compute statistics
    train_features = {}
    val_features = {}
    train_features["model_stealing"] = ModelStealer.compute_dataloader_model_stealing(
        dataloader=training_dataloader, projector=projector,
        num_batches=math.ceil(args.num_samples/args.bs),device=device,
        model_half=args.model_half
    )
    val_features["model_stealing"] = ModelStealer.compute_dataloader_model_stealing(
        dataloader=validation_dataloader, projector=projector,
        num_batches=math.ceil(args.num_samples/args.bs),device=device,
        model_half=args.model_half
    )
    ModelStealer.unload_model()

    torch.save(train_features,f"{args.experiment_name}_train.pt")
    torch.save(val_features,f"{args.experiment_name}_val.pt")

    end = time.perf_counter()

    logger.info(f"- Experiment {args.experiment_name} took {end-start} seconds.")

if __name__ == "__main__":
    main()

    