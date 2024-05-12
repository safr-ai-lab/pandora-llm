from math import prod
from sklearn import random_projection
from joblib import dump, load
from llmprivacy.utils.attack_utils import *
from llmprivacy.utils.dataset_utils import *
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from torch.utils.data import DataLoader
import torch
import os
import subprocess
from trak.projectors import CudaProjector, ProjectionType, ChunkedCudaProjector, BasicProjector
import time
from accelerate.utils import set_seed
import argparse

"""
This script serves a few functions:
1) It creates the JL'ized gradients with respect to train/val data. By default it will project each layer ot some fixed num dimensions (proj_each_layer_to; default=3). 

[Example] python create_jl_data.py --model_name EleutherAI/pythia-1b-deduped --pack --num_samples 10 --proj_each_layer_to 4 --wrt theta --model_half --project_type normal

2) It creates the JL'ized embedding layer (up to symmetries) that is used in our gray-box model-stealing attack. This info also contains p=1,2,inf norms of those gradients.

[Example] python create_jl_data.py --model_name EleutherAI/pythia-1b-deduped --pack --num_samples 10 --last_layer --project_type normal
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser()

    # Model Arguments
    parser.add_argument('--model_name', action="store", type=str, required=True, help='Huggingface model name')
    parser.add_argument('--model_revision', action="store", type=str, required=False, help='Model revision. If not specified, uses main.')
    parser.add_argument('--model_cache_dir', action="store", type=str, required=False, help='Model cache directory. If not specified, uses main.')

    # Dataset arguments 
    parser.add_argument('--pack', action="store_true", required=False, help='Pack validation set')
    parser.add_argument('--num_samples', action="store", type=int, required=True, help='Dataset size')
    # parser.add_argument('--sample_length', action="store", type=int, required=False, help='Truncate number of tokens')
    parser.add_argument('--min_length', action="store", type=int, required=False, default=20, help='Min number of tokens')
    parser.add_argument('--train_pt', action="store", required=False, help='.pt file of train dataset (not dataloader) - if using own data')
    parser.add_argument('--val_pt', action="store", required=False, help='.pt file of val dataset (not dataloader) - if using own data')

    # Running Arguments
    parser.add_argument('--seed', action="store", type=int, required=False, default=229, help='Seed')
    parser.add_argument('--proj_seed', action="store", type=int, required=False, default=229, help='Seed for random projection')
    parser.add_argument('--proj_each_layer_to', action="store", type=int, required=False, default=3, help='When JLing each layer, project to this dimension.')
    parser.add_argument('--bs', action="store", type=int, required=False, default=1, help='Batch size')
    parser.add_argument('--wrt', action="store", type=str, required=False, default="x", help='Take gradient with respect to {x, theta}')    
    parser.add_argument('--accelerate', action="store_true", required=False, help='Use accelerate. Not supported.')
    parser.add_argument('--model_half', action="store_true", required=False, help='Use half precision (fp16). 1 for use; 0 for not.')
    parser.add_argument('--project_type', action="store", type=str,required=False,default=False,help='type of projection (rademacher or normal)')
    parser.add_argument('--last_layer', action="store_true", required=False,default=False,help='last layer only (Carlini et al. 2024 extraction experiment)')
    parser.add_argument('--last_layer_proj_dim', action="store", type=int, required=False,default=4096,help='dimension of projection of last layer gradients')

    # Experimental flags
    parser.add_argument('--no_rotation', action="store_true", required=False,default=False,help='No random rotation of embedding projection layer (for debugging)')
    parser.add_argument('--total_dim', action="store", required=False, default=-1, help='Specify total number of features to save. Dimensionality reduction will intelligently work around this.')
    args = parser.parse_args()

    accelerator = Accelerator() if args.accelerate else None
    set_seed(args.seed)

    # Title and setup
    if args.last_layer:
        title_str = f"Data/lastlayer_model={args.model_name.replace('/','-')}_samp={args.num_samples}_seed={args.seed}_projseed={args.proj_seed}_half={args.model_half}"
    else:
        if args.total_dim >= 0:
            title_str = f"Data/gradjl_model={args.model_name.replace('/','-')}_samp={args.num_samples}_wrt={args.wrt}_totalproj={args.total_dim}_seed={args.seed}_projseed={args.proj_seed}_half={args.model_half}"
            print("total_dim is not yet supported; exiting....")
            sys.exit(0)
        else:
            title_str = f"Data/gradjl_model={args.model_name.replace('/','-')}_samp={args.num_samples}_wrt={args.wrt}_projeach={args.proj_each_layer_to}_seed={args.seed}_projseed={args.proj_seed}_half={args.model_half}"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, revision=args.model_revision, cache_dir=args.model_cache_dir)

    # Load training data
    if args.train_pt:
        logger.info("You are using a self-specified training dataset...")
        fixed_input = args.train_pt + ".pt" if not args.train_pt.endswith(".pt") else args.train_pt
        training_dataset = torch.load(fixed_input)[:args.num_samples]
        training_dataloader = DataLoader(training_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))
    else:
        training_dataset = load_train_pile_random(number=args.num_samples, seed=args.seed, num_splits=1, min_length=args.min_length, deduped="deduped" in args.model_name)[0]
        training_dataloader = DataLoader(training_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))
        if accelerator is not None: # for subprocess call
            args.train_pt = "data/JL/train_dataset.pt"
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
            args.val_pt = "data/JL/val_dataset.pt"
            torch.save(validation_dataset,args.val_pt)

    # Initialization
    embedding_layer = model.get_input_embeddings().weight
    max_length = AutoConfig.from_pretrained(args.model_name).max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    ## Set up projectors (JL objects from TRAK)
    projectors = {}                 
    assert args.project_type in ["rademacher","normal"]          
    
    if args.last_layer: # Gray-box attack on [MLP] -> Logits layer
        input_size = next(model.parameters()).shape[1] * next(model.parameters()).shape[0]
        projectors["embed_out"] = CudaProjector(input_size, args.last_layer_proj_dim, 
                                                        args.proj_seed, ProjectionType(args.project_type), 'cuda', 8)

        if args.no_rotation: # debugging, just use the godeye view
            projectors["random_basis_change"] = torch.eye(next(model.parameters()).shape[1])
        else:
            ## Generate data to run Carlini data extraction attack
            # mechanically, this is an equivalent projection to the carlini paper

            svd_dataset = load_val_pile(number=next(model.parameters()).shape[1], seed=314159, num_splits=1, window=2048 if args.pack else 0)[0]
            svd_dataloader = DataLoader(svd_dataset, batch_size = 1, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))
            
            dataloader_logits = compute_dataloader_logits_embedding(model, svd_dataloader, device, half=args.model_half).T.float().to(device)
            last_layer = [m for m in model.parameters()][-1]
            
            ## Generate matrix U @ torch.diag(S) which is equal to embedding projection up to symmetries
            U, S, _ = torch.linalg.svd(dataloader_logits,full_matrices=False)
            svd_embedding_projection_layer = U @ torch.diag(S)

            ## Identify base change to convert regular gradients to gradients we can access
            base_change = torch.linalg.pinv(last_layer.float()) @ svd_embedding_projection_layer
            projectors["random_basis_change"] = torch.linalg.inv(base_change).T   

        train_info = compute_dataloader_basis_changes(model, training_dataloader, projectors, device=device, nbatches=args.num_samples, half=args.model_half).cpu() 
        val_info = compute_dataloader_basis_changes(model, validation_dataloader, projectors, device=device, nbatches=args.num_samples, half=args.model_half).cpu() 
    else: # JL of all layers
        if args.total_dim >= 0:
            print(f"In total, the number of features will be: {args.total_dim}. This feature is not yet implemented.")
            pass
        else:
            print(f"In total, the number of features will be: {sum(1 for _ in model.named_parameters()) * args.proj_each_layer_to}.")
            ## Project each type of data with a JL dimensionality reduction
            projectors["x"] = BasicProjector(next(model.parameters()).shape[1]**2, args.proj_each_layer_to, args.proj_seed, args.project_type, 'cuda', 1)
            
            for i, (name,param) in enumerate(model.named_parameters()):
                projectors[(i,name)] = BasicProjector(prod(param.size()), args.proj_each_layer_to, args.proj_seed, args.project_type, 'cuda', 1)
        
            ## Get data
            train_info = compute_dataloader_jl(model, embedding_layer, training_dataloader, projectors, device=device, nbatches=args.num_samples, half=args.model_half).cpu() 
            val_info = compute_dataloader_jl(model, embedding_layer, validation_dataloader, projectors, device=device, nbatches=args.num_samples, half=args.model_half).cpu() 

    # Data
    train_dict = {"jl-train": train_info}
    val_dict = {"jl-val": val_info}

    # Save Information
    directory_path = "Data"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists. Using it!")
    
    torch.save(train_info, f"{title_str}_train.pt")
    torch.save(val_info, f"{title_str}_val.pt")

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"Total Elapsed Time: {end_time-start_time} seconds")