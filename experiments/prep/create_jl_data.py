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
import sys

"""
This script serves a few functions:
1) It creates the JL'ized gradients with respect to train/val data. By default it will project each layer ot some fixed num dimensions (proj_each_layer_to; default=3). 

[Example] python create_jl_data.py --model_name EleutherAI/pythia-1b-deduped --pack --num_samples 10 --proj_each_layer_to 4 --wrt theta --model_half --project_type normal

2) It creates the JL'ized embedding layer (up to symmetries) that is used in our gray-box model-stealing attack. This info also contains p=1,2,inf norms of those gradients.

[Example] python create_jl_data.py --model_name EleutherAI/pythia-1b-deduped --pack --num_samples 10 --last_layer --project_type normal

3) It creates the enhanced JL features with full projections (need to use --mode for this):

[Example] python create_jl_data.py --model_name EleutherAI/pythia-1.4b-deduped --model_revision step98000 --num_samples 2000 --bs 1 --mode 1 --project_type rademacher --start_index 150000
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

def balanced_partition(sizes, num_groups):
    # Pair each size with its original index
    sizes_with_indices = list(enumerate(sizes))
    # Sort sizes in descending order while keeping track of indices
    sizes_with_indices.sort(key=lambda x: x[1], reverse=True)
    
    # Initialize groups and their sums
    groups = [[] for _ in range(num_groups)]
    group_sums = [0] * num_groups
    group_indices = [[] for _ in range(num_groups)]
    
    # Assign each size to the group with the smallest current sum
    for index, size in sizes_with_indices:
        min_index = group_sums.index(min(group_sums))
        groups[min_index].append(size)
        group_indices[min_index].append(index)
        group_sums[min_index] += size

    return groups, group_sums, group_indices

def main():
    parser = argparse.ArgumentParser()

    # Model Arguments
    parser.add_argument('--model_name', action="store", type=str, required=True, help='Huggingface model name')
    parser.add_argument('--model_revision', action="store", type=str, required=False, help='Model revision. If not specified, uses main.')
    parser.add_argument('--model_cache_dir', action="store", type=str, required=False, help='Model cache directory. If not specified, uses main.')

    # Dataset arguments 
    parser.add_argument('--pack', action="store_true", required=False, help='Pack validation set')
    parser.add_argument('--num_samples', action="store", type=int, required=True, help='Dataset size')
    parser.add_argument('--min_length', action="store", type=int, required=False, default=20, help='Min number of tokens')
    parser.add_argument('--train_pt', action="store", required=False, help='.pt file of train dataset (not dataloader) - if using own data')
    parser.add_argument('--val_pt', action="store", required=False, help='.pt file of val dataset (not dataloader) - if using own data')
    parser.add_argument('--start_index', action="store", type=int, required=False, default=0, help='Slice dataset starting from this index')

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
    parser.add_argument('--mode', action="store", type=int, required=False,default=0,help='Mode = 1 -> Fancy JL features. Otherwise, omit this flag.')
    parser.add_argument('--jl_enhance_num_split', action="store", type=int, required=False, default=8, help='Num groups of layers to try dividing into before JL projection.')

    # Experimental flags
    parser.add_argument('--quiet', action="store_true", required=False, default=False, help="Hide print of Shapes & Samples.")
    parser.add_argument('--no_rotation', action="store_true", required=False,default=False,help='No random rotation of embedding projection layer (for debugging)')
    args = parser.parse_args()

    accelerator = Accelerator() if args.accelerate else None
    set_seed(args.seed)
    quiet = args.quiet
           
    title_str = f"model={args.model_name.replace('/','-')}_samp={args.num_samples}_seed={args.seed}_projseed={args.proj_seed}_half={args.model_half}_start={args.start_index}"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, revision=args.model_revision, cache_dir=args.model_cache_dir)

    # Initialization
    embedding_layer = model.get_input_embeddings().weight
    max_length = AutoConfig.from_pretrained(args.model_name).max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load training data
    if args.train_pt:
        logger.info("You are using a self-specified training dataset...")
        fixed_input = args.train_pt + ".pt" if not args.train_pt.endswith(".pt") else args.train_pt
        training_dataset = torch.load(fixed_input)[args.start_index:args.start_index+args.num_samples]
        training_dataloader = DataLoader(training_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))
    else:
        training_dataset = load_train_pile_random(number=args.num_samples,start_index=args.start_index,seed=args.seed,num_splits=1,min_length=args.min_length,deduped="deduped" in args.model_name)[0]
        training_dataloader = DataLoader(training_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))
        if accelerator is not None: # for subprocess call
            args.train_pt = "Data/JL/train_dataset.pt"
            torch.save(training_dataset,args.train_pt)

    # Save Information
    directory_path = "Data"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists. Using it!")    

    # Load validation data
    if args.val_pt:
        fixed_input = args.val_pt + ".pt" if not args.val_pt.endswith(".pt") else args.val_pt
        logger.info("You are using a self-specified validation dataset...")
        validation_dataset = torch.load(fixed_input)[:args.num_samples]
        validation_dataloader = DataLoader(validation_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))
    else:
        validation_dataset = load_val_pile(number=args.num_samples,start_index=args.start_index,seed=args.seed,num_splits=1,window=2048 if args.pack else 0)[0]
        validation_dataloader = DataLoader(validation_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))
        if accelerator is not None: # for subprocess call
            args.val_pt = "Data/JL/val_dataset.pt"
            torch.save(validation_dataset,args.val_pt)

    if not quiet: # e.g. for verifying that data seed is right
        print("Sample Train Point:")
        print(training_dataset[0])
        print("Sample Val Point:")
        print(validation_dataset[0])
    
    ## Set up projectors (JL objects from TRAK)
    projectors = {}                 
    assert args.project_type in ["rademacher","normal"]
    
    if args.mode == 1:
        print("Mode = 1, so computing Carlini JL Gray Box features + JL of gradients...")
        ## CARLINI INFO
        input_size = next(model.parameters()).shape[1] * next(model.parameters()).shape[0]
        projectors["embed_out"] = CudaProjector(input_size, 512, 
                                                        args.proj_seed, ProjectionType(args.project_type), 'cuda', 8)

        svd_dataset = load_val_pile(number=next(model.parameters()).shape[1], seed=314159, num_splits=1, window=2048 if args.pack else 0)[0]
        svd_dataloader = DataLoader(svd_dataset, batch_size = 1, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))
        
        # Computing Logits for svd_dataloader
        print("Computing logits for svd_dataloader")
        if os.path.exists("logits.pt"):
            print("Loading....")
            dataloader_logits = torch.load("logits.pt").to('cpu')
        else:
            dataloader_logits = compute_dataloader_logits_embedding(model, svd_dataloader, device, half=args.model_half).T.float().to(device)
            torch.save(dataloader_logits, "logits.pt")
        last_layer = [m for m in model.parameters()][-1]
        
        ## Generate matrix U @ torch.diag(S) which is equal to embedding projection up to symmetries
        U, S, _ = torch.linalg.svd(dataloader_logits,full_matrices=False)
        svd_embedding_projection_layer = U @ torch.diag(S)

        ## Identify base change to convert regular gradients to gradients we can access
        print("Computing random basis change")
        base_change = torch.linalg.pinv(last_layer.float()).to('cpu') @ svd_embedding_projection_layer.to('cpu')
        projectors["random_basis_change"] = torch.linalg.inv(base_change).T  

        print("Computing Carlini Gray Box Features")
        train_carlini_info = compute_dataloader_basis_changes(model, training_dataloader, projectors, device=device, nbatches=args.num_samples, half=args.model_half).cpu() 
        torch.save(train_carlini_info, f"Data/carlini_train_{title_str}.pt")
        val_carlini_info = compute_dataloader_basis_changes(model, validation_dataloader, projectors, device=device, nbatches=args.num_samples, half=args.model_half).cpu() 
        torch.save(val_carlini_info, f"Data/carlini_val_{title_str}.pt")

        if not quiet:
            print(f"Train Carlini Info: {train_carlini_info.shape}")
            print(f"Val Carlini Info: {val_carlini_info.shape}")

        ## JL Enahnced Features
        NUM = args.jl_enhance_num_split
        sizes = []
        for i, (name,param) in enumerate(model.named_parameters()):
            sizes.append(prod(param.size()))
        
        groups, sums, indices = balanced_partition(sizes, NUM)
        if not quiet:
            print(f"Split groups: {groups}")
            print(f"Split sums: {sums}")
            print(f"Split group indices: {indices}")

        for i in range(NUM):
            projectors[i] = CudaProjector(sums[i], 512, args.proj_seed, ProjectionType(args.project_type), 'cuda', 32)
        
        projectors["x"] = BasicProjector(next(model.parameters()).shape[1]**2, 32, args.proj_seed, args.project_type, 'cuda', 1)

        train_jl_big = compute_dataloader_jl_enhanced(model, embedding_layer, training_dataloader, projectors, indices, device=device).cpu() 
        torch.save(train_jl_big, f"Data/jllayers_train_{title_str}.pt")

        val_jl_big = compute_dataloader_jl_enhanced(model, embedding_layer, validation_dataloader, projectors, indices, device=device).cpu() 
        torch.save(val_jl_big, f"Data/jllayers_val_{title_str}.pt")

        # Save
        if not quiet:
            print(f"Train JL Big: {train_jl_big.shape}") # e.g. print(f"Train JL Big: {train_jl_big.shape}")
            print(f"Val JL Big: {val_jl_big.shape}")

        train_info = torch.cat((train_carlini_info, train_jl_big), dim=1)
        val_info = torch.cat((val_carlini_info, val_jl_big), dim=1)
        finalsavetrain = f"enhanceJL_train_{title_str}.pt"
        finalsaveval = f"enhanceJL_val_{title_str}.pt"

        if not quiet:
            print(f"Train JL + Carlini Concatenated Shape: {train_info.shape}")
            print(f"Val JL + Carlini Concatenated Shape: {val_info.shape}")
    elif args.mode == 2:
        print(f"Mode = 2, so not including Carlini JL features...")
        ## JL Enhanced Features
        NUM = args.jl_enhance_num_split
        sizes = []
        for i, (name,param) in enumerate(model.named_parameters()):
            sizes.append(prod(param.size()))
        
        groups, sums, indices = balanced_partition(sizes, NUM)
        if not quiet:
            print(f"Split groups: {groups}")
            print(f"Split sums: {sums}")
            print(f"Split group indices: {indices}")

        for i in range(NUM):
            projectors[i] = CudaProjector(sums[i], 512, args.proj_seed, ProjectionType(args.project_type), 'cuda', 32)
        
        projectors["x"] = BasicProjector(next(model.parameters()).shape[1]**2, 32, args.proj_seed, args.project_type, 'cuda', 1)

        train_info = compute_dataloader_jl_enhanced(model, embedding_layer, training_dataloader, projectors, indices, device=device).cpu() 
        val_info = compute_dataloader_jl_enhanced(model, embedding_layer, validation_dataloader, projectors, indices, device=device).cpu() 

        finalsavetrain = f"Data/jllayers_train_{title_str}.pt"
        finalsaveval = f"Data/jllayers_val_{title_str}.pt"

        if not quiet:
            print(f"Train JL Shape: {train_info.shape}")
            print(f"Val JL Shape: {val_info.shape}")

    elif args.last_layer: # Gray-box attack on [MLP] -> Logits layer
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

        finalsavetrain = f"Data/carliniGray_train_{title_str}.pt"
        finalsaveval = f"Data/carliniGray_val_{title_str}.pt"

        train_info = compute_dataloader_basis_changes(model, training_dataloader, projectors, device=device, nbatches=args.num_samples, half=args.model_half).cpu() 
        val_info = compute_dataloader_basis_changes(model, validation_dataloader, projectors, device=device, nbatches=args.num_samples, half=args.model_half).cpu() 

    else: # JL of all layers
        print(f"In total, the number of features will be: {sum(1 for _ in model.named_parameters()) * args.proj_each_layer_to}.")
        ## Project each type of data with a JL dimensionality reduction
        projectors["x"] = BasicProjector(next(model.parameters()).shape[1]**2, args.proj_each_layer_to, args.proj_seed, args.project_type, 'cuda', 1)
        
        for i, (name,param) in enumerate(model.named_parameters()):
            projectors[(i,name)] = BasicProjector(prod(param.size()), args.proj_each_layer_to, args.proj_seed, args.project_type, 'cuda', 1)
    
        ## Get data
        train_info = compute_dataloader_jl(model, embedding_layer, training_dataloader, projectors, device=device, nbatches=args.num_samples, half=args.model_half).cpu() 
        val_info = compute_dataloader_jl(model, embedding_layer, validation_dataloader, projectors, device=device, nbatches=args.num_samples, half=args.model_half).cpu() 
        
        finalsavetrain = f"Data/JL_train_to{args.proj_each_layer_to}_{title_str}.pt"
        finalsaveval = f"Data/JL_val_to{args.proj_each_layer_to}_{title_str}.pt"

    # Data
    print(f"Saving to... {finalsavetrain} and {finalsaveval}")
    train_dict = {"data": train_info}
    val_dict = {"data": val_info}
    
    torch.save(train_info, finalsavetrain)
    torch.save(val_info, finalsaveval)

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"Total Elapsed Time: {end_time-start_time} seconds")