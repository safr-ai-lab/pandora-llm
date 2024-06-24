import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments, Trainer
import time
import transformers
import argparse
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from itertools import groupby
import os
import numpy as np
from transformers.generation.utils import GenerationMixin, GenerationConfig
from pandora_llm.utils.generation_utils import *

"""
Create data for non-supervised extraction attacks.

[Using a FT model]
python nonsupervised_extract_data_gen.py --model_name ../../../1.4b_extraction/FineTune/pythia-1.4b-deduped-ft/checkpoint-21868
    --num_generations 10 --prefixes prefixes_pileval_train.npy --suffixes suffixes_pileval_train.npy 
    --n_samples 10 --k 50 --x 50 --top_k 24 --top_p 0.8 --typical_p 0.9 --temperature 0.58 --repetition_penalty 1.04

[Using a base model]
python nonsupervised_extract_data_gen.py --model_name EleutherAI/pythia-70m-deduped 
    --num_generations 10 --prefixes prefixes_pileval_train.npy --suffixes suffixes_pileval_train.npy 
    --n_samples 10 --k 50 --x 50 --top_k 24 --top_p 0.8 --typical_p 0.9 --temperature 0.58 --repetition_penalty 1.04
"""

def main():
    parser = argparse.ArgumentParser()

    # Model Loading
    parser.add_argument('--model_name', action="store", type=str, required=True, help='Huggingface model name')
    parser.add_argument('--model_revision', action="store", type=str, required=False, help='Model revision. If not specified, uses main.')
    parser.add_argument('--model_cache_dir', action="store", type=str, required=False, help='Model cache directory. If not specified, uses main.')

    # Running parameters 
    parser.add_argument('--num_generations', action="store", type=int, required=True, help="Number Generations to do.")
    parser.add_argument('--prefixes', action="store", required=False, help='.npy file of prefixes. Not passing this in indicates you want to use the Pile Train data.')
    parser.add_argument('--suffixes', action="store", required=False, help='.npy file of suffixes. Not passing this in indicates you want to use the Pile Train data.')
    parser.add_argument('--seed', action="store", type=int, required=False, default=229, help='Seed')
    parser.add_argument('--bs', action="store", type=int, required=False, default=1, help='Batch size of generation. Only 1 is supported.')

    # Generation configs
    parser.add_argument('--n_samples', action="store", type=int, required=True, help='Number of prefixes to run generations on.')
    parser.add_argument('--k', action="store", type=int, default=50, required=False, help="For generation, do K-length prefix and X-length suffix where K+X <= 100.")
    parser.add_argument('--x', action="store", type=int, default=50, required=False, help="For generation, do K-length prefix and X-length suffix where K+X <= 100.")
    parser.add_argument('--top_k', action="store", type=int, default=10, required=False, help='Top k sampling for generation (number of tokens to choose from)')
    parser.add_argument('--top_p', action="store", type=float, default=1.0, required=False, help='Top p / nucleus eta sampling for generation (choose tokens until probability adds up to p)')
    parser.add_argument('--typical_p', action="store", type=float, default=1.0, required=False, help='Typical p / phi sampling for generation (choose locally typical tokens until probability adds up to p)')
    parser.add_argument('--temperature', action="store", type=float, default=1.0, required=False, help='Higher temperature, more diversity - the value used to modulate the next token probabilities.')
    parser.add_argument('--repetition_penalty', action="store", type=float, default=1.0, required=False, help='The parameter for repetition penalty. 1.0 means no penalty.')
    parser.add_argument('--p_threshold', action="store", type=float, required=False, help='Probability threshold to attempt generation from')
    args = parser.parse_args()

    # accelerator = Accelerator() if args.accelerate else None
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Title
    title_str = f"samp={args.n_samples}_gen={args.num_generations}_k={args.k}_x={args.x}"

    # Init Model/Tokenizer
    if args.model_name:
        args.model_cache_dir = args.model_cache_dir if args.model_cache_dir is not None else f"models/{args.model_name.replace('/','-')}"
        model = AutoModelForCausalLM.from_pretrained(args.model_name, revision=args.model_revision, cache_dir=args.model_cache_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name) # e.g. "EleutherAI/pythia-70m-deduped"
        max_length = AutoConfig.from_pretrained(args.model_name).max_position_embeddings
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_model)
        tokenizer = AutoTokenizer.from_pretrained(args.load_model) # e.g. "EleutherAI/pythia-70m-deduped"
        max_length = AutoConfig.from_pretrained(args.load_model).max_position_embeddings # might not work

    ## Data Prep
    if args.prefixes:
        prefixes = torch.tensor(np.load(args.prefixes).astype(np.int64)[:args.n_samples],dtype=torch.int64)
        suffixes = torch.tensor(np.load(args.suffixes).astype(np.int64)[:args.n_samples],dtype=torch.int64)
    else:
        dataset = load_pile_data(args.after, args.n_samples, args.seed, tokenizer, args.k, args.x)
    
    ## P_Threshold
    if args.p_threshold:
        probabilities = np.array(compute_actual_prob(prefixes, suffixes, args, model, tokenizer, device, title=f"{gen_title}_probs", kx_tup=(args.k, args.x)))

    tokenizer.pad_token = tokenizer.eos_token # not set by default

    # Generation config
    generation_config = GenerationConfig()
    generation_config.update(**{
        "min_length":100,
        "max_length":100,
        "do_sample":True,
        "pad_token_id":tokenizer.pad_token_id,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "typical_p": args.typical_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
    })

    # Make `Generation` directory
    directory_path = "Generation"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists. Using it!")

    # Generation! 
    generations = generate_suffixes(
        model=model,
        prefixes=DataLoader(prefixes,batch_size=args.bs),
        generation_config=generation_config,
        trials=args.num_generations,
        accelerate=False
    )

    # Shenanigans (could be cleaned up)
    generations = generations.reshape(prefixes.shape[0],-1,generations.shape[-1])
    generations = np.concatenate((generations,np.concatenate((prefixes,suffixes),axis=1)[:,None,:]),axis=1)
    generations = generations.reshape(-1,generations.shape[-1])
    generations = generations.reshape(prefixes.shape[0],-1,generations.shape[-1])

    with open(f"Generation/{title_str}.npy", 'wb') as f:
        print("Saving!")
        np.save(f,generations)
        
if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"Total Elapsed Time: {end_time-start_time} seconds")