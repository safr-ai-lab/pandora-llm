import os
import time
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.generation.utils import GenerationConfig
from accelerate import Accelerator
from accelerate.utils import set_seed
from llmprivacy.utils.dataset_utils import collate_fn, load_val_pile
from llmprivacy.utils.generation_utils import generate_suffixes, compute_dataloader_suffix_probability
from llmprivacy.utils.plot_utils import plot_probabilities, plot_probabilities_plotly
from llmprivacy.utils.log_utils import get_my_logger
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Sample command line prompt (no acceleration)
python run_generation.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --n_samples 1000 --pack --seed 229
Sample command laine prompt (with acceleration)
accelerate launch run_generation.py --accelerate --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --n_samples 1000 --pack --seed 229
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
    parser.add_argument('--data', action="store", type=str, required=False, help='.pt file of dataset (not dataloader)')
    parser.add_argument('--num_samples', action="store", type=int, required=True, help='Dataset size')
    parser.add_argument('--start_index', action="store", type=int, required=False, default=0, help='Slice dataset starting from this index')
    parser.add_argument('--bs', action="store", type=int, required=False, default=1, help='Batch size')
    # Generation Arguments
    parser.add_argument('--prefix_length', action="store", type=int, required=False, help='Prefix length')
    parser.add_argument('--suffix_length', action="store", type=int, required=False, help='Suffix length')
    parser.add_argument('--num_generations', action="store", type=int, default=10, required=False, help='How many generations per prompt')
    parser.add_argument('--top_k', action="store", type=int, default=10, required=False, help='Top k sampling for generation (number of tokens to choose from)')
    parser.add_argument('--top_p', action="store", type=float, default=1.0, required=False, help='Top p / nucleus eta sampling for generation (choose tokens until probability adds up to p)')
    parser.add_argument('--typical_p', action="store", type=float, default=1.0, required=False, help='Typical p / phi sampling for generation (choose locally typical tokens until probability adds up to p)')
    parser.add_argument('--temperature', action="store", type=float, default=1.0, required=False, help='Higher temperature, more diversity - the value used to modulate the next token probabilities.')
    parser.add_argument('--repetition_penalty', action="store", type=float, default=1.0, required=False, help='The parameter for repetition penalty. 1.0 means no penalty.')
    parser.add_argument('--skip_probability', action="store_true", required=False, help='Whether to skip probability computation')
    parser.add_argument('--skip_generation', action="store_true", required=False, help='Whether to skip generations')
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
            (f"Generations_{args.model_name.replace('/','-')}") +
            (f"_{args.model_revision.replace('/','-')}" if args.model_revision is not None else "") +
            # (f"_{args.data.replace('/','-')}") + 
            (f"_k={args.prefix_length}_m={args.suffix_length}") +
            (f"_N={args.num_samples}_S={args.start_index}_seed={args.seed}") +
            (f"_tag={args.tag}" if args.tag is not None else "") +
            (f"_extract")
        )
        args.experiment_name = f"results/Generations/{args.experiment_name}/{args.experiment_name}"
    os.makedirs(os.path.dirname(args.experiment_name), exist_ok=True)
    logger = get_my_logger(log_file=f"{args.experiment_name}.log")
    with open(f"{args.experiment_name}_args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    ####################################################################################################
    # LOAD DATA
    ####################################################################################################
    start = time.perf_counter()
    logger.info("Loading Data")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)    
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load data
    if args.data:
        logger.info("You are using a self-specified dataset...")
        dataset = torch.load(args.data)[args.start_index:args.start_index+args.num_samples]
        dataloader = DataLoader(dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=args.prefix_length+args.suffix_length))
    else:
        dataset = load_val_pile(number=args.num_samples,start_index=args.start_index,seed=args.seed,num_splits=1,window=0)[0]
        dataloader = DataLoader(dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=args.prefix_length+args.suffix_length))

    end = time.perf_counter()
    logger.info(f"- Dataset loading took {end-start} seconds.")
    ####################################################################################################
    # RUN GENERATIONS
    ####################################################################################################
    start = time.perf_counter()
    logger.info("Loading Model")

    model = AutoModelForCausalLM.from_pretrained(args.model_name, revision=args.model_revision, cache_dir=args.model_cache_dir).to(device)

    generation_config = GenerationConfig(**{
        "min_length": args.prefix_length+args.suffix_length,
        "max_length": args.prefix_length+args.suffix_length,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "typical_p": args.typical_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
    })

    if not args.skip_probability:
        logger.info("Computing True Suffix Probabilities")
        probabilities = compute_dataloader_suffix_probability(
            model=model,
            dataloader=dataloader,
            prefix_length=args.prefix_length,
            generation_config=generation_config,
        )
        torch.save(probabilities,f"{args.experiment_name}_probabilities.pt")
        plot_probabilities(probabilities, plot_title=args.experiment_name, log_scale=False, show_plot=False, save_name=args.experiment_name)
        plot_probabilities(probabilities, plot_title=args.experiment_name, log_scale=True, show_plot=False, save_name=args.experiment_name+"_log")
        plot_probabilities_plotly(probabilities, plot_title=args.experiment_name, log_scale=False, show_plot=False, save_name=args.experiment_name)
        plot_probabilities_plotly(probabilities, plot_title=args.experiment_name, log_scale=True, show_plot=False, save_name=args.experiment_name+"_log")

    if not args.skip_generation:
        logger.info("Generating Suffixes")
        generations = generate_suffixes(
            model=model,
            dataloader=dataloader,
            prefix_length=args.prefix_length,
            generation_config=generation_config,
            num_generations=args.num_generations,
            accelerate=False
        )
        torch.save(torch.from_numpy(generations).long(),f"{args.experiment_name}_generations.pt")
        torch.save(torch.from_numpy(np.array([sample for batch in dataloader for sample in batch["input_ids"]])).long(),f"{args.experiment_name}_true.pt")

    end = time.perf_counter()
    logger.info(f"- Experiment {args.experiment_name} took {end-start} seconds.")

if __name__ == "__main__":
    main()