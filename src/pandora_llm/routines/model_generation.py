import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
from ..utils.generation_utils import generate_suffixes


def main():
    ####################################################################################################
    # SETUP
    ####################################################################################################
    parser = argparse.ArgumentParser()
    # Model Arguments
    parser.add_argument('--args_path', action="store", type=str, required=True, help='Training args path')
    parser.add_argument('--model_name', action="store", type=str, required=True, help='Huggingface model name')
    parser.add_argument('--model_revision', action="store", type=str, required=False, help='Model revision. If not specified, uses main.')
    parser.add_argument('--model_cache_dir', action="store", type=str, required=False, help='Model cache directory. If not specified, uses main.')
    parser.add_argument('--save_path', action="store", type=str, required=True, help="Save path")
    # Data Arguments
    parser.add_argument('--prefixes', action="store", type=str, required=True, help='Path to prefixes')
    # Generation Arguments
    parser.add_argument('--gen_config', action="store", type=str, required=True, help='GenerationConfig path')
    parser.add_argument('--num_trials', action="store", type=int, required=True, help='Number of Trials')
    # Device Arguments
    parser.add_argument('--device', action="store", type=str, required=False, help='Device')
    parser.add_argument('--accelerate', action="store_true", required=False, help='Use accelerate')
    parser.add_argument('--model_half', action="store_true", required=False, help='Use half precision (fp16)')
    args = parser.parse_args()
    ####################################################################################################
    # GENERATE FROM MODEL
    ####################################################################################################
    accelerator = Accelerator() if args.accelerate else None

    model = AutoModelForCausalLM.from_pretrained(args.model_path, revision=args.model_revision, cache_dir=args.cache_dir)
    prefixes = DataLoader(torch.tensor(np.load(args.prefixes)[:args.n_samples], dtype=torch.int64), batch_size=args.bs)
    generation_config = torch.load(args.gen_config)
    
    if accelerator is not None:
        model, prefixes = accelerator.prepare(model, prefixes)

    generations = generate_suffixes(
        model=model,
        prefixes=prefixes,
        generation_config=generation_config,
        trials=args.num_trials,
        accelerate=args.accelerate,
    )

    if accelerator is None or accelerator.is_main_process:
        torch.save(generations,args.save_path)

if __name__ == "__main__":
    main()