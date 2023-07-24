import torch
from torch.utils.data import DataLoader
import transformers
import datasets
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from tqdm import tqdm
from attack_utils import *
from dataset_utils import *
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', action="store", type=str, required=True, help='Model Path')
    parser.add_argument('--model_revision', action="store", type=str, required=False, default="main", help='Model Revision')
    parser.add_argument('--cache_dir', action="store", type=str, required=False, help='Model Cache Dir')
    parser.add_argument('--save_path', action="store", type=str, required=True, help="Save path")
    parser.add_argument('--model_half', action="store_true", required=False, help='Use half precision (fp16)')
    args = parser.parse_args()

    model = GPTNeoXForCausalLM.from_pretrained(args.model_path, revision=args.model_revision, cache_dir=args.cache_dir)
    if args.model_half:
        model.half()
    torch.save(model.get_input_embeddings().weight,args.save_path)

if __name__ == "__main__":
    main()