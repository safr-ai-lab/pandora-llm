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
from accelerate import Accelerator
import dill
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from transformers.trainer_utils import get_last_checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainer_path', action="store", type=str, required=True, help='Trainer Path')
    parser.add_argument('--save_path', action="store", type=str, required=True, help="Save path")
    parser.add_argument('--model_path', action="store", type=str, required=True, help="Model path")
    parser.add_argument('--model_revision', action="store", type=str, required=False, help="Model revision")
    parser.add_argument('--cache_dir', action="store", type=str, required=False, help="Model cache dir")
    parser.add_argument('--accelerate', action="store_true", required=False, help='Use accelerate')
    args = parser.parse_args()

    with open(args.trainer_path, 'rb') as f:
        trainer = dill.load(f)
    trainer.train()
    trainer.save_model(args.save_path)
    if args.accelerate:
        model = GPTNeoXForCausalLM.from_pretrained(args.model_path,revision=args.model_revision,cache_dir=args.cache_dir)
        fp32_model = load_state_dict_from_zero_checkpoint(model,checkpoint_dir=get_last_checkpoint(trainer.args.output_dir))
        fp32_model.save_pretrained(args.save_path)

if __name__ == "__main__":
    main()