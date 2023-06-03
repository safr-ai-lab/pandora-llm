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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', action="store", type=str, required=True, help='Model Path')
    parser.add_argument('--model_revision', action="store", type=str, required=False, default="main", help='Model Revision')
    parser.add_argument('--cache_dir', action="store", type=str, required=False, help='Model Cache Dir')
    parser.add_argument('--dataset_path', action="store", type=str, required=True, help='Dataset path')
    parser.add_argument('--device', action="store", type=str, required=False, help='Device')
    parser.add_argument('--n_samples', action="store", type=int, required=True, help='Number of samples')
    parser.add_argument('--bs', action="store", type=int, required=True, help='Batch size')
    parser.add_argument('--save_path', action="store", type=str, required=True, help="Save path")
    parser.add_argument('--accelerate', action="store_true", required=False, help='Use accelerate')
    parser.add_argument('--model_half', action="store_true", required=False, help='Use half precision (fp16)')
    args = parser.parse_args()

    accelerator = Accelerator() if args.accelerate else None

    model = GPTNeoXForCausalLM.from_pretrained(args.model_path, revision=args.model_revision, cache_dir=args.cache_dir)
    max_length = model.config.max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = torch.load(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, length=max_length))
    
    if accelerator is not None:
        model, dataloader = accelerator.prepare(model, dataloader)

    loss = compute_dataloader_cross_entropy(model, dataloader, device=args.device, nbatches=args.n_samples, accelerator=accelerator, half=args.model_half).detach().cpu()
    
    if accelerator is None or accelerator.is_main_process:
        torch.save(loss,args.save_path)

if __name__ == "__main__":
    main()