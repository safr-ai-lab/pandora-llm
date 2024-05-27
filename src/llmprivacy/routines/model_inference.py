import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed
from ..attacks.LOSS import compute_dataloader_cross_entropy
from ..utils.dataset_utils import collate_fn

def main():
    ####################################################################################################
    # SETUP
    ####################################################################################################
    parser = argparse.ArgumentParser()
    # Model Arguments
    parser.add_argument('--model_name', action="store", type=str, required=True, help='Huggingface model name')
    parser.add_argument('--model_revision', action="store", type=str, required=False, help='Model revision. If not specified, uses main.')
    parser.add_argument('--model_cache_dir', action="store", type=str, required=False, help='Model cache directory. If not specified, uses main.')
    # Dataset Arguments
    parser.add_argument('--dataset_path', action="store", type=str, required=True, help='Dataset path')
    parser.add_argument('--already_tokenized', action="store_true", required=False, help='Skip collation - use if dataset is already tokenized')
    parser.add_argument('--n_samples', action="store", type=int, required=False, help='Number of samples')
    parser.add_argument('--bs', action="store", type=int, required=True, help='Batch size')
    parser.add_argument('--save_path', action="store", type=str, required=True, help="Save path")
    # Device Arguments
    parser.add_argument('--seed', action="store", type=int, required=False, help='Seed')
    parser.add_argument('--device', action="store", type=str, required=False, help='Device')
    parser.add_argument('--accelerate', action="store_true", required=False, help='Use accelerate')
    parser.add_argument('--model_half', action="store_true", required=False, help='Use half precision (fp16)')
    args = parser.parse_args()

    set_seed(args.seed)
    accelerator = Accelerator() if args.accelerate else None
    ####################################################################################################
    # INFERENCE
    ####################################################################################################
    model = AutoModelForCausalLM.from_pretrained(args.model_path, revision=args.model_revision, cache_dir=args.cache_dir)
    max_length = model.config.max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = torch.load(args.dataset_path)
    if not args.already_tokenized:
        dataloader = DataLoader(dataset, batch_size=args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, length=max_length))
    else:
        dataloader = DataLoader(dataset, batch_size=args.bs)

    if accelerator is not None:
        model, dataloader = accelerator.prepare(model, dataloader)

    loss = compute_dataloader_cross_entropy(model, dataloader, device=args.device, nbatches=args.n_samples, accelerator=accelerator, half=args.model_half).detach().cpu()
    
    if accelerator is None:
        torch.save(loss,args.save_path)
    else:
        with accelerator.main_process_first():
            torch.save(loss,args.save_path)

if __name__ == "__main__":
    main()