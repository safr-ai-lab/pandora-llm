import argparse
import torch
from transformers import AutoModelForCausalLM

def main():
    ####################################################################################################
    # SETUP
    ####################################################################################################
    parser = argparse.ArgumentParser()
    # Model Arguments
    parser.add_argument('--model_name', action="store", type=str, required=True, help='Huggingface model name')
    parser.add_argument('--model_revision', action="store", type=str, required=False, help='Model revision. If not specified, uses main.')
    parser.add_argument('--model_cache_dir', action="store", type=str, required=False, help='Model cache directory. If not specified, uses main.')
    parser.add_argument('--save_path', action="store", type=str, required=True, help="Save path")
    # Device Arguments
    parser.add_argument('--model_half', action="store_true", required=False, help='Use half precision (fp16)')
    args = parser.parse_args()
    ####################################################################################################
    # RETRIEVE EMBEDDING
    ####################################################################################################
    model = AutoModelForCausalLM.from_pretrained(args.model_path, revision=args.model_revision, cache_dir=args.cache_dir)
    if args.model_half:
        model.half()
    torch.save(model.get_input_embeddings().weight,args.save_path)

if __name__ == "__main__":
    main()