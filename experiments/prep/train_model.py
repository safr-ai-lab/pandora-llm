import time
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments, Trainer
import argparse
from accelerate import Accelerator
from accelerate.utils import set_seed
import torch
from pandora_llm.utils.dataset_utils import collate_fn

"""
Script to fine-tune a model on certain # of points, for certain # of epochs. 

Example [on a base model]:
python train_model.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --train_pt pileval_train_dl.pt --val_pt pileval_val_dl.pt --n_train_epochs 1

Example [on an already ft'ed model]:
python train_model.py --load_model FineTune/70mtest --seed 229 --train_pt pileval_train_dl.pt --val_pt pileval_val_dl.pt --n_train_epochs 1 --title 70mtest2
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', action='store', type=str, required=True, help="Title of experiment.")

    # Default arguments
    parser.add_argument('--model_name', action="store", type=str, required=False, help='Name of model, if starting from base image.')
    parser.add_argument('--model_revision', action="store", type=str, required=False, help='Model revision. If not specified, uses main.')
    parser.add_argument('--model_cache_dir', action="store", type=str, required=False, help='Model cache directory. If not specified, uses main.')

    parser.add_argument('--load_model', action="store", type=str, required=False, help='Pre-load (a possibly already ft-ed) model, to FT more. Should have tokenizer in base dir.')
    parser.add_argument('--accelerate', action="store_true", required=False, help='Use accelerate. Not currently supported.')
    parser.add_argument('--seed', action="store", type=int, required=False, default=229, help='Seed')

    # Fine-tuning 
    parser.add_argument('--train_pt', action="store", type=str, required=False, help='pt file of train string array (for fine-tuning attacks)')
    parser.add_argument('--val_pt', action="store", type=str, required=False, help='pt file of val string array (for fine-tuning attacks)')
    parser.add_argument('--n_train_epochs', action="store", type=int, required=False, help='Num train epochs for fine-tuning.')
    parser.add_argument('--ft_bs', action="store", type=int, required=False, default=1, help='batch size when fine-tuning the model')

    args = parser.parse_args()

    ## Other parameters
    seed = args.seed
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## Launch Checks
    assert (args.model_name or args.load_model), "Load a HF hub model or a saved one."
    if args.model_name:
        args.model_cache_dir = args.model_cache_dir if args.model_cache_dir is not None else f"models/{args.model_name.replace('/','-')}"
        model = AutoModelForCausalLM.from_pretrained(args.model_name, revision=args.model_revision, cache_dir=args.model_cache_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name) # e.g. "EleutherAI/pythia-70m-deduped"
        max_length = AutoConfig.from_pretrained(args.model_name).max_position_embeddings
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_model)
        tokenizer = AutoTokenizer.from_pretrained(args.load_model) # e.g. "EleutherAI/pythia-70m-deduped"
        max_length = AutoConfig.from_pretrained(args.load_model).max_position_embeddings # might not work

    tokenizer.pad_token = tokenizer.eos_token # not set by default

    training_args = TrainingArguments(output_dir=f"FineTune/{args.title}",
        do_train=True,
        do_eval=True,
        num_train_epochs=args.n_train_epochs,
        per_device_train_batch_size=args.ft_bs,
        per_device_eval_batch_size=args.ft_bs,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch", # this means a model is saved every epoch
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        load_best_model_at_end=False,
        fp16=True,
        deepspeed='ds_config_zero3.json' if args.accelerate else None
        )

    # Load Data
    print("Loading Data")
    print("You are using a self-specified training dataset...")
    fixed_input = args.train_pt + ".pt" if not args.train_pt.endswith(".pt") else args.train_pt
    training_dataset = torch.load(fixed_input)
    print("You are using a self-specified validation dataset...")
    fixed_input = args.val_pt + ".pt" if not args.val_pt.endswith(".pt") else args.val_pt
    validation_dataset = torch.load(fixed_input)

    trainer = Trainer(model=model,
                args=training_args,
                train_dataset=training_dataset,
                eval_dataset=validation_dataset,
                tokenizer=tokenizer,
                data_collator=lambda batch: collate_fn(batch, tokenizer, max_length),
            )

    trainer.train()
    trainer.save_model(f"FineTune/{args.title}")
    tokenizer.save_pretrained(f"FineTune/{args.title}")

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"Total Elapsed Time: {end_time-start_time} seconds")