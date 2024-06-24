import os
import time
import json
import subprocess
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments, Trainer
from accelerate import Accelerator
from accelerate.utils import set_seed
from pandora_llm.utils.dataset_utils import collate_fn, load_val_pile
from pandora_llm.utils.log_utils import get_my_logger
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Sample command line prompt (no acceleration)
python run_loss.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --n_samples 1000 --pack --seed 229
Sample command laine prompt (with acceleration)
accelerate launch run_loss.py --accelerate --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --n_samples 1000 --pack --seed 229
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
    parser.add_argument('--num_samples', action="store", type=int, required=True, help='Dataset size')
    parser.add_argument('--start_index', action="store", type=int, required=False, default=0, help='Slice dataset starting from this index')
    parser.add_argument('--bs', action="store", type=int, required=False, default=1, help='Batch size')
    parser.add_argument('--pack', action="store_true", required=False, help='Pack validation set')
    # Fine-tuning 
    parser.add_argument('--train_pt', action="store", type=str, required=False, help='pt file of train string array (for fine-tuning attacks)')
    parser.add_argument('--val_pt', action="store", type=str, required=False, help='pt file of val string array (for fine-tuning attacks)')
    parser.add_argument('--num_epochs', action="store", type=int, required=False, default=1, help='Num train epochs for fine-tuning.')
    parser.add_argument('--learning_rate', action="store", type=float, required=False, default=5e-5, help='Learning rate')
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
            (f"{args.model_name.replace('/','-')}") +
            (f"_{args.model_revision.replace('/','-')}" if args.model_revision is not None else "") +
            (f"_N={args.num_samples}_S={args.start_index}_seed={args.seed}") +
            (f"_tag={args.tag}" if args.tag is not None else "")
        )
        args.experiment_name = f"models/FineTune/{args.experiment_name}/{args.experiment_name}"
    os.makedirs(os.path.dirname(args.experiment_name), exist_ok=True)
    logger = get_my_logger(log_file=f"{args.experiment_name}.log")
    with open(f"{args.experiment_name}_args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    ####################################################################################################
    # LOAD DATA
    ####################################################################################################
    start = time.perf_counter()

    max_length = AutoConfig.from_pretrained(args.model_name).max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    logger.info("Loading Data")    

    # Load data
    if args.train_pt and args.val_pt:
        logger.info("You are using a self-specified validation dataset...")
        
        fixed_input = args.train_pt + ".pt" if not args.train_pt.endswith(".pt") else args.train_pt
        training_dataset = torch.load(fixed_input)[args.start_index:args.start_index+args.num_samples]
        
        fixed_input = args.val_pt + ".pt" if not args.val_pt.endswith(".pt") else args.val_pt
        validation_dataset = torch.load(fixed_input)[args.start_index:args.start_index+args.num_samples]
    else:
        training_dataset, validation_dataset = load_val_pile(number=2*args.num_samples,start_index=args.start_index,seed=args.seed,num_splits=2,window=2048 if args.pack else 0)

    end = time.perf_counter()
    logger.info(f"- Dataset loading took {end-start} seconds.")
    ####################################################################################################
    # FINE-TUNE MODEL
    ####################################################################################################
    start = time.perf_counter()
    training_args = TrainingArguments(
        output_dir=args.experiment_name,
        do_train=True,
        do_eval=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        load_best_model_at_end=False,
        fp16=False,
        deepspeed='ds_config_zero3.json' if args.accelerate else None
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name,padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name,revision=args.model_revision,cache_dir=args.model_cache_dir).to(device)
    max_length = AutoConfig.from_pretrained(args.model_name).max_position_embeddings
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length),
    )
    trainer.train()
    trainer.save_model(args.experiment_name)
    tokenizer.save_pretrained(args.experiment_name)

    end = time.perf_counter()
    logger.info(f"- Fine-tuning took {end-start} seconds.")

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"Total Elapsed Time: {end_time-start_time} seconds")