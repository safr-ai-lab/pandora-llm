import os
import time
import math
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments, Trainer
from llmprivacy.utils.attack_utils import *
from llmprivacy.utils.dataset_utils import *
from llmprivacy.utils.log_utils import get_my_logger
from llmprivacy.attacks.ZLIBLoRa import ZLIBLoRa
from accelerate import Accelerator
from accelerate.utils import set_seed
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Sample command line prompt (no acceleration)
python run_zliblora.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --n_samples 1000 --pack --seed 229
Sample command line prompt (with acceleration)
accelerate launch run_zliblora.py --accelerate --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --n_samples 1000 --pack --seed 229
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
    parser.add_argument('--min_length', action="store", type=int, required=False, default=20, help='Min number of tokens (filters)')
    parser.add_argument('--max_length', action="store", type=int, required=False, help='Max number of tokens (truncates)')
    parser.add_argument('--pack', action="store_true", required=False, help='Pack validation set')
    parser.add_argument('--train_pt', action="store", required=False, help='.pt file of train dataset (not dataloader)')
    parser.add_argument('--val_pt', action="store", required=False, help='.pt file of val dataset (not dataloader)')
    # Device Arguments
    parser.add_argument('--seed', action="store", type=int, required=False, default=229, help='Seed')
    parser.add_argument('--accelerate', action="store_true", required=False, help='Use accelerate')
    parser.add_argument('--model_half', action="store_true", required=False, help='Use half precision (fp16). 1 for use; 0 for not.')
    args = parser.parse_args()
    
    accelerator = Accelerator() if args.accelerate else None
    set_seed(args.seed)

    os.makedirs("results/ZLIBLoRa", exist_ok=True)
    args.model_cache_dir = args.model_cache_dir if args.model_cache_dir is not None else f"models/{args.model_name.replace('/','-')}"
    args.experiment_name = args.experiment_name if args.experiment_name is not None else (
        (f"results/ZLIBLoRa/ZLIBLoRa_{args.model_name.replace('/','-')}") +
        (f"_{args.model_revision.replace('/','-')}" if args.model_revision is not None else "") +
        (f"_N={args.num_samples}_S={args.start_index}_seed={args.seed}") +
        (f"_tag={args.tag}" if args.tag is not None else "")
    )
    logger = get_my_logger(log_file=f"{args.experiment_name}.log")
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
        training_dataloader = DataLoader(training_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))
        
        fixed_input = args.val_pt + ".pt" if not args.val_pt.endswith(".pt") else args.val_pt
        validation_dataset = torch.load(fixed_input)[args.start_index:args.start_index+args.num_samples]
        validation_dataloader = DataLoader(validation_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))
    else:
        training_dataset, validation_dataset = load_val_pile(number=2*args.num_samples, seed=args.seed, num_splits=2, window=2048 if args.pack else 0)
        training_dataloader = DataLoader(training_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))
        validation_dataloader = DataLoader(validation_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))

    end = time.perf_counter()
    logger.info(f"- Dataset loading took {end-start} seconds.")
    ####################################################################################################
    # FINE-TUNE MODEL
    ####################################################################################################
    start = time.perf_counter()
    training_args = TrainingArguments(output_dir=f"models/ZLIBLoRa/{args.model_name.replace('/','-')}-ft",
                                      do_train=True,
                                      do_eval=True,
                                      num_train_epochs=1,
                                      per_device_train_batch_size=args.bs,
                                      per_device_eval_batch_size=args.bs,
                                      evaluation_strategy="epoch",
                                      logging_strategy="epoch",
                                      save_strategy="epoch",
                                      gradient_accumulation_steps=1,
                                      gradient_checkpointing=False,
                                      load_best_model_at_end=True,
                                      fp16=False,
                                      deepspeed='ds_config_zero3.json' if args.accelerate else None
                                      )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not args.accelerate:
        model = AutoModelForCausalLM.from_pretrained(args.model_name,revision=args.model_revision,cache_dir=args.model_cache_dir).to(device)
        max_length = AutoConfig.from_pretrained(args.model_name).max_position_embeddings
        trainer = Trainer(model=model,
                    args=training_args,
                    train_dataset=training_dataset,
                    eval_dataset=validation_dataset,
                    tokenizer=tokenizer,
                    data_collator=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length),
                )
        trainer.train()
        trainer.save_model(f"models/ZLIBLoRa/{args.model_name.replace('/','-')}-ft")
    else:
        torch.save(training_dataset,"train_data.pt")
        torch.save(validation_dataset,"val_data.pt")
        torch.save(training_args,"models/ZLIBLoRa/train_args.pt")
        subprocess.call(["accelerate", "launch", "-m", "src.scripts.model_train",
            "--args_path", "models/ZLIBLoRa/train_args.pt",
            "--save_path", f"models/ZLIBLoRa/{args.model_name.replace('/','-')}-ft",
            "--model_path", args.model_name,
            "--model_revision", args.model_revision,
            "--model_cache_dir", args.model_cache_dir,
            "--train_pt", args.train_pt if args.train_pt is not None else "train_data.pt",
            "--val_pt",  args.val_pt if args.val_pt is not None else "train_data.pt",
            "--accelerate"
            ]
        )
    end = time.perf_counter()
    logger.info(f"- Fine-tuning took {end-start} seconds.")
    ####################################################################################################
    # RUN ATTACK
    ####################################################################################################
    start = time.perf_counter()
    logger.info("Running Attack")

    # Initialize attack
    ZLIBLoRaer = ZLIBLoRa(f"models/ZLIBLoRa/{args.model_name.replace('/','-')}-ft")
    
    # Compute statistics
    ZLIBLoRaer.load_model()
    train_statistics = ZLIBLoRaer.compute_statistic(training_dataloader,num_samples=args.num_samples,device=device,model_half=args.model_half,accelerator=accelerator)
    torch.save(train_statistics,f"{args.experiment_name}_train.pt")    
    val_statistics = ZLIBLoRaer.compute_statistic(validation_dataloader,num_samples=args.num_samples,device=device,model_half=args.model_half,accelerator=accelerator)
    torch.save(val_statistics,f"{args.experiment_name}_val.pt")
    ZLIBLoRaer.unload_model()

    # Plot ROCs
    ZLIBLoRaer.attack_plot_ROC(train_statistics, val_statistics, title=args.experiment_name, log_scale=False, show_plot=False)
    ZLIBLoRaer.attack_plot_ROC(train_statistics, val_statistics, title=args.experiment_name, log_scale=True, show_plot=False)

    end = time.perf_counter()

    logger.info(f"- Experiment {args.experiment_name} took {end-start} seconds.")

if __name__ == "__main__":
    main()