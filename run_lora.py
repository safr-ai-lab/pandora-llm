import torch
from torch.utils.data import DataLoader
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments, Trainer
from attack_utils import *
from dataset_utils import *
from LoRa import LoRa
import time
import argparse

"""
Sample command line prompt (no acceleration)
python run_lora.py --mod_size 70m --n_samples 1000
Sample command line prompt (with acceleration)
python run_lora.py --mod_size 70m --n_samples 1000 --accelerate
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mod_size', action="store", type=str, required=True, help='Pythia Model Size')
    parser.add_argument('--checkpoint', action="store", type=str, required=False, help='Model revision. If not specified, use last checkpoint.')
    parser.add_argument('--deduped', action="store_true", required=False, help='Use deduped models')
    parser.add_argument('--pack', action="store_true", required=False, help='Pack validation set')
    parser.add_argument('--bs', action="store", type=int, required=False, default=1, help='Batch size')
    parser.add_argument('--seed', action="store", type=int, required=False, default=229, help='Seed')
    parser.add_argument('--n_samples', action="store", type=int, required=True, help='Dataset size')
    parser.add_argument('--sample_length', action="store", type=int, required=False, help='Truncate number of tokens')
    parser.add_argument('--accelerate', action="store_true", required=False, help='Use accelerate')
    parser.add_argument('--model_half', action="store_true", required=False, help='Use half precision (fp16). 1 for use; 0 for not.')
    args = parser.parse_args()

    if args.model_half and args.accelerate:
        print("WARNING: training in half precision is not supported yet!!!")

    ## Other parameters
    model_revision = args.checkpoint
    seed = args.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_title = f"pythia-{args.mod_size}" + ("-deduped" if args.deduped else "")
    model_name = "EleutherAI/" + model_title
    model_cache_dir = "./"+ model_title + ("/"+model_revision if args.checkpoint else "")

    ## Load model and training and validation dataset
    start = time.perf_counter()

    model = GPTNeoXForCausalLM.from_pretrained(model_name,revision=model_revision,cache_dir=model_cache_dir)
    max_length = AutoConfig.from_pretrained(model_name).max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading Data")
    if args.pack:
        dataset = load_val_pile_packed(number=args.n_samples, seed=seed, num_splits=2)
    else:
        dataset = load_val_pile(number=args.n_samples, seed=seed, num_splits=2)
    training_dataset, validation_dataset = dataset[0], dataset[1]

    if args.accelerate:
        torch.save(training_dataset,"train_data.pt")
        torch.save(validation_dataset,"val_data.pt")

    training_dataloader = DataLoader(training_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, length=max_length))
    validation_dataloader = DataLoader(validation_dataset, batch_size = args.bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, length=max_length))

    ## Run LoRa attack

    training_args = TrainingArguments(output_dir=f"LoRa/{model_title}-ft",
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
    trainer = Trainer(model=model,
                    args=training_args,
                    train_dataset=training_dataset,
                    eval_dataset=validation_dataset,
                    tokenizer=tokenizer,
                    data_collator=lambda batch: collate_fn(batch, tokenizer=tokenizer, length=max_length),
                    )

    config_lora = {
        "trainer": trainer,
        "training_dl": training_dataloader,
        "validation_dl": validation_dataloader,
        "tokenizer": tokenizer,
        "device": device,
        "n_batches": args.n_samples,
        "n_samples": args.sample_length,
        "bs": args.bs,
        "accelerate": args.accelerate,
        "model_half": args.model_half
    }

    ## Stopwatch for testing MoPe runtime
    end = time.perf_counter()
    print(f"- Code initialization time was {end-start} seconds.")

    start = time.perf_counter()

    LoRaer = LoRa(model_name, model_revision=model_revision, cache_dir=model_cache_dir)
    LoRaer.inference(config_lora)

    LoRaer.attack_plot_ROC(log_scale = False, show_plot=False)
    LoRaer.attack_plot_ROC(log_scale = True, show_plot=False)
    LoRaer.save()

    end = time.perf_counter()
    print(f"- LoRa at {args.mod_size} took {end-start} seconds.")

if __name__ == "__main__":
    main()