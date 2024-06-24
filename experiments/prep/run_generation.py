import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments, Trainer
from src.utils.attack_utils import *
from src.utils.dataset_utils import *
from src.utils.generation_utils import *
from src.attacks.LOSS import LOSS
from src.attacks.MoPe import MoPe
from src.attacks.LoRa import LoRa
import time
import argparse
from accelerate import Accelerator
from accelerate.utils import set_seed
import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import TrainerCallback
import warnings
from itertools import chain

"""
Sample Commands. There are two ways of submission, interactive (you can see outputs in front of you) and batch (submit job, it disappears into ether until done).

No DeepSpeed, Interactive
bsub -q gpu -gpu "num=1:aff=yes:mode=exclusive_process:mps=no" -R "rusage[mem=30000]" -Is -M 30G python run_generation.py --prefixes prefixes_pileval_train.npy --suffixes suffixes_pileval_train.npy --mod_size 1b --checkpoint step98000 --deduped --n_samples 25 --suffix_length 50 --bs 1 --attack LoRa --train_pt pileval_train_1000_dl.pt --val_pt pileval_val_1000_dl.pt --n_iterations 500 --top_k 24 --top_p 0.8 --typical_p 0.9 --temperature 0.58 --repetition_penalty 1.04 --n_train_epochs 3 --ft_bs 1 

DeepSpeed, Batch - 1 GPU - make sure config is adjusted for this!
bsub -q gpu -gpu "num=1:aff=yes:mode=exclusive_process:mps=no" -W 14:00 -M 55000 -hl accelerate launch run_generation.py --prefixes prefixes_pileval_train.npy --suffixes suffixes_pileval_train.npy --mod_size 2.8b --checkpoint step98000 --deduped --n_samples 25 --suffix_length 50 --bs 1 --attack LoRa --train_pt pileval_train_1000_dl.pt --val_pt pileval_val_1000_dl.pt --n_iterations 10 --top_k 24 --top_p 0.8 --typical_p 0.9 --temperature 0.58 --repetition_penalty 1.04 --n_train_epochs 3 --ft_bs 1 --accelerate 

2 GPUs - make sure config is adjusted for this!
bsub -q gpu -gpu "num=2:aff=yes:mode=exclusive_process:mps=no" -W 14:00 -M 55000 -hl accelerate launch run_generation.py --prefixes prefixes_pileval_train.npy --suffixes suffixes_pileval_train.npy --mod_size 2.8b --checkpoint step98000 --deduped --n_samples 25 --suffix_length 50 --bs 1 --attack LoRa --train_pt pileval_train_1000_dl.pt --val_pt pileval_val_1000_dl.pt --n_iterations 10 --top_k 24 --top_p 0.8 --typical_p 0.9 --temperature 0.58 --repetition_penalty 1.04 --n_train_epochs 3 --ft_bs 1 --accelerate 

Using existing model (--load_model + --do_gen) and running only generations, No DS, Batch
bsub -q gpu -gpu "num=1:aff=yes:mode=exclusive_process:mps=no" -W 14:00 -M 30000 -hl python run_generation.py --prefixes prefixes_pileval_train.npy --suffixes suffixes_pileval_train.npy --mod_size 1b --checkpoint step98000 --deduped --n_samples 100 --suffix_length 50 --bs 1 --attack LoRa --train_pt pileval_train_1000_dl.pt --val_pt pileval_val_1000_dl.pt --n_iterations 100 --top_k 24 --top_p 0.8 --typical_p 0.9 --temperature 0.9 --repetition_penalty 1.04 --ft_bs 1 --load_model /export/projects4/sneel_pandora_llmteam/1b_pileval_500_samp_25/FineTune/pythia-1b-deduped-ft/checkpoint-2000 --do_gen 

Using existing model + training another epoch + running results, No DS, Batch
bsub -q gpu -gpu "num=1:aff=yes:mode=exclusive_process:mps=no" -R "rusage[mem=30000]" -W 24:00 -M 30000 -hl python run_generation.py --prefixes prefixes_pm_train_last_500.npy --suffixes suffixes_pm_train_last_500.npy --mod_size 1b --checkpoint step98000 --deduped --n_samples 25 --suffix_length 50 --bs 1 --attack LoRa --n_iterations 500 --top_k 24 --top_p 0.8 --typical_p 0.9 --temperature 0.58 --repetition_penalty 1.04 --n_train_epochs 1 --ft_bs 1 --load_model /export/projects4/sneel_pandora_llmteam/1b_Generation_Testing/FineTune/pythia-1b-deduped-ft/checkpoint-1000 --train_pt pubmed_train_1000_dl.pt --val_pt pubmed_val_1000_dl.pt


Sample command line prompts:

LOSS
python run_generation.py \
--prefixes data/ckpt_pre_suf/prefixes_ckpt_step89999.npy \
--suffixes data/ckpt_pre_suf/suffixes_ckpt_step89999.npy \
--mod_size 70m --checkpoint step90000 --deduped \
--n_samples 10 --suffix_length 50 --bs 1 --attack LOSS \
--n_iterations 3 --top_k 24 --top_p 0.8 --typical_p 0.9 --temperature 0.58 --repetition_penalty 1.04 \
--accelerate

MoPe
python run_generation.py \
--prefixes data/ckpt_pre_suf/prefixes_ckpt_step89999.npy \
--suffixes data/ckpt_pre_suf/suffixes_ckpt_step89999.npy \
--mod_size 70m --checkpoint step90000 --deduped \
--n_samples 10 --suffix_length 50 --bs 1 --attack MoPe \
--n_models 2 --noise_type 1 --sigma 1e-3 \
--n_iterations 3 --top_k 24 --top_p 0.8 --typical_p 0.9 --temperature 0.58 --repetition_penalty 1.04 \
--accelerate

LoRa Checkpoint
python run_generation.py \
--prefixes data/ckpt_pre_suf/prefixes_ckpt_step89999.npy \
--suffixes data/ckpt_pre_suf/suffixes_ckpt_step89999.npy \
--mod_size 70m --checkpoint step90000 --deduped \
--n_samples 10 --suffix_length 50 --bs 1 --attack LoRa \
--base_ckpt step89000 --ft_ckpt step90000 \
--n_iterations 3 --top_k 24 --top_p 0.8 --typical_p 0.9 --temperature 0.58 --repetition_penalty 1.04 \
--accelerate

LoRa FT
python run_generation.py \
--prefixes data/pile_val_ft/prefixes_pileval_train.npy \
--suffixes data/pile_val_ft/suffixes_pileval_train.npy \
--mod_size 70m --checkpoint step90000 --deduped \
--n_samples 10 --suffix_length 50 --bs 1 --attack LoRa \
--train_pt data/pile_val_ft/pileval_train_1000_dl.pt --val_pt data/pile_val_ft/pileval_val_1000_dl.pt \
--n_train_epochs 3 --ft_bs 1 \
--n_iterations 3 --top_k 24 --top_p 0.8 --typical_p 0.9 --temperature 0.58 --repetition_penalty 1.04 \
--accelerate

LoRa
python run_generation.py \
--prefixes data/pile_val_ft/prefixes_pileval_train.npy \
--suffixes data/pile_val_ft/suffixes_pileval_train.npy \
--mod_size 1b --checkpoint step98000 --deduped \
--n_samples 100 --suffix_length 50 --bs 1 --attack LoRa \
--n_iterations 3 --p_threshold 0.05 --top_k 24 --top_p 0.8 --typical_p 0.9 --temperature 0.58 --repetition_penalty 1.04 \
--compute_actual_prob --do_generation --load_model /export/projects4/sneel_pandora_llmteam/OldExtractionCode/1b_4by4_experiments/pileval_1000_4/FineTune/pythia-1b-deduped-ft/checkpoint-4000/ \
--accelerate
"""

def compute_prob_do_gen_run_results(
        args, 
        model_name, 
        model_revision, 
        model_cache_dir, 
        prefixes, 
        suffixes, 
        tokenizer, 
        device, 
        attack, 
        kwargs, 
        model,
        gen_title=""):
    """
    - all arguments the same as the respective named in main()
    - generate() returns "losses" and "base_losses". Originally these were the MoPe and LOSS statistics, but more generally
    losses are what we rank over, and base_losses are something we use as a comparison. Handily, we can then use base_losses with results()!
    - LOSS returns loss/loss, MoPe returns MoPe/loss, and LoRa returns LoRa/ft-loss. 
    """
    # First compute probabilities if specified to
    probabilities = None
    if args.compute_actual_prob:
        probabilities = np.array(compute_actual_prob(prefixes, suffixes, args, model, tokenizer, device, title=f"{gen_title}_probs", kx_tup=(args.k, args.x)))

    if args.do_generation:
        # Then generate as long as probability is > p*
        if probabilities is not None:
            prefixes = prefixes[probabilities>args.p_threshold]
            suffixes = suffixes[probabilities>args.p_threshold]
        generation_config = GenerationConfig()
        generation_config.update(**{
            "min_length":100,
            "max_length":100,
            "do_sample":True,
            "pad_token_id":tokenizer.pad_token_id,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "typical_p": args.typical_p,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty,
        })

        if not args.accelerate:
            if args.attack=="LoRa":
                model = attack.get_ft_model().to(device)
            else:
                model = attack.get_model().to(device)
            generations = generate_suffixes(
                model=model,
                prefixes=DataLoader(prefixes,batch_size=args.bs),
                generation_config=generation_config,
                trials=args.n_iterations,
                accelerate=False
            )
            generations = generations.reshape(prefixes.shape[0],-1,generations.shape[-1])
            generations = np.concatenate((generations,np.concatenate((prefixes,suffixes),axis=1)[:,None,:]),axis=1)
            generations = generations.reshape(-1,generations.shape[-1])
            del model
            losses = attack.compute_statistic(
                dataloader=DataLoader(torch.tensor(generations,dtype=torch.int64),batch_size=args.bs),
                device=device,
                accelerator=False
            )
            generations = generations.reshape(prefixes.shape[0],-1,generations.shape[-1])
            losses = losses.reshape(prefixes.shape[0],-1).numpy()
            with open('Generation/generations.npy', 'wb') as f:
                np.save(f,generations)
            with open('Generation/losses.npy', 'wb') as f:
                np.save(f,losses)
        else:
            torch.save(generation_config,"Generation/config.pt")
            subprocess.call(["accelerate", "launch", "-m", "src.scripts.model_generation",
                "--model_path", model_name,
                "--model_revision", model_revision,
                "--cache_dir", model_cache_dir,
                "--prefixes", args.prefixes,
                "--n_samples", str(args.n_samples),
                "--bs", str(args.bs),
                "--gen_config", "Generation/config.pt",
                "--num_trials", str(args.n_iterations),
                "--save_path", 'Generation/generations.pt',
                "--accelerate",
                ]
            )
            generations = torch.load("Generation/generations.pt")
            generations = torch.cat((generations,torch.tensor(suffixes)),dim=1)
            torch.save(generations,"Generation/generations.pt")
            subprocess.call((["accelerate", "launch"] if args.attack=="LOSS" else ["python"])+["-m", "src.scripts.model_attack",
                "--model_path", model_name,
                "--model_revision", model_revision,
                "--cache_dir", model_cache_dir,
                "--attack", args.attack,
                "--dataset_path", "Generation/generations.pt",
                "--bs", str(args.bs),
                "--save_path", 'Generation/losses.pt',
                "--accelerate",
                ]+list(chain.from_iterable([[f"--{key}",f"{value}"] for key,value in kwargs.items()]))
            )
            generations = torch.load("Generation/generations.pt")
            losses = torch.load("Generation/losses.pt")
            generations = generations.reshape(prefixes.shape[0],-1,generations.shape[-1])
            losses = losses.reshape(prefixes.shape[0],-1).numpy()
            with open('Generation/generations.npy', 'wb') as f:
                np.save(f,generations)
            with open('Generation/losses.npy', 'wb') as f:
                np.save(f,losses)

    # Run correct type of generation
    if args.attack == "LoRa" or args.attack == "ft-loss":
        results(generations,losses,losses,prefixes,suffixes,args.n_samples,args.n_iterations,args.suffix_length,tokenizer, "LoRa", "ft-loss", title=gen_title if gen_title else "LoRa", probabilities=probabilities)
    elif args.attack == "MoPe":
        results(generations,losses,losses,prefixes,suffixes,args.n_samples,args.n_iterations,args.suffix_length,tokenizer, "MoPe", "LOSS", title=gen_title if gen_title else "MoPe", probabilities=probabilities)
    elif args.attack == "LOSS":
        results(generations,losses,losses,prefixes,suffixes,args.n_samples,args.n_iterations,args.suffix_length,tokenizer, "LOSS", "LOSS", title=gen_title if gen_title else "LOSS", probabilities=probabilities)


def main():
    parser = argparse.ArgumentParser()

    # Default arguments
    parser.add_argument('--prefixes', action="store", required=True, help='.npy file of prefixes')
    parser.add_argument('--suffixes', action="store", required=True, help='.npy file of suffixes')
    parser.add_argument('--mod_size', action="store", type=str, required=True, help='Pythia Model Size')
    parser.add_argument('--checkpoint', action="store", type=str, required=False, help='Model revision. If not specified, use last checkpoint.')
    parser.add_argument('--deduped', action="store_true", required=False, help='Use deduped models. Otherwise, do not.')
    parser.add_argument('--n_samples', action="store", type=int, required=True, help='Number of prefixes to run generations on. Takes last n_samples from prefixes npy file.')
    parser.add_argument('--n_iterations', action="store", type=int, required=False, help='How many generations per prompt')
    parser.add_argument('--suffix_length', action="store", type=int, required=False, default=50, help='Length of suffix. Default 50.')
    parser.add_argument('--attack', action="store", type=str, required=True, help='Attack type (LOSS or MoPe)')
    parser.add_argument('--write', action='store', required=False, help='Create logging files for data. Usually not needed.')
    parser.add_argument('--accelerate', action="store_true", required=False, help='Use accelerate')
    parser.add_argument('--seed', action="store", type=int, required=False, default=229, help='Seed')
    # parser.add_argument('--model_half', action="store_true", required=False, help='Use half precision (fp16). 1 for use; 0 for not.') # TODO - this argument isn't used right now.
    parser.add_argument('--bs', action="store", type=int, required=False, default=1, help='Batch size of generation.')
    parser.add_argument('--suppress_warnings', action="store_true", required=False, help="Suppress warnings when flag is activated. Not recommended for normal use, but possibly useful in terminal tmux sessions in case buffer runs out.")
    parser.add_argument('--top_k', action="store", type=int, default=10, required=False, help='Top k sampling for generation (number of tokens to choose from)')
    parser.add_argument('--top_p', action="store", type=float, default=1.0, required=False, help='Top p / nucleus eta sampling for generation (choose tokens until probability adds up to p)')
    parser.add_argument('--typical_p', action="store", type=float, default=1.0, required=False, help='Typical p / phi sampling for generation (choose locally typical tokens until probability adds up to p)')
    parser.add_argument('--temperature', action="store", type=float, default=1.0, required=False, help='Higher temperature, more diversity - the value used to modulate the next token probabilities.')
    parser.add_argument('--repetition_penalty', action="store", type=float, default=1.0, required=False, help='The parameter for repetition penalty. 1.0 means no penalty.')
    parser.add_argument('--compute_actual_prob', action="store_true", required=False, help='Whether to compute the probability')
    parser.add_argument('--do_generation', action="store_true", required=False, help='Whether to compute the probability')
    parser.add_argument('--p_threshold', action="store", type=float, help='Probability threshold to attempt generation from')

    # MoPe specific arguments 
    parser.add_argument('--n_models', action="store", type=int, required=False, help='Number of new models')
    parser.add_argument('--sigma', action="store", type=float, required=False, help='Noise standard deviation')
    parser.add_argument('--noise_type', action="store", type=int, default=1, required=False, help='Noise to add to model. Options: 1 = Gaussian, 2 = Rademacher, 3+ = user-specified (see README). If not specified, use Gaussian noise.')

    # Fine-tuning with new data ("FT" attack)
    parser.add_argument('--statistic', action="store", type=str, required=False, help="membership statistic to rank generations. LoRa = LoRa and ft-loss")
    parser.add_argument('--train_pt', action="store", type=str, required=False, help='pt file of train string array (for fine-tuning attacks)')
    parser.add_argument('--val_pt', action="store", type=str, required=False, help='pt file of val string array (for fine-tuning attacks)')
    parser.add_argument('--n_train_epochs', action="store", type=int, required=False, help='Num train epochs for fine-tuning.')
    parser.add_argument('--ft_bs', action="store", type=int, required=False, help='batch size when fine-tuning the model')
    parser.add_argument('--do_gen_every_X', action="store", type=int, required=False, help='If fine tuning for more than 1 epoch, run generation after every X epochs, where n_train_epochs (mod X) must equal 0.')
    # parser.add_argument('--load_models', action="store", type=str, required=False, help='If running do_gen_every_X retrospectively, then load in model directory here (e.g. FineTune/pythia-70m-ft) with fted model copies.')
    parser.add_argument('--load_model', action="store", type=str, required=False, help='Pre-load (a possibly already ft-ed) model, either to fine-tune on top of it or just run generation (do_gen).')
    parser.add_argument('--k', action="store", type=int, default=50, required=False, help="For generation, do K-length prefix and X-length suffix where K+X <= 100.")
    parser.add_argument('--x', action="store", type=int, default=50, required=False, help="For generation, do K-length prefix and X-length suffix where K+X <= 100.")

    # Using checkpoints as FTing 
    parser.add_argument('--base_ckpt', action="store", type=str, required=False, help='Checkpoint (e.g. step98000) of base model.')
    parser.add_argument('--ft_ckpt', action="store", type=str, required=False, help='Checkpoint (e.g. step98000) of quote-unquote fine-tuned model.')
    
    args = parser.parse_args()

    ## Other parameters
    seed = args.seed
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_revision = args.checkpoint
    model_title = f"pythia-{args.mod_size}" + ("-deduped" if args.deduped else "")
    model_name = "EleutherAI/" + model_title
    model_cache_dir = "./"+ model_title + ("/"+model_revision if args.checkpoint else "")

    ## Load model and training and validation dataset
    prefixes = torch.tensor(np.load(args.prefixes).astype(np.int64)[:args.n_samples],dtype=torch.int64)
    suffixes = torch.tensor(np.load(args.suffixes).astype(np.int64)[:args.n_samples],dtype=torch.int64)

    tokenizer = AutoTokenizer.from_pretrained(model_name) # e.g. "EleutherAI/pythia-70m-deduped"
    tokenizer.pad_token = tokenizer.eos_token # not set by default

    kwargs = {}
    if args.attack=="LOSS":
        attack = LOSS(model_name, model_revision=model_revision, cache_dir=model_cache_dir)
    elif args.attack=="MoPe":
        attack = MoPe(model_name, model_revision=model_revision, cache_dir=model_cache_dir)
        kwargs = {
            "n_new_models": args.n_models,
            "noise_stdev": args.sigma,
            "noise_type": args.noise_type
        }
    elif args.attack=="LoRa":
        if args.base_ckpt and args.ft_ckpt: # checkpoint setting
            attack = LoRa(
                model_path=model_name,
                model_revision=args.base_ckpt,
                cache_dir=model_cache_dir,
                ft_model_path=model_name,
                ft_model_revision=args.ft_ckpt,
                ft_cache_dir=model_cache_dir
            )
        elif args.load_model: # preload a ft model to do generations on
            attack = LoRa(
                model_path=model_name,
                model_revision=model_revision,
                cache_dir=model_cache_dir,
                ft_model_path=args.load_model,    
            )
            model = AutoModelForCausalLM.from_pretrained(model_name,revision=model_revision,cache_dir=model_cache_dir).to(device)
        else: # fine-tune a model (possibly pre-loaded)
            # Fine-Tune Model
            training_args = TrainingArguments(output_dir=f"FineTune/{model_title}-ft",
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
            if not args.accelerate:
                # Load Data
                print("Loading Data")
                print("You are using a self-specified training dataset...")
                fixed_input = args.train_pt + ".pt" if not args.train_pt.endswith(".pt") else args.train_pt
                training_dataset = torch.load(fixed_input)
                print("You are using a self-specified validation dataset...")
                fixed_input = args.val_pt + ".pt" if not args.val_pt.endswith(".pt") else args.val_pt
                validation_dataset = torch.load(fixed_input)

                model = AutoModelForCausalLM.from_pretrained(model_name,revision=model_revision,cache_dir=model_cache_dir).to(device)
                max_length = AutoConfig.from_pretrained(model_name).max_position_embeddings
                trainer = Trainer(model=model,
                            args=training_args,
                            train_dataset=training_dataset,
                            eval_dataset=validation_dataset,
                            tokenizer=tokenizer,
                            data_collator=lambda batch: collate_fn(batch, tokenizer=tokenizer, length=max_length),
                        )
                trainer.train()
                trainer.save_model(f"FineTune/{model_title}-ft")
            else:
                torch.save(training_args,"FineTune/train_args.pt")
                subprocess.call(["accelerate", "launch", "-m", "src.scripts.model_train",
                    "--args_path", "FineTune/train_args.pt",
                    "--save_path", f"FineTune/{model_title}-ft",
                    "--model_path", model_name,
                    "--model_revision", model_revision,
                    "--cache_dir", model_cache_dir,
                    "--train_pt", args.train_pt,
                    "--val_pt", args.val_pt,
                    "--accelerate"
                    ]
                )
            attack = LoRa(
                model_path=model_name,
                model_revision=model_revision,
                cache_dir=model_cache_dir,
                ft_model_path=f"FineTune/{model_title}-ft",    
            )
        kwargs = {
            "model_path": attack.model_path,
            "model_revision": attack.model_revision,
            "cache_dir": attack.cache_dir,
            "ft_model_path": attack.ft_model_path,
            "ft_model_revision": attack.ft_model_revision,
            "ft_cache_dir": attack.ft_cache_dir,
        }

    if args.do_gen_every_X:
        checkpoints = os.listdir(f"FineTune/{model_title}-ft")
        checkpoints = sorted([int(a.split("-")[1]) for a in checkpoints if "checkpoint-" in a]) # get checkpoints from epochs
        for epochno in range(args.do_gen_every_X-1, args.n_train_epochs, args.do_gen_every_X):
            print(f"Computing probabilities and running generation/attack for epoch {epochno+1}.")
            compute_prob_do_gen_run_results(args, model_name, model_revision, model_cache_dir, prefixes, suffixes, tokenizer, device, attack, kwargs, model, gen_title=f"{attack.model_name}-{epochno+1}")
    else: # only run generation at end
        compute_prob_do_gen_run_results(args, model_name, model_revision, model_cache_dir, prefixes, suffixes, tokenizer, device, attack, kwargs, model, gen_title=f"{attack.model_name}")

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"Total Elapsed Time: {end_time-start_time} seconds")