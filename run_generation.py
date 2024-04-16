import sys
sys.path.append("./attacks/")
sys.path.append("./utils/")

import torch
from transformers import AutoModelForCausalLM, GPTNeoXForCausalLM, AutoTokenizer, AutoConfig
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments, Trainer
from attack_utils import *
from dataset_utils import *
from LOSS import LOSS
from MoPe import MoPe
from LoRa import LoRa
import time
import argparse
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings
from tqdm import tqdm

"""
Sample Command. 

python run_generation.py --prefixes prefixes_pileval_train.npy --suffixes suffixes_pileval_train.npy --mod_size 1b --checkpoint step98000 --deduped --n_samples 25 --suffix_length 50 --bs 1 --attack LoRa --train_pt pileval_train_1000_dl.pt --val_pt pileval_val_1000_dl.pt --n_iterations 500 --top_k 24 --top_p 0.8 --typical_p 0.9 --temperature 0.58 --repetition_penalty 1.04 --n_train_epochs 4 --ft_bs 1 
"""

def lin_combo(generations,losses,base_losses,suffixes,n_samples,suffix_length,num_lambdas=1000):
    """
    """
    axis0 = np.arange(generations.shape[0])
    print(losses.shape)
    losses = (losses-np.mean(losses))/np.std(losses)
    base_losses= (base_losses-np.mean(base_losses))/np.std(base_losses)
    best_lambda = -1
    best_precision = -1
    for lambd in np.linspace(0,1,num_lambdas):
        combined = base_losses*lambd+(1-lambd)*losses
        axis1 = combined.argmin(1).reshape(-1)
        best_suffixes = generations[axis0, axis1, -suffix_length:]
        precision = np.sum(np.all(best_suffixes==suffixes, axis=-1)) / n_samples
        if precision>best_precision:
            best_precision = precision
            best_lambda = lambd
    return best_lambda

def results(generations,losses,base_losses,prefixes,suffixes,n_samples,n_iterations,suffix_length,tokenizer,name1,name2,title="",probabilities=None):  
    """
    Method to generate extraction results. 
    - generations is the array of generations created 
    - losses are the PRIMARY membership statistic we rank over (e.g. MoPe stat, LoRa stat)
    - base_losses are another membership statistic. We don't rank by these, but we do compare (e.g. LOSS)
    - prefixes / suffixes are the prefixes/suffixes.
    - n_samples = # samples (prefixes/suffixes) tested, n_iterations = # generations, suffix_length = 50 by default 
    - name1 and name2 are the names of losses and base_losses 
    """ 

    # Metrics obtained by choosing the best suffix per prefix per `losses` statistic
    axis0 = np.arange(generations.shape[0])
    axis1 = losses.argmin(1).reshape(-1)
    axis1_baselosses = base_losses.argmin(1).reshape(-1)
    best_suffixes = generations[axis0, axis1, -suffix_length:]
    precision = np.sum(np.all(best_suffixes==suffixes, axis=-1)) / n_samples
    hamming = np.sum(np.sum(best_suffixes==suffixes, axis=-1) / suffix_length) / n_samples
    print(f"Exact Match Accuracy (Precision): {precision:.4g}")
    print(f"Token Level Accuracy (Hamming): {hamming:.4g}")
    i = 0

    # Print best guesses based on losses stat, with corresponding base_losses stat for each one
    with open(f"Generation/{title}_best_guesses.txt","w") as f:
        for row in range(len(prefixes)):
            prefix = tokenizer.decode(prefixes[row])
            guess = tokenizer.decode(best_suffixes[row])
            suffix = tokenizer.decode(suffixes[row])
            f.write(f"Example {row}\n")
            f.write("Prefix: "+prefix.replace("\n","")+"\n")
            f.write("Suffix: "+suffix.replace("\n","")+"\n")
            f.write("Guess: "+guess.replace("\n","")+"\n")
            f.write(f"{name1}: {losses[row][axis1[row]][0]} | {name2} stat: {base_losses[row][axis1_baselosses[row]][0]}\n\n")
            i += 1
    
    # Now I want to print all guesses with corresponding info
    with open(f"Generation/{title}_all_guesses.txt","w") as f:
        for row in range(len(prefixes)):
            prefix = tokenizer.decode(prefixes[row])
            suffix = tokenizer.decode(suffixes[row])
            f.write(f"Example {row}\n")
            f.write("Prefix: "+prefix.replace("\n","")+"\n")
            f.write("Suffix: "+suffix.replace("\n","")+"\n")
            for j in range(n_iterations):
                genstr = tokenizer.decode(generations[row, j, -suffix_length:])
                f.write(f"\nGuess {j}: "+genstr.replace("\n","")+"\n")
                f.write(f"{name1}: {losses[row][j][0]} | {name2} stat: {base_losses[row][j][0]}\n")
            f.write("\n\n")

    # Write csv with the following format: 
    # samp_num, gen_num, loss, base_loss, prefix_tok, prefix_str, guess_suffix_tok, guess_suffix_str, true_suffix_tok, true_suffix_str, hamming (0 to 1)
    with open(f"Generation/{title}_all_info.tsv", "w") as f:
        f.write(f"samp_num\tgen_num\t{name1}\t{name2}\t\"prefix_tok\"\t\"prefix_str\"\t\"guess_suffix_tok\"\t\"guess_suffix_str\"\t\"true_suffix_tok\"\t\"true_suffix_str\"\thamming0to1\ttrue_suffix_prob\n")
        for samp_num in range(len(prefixes)):
            prefix_tok = list(prefixes[samp_num])
            prefix_str = tokenizer.decode(prefixes[samp_num]).replace("\n", "[newline]").replace("\t", "[tab]").replace("    ", "[tab]")
            true_suffix_tok = list(suffixes[samp_num])
            true_suffix_str = tokenizer.decode(suffixes[samp_num]).replace("\n", "[newline]").replace("\t", "[tab]").replace("    ", "[tab]")
            for gen_num in range(n_iterations):
                guess_suffix_tok = list(generations[samp_num, gen_num, -suffix_length:])
                guess_suffix_str = tokenizer.decode(guess_suffix_tok).replace("\n", "[newline]").replace("\t", "[tab]").replace("    ", "[tab]")
                loss = losses[samp_num][gen_num][0]
                base_loss = base_losses[samp_num][gen_num][0]
                # print(f"Guess Suffix: {guess_suffix_tok}")
                # print(f"True Suffix: {true_suffix_tok}")
                hamming_dist = sum(g == t for g, t in zip(guess_suffix_tok, true_suffix_tok)) / suffix_length
                f.write(f"{samp_num}\t{gen_num}\t{loss}\t{base_loss}\t\"{prefix_tok}\"\t\"{prefix_str}\"\t\"{guess_suffix_tok}\"\t\"{guess_suffix_str}\"\t\"{true_suffix_tok}\"\t\"{true_suffix_str}\"\t{hamming_dist}\t{probabilities[samp_num] if probabilities is not None else np.nan}\n")
    
    # Metrics obtained by looking at all suffixes
    precision_multi = 0
    for i in range(generations.shape[0]):
        if np.sum(np.all(generations[i,:,-suffix_length:] == suffixes[i],axis=-1)):
            precision_multi += 1
    precision_multi = precision_multi/generations.shape[0]
    print(f"Any Suffix Exact Match Accuracy: {precision_multi:.4g}")

    # Metrics obtained by ranking all the suffixes as a whole
    generations, losses, base_losses = generations.reshape(-1,generations.shape[-1]), losses.flatten(), base_losses.flatten()
    prefix_nums = np.concatenate(tuple(np.array(range(n_samples))[:,np.newaxis] for _ in range(n_iterations)),1).flatten()
    order = losses.argsort()

    # Write result for google extraction challenge
    with open(f"Generation/{title}_generation_result.csv","w") as f:
        writer = csv.writer(f)
        writer.writerow(["Example ID", "Suffix Guess", name1, name2, "Correct"])
        for exid, generation, loss, base in zip(prefix_nums[order], generations[order], losses[order], base_losses[order]):
            writer.writerow([exid,str(list(generation[-suffix_length:])).replace(" ", ""),loss,base,np.all(suffixes[exid] == generation[-suffix_length:])])

    # Measure recall at 100 errors
    total = 0
    correct = 0
    recall = []
    errors = []
    bad_guesses = 0
    answer = 0

    for exid, generation in zip(prefix_nums[order],generations[order]):
        total+=1
        if np.all(suffixes[exid] == generation[-suffix_length:]):
            correct+=1
            recall.append(correct/total)
            errors.append(bad_guesses)
            if bad_guesses < 100:
                answer = correct/total
        else:
            bad_guesses += 1

    print("Recall at 100 Errors", answer)
    plt.plot(errors, recall)
    plt.xlabel("Number of bad guesses")
    plt.ylabel("Recall")
    plt.title("Error-Recall Curve")
    plt.grid(which="both",alpha=0.2)
    plt.savefig(f"Generation/{title}_error_curve.png")
    # plt.semilogx()
    # plt.savefig(f"Generation/{title}_error_curve_log.png")

    precision_scores, recall_scores, thresholds = precision_recall_curve([np.all(suffixes[exid] == generation[-suffix_length:]) for exid, generation in zip(prefix_nums[order],generations[order])], -losses[order])
    plt.figure()
    plt.plot(recall_scores,precision_scores)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(which="both",alpha=0.2)
    plt.savefig(f"Generation/{title}_pr_curve.png")
    # plt.semilogy()
    # plt.savefig(f"Generation/{title}_pr_curve_log.png")

    with open(f"Generation/{title}_metrics_wout_AUC.txt","w") as f:
        f.write(f"Precision: {precision:4g}\nHamming: {hamming:4g}\nMultiprecision: {precision_multi:4g}\nRecall@100: {answer:4g}")

    # Plot overall ROC curve
    roc = False
    if roc:
        fpr, tpr, thresholds = roc_curve([np.all(suffixes[exid] == generation[-suffix_length:]) for exid, generation in zip(prefix_nums[order],generations[order])], -losses[order])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.4f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.grid(which="both",alpha=0.2)
        save_name = f"Generation/{title}_generation_roc.png"
        plt.title("Generation Attack ROC")
        plt.legend(loc="lower right")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        print(f"AUC of Experiment {title}\n{roc_auc}")
        plt.savefig(save_name, bbox_inches="tight")
        plt.show()
        save_name = f"Generation/{title}_generation_roc_log.png"
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.4f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.grid(which="both",alpha=0.2)
        plt.xscale("log",base=10,subs=list(range(11)))
        plt.yscale("log",base=10,subs=list(range(11)))
        plt.xlim(9e-4,1.1)
        plt.ylim(9e-4,1.1)
        plt.title("Generation Attack ROC")
        plt.legend(loc="lower right")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig(save_name, bbox_inches="tight")
        plt.show()

        with open(f"Generation/{title}_metrics.txt","w") as f:
            f.write(f"Precision: {precision:4g}\nHamming: {hamming:4g}\nMultiprecision: {precision_multi:4g}\nRecall@100: {answer:4g}\nAUC: {roc_auc}")


    return recall, errors


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
    parser.add_argument('--base_model', action="store", type=str, required=False, help="Path to files for a base model to avoid redownloading or to test attacks in specialized config. Good for LOSS and FT/LoRa attacks; don't use for MoPe.")
    parser.add_argument('--top_k', action="store", type=int, default=10, required=False, help='Top k sampling for generation (number of tokens to choose from)')
    parser.add_argument('--top_p', action="store", type=float, default=1.0, required=False, help='Top p / nucleus eta sampling for generation (choose tokens until probability adds up to p)')
    parser.add_argument('--typical_p', action="store", type=float, default=1.0, required=False, help='Typical p / phi sampling for generation (choose locally typical tokens until probability adds up to p)')
    parser.add_argument('--temperature', action="store", type=float, default=1.0, required=False, help='Higher temperature, more diversity - the value used to modulate the next token probabilities.')
    parser.add_argument('--repetition_penalty', action="store", type=float, default=1.0, required=False, help='The parameter for repetition penalty. 1.0 means no penalty.')
    parser.add_argument('--compute_actual_prob', action="store_true", required=False, help='Whether to compute the probability')

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
    parser.add_argument('--load_model', action="store", type=str, required=False, help='Pre-load (a possibly already ft-ed) model, either to fine-tune on top of it or just run generation (do_gen).')
    parser.add_argument('--do_gen', action="store_true", required=False, help="Do generations with some pre-loaded model (via --load_model) in FT setting. Then run results!")

    # Using checkpoints as FTing 
    parser.add_argument('--base_ckpt', action="store", type=str, required=False, help='Checkpoint (e.g. step98000) of base model.')
    parser.add_argument('--ft_ckpt', action="store", type=str, required=False, help='Checkpoint (e.g. step98000) of quote-unquote fine-tuned model.')
    
    args = parser.parse_args()

    if args.suppress_warnings:
        warnings.filterwarnings("ignore")
    
    relevant_title = f"{args.mod_size}_nsamples={args.n_samples}_niterations={args.n_iterations}"

    ## Other parameters
    seed = args.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_revision = args.checkpoint
    model_title = f"pythia-{args.mod_size}" + ("-deduped" if args.deduped else "")
    model_name = "EleutherAI/" + model_title
    model_cache_dir = "./"+ model_title + ("/"+model_revision if args.checkpoint else "")

    ## Load model and training and validation dataset
    prefixes = np.load(args.prefixes).astype(np.int64)[-args.n_samples:]
    suffixes = np.load(args.suffixes).astype(np.int64)[-args.n_samples:]

    tokenizer = AutoTokenizer.from_pretrained(model_name) # e.g. "EleutherAI/pythia-70m-deduped"
    tokenizer.pad_token = tokenizer.eos_token # not set by default

    # Make `Generation` directory
    directory_path = "Generation"

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists. Using it!")

    if args.attack=="LOSS":
        if args.base_model:
            model_name = args.base_model
        attack = LOSS(model_name)
        attack_config = {
            "suffix_length": args.suffix_length,
            "bs": args.bs,
            "device": device,
            "tokenizer": tokenizer,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "typical_p": args.typical_p,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty,
        }
    elif args.attack=="MoPe":
        attack = MoPe(model_name)
        attack_config = {
            "suffix_length": args.suffix_length,
            "bs": args.bs,
            "device": device,
            "tokenizer": tokenizer,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "typical_p": args.typical_p,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty,
            "n_models": args.n_models,
            "sigma": args.sigma,
            "noise_type": args.noise_type
        }
    elif args.attack in ["LoRa", "lora", "ft-loss"]: 
        attack = LoRa(model_name) # the attack is called LoRa, but it's just as possible to rank by ft-loss since that's computed as part of LoRa
        attack_config = {
            "suffix_length": args.suffix_length,
            "bs": args.bs,
            "device": device,
            "tokenizer": tokenizer,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "typical_p": args.typical_p,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty,
        }

        if args.base_ckpt and args.ft_ckpt: # checkpoint setting
            base_model = GPTNeoXForCausalLM.from_pretrained(model_name,
                                                       revision=args.base_ckpt,
                                                        cache_dir=model_cache_dir).to(device)
            model = GPTNeoXForCausalLM.from_pretrained(model_name,
                                                       revision=args.ft_ckpt,
                                                       cache_dir=model_cache_dir).to(device)
        elif args.load_model: # preload a ft model to do generations on
            if args.base_model:
                base_model = AutoModelForCausalLM.from_pretrained(base_model)
            else:
                base_model = AutoModelForCausalLM.from_pretrained(model_name, revision=model_revision, cache_dir=model_cache_dir).to(device)
            model = GPTNeoXForCausalLM.from_pretrained(args.load_model).to(device)
            ft_name = args.load_model.strip("/").split('/')[-1]
            relevant_title = f"{args.mod_size}_{ft_name}_nsamples={args.n_samples}_niterations={args.n_iterations}"
        else: # fine-tune a model (possibly pre-loaded)
            if args.base_model:
                base_model = AutoModelForCausalLM.from_pretrained(base_model)
            else:
                base_model = AutoModelForCausalLM.from_pretrained(model_name, revision=model_revision, cache_dir=model_cache_dir).to(device)
            if args.load_model:
                model = GPTNeoXForCausalLM.from_pretrained(args.load_model).to(device)
            else:
                model = GPTNeoXForCausalLM.from_pretrained(model_name,revision=model_revision,cache_dir=model_cache_dir)

            # Load Data
            print("Loading Data")
            print("You are using a self-specified training dataset...")
            fixed_input = args.train_pt + ".pt" if not args.train_pt.endswith(".pt") else args.train_pt
            training_dataset = torch.load(fixed_input)
            max_length = AutoConfig.from_pretrained(model_name).max_position_embeddings

            print("You are using a self-specified validation dataset...")
            fixed_input = args.val_pt + ".pt" if not args.val_pt.endswith(".pt") else args.val_pt
            validation_dataset = torch.load(fixed_input)

            # Fine-Tune Model
            training_args = TrainingArguments(output_dir=f"FineTune/{model_title}-ft",
                                                    do_train=True,
                                                    do_eval=True,
                                                    optim="adafactor",
                                                    num_train_epochs=args.n_train_epochs,
                                                    per_device_train_batch_size=args.ft_bs,
                                                    per_device_eval_batch_size=args.ft_bs,
                                                    evaluation_strategy="epoch",
                                                    logging_strategy="epoch",
                                                    save_strategy="epoch", # this means a model is saved every epoch
                                                    gradient_accumulation_steps=2,
                                                    gradient_checkpointing=True,
                                                    load_best_model_at_end=False,
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
            trainer.train()

    # Generations and attack
    # There's a bit of weird code here, so let's unpack it. 
    # generate() returns "losses" and "base_losses". Originally these were the MoPe and LOSS statistics, but more generally
    # losses are what we rank over, and base_losses are something we use as a comparison. Handily, we can then use base_losses with results()!
    # LOSS returns loss/loss, MoPe returns MoPe/loss, and LoRa returns LoRa/ft-loss. 
        
    # First compute probabilities if specified to
    karray = [2, 4, 8]
    xarray = [25, 50]

    probabilities = None
    if args.compute_actual_prob:
        probabilities = []
        generation_config = model.generation_config
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
        for off in tqdm(range(0, len(prefixes), args.bs)):
            prompt_batch = np.concatenate((prefixes[off:off+args.bs],suffixes[off:off+args.bs]),axis=1)
            prompt_batch = np.stack(prompt_batch, axis=0)
            input_ids = torch.tensor(prompt_batch, dtype=torch.int64)
            with torch.no_grad():
                logits = model(input_ids.to(device)).logits.cpu()
                probs = calculate_sentence_probability(logits.cpu(),input_ids.cpu(),condition_from_index=49,generation_config=generation_config,verbose=False)
                probabilities.extend(probs.tolist())
        prob_title = f"Generation/{relevant_title}-probabilities.npy"
        proportion_title = f"Generation/{relevant_title}-proportions.txt"
        # np.save(f"Generation/{prob_title}",np.array(probabilities))
        with open(prob_title, 'wb') as f:
            np.save(f, np.array(probabilities))
        thresholds = [0.001, 0.01, 0.05, 0.1]
        proportions = [len([i for i in probabilities if i > a]) for a in thresholds]
        with open(proportion_title, "w") as file:
            for i in range(len(proportions)):
                file.write(f"{thresholds[i]}: {proportions[i]} / {args.n_samples} = {proportions[i] / args.n_samples} \n")

        print("Theoretical",probabilities)
        plt.xlabel("Probabilities")
        plt.ylabel(f"Number of points (out of {args.n_samples} samples)")
        plt.title(f"Probabilities Histogram")
        plt.hist(probabilities, bins=30, range=(0, 1))
        plt.savefig(f"Generation/{relevant_title}.png")
        plt.clf()

        plt.xlabel("Probabilities")
        plt.ylabel(f"Number of points (out of {args.n_samples} samples)")
        plt.title(f"Log Probabilities Histogram")
        plt.xlim(-0.01, 1.01)
        plt.hist(probabilities, bins=30, range=(0,1))
        plt.yscale('log')
        plt.savefig(f"Generation/{relevant_title}_LOG.png")
        plt.clf()

    if args.do_gen:
        # Jason's prob filter
        #if probabilities is not None:
        #    probabilities = np.array(probabilities)
        #    prefixes = prefixes[(probabilities>0.001) and (probabilities<0.5)]
        #    suffixes = suffixes[(probabilities>0.001) and (probabilities<0.5)]

        if args.attack == "MoPe" or args.attack == "LOSS":
            all_generations, all_losses, all_base_losses = [], [], []
            for trial in range(args.n_iterations):
                print(f"Generation {trial+1}/{args.n_iterations}")
                generations, losses, base_losses = attack.generate(prefixes,suffixes,attack_config)
                all_generations.append(generations)
                all_losses.append(losses)
                all_base_losses.append(base_losses)
            generations = np.stack(all_generations, axis=1)
            losses = np.stack(all_losses, axis=1)
            base_losses = np.stack(all_base_losses, axis=1)

            with open('Generation/generations.npy', 'wb') as f:
                np.save(f,generations)
            with open('Generation/losses.npy', 'wb') as f:
                np.save(f,losses)
            with open('Generation/base_losses.npy', 'wb') as f:
                np.save(f,base_losses)

            try:
                # MoPe
                results(generations,losses,base_losses,prefixes,suffixes,args.n_samples,args.n_iterations,args.suffix_length,tokenizer, "MoPe", "LOSS", title="mope", probabilities=probabilities)
                
                # LOSS
                results(generations,base_losses,base_losses,prefixes,suffixes,args.n_samples,args.n_iterations,args.suffix_length,tokenizer, "LOSS", "LOSS", title="loss", probabilities=probabilities)
            except:
                print("This catching some bug with results() with MoPe. Let's try LOSS...")
                results(generations,base_losses,base_losses,prefixes,suffixes,args.n_samples,args.n_iterations,args.suffix_length,tokenizer, "LOSS", "LOSS", title="loss", probabilities=probabilities)
            
            # Lin Combo Code
            lambd = lin_combo(generations,losses,base_losses,suffixes,args.n_samples,args.suffix_length)
            print(f"Best Lambda: {lambd}")
            losses = (losses-np.mean(losses))/np.std(losses)
            base_losses= (base_losses-np.mean(base_losses))/np.std(base_losses)

            # Run results
            results(generations,losses*(1-lambd)+lambd*base_losses,base_losses,prefixes,suffixes,args.n_samples,args.n_iterations,args.suffix_length,tokenizer,title=f"combo_{lambd}", probabilities=probabilities)
        else:
            all_generations, all_losses, all_base_losses = [], [], [] # generations, lora_stat, loss (base model)
            for trial in range(args.n_iterations):
                print(f"Generation {trial+1}/{args.n_iterations}")
                generations, losses, base_losses = attack.generate(prefixes, suffixes, attack_config, model, base_model)
                all_generations.append(generations)
                all_losses.append(losses)
                all_base_losses.append(base_losses)
            generations = np.stack(all_generations, axis=1)
            losses = np.stack(all_losses, axis=1)
            base_losses = np.stack(all_base_losses, axis=1)

            with open(f'Generation/{relevant_title}_generations.npy', 'wb') as f:
                np.save(f, generations)
            with open(f'Generation/{relevant_title}_losses.npy', 'wb') as f:
                np.save(f, losses)
            with open(f'Generation/{relevant_title}_base_losses.npy', 'wb') as f:
                np.save(f, base_losses)

        try: # this structure allows us to get both the ft-loss and LoRa results from one run! 
            print("LoRa Results:")
            results(generations,losses,base_losses,prefixes,suffixes,args.n_samples,args.n_iterations,args.suffix_length,tokenizer, "LoRa", "ft-loss", title=f"{relevant_title}_lora", probabilities=probabilities)
            print("ft-loss Results:")
            results(generations,base_losses,base_losses,prefixes,suffixes,args.n_samples,args.n_iterations,args.suffix_length,tokenizer, "ft-loss", "ft-loss", title=f"{relevant_title}_ft-loss", probabilities=probabilities)
        except:
            print("Some bug is caught with LoRa results. Now creating ft-loss Results:")
            results(generations,base_losses,base_losses,prefixes,suffixes,args.n_samples,args.n_iterations,args.suffix_length,tokenizer, "ft-loss", "ft-loss", title=f"{relevant_title}_ft-loss", probabilities=probabilities)
    
    # Actual Spaghetti Code
    if args.compute_actual_prob:
        for k in karray:
            for x in xarray:
                probabilities = []
                generation_config = model.generation_config
                generation_config.update(**{
                    "min_length":k+x,
                    "max_length":k+x,
                    "do_sample":True,
                    "pad_token_id":tokenizer.pad_token_id,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "typical_p": args.typical_p,
                    "temperature": args.temperature,
                    "repetition_penalty": args.repetition_penalty,
                })
                for off in tqdm(range(0, len(prefixes), args.bs)):
                    prompt_batch = np.concatenate((prefixes[off:off+args.bs],suffixes[off:off+args.bs]),axis=1)
                    prompt_batch = np.stack(prompt_batch, axis=0)
                    input_ids = torch.tensor(prompt_batch, dtype=torch.int64)
                    input_ids = torch.tensor(prompt_batch, dtype=torch.int64)
                    input_ids = input_ids[:, :(k + x)]
                    with torch.no_grad():
                        logits = model(input_ids.to(device)).logits.cpu()
                        probs = calculate_sentence_probability(logits.cpu(),input_ids.cpu(),condition_from_index=k,generation_config=generation_config,verbose=False)
                        probabilities.extend(probs.tolist())
                    
                title_here = f"{relevant_title}_k={k}_x={x}"
                prob_title = f"Generation/{title_here}-probabilities.npy"
                proportion_title = f"Generation/{title_here}-proportions.txt"
                with open(prob_title, 'wb') as f:
                    np.save(f, np.array(probabilities))
                
                thresholds = [0.001, 0.01, 0.05, 0.1]
                proportions = [len([i for i in probabilities if i > a]) for a in thresholds]
                with open(proportion_title, "w") as file:
                    for i in range(len(proportions)):
                        file.write(f"{thresholds[i]}: {proportions[i]} / {args.n_samples} = {proportions[i] / args.n_samples}\n")

                print(f"Theoretical Probs {title_here}",probabilities)
                plt.xlabel("Probabilities")
                plt.ylabel("Number of points (out of 500 samples)")
                plt.title(f"Probabilities Histogram")
                plt.hist(probabilities, bins=30, range=(0, 1))
                plt.savefig(f"Generation/{title_here}.png")
                plt.clf()

                plt.xlabel("Probabilities")
                plt.ylabel("Number of points (out of 500 samples)")
                plt.title(f"Log Probabilities Histogram")
                plt.xlim(-0.01, 1.01)
                plt.hist(probabilities, bins=30, range=(0,1))
                plt.yscale('log')
                plt.savefig(f"Generation/{title_here}_LOG.png")
                plt.clf()
   
if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"Total Elapsed Time: {end_time-start_time} seconds")
