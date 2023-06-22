import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, GPTNeoXForCausalLM, AutoTokenizer, AutoConfig
from attack_utils import *
from dataset_utils import *
from LOSS import LOSS
from MoPe import MoPe
import time
import argparse
from accelerate import Accelerator
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

"""
Sample command line prompt
python run_generation.py --prefixes train_prefix.npy --suffixes train_suffix.npy --mod_size 125m --n_samples 128 --n_iterations 4 --bs 16 --top_k 24 --top_p 0.8 --typical_p 0.9 --temperature 0.58 --repetition_penalty 1.04 --attack MoPe --n_models 5 --sigma 0.005
"""    

def results(generations,losses,base_losses,prefixes,suffixes,n_samples,n_iterations,suffix_length,tokenizer):
    # Metrics obtained by choosing the best suffix per prefix
    axis0 = np.arange(generations.shape[0])
    axis1 = losses.argmin(1).reshape(-1)
    best_suffixes = generations[axis0, axis1, -suffix_length:]
    precision = np.sum(np.all(best_suffixes==suffixes, axis=-1)) / n_samples
    hamming = np.sum(np.sum(best_suffixes==suffixes, axis=-1) / suffix_length) / n_samples
    print(f"Exact Match Accuracy (Precision): {precision:.4g}")
    print(f"Token Level Accuracy (Hamming): {hamming:.4g}")
    with open("Generation/guesses.txt","w") as f:
        for row in range(len(prefixes)):
            prefix = tokenizer.decode(prefixes[row])
            guess = tokenizer.decode(best_suffixes[row])
            suffix = tokenizer.decode(suffixes[row])
            f.write(f"Example {row}\n")
            f.write("Prefix: "+prefix.replace("\n","")+"\n")
            f.write("Suffix: "+suffix.replace("\n","")+"\n")
            f.write("Guess: "+guess.replace("\n","")+"\n\n")

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
    with open("Generation/generation_result.csv","w") as f:
        writer = csv.writer(f)
        writer.writerow(["Example ID", "Suffix Guess", "MoPe", "LOSS", "Correct"])
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
    plt.savefig("Generation/error_curve.png")
    plt.semilogx()
    plt.savefig("Generation/error_curve_log.png")

    precision_scores, recall_scores, thresholds = precision_recall_curve([np.all(suffixes[exid] == generation[-suffix_length:]) for exid, generation in zip(prefix_nums[order],generations[order])], -losses[order])
    plt.figure()
    plt.plot(recall_scores,precision_scores)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(which="both",alpha=0.2)
    plt.savefig("Generation/pr_curve.png")
    plt.semilogy()
    plt.savefig("Generation/pr_curve_log.png")

    # Plot overall ROC curve
    fpr, tpr, thresholds = roc_curve([np.all(suffixes[exid] == generation[-suffix_length:]) for exid, generation in zip(prefix_nums[order],generations[order])], -losses[order])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.grid(which="both",alpha=0.2)
    title = "Generation Attack ROC"
    save_name = "Generation/generation_roc.png"
    plt.title(title)
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    print(f"AUC of Experiment {title}\n{roc_auc}")
    plt.savefig(save_name, bbox_inches="tight")
    plt.show()
    save_name = "Generation/generation_roc_log.png"
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.grid(which="both",alpha=0.2)
    plt.xscale("log",base=10,subs=list(range(11)))
    plt.yscale("log",base=10,subs=list(range(11)))
    plt.xlim(9e-4,1.1)
    plt.ylim(9e-4,1.1)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(save_name, bbox_inches="tight")
    plt.show()

    with open("Generation/metrics.txt","w") as f:
        f.write(f"Precision: {precision:4g}\nHamming: {hamming:4g}\nMultiprecision: {precision_multi:4g}\nRecall@100: {answer:4g}\nAUC: {roc_auc}")

    return recall, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefixes', action="store", required=True, help='.npy file of prefixes')
    parser.add_argument('--suffixes', action="store", required=True, help='.npy file of suffixes')
    parser.add_argument('--mod_size', action="store", type=str, required=True, help='Pythia Model Size')
    parser.add_argument('--n_samples', action="store", type=int, required=True, help='Dataset size')
    parser.add_argument('--n_iterations', action="store", type=int, required=False, help='How many generations per prompt')
    parser.add_argument('--suffix_length', action="store", type=int, required=False, default=50, help='Length of suffix')
    parser.add_argument('--seed', action="store", type=int, required=False, default=229, help='Seed')
    parser.add_argument('--bs', action="store", type=int, required=False, default=1, help='Batch size')
    parser.add_argument('--accelerate', action="store_true", required=False, help='Use accelerate')
    parser.add_argument('--model_half', action="store_true", required=False, help='Use half precision (fp16). 1 for use; 0 for not.')
    parser.add_argument('--attack', action="store", type=str, required=True, help='Attack type (LOSS or MoPe)')
    parser.add_argument('--top_k', action="store", type=int, default=10, required=False, help='Top k sampling for generation (number of tokens to choose from)')
    parser.add_argument('--top_p', action="store", type=float, default=1.0, required=False, help='Top p / nucleus eta sampling for generation (choose tokens until probability adds up to p)')
    parser.add_argument('--typical_p', action="store", type=float, default=1.0, required=False, help='Typical p / phi sampling for generation (choose locally typical tokens until probability adds up to p)')
    parser.add_argument('--temperature', action="store", type=float, default=1.0, required=False, help='Higher temperature, more diversity - the value used to modulate the next token probabilities.')
    parser.add_argument('--repetition_penalty', action="store", type=float, default=1.0, required=False, help='The parameter for repetition penalty. 1.0 means no penalty.')
    parser.add_argument('--n_models', action="store", type=int, required=False, help='Number of new models')
    parser.add_argument('--sigma', action="store", type=float, required=False, help='Noise standard deviation')
    parser.add_argument('--noise_type', action="store", type=int, default=1, required=False, help='Noise to add to model. Options: 1 = Gaussian, 2 = Rademacher, 3+ = user-specified (see README). If not specified, use Gaussian noise.')
    args = parser.parse_args()

    accelerator = Accelerator() if args.accelerate else None

    ## Other parameters
    seed = args.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_title = f"gpt-neo-{args.mod_size}"
    model_name = "EleutherAI/" + model_title

    ## Load model and training and validation dataset
    prefixes = np.load(args.prefixes).astype(np.int64)[-args.n_samples:]
    suffixes = np.load(args.suffixes).astype(np.int64)[-args.n_samples:]

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7b")

    if args.attack=="LOSS":
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
    else:
        raise NotImplementedError()

    all_generations, all_losses, all_base_losses = [], [], []
    for trial in range(args.n_iterations):
        print(f"Generation {trial+1}/{args.n_iterations}")
        generations, losses, base_losses = attack.generate(prefixes,attack_config)
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

    results(generations,losses,base_losses,prefixes,suffixes,args.n_samples,args.n_iterations,args.suffix_length,tokenizer)

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"Total Elapsed Time: {end_time-start_time} seconds")