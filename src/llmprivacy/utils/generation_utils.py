from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from typing import Optional
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
from transformers.generation.utils import GenerationMixin, GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList

def generate_suffixes(
    model: AutoModelForCausalLM,
    prefixes: DataLoader,
    generation_config: GenerationConfig,
    trials: int,
    accelerate: bool,
) -> np.ndarray:
    generations = []
    if not accelerate:
        device = next(model.parameters()).device
    for trial in range(trials):
        for input_ids in tqdm(prefixes):
            with torch.no_grad():
                generated_tokens = model.generate(
                    inputs=input_ids if accelerate else input_ids.to(device),
                    generation_config=generation_config
                ).cpu().detach()
                generations.extend(generated_tokens.numpy())    
    return np.array(generations)

def calculate_sentence_probability(
    logits: torch.FloatTensor,
    input_ids: torch.LongTensor,
    condition_from_index: Optional[int] = 0,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
    verbose: Optional[bool] = False,
    **kwargs,
) -> torch.FloatTensor:
    """
    Calculates the probability that a sentence is decoded from the logits with given decoding strategy
    Uses logic of transformers/generation/utils.py
    """
    if generation_config is None:
        generation_config = GenerationConfig()
    generation_config.update(**kwargs)
    generation_config.validate()

    if generation_config.pad_token_id is None:
        raise ValueError("No pad_token set")

    batch_size = input_ids.size(0)
    sentence_probability = torch.zeros(batch_size) # may want to do log space and use log_softmax

    logits_processor = GenerationMixin()._get_logits_processor(generation_config=generation_config,
            input_ids_seq_length=input_ids.size(1),
            encoder_input_ids=None,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,)
    logits_warper = GenerationMixin()._get_logits_warper(generation_config=generation_config)
    for pos in range(condition_from_index,input_ids.size(1) - 1):
        # Extract logits for the current token
        current_logits = logits[:, pos, :]
        prev_input_ids = input_ids[:,:(pos+1)]
        current_input_ids = input_ids[:,(pos+1)]
        current_logits = logits_processor(prev_input_ids, current_logits)
        current_logits = logits_warper(prev_input_ids, current_logits)
        if generation_config.do_sample:
            # Sample
            probs = F.log_softmax(current_logits, dim=-1)
        else:
            # Greedy decoding
            probs = torch.full((current_logits.shape[0],current_logits.shape[1]),-float("Inf"))
            probs[:,torch.argmax(current_logits, dim=-1)] = 0

        # Ignore Padding
        probs[:,generation_config.pad_token_id] = 0
        # if verbose:
        #     print(tokenizer.batch_decode(current_input_ids),torch.exp(probs.gather(1,current_input_ids.unsqueeze(1)).squeeze(-1)))
        sentence_probability += probs.gather(1,current_input_ids.unsqueeze(1)).squeeze(-1)

    return torch.exp(sentence_probability)

def compute_actual_prob(prefixes, suffixes, args, model, tokenizer, device, title=None, kx_tup=(50,50)):
    """
    Computes the actual probability of a suffix given a prefix and a model. 

    kx_tup expands the capabilities of this code. It changes the testing, given the same 50+50 input data, 
    to do K tokens of prefixes and X tokens of suffixes.
    """
    probabilities = []
    generation_config = model.generation_config
    generation_config.update(**{
        "min_length":kx_tup[0]+kx_tup[1],
        "max_length":kx_tup[0]+kx_tup[1],
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
        input_ids = input_ids[:, :(kx_tup[0] + kx_tup[1])]
        with torch.no_grad():
            logits = model(input_ids.to(device)).logits.cpu()
            probs = calculate_sentence_probability(logits.cpu(),input_ids.cpu(),condition_from_index=kx_tup[0],generation_config=generation_config,verbose=False)
            probabilities.extend(probs.tolist())
    output_title = f"{title}-probs" if title else "probabilities.npy"
    np.save(f"Generation/{output_title}", np.array(probabilities))
    print("Theoretical Probabilities: ", probabilities)
    return probabilities

def lin_combo(generations,losses,base_losses,suffixes,n_samples,suffix_length,num_lambdas=1000):
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
            f.write("Guess: "+guess.replace("\n","")+"\n\n")
            f.write(f"{name1}: {losses[row][axis1[row]]} | {name2} stat: {base_losses[row][axis1_baselosses[row]]}")
            i += 1
    
    # Now I want to print all guesses with corresponding info
    with open(f"Generation/{title}_all_guesses.txt","w") as f:
        for row in range(len(prefixes)):
            prefix = tokenizer.decode(prefixes[row])
            suffix = tokenizer.decode(suffixes[row])
            f.write(f"Example {row}\n")
            f.write("Prefix: "+prefix.replace("\n","")+"\n\n")
            f.write("Suffix: "+suffix.replace("\n","")+"\n\n")
            for j in range(n_iterations):
                genstr = tokenizer.decode(generations[row, j, -suffix_length:])
                f.write(f"Guess {j}: "+genstr.replace("\n","")+"\n\n")
                f.write(f"{name1}: {losses[row][j]} | {name2} stat: {base_losses[row][j]}\n")
            f.write("\n")

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
                loss = losses[samp_num][gen_num]
                base_loss = base_losses[samp_num][gen_num]
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
    prefix_nums = np.concatenate(tuple(np.array(range(n_samples))[:,np.newaxis] for _ in range(generations.shape[1])),1).flatten()
    generations, losses, base_losses = generations.reshape(-1,generations.shape[-1]), losses.flatten(), base_losses.flatten()
    order = losses.argsort()

    # Write result for google extraction challenge
    with open(f"Generation/{title}_generation_result.csv","w") as f:
        writer = csv.writer(f)
        writer.writerow(["Example ID", "Suffix Guess", name1, name2, "Correct"])
        for exid, generation, loss, base in zip(prefix_nums[order], generations[order], losses[order], base_losses[order]):
            writer.writerow([exid,str(list(generation[-suffix_length:])).replace(" ", ""),loss,base,exid==suffixes.shape[0] or np.all(suffixes[exid] == generation[-suffix_length:])])

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