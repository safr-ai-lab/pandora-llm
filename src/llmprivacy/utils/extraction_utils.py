import csv
import numpy as np
import pandas as pd

def compute_extraction_metrics(ground_truth,generations,ground_truth_statistics,generations_statistics,prefix_length,suffix_length,tokenizer,title=None,statistic_name=None,ground_truth_probabilities=None):
    """
    Computes all extraction metrics
    """
    # Metrics obtained by choosing the best suffix per prefix
    num_samples = ground_truth.shape[0]
    num_generations = generations.shape[1]
    axis0 = np.arange(num_samples)
    axis1 = generations_statistics.argmin(1).reshape(-1)

    datadict = {}
    datadict |= {f"ground_truth_{statistic_name}": ground_truth_statistics}
    datadict |= {f"best_generation_{statistic_name}": generations_statistics[axis0,axis1].tolist()}
    datadict |= {f"generations_{i}_{statistic_name}": generations_statistics[:,i] for i in range(num_generations)}

    datadict |= {"prefix": [tokenizer.decode(ground_truth[row,:prefix_length]) for row in range(num_samples)]}
    datadict |= {"ground_truth_suffix_text": [tokenizer.decode(ground_truth[row,-suffix_length:]) for row in range(num_samples)]}
    datadict |= {"best_generation_suffix_text": [tokenizer.decode(row) for row in generations[axis0,axis1,-suffix_length:]]}
    datadict |= {f"generations_{i}_suffix_text": [tokenizer.decode(generations[row,i,-suffix_length:]) for row in range(num_samples)] for i in range(num_generations)}

    datadict |= {"ground_truth_suffix_tokens": ground_truth[:,-suffix_length:].tolist()}
    datadict |= {"best_generation_suffix_tokens": generations[axis0,axis1,-suffix_length:].tolist()}
    datadict |= {f"generations_{i}_suffix_tokens": generations[:,i,-suffix_length:].tolist() for i in range(num_generations)}

    if ground_truth_probabilities is not None:
        datadict |= {"ground_truth_suffix_probability": ground_truth_probabilities}

    df = pd.DataFrame(datadict)

    df["exact_match"] = df["ground_truth_suffix_tokens"]==df["best_generation_suffix_tokens"]
    df["token_match"] = [(np.array(df["ground_truth_suffix_tokens"][row])==np.array(df["best_generation_suffix_tokens"][row])).mean() for row in range(num_samples)]
    df["any_exact_match"] = df.apply(lambda row: any(row[f"generations_{i}_suffix_tokens"] == row["ground_truth_suffix_tokens"] for i in range(num_generations)), axis=1)
    df["highest_token_match"] = df.apply(lambda row: max((np.array(row["ground_truth_suffix_tokens"]) == np.array(row[f"generations_{i}_suffix_tokens"])).mean() for i in range(num_generations)), axis=1)
    df["ground_truth_better_than"] = df.apply(lambda row: sum(row[f"ground_truth_{statistic_name}"] <= row[f"generations_{i}_{statistic_name}"] for i in range(num_generations)) / num_generations, axis=1)
    df["ground_truth_best"] = df['ground_truth_better_than']==1
    print(f"Exact Match Accuracy (Precision): {df['exact_match'].mean():.4g}")
    print(f"Token Level Accuracy (Hamming): {df['token_match'].mean():.4g}")
    print(f"Any Exact Match Accuracy (Multiprecision): {df['any_exact_match'].mean():.4g}")
    print(f"Highest Token Level Accuracy (Multihamming): {df['highest_token_match'].mean():.4g}")
    print(f"Average Proportion of Generations True Suffix is Better Than (Distinguishability Given Generated): {df['ground_truth_better_than'].mean():.4g}")
    print(f"True Suffix is Best (Accuracy Given Generated): {df['ground_truth_best'].mean():.4g}")

    return df

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