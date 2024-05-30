import csv
import numpy as np
import pandas as pd

def compute_extraction_metrics(ground_truth,generations,ground_truth_statistics,generations_statistics,prefix_length,suffix_length,tokenizer,title=None,statistic_name=None,ground_truth_probabilities=None):
    """
    Computes all extraction metrics
    """
    num_samples = ground_truth.shape[0]
    num_generations = generations.shape[1]
    # Get best generation for each sample
    axis0 = np.arange(num_samples)
    axis1 = generations_statistics.argmin(1).reshape(-1)

    datadict = {}
    # Statistics
    datadict |= {f"ground_truth_{statistic_name}": ground_truth_statistics}
    datadict |= {f"best_generation_{statistic_name}": generations_statistics[axis0,axis1].tolist()}
    datadict |= {f"generations_{i}_{statistic_name}": generations_statistics[:,i] for i in range(num_generations)}
    # Text
    datadict |= {"prefix": [tokenizer.decode(ground_truth[row,:prefix_length]).replace("\n","\\n") for row in range(num_samples)]}
    datadict |= {"ground_truth_suffix_text": [tokenizer.decode(ground_truth[row,-suffix_length:]).replace("\n","\\n") for row in range(num_samples)]}
    datadict |= {"best_generation_suffix_text": [tokenizer.decode(row).replace("\n","\\n") for row in generations[axis0,axis1,-suffix_length:]]}
    datadict |= {f"generations_{i}_suffix_text": [tokenizer.decode(generations[row,i,-suffix_length:]).replace("\n","\\n") for row in range(num_samples)] for i in range(num_generations)}
    # Tokens
    datadict |= {"ground_truth_suffix_tokens": ground_truth[:,-suffix_length:].tolist()}
    datadict |= {"best_generation_suffix_tokens": generations[axis0,axis1,-suffix_length:].tolist()}
    datadict |= {f"generations_{i}_suffix_tokens": generations[:,i,-suffix_length:].tolist() for i in range(num_generations)}

    df = pd.DataFrame(datadict)

    # Compute Metrics
    df["exact_match"] = df["ground_truth_suffix_tokens"]==df["best_generation_suffix_tokens"]
    df["token_match"] = [(np.array(df["ground_truth_suffix_tokens"][row])==np.array(df["best_generation_suffix_tokens"][row])).mean() for row in range(num_samples)]
    df["any_exact_match"] = df.apply(lambda row: any(row[f"generations_{i}_suffix_tokens"] == row["ground_truth_suffix_tokens"] for i in range(num_generations)), axis=1)
    df["highest_token_match"] = df.apply(lambda row: max((np.array(row["ground_truth_suffix_tokens"]) == np.array(row[f"generations_{i}_suffix_tokens"])).mean() for i in range(num_generations)), axis=1)
    df["ground_truth_better_than"] = df.apply(lambda row: sum(row[f"ground_truth_{statistic_name}"] <= row[f"generations_{i}_{statistic_name}"] for i in range(num_generations)) / num_generations, axis=1)
    df["ground_truth_best"] = df['ground_truth_better_than']==1
    if ground_truth_probabilities is not None:
        df["ground_truth_suffix_probability"] = ground_truth_probabilities
    df = df[df.columns[(np.arange(len(df.columns))-(6 if ground_truth_probabilities is None else 7))%len(df.columns)]]

    metrics = {
        "precision": df['exact_match'].mean(),
        "hamming": df['token_match'].mean(),
        "multiprecision": df['any_exact_match'].mean(),
        "multihamming": df['highest_token_match'].mean(),
        "betterthan": df['ground_truth_better_than'].mean(),
        "best": df['ground_truth_best'].mean(),
    }

    print(f"Exact Match Accuracy (Precision): {metrics['precision']:.4g}")
    print(f"Token Level Accuracy (Hamming): {metrics['hamming']:.4g}")
    print(f"Any Exact Match Accuracy (Multiprecision): {metrics['multiprecision']:.4g}")
    print(f"Highest Token Level Accuracy (Multihamming): {metrics['multihamming']:.4g}")
    print(f"Average Proportion of Generations True Suffix is Better Than (Distinguishability Given Generated): {metrics['betterthan']:.4g}")
    print(f"True Suffix is Best (Accuracy Given Generated): {metrics['best']:.4g}")

    # Write to files
    ## Full CSV
    df.to_csv(f"{title}_full.csv",index=False)
    ## More human-legible json
    with open(f"{title}_records.json","w") as f:
        f.write(
            df.astype(
                {'ground_truth_suffix_tokens':'str','best_generation_suffix_tokens':'str'}|{f'generations_{i}_suffix_tokens':'str' for i in range(num_generations)}
            ).to_json(orient="records",lines=False,indent=4)
        )
    
    # Flattened version for overall ranking
    rows = []
    rows_with_ground_truth = []
    for idx, row in df.iterrows():
        for i in range(num_generations):
            new_row = {
                "original_index": idx,
                "generation_index": i,
                "exact_match": row["exact_match"],
                "token_match": row["token_match"],
                f"ground_truth_{statistic_name}": row[f"ground_truth_{statistic_name}"],
                f"generation_{statistic_name}": row[f"generations_{i}_{statistic_name}"],
                "prefix": row["prefix"],
                "ground_truth_suffix_text": row["ground_truth_suffix_text"],
                "generation_suffix_text": row[f"generations_{i}_suffix_text"],
                "ground_truth_suffix_tokens": row["ground_truth_suffix_tokens"],
                "generation_suffix_tokens": row[f"generations_{i}_suffix_tokens"],
            }
            rows.append(new_row)
            rows_with_ground_truth.append(new_row)
        # Also append ground truth
        rows_with_ground_truth.append({
            "original_index": idx,
            "generation_index": -1,
            "exact_match": 1.,
            "token_match": 1.,
            f"ground_truth_{statistic_name}": row[f"ground_truth_{statistic_name}"],
            f"generation_{statistic_name}": row[f"ground_truth_{statistic_name}"],
            "prefix": row["prefix"],
            "ground_truth_suffix_text": row["ground_truth_suffix_text"],
            "generation_suffix_text": row["ground_truth_suffix_text"],
            "ground_truth_suffix_tokens": row["ground_truth_suffix_tokens"],
            "generation_suffix_tokens": row["ground_truth_suffix_text"],
        })
    flattened_df = pd.DataFrame(rows).sort_values(by=f"generation_{statistic_name}").reset_index(drop=True)
    flattened_df.to_csv(f"{title}_flattened.csv",index=False)

    # Error-recall
    did_solve = np.zeros(num_samples)
    recall = []
    errors = []
    bad_guesses = 0
    answer = None
    for exid, is_correct in zip(flattened_df["original_index"],flattened_df["exact_match"]):
        if is_correct:
            did_solve[int(exid)] = 1
            recall.append(np.mean(did_solve))
            errors.append(bad_guesses)
            if bad_guesses < 100:
                answer = np.mean(did_solve)
        else:
            bad_guesses += 1
    print("Recall at 100 errors", answer)
            
    plt.plot(errors, recall)

    plt.semilogx()
    plt.xlabel("Number of bad guesses")
    plt.ylabel("Recall")

    # Precision-recall curve using scikit-learn
    precision, recall, _ = precision_recall_curve(flattened_df["exact_match"], flattened_df[f"generation_{statistic_name}"])
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, marker='o', linestyle='-', color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()

    # ROC-AUC curve
    fpr, tpr, thresholds = roc_curve(flattened_df["exact_match"], flattened_df[f"generation_{statistic_name}"])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.grid(which="both",alpha=0.2)
    plt.title("Generation Attack ROC")
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    flattened_df_w_true = pd.DataFrame(rows_with_ground_truth).sort_values(by=f"generation_{statistic_name}")
    flattened_df_w_true.to_csv(f"{title}_flattened_w_true.csv",index=False)

    ## Metrics json
    with open(f"{title}_metrics.json","w") as f:
        json.dump(metrics,f,indent=4)

    return df, flattened_df