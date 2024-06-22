import json
import numpy as np
import pandas as pd
import torch
from ..utils.plot_utils import plot_error_recall, plot_error_recall_plotly, plot_precision_recall, plot_precision_recall_plotly, plot_ROC_single, plot_ROC_single_plotly

def compute_extraction_metrics(ground_truth,generations,ground_truth_statistics,generations_statistics,prefix_length,suffix_length,tokenizer,title=None,statistic_name=None,ground_truth_probabilities=None):
    """
    Computes all extraction metrics

    Args:
        ground_truth (list[list[int]]): list of input ids for each ground truth (N x L)
        generations (list[list[list[int]]]): list of input ids for each generation (N x G x L)
        ground_truth_statistics (list[float]): list of statistics for each ground truth (N)
        generations_statistics (list[list[float]]): list of statistics for each generation (N x G)
        prefix_length (int): number of prefix input ids
        suffix_length (int): number of suffix input_ids
        tokenizer (AutoTokenizer): tokenizer used to create the input ids
        title (str): title of the experiment
        statistic_name (str): name of the statistic
        ground_truth_probabilities (list[float]): list of probabilities of generating each ground truth suffix from prefix
    """
    ####################################################################################################
    # Generate dataframe of prefix to generations
    ####################################################################################################
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

    metrics = { # add CI
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

    ####################################################################################################
    # Generate flattened dataframe for Google-LLM Extraction challenge
    ####################################################################################################    
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
            "exact_match": True,
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
    with open(f"{title}_flattened_records.json","w") as f:
        f.write(
            flattened_df.astype(
                {'ground_truth_suffix_tokens':'str','generation_suffix_tokens':'str'}
            ).to_json(orient="records",lines=False,indent=4)
        )

    metrics["recalls@errors"], metrics["recall@errors_SE"] = plot_error_recall(torch.from_numpy(flattened_df["original_index"].to_numpy()),torch.from_numpy(flattened_df["exact_match"].to_numpy()),plot_title=title,save_name=title,log_scale=False,show_plot=False)
    metrics["recalls@errors"] = metrics["recalls@errors"].tolist()
    metrics["recall@errors_SE"] = metrics["recall@errors_SE"].tolist()
    plot_error_recall(torch.from_numpy(flattened_df["original_index"].to_numpy()),torch.from_numpy(flattened_df["exact_match"].to_numpy()),plot_title=title,save_name=title+"_log",log_scale=True,show_plot=False)
    # plot_error_recall_plotly(torch.from_numpy(flattened_df["original_index"].to_numpy()),torch.from_numpy(flattened_df["exact_match"].to_numpy()),plot_title=title,save_name=title,log_scale=False,show_plot=False)
    # plot_error_recall_plotly(torch.from_numpy(flattened_df["original_index"].to_numpy()),torch.from_numpy(flattened_df["exact_match"].to_numpy()),plot_title=title,save_name=title+"_log",log_scale=True,show_plot=False)

    metrics["AP"], metrics["P@R"], metrics["AP_SE"], metrics["P@R_SE"] = plot_precision_recall(torch.from_numpy(flattened_df["exact_match"].to_numpy()),torch.from_numpy(flattened_df[f"generation_{statistic_name}"].to_numpy()),plot_title=title,save_name=title,log_scale=False,show_plot=False)
    plot_precision_recall(torch.from_numpy(flattened_df["exact_match"].to_numpy()),torch.from_numpy(flattened_df[f"generation_{statistic_name}"].to_numpy()),plot_title=title,save_name=title+"_log",log_scale=True,show_plot=False)
    # plot_precision_recall_plotly(torch.from_numpy(flattened_df["exact_match"].to_numpy()),torch.from_numpy(flattened_df[f"generation_{statistic_name}"].to_numpy()),plot_title=title,save_name=title,log_scale=False,show_plot=False)
    # plot_precision_recall_plotly(torch.from_numpy(flattened_df["exact_match"].to_numpy()),torch.from_numpy(flattened_df[f"generation_{statistic_name}"].to_numpy()),plot_title=title,save_name=title+"_log",log_scale=True,show_plot=False)

    metrics["AUC"], metrics["TPR@FPR"], metrics["AUC_SE"], metrics["TPR@FPR_SE"] = plot_ROC_single(torch.from_numpy(flattened_df["exact_match"].to_numpy()), torch.from_numpy(flattened_df[f"generation_{statistic_name}"].to_numpy()),plot_title=title,save_name=title,log_scale=False,show_plot=False)
    plot_ROC_single(torch.from_numpy(flattened_df["exact_match"].to_numpy()), torch.from_numpy(flattened_df[f"generation_{statistic_name}"].to_numpy()),plot_title=title,save_name=title+"_log",log_scale=True,show_plot=False)
    # plot_ROC_single_plotly(torch.from_numpy(flattened_df["exact_match"].to_numpy()), torch.from_numpy(flattened_df[f"generation_{statistic_name}"].to_numpy()),plot_title=title,save_name=title,log_scale=False,show_plot=False)
    # plot_ROC_single_plotly(torch.from_numpy(flattened_df["exact_match"].to_numpy()), torch.from_numpy(flattened_df[f"generation_{statistic_name}"].to_numpy()),plot_title=title,save_name=title+"_log",log_scale=True,show_plot=False)

    flattened_df_w_true = pd.DataFrame(rows_with_ground_truth).sort_values(by=f"generation_{statistic_name}")
    flattened_df_w_true.to_csv(f"{title}_flattened_w_true.csv",index=False)
    with open(f"{title}_flattened_w_true_records.json","w") as f:
        f.write(
            flattened_df_w_true.astype(
                {'ground_truth_suffix_tokens':'str','generation_suffix_tokens':'str'}
            ).to_json(orient="records",lines=False,indent=4)
        )

    metrics["w_true_recalls@errors"], metrics["w_true_recall@errors_SE"] = plot_error_recall(torch.from_numpy(flattened_df_w_true["original_index"].to_numpy()),torch.from_numpy(flattened_df_w_true["exact_match"].to_numpy()),plot_title=title+"_w_true",save_name=title+"_w_true",log_scale=False,show_plot=False)
    metrics["w_true_recalls@errors"] = metrics["w_true_recalls@errors"].tolist()
    metrics["w_true_recall@errors_SE"] = metrics["w_true_recall@errors_SE"].tolist()
    plot_error_recall(torch.from_numpy(flattened_df_w_true["original_index"].to_numpy()),torch.from_numpy(flattened_df_w_true["exact_match"].to_numpy()),plot_title=title+"_w_true",save_name=title+"_w_true"+"_log",log_scale=True,show_plot=False)
    # plot_error_recall_plotly(torch.from_numpy(flattened_df_w_true["original_index"].to_numpy()),torch.from_numpy(flattened_df_w_true["exact_match"].to_numpy()),plot_title=title+"_w_true",save_name=title+"_w_true",log_scale=False,show_plot=False)
    # plot_error_recall_plotly(torch.from_numpy(flattened_df_w_true["original_index"].to_numpy()),torch.from_numpy(flattened_df_w_true["exact_match"].to_numpy()),plot_title=title+"_w_true",save_name=title+"_w_true"+"_log",log_scale=True,show_plot=False)

    metrics["w_true_AP"], metrics["w_true_P@R"], metrics["w_true_AP_SE"], metrics["w_true_P@R_SE"] = plot_precision_recall(torch.from_numpy(flattened_df_w_true["exact_match"].to_numpy()),torch.from_numpy(flattened_df_w_true[f"generation_{statistic_name}"].to_numpy()),plot_title=title+"_w_true",save_name=title+"_w_true",log_scale=False,show_plot=False)
    plot_precision_recall(torch.from_numpy(flattened_df_w_true["exact_match"].to_numpy()),torch.from_numpy(flattened_df_w_true[f"generation_{statistic_name}"].to_numpy()),plot_title=title+"_w_true",save_name=title+"_w_true"+"_log",log_scale=True,show_plot=False)
    # plot_precision_recall_plotly(torch.from_numpy(flattened_df_w_true["exact_match"].to_numpy()),torch.from_numpy(flattened_df_w_true[f"generation_{statistic_name}"].to_numpy()),plot_title=title+"_w_true",save_name=title+"_w_true",log_scale=False,show_plot=False)
    # plot_precision_recall_plotly(torch.from_numpy(flattened_df_w_true["exact_match"].to_numpy()),torch.from_numpy(flattened_df_w_true[f"generation_{statistic_name}"].to_numpy()),plot_title=title+"_w_true",save_name=title+"_w_true"+"_log",log_scale=True,show_plot=False)

    metrics["w_true_AUC"], metrics["w_true_TPR@FPR"], metrics["w_true_AUC_SE"], metrics["w_true_TPR@FPR_SE"] = plot_ROC_single(torch.from_numpy(flattened_df_w_true["exact_match"].to_numpy()), torch.from_numpy(flattened_df_w_true[f"generation_{statistic_name}"].to_numpy()),plot_title=title+"_w_true",save_name=title+"_w_true",log_scale=False,show_plot=False)
    plot_ROC_single(torch.from_numpy(flattened_df_w_true["exact_match"].to_numpy()), torch.from_numpy(flattened_df_w_true[f"generation_{statistic_name}"].to_numpy()),plot_title=title+"_w_true",save_name=title+"_w_true"+"_log",log_scale=True,show_plot=False)
    # plot_ROC_single_plotly(torch.from_numpy(flattened_df_w_true["exact_match"].to_numpy()), torch.from_numpy(flattened_df_w_true[f"generation_{statistic_name}"].to_numpy()),plot_title=title+"_w_true",save_name=title+"_w_true",log_scale=False,show_plot=False)
    # plot_ROC_single_plotly(torch.from_numpy(flattened_df_w_true["exact_match"].to_numpy()), torch.from_numpy(flattened_df_w_true[f"generation_{statistic_name}"].to_numpy()),plot_title=title+"_w_true",save_name=title+"_w_true"+"_log",log_scale=True,show_plot=False)

    ## Metrics json
    with open(f"{title}_metrics.json","w") as f:
        json.dump(metrics,f,indent=4)

    return flattened_df