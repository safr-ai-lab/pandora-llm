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
    def plot_error_recall(prefix_index,correct,plot_title,recall_at=100,log_scale=False,show_plot=True,save_name=None):
        # recall computation from google-research/lm-extraction-benchmark
        did_solve = np.zeros(len(np.unique(prefix_index)))
        recall = []
        errors = []
        bad_guesses = 0
        answer = None
        for exid, is_correct in zip(prefix_index,correct):
            if is_correct:
                did_solve[int(exid)] = 1
            else:
                bad_guesses += 1
            recall.append(np.mean(did_solve))
            errors.append(bad_guesses)
            if bad_guesses < 100:
                answer = np.mean(did_solve)
        print("Recall at 100 errors", answer)
        plt.figure(dpi=300)        
        plt.plot(errors, recall, label=f"Error-Recall (Recall@{recall_at}={answer:.4f})")
        plt.axvline(recall_at,alpha=0.5,c="red")
        if log_scale:
            plt.semilogx()
        plt.title(plot_title)
        plt.xlabel("Number of Errors")
        plt.ylabel("Recall")
        plt.legend(loc='lower right')
        plt.minorticks_on()
        plt.grid(which="major",alpha=0.2)
        plt.grid(which="minor",alpha=0.1)
        if save_name is not None:
            plt.savefig(save_name+"_error_recall.png", bbox_inches="tight")
            plt.savefig(save_name+"_error_recall.pdf", bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close()
        return answer
    metrics["recall@100"] = plot_error_recall(flattened_df["original_index"].to_numpy(),flattened_df["exact_match"].to_numpy(),"test")
    plot_error_recall(flattened_df["original_index"].to_numpy(),flattened_df["exact_match"].to_numpy(),"test",log_scale=True)

    # Precision-recall curve using scikit-learn
    def plot_precision_recall(ground_truth,predictions,plot_title,ci=True,num_bootstraps=1000,recalls=None,log_scale=False, show_plot=True, save_name=None, lims=None, color='darkorange'):
        n_points = len(ground_truth)

        precision, recall, thresholds = precision_recall_curve(ground_truth, predictions)
        average_precision = average_precision_score(ground_truth, predictions)
        if recalls is None:
            recalls = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        precision_at_recalls = [precision[np.max(np.argwhere(recall<=recall_val))] for recall_val in recalls]

        if ci:
            recall_range = np.linspace(0, 1, n_points)
            def precision_recall_statistic(data,axis):
                ground_truth = data[0,0,:].T
                predictions = data[1,0,:].T
                precision, recall, _ = precision_recall_curve(ground_truth, predictions)
                average_precision = average_precision_score(ground_truth, predictions)
                precision_range = np.interp(recall_range,recall,precision)
                return np.array([[average_precision]+precision_range.tolist()]).T
            
            data = torch.cat((ground_truth[:,None],predictions[:,None]),dim=1)
            bootstrap_result = bootstrap((data,), precision_recall_statistic, confidence_level=0.95, n_resamples=num_bootstraps, batch=1, method='percentile',axis=0)
            average_precision_se = bootstrap_result.standard_error[0]
            precision_se = [bootstrap_result.standard_error[np.max(np.argwhere(recall<=recall_val))] for recall_val in recalls]
        
        plt.figure(dpi=300)        
        if not log_scale:
            plt.plot(recall, precision, label=f'AP = {average_precision:0.4f}',c=color)
            plt.xlim([0,1] if lims is None else lims)
            plt.ylim([0,1] if lims is None else lims)
        else:
            plt.loglog(recall, precision, label=f'AP = {average_precision:0.4f}',c=color)
            plt.xlim([10**(-int(np.log10(n_points))),1] if lims is None else lims)
            plt.ylim([10**(-int(np.log10(n_points))),1] if lims is None else lims)
        if ci:
            plt.fill_between(recall_range,bootstrap_result.confidence_interval.low[1:],bootstrap_result.confidence_interval.high[1:],alpha=0.1,color=color)
        plt.title(plot_title)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.minorticks_on()
        plt.grid(which="major",alpha=0.2)
        plt.grid(which="minor",alpha=0.1)
        if save_name is not None:
            plt.savefig(save_name+"_pr.png", bbox_inches="tight")
            plt.savefig(save_name+"_pr.pdf", bbox_inches="tight")
            if ci:
                df = pd.DataFrame([average_precision,average_precision_se,precision_at_recalls,precision_se]).T
                df = df.rename(columns={0:"AP",1:"AP_SE"})
                df[[f'Precision@{recall_val}' for recall_val in recalls]] = pd.DataFrame(df[2].tolist(), index=df.index)
                df[[f'Precision@{recall_val}_SE' for recall_val in recalls]] = pd.DataFrame(df[3].tolist(), index=df.index)
                df = df.drop(columns=[2,3])
            else:
                df = pd.DataFrame([average_precision,precision_at_recalls]).T
                df = df.rename(columns={0:"AP"})
                df[[f'Precision@{recall_val}' for recall_val in recalls]] = pd.DataFrame(df[2].tolist(), index=df.index)
                df = df.drop(columns=[1])
            output = io.StringIO()
            df.to_csv(output,sep="\t")
            print(output.getvalue())
            df.to_csv(save_name+"_pr.csv")
        if show_plot:
            plt.show()
        plt.close()
        if ci:
            return average_precision, precision_at_recalls, average_precision_se, precision_se
        else:
            return average_precision, precision_at_recalls
        
    metrics["AP"], metrics["P@R"], metrics["AP_SE"], metrics["P@R_SE"] = plot_precision_recall(torch.from_numpy(flattened_df["exact_match"].to_numpy()),torch.from_numpy(flattened_df[f"generation_{statistic_name}"].to_numpy()),"test",log_scale=False)
    plot_precision_recall(torch.from_numpy(flattened_df["exact_match"].to_numpy()),torch.from_numpy(flattened_df[f"generation_{statistic_name}"].to_numpy()),"test",log_scale=False)

    # # ROC-AUC curve
    def plot_ROC_single(ground_truth, predictions, plot_title, keep_first=None, ci=True, num_bootstraps=1000, fprs=None, log_scale=False, show_plot=True, save_name=None, lims=None, color='darkorange'):
        n_points = len(ground_truth)
        fpr, tpr, thresholds = roc_curve(ground_truth,predictions)
        roc_auc = auc(fpr, tpr)

        if fprs is None:
            fprs = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        tpr_at_fprs = [tpr[np.max(np.argwhere(fpr<=fpr_val))] for fpr_val in fprs]

        # Compute CI
        if ci:
            fpr_range = np.linspace(0, 1, n_points)
            def auc_statistic(data,axis):
                ground_truth = data[0,0,:].T
                predictions = data[1,0,:].T
                fpr, tpr, thresholds = roc_curve(ground_truth, predictions)
                roc_auc = auc(fpr, tpr)
                tpr_range = np.interp(fpr_range,fpr,tpr)
                return np.array([[roc_auc]+tpr_range.tolist()]).T
            
            data = torch.cat((ground_truth[:,None],predictions[:,None]),dim=1)
            bootstrap_result = bootstrap((data,), auc_statistic, confidence_level=0.95, n_resamples=num_bootstraps, batch=1, method='percentile',axis=0)
            auc_se = bootstrap_result.standard_error[0]
            tpr_se = [bootstrap_result.standard_error[np.max(np.argwhere(fpr_range<=fpr_val))] for fpr_val in fprs]

        # Plot
        plt.figure(figsize=(7,7),dpi=300)
        plt.plot([0, 1], [0, 1], linestyle="--", c="k")
        if not log_scale:
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:0.4f}',c=color)
            plt.xlim([0,1] if lims is None else lims)
            plt.ylim([0,1] if lims is None else lims)
        else:
            plt.loglog(fpr, tpr, label=f'AUC = {roc_auc:0.4f}',c=color)
            plt.xlim([10**(-int(np.log10(n_points))),1] if lims is None else lims)
            plt.ylim([10**(-int(np.log10(n_points))),1] if lims is None else lims)
        if ci:
            plt.fill_between(fpr_range,bootstrap_result.confidence_interval.low[1:],bootstrap_result.confidence_interval.high[1:],alpha=0.1,color=color)
        plt.title(plot_title)
        plt.legend(loc="lower right")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.minorticks_on()
        plt.grid(which="major",alpha=0.2)
        plt.grid(which="minor",alpha=0.1)
        if save_name is not None:
            plt.savefig(save_name+"_roc.png", bbox_inches="tight")
            plt.savefig(save_name+"_roc.pdf", bbox_inches="tight")
            if ci:
                df = pd.DataFrame([roc_auc,auc_se,tpr_at_fprs,tpr_se]).T
                df = df.rename(columns={0:"AUC",1:"AUC_SE"})
                df[[f'TPR@{fpr_val}' for fpr_val in fprs]] = pd.DataFrame(df[2].tolist(), index=df.index)
                df[[f'TPR@{fpr_val}_SE' for fpr_val in fprs]] = pd.DataFrame(df[3].tolist(), index=df.index)
                df = df.drop(columns=[2,3])
            else:
                df = pd.DataFrame([roc_auc,tpr_at_fprs]).T
                df = df.rename(columns={0:"AUC"})
                df[[f'TPR@{fpr_val}' for fpr_val in fprs]] = pd.DataFrame(df[2].tolist(), index=df.index)
                df = df.drop(columns=[1])
            output = io.StringIO()
            df.to_csv(output,sep="\t")
            print(output.getvalue())
            df.to_csv(save_name+"_roc.csv")
        if show_plot:
            plt.show()
        plt.close()
        if ci:
            return roc_auc, tpr_at_fprs, auc_se, tpr_se
        else:
            return roc_auc, tpr_at_fprs
    metrics["AUC"], metrics["TPR@FPR"], metrics["AUC_SE"], metrics["TPR@FPR_SE"] = plot_ROC_single(torch.from_numpy(flattened_df["exact_match"].to_numpy()), torch.from_numpy(flattened_df[f"generation_{statistic_name}"].to_numpy()),"test",log_scale=False)
    plot_ROC_single(torch.from_numpy(flattened_df["exact_match"].to_numpy()), torch.from_numpy(flattened_df[f"generation_{statistic_name}"].to_numpy()),"test",log_scale=True)

    flattened_df_w_true = pd.DataFrame(rows_with_ground_truth).sort_values(by=f"generation_{statistic_name}")
    flattened_df_w_true.to_csv(f"{title}_flattened_w_true.csv",index=False)

    ## Metrics json
    with open(f"{title}_metrics.json","w") as f:
        json.dump(metrics,f,indent=4)

    return flattened_df