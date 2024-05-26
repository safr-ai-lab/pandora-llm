import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objects as go
from scipy.stats import bootstrap
from sklearn.metrics import roc_curve, auc
import torch

def plot_hist(train_statistics, val_statistics, plot_title, keep_first=None, show_plot=True, save_name=None, bins=20):
    """
    Plot histogram of membership inference statistics on train and validation datasets

    Args:
        train_statistics (list[float]): list of train statistics
        val_statistics (list[float]): list of val statistics
        plot_title (str): title of the plot
        keep_first (int): compute only for the first keep_first number of samples
        show_plot (bool): whether to show the plot
        save_name (str): save path for plot (without extension); does not save unless save_name is specified
    """
    # Preprocess
    train_statistics = torch.as_tensor(train_statistics).flatten()[:keep_first]
    train_statistics = train_statistics[~train_statistics.isnan()]
    val_statistics = torch.as_tensor(val_statistics).flatten()[:keep_first]
    val_statistics = val_statistics[~val_statistics.isnan()]

    # Plot
    plt.figure()
    plt.hist(train_statistics, bins=bins, alpha=0.5, label='training')
    plt.hist(val_statistics, bins=bins, alpha=0.5, label='validation')
    plt.legend(loc='upper right')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(plot_title)
    if save_name is not None:
        plt.savefig(save_name+"_hist.png", bbox_inches="tight")
    if show_plot:
        plt.show()

def plot_ROC(train_statistics, val_statistics, plot_title, keep_first=None, ci=True, num_bootstraps=1000, fprs=None, log_scale=False, show_plot=True, save_name=None, lims=None, color='darkorange'):
    '''
    Plots ROC curve with train and validation test statistics. Also saves TPRs at FPRs.
    
    **Note that we assume train statistic < test statistic. Negate before using if otherwise.**

    Args:
        train_statistics (list[float]): list of train statistics
        val_statistics (list[float]): list of val statistics
        plot_title (str): title of the plot
        ci (bool): compute confidence intervals. Default: True
        num_bootstraps (int): number of bootstraps for confidence interval
        keep_first (int): compute only for the first keep_first number of samples
        show_plot (bool): whether to show the plot
        save_name (str): save path for plot and scores (without extension); does not save unless save_name is specified
        log_scale (bool): whether to plot on log-log scale
        lims (list): argument to xlim and ylim
        fprs (list[float]): return TPRs at given FPRs. If unspecified, calculates at every 0.1 increment
        color (str): color
    
    Returns:
        auc (float): the ROC-AUC score
        tpr_at_fprs (list[float]): the tprs at the given fprs
    '''
    # Preprocess
    train_statistics = torch.as_tensor(train_statistics).flatten()[:keep_first]
    train_statistics = train_statistics[~train_statistics.isnan()]
    val_statistics = torch.as_tensor(val_statistics).flatten()[:keep_first]
    val_statistics = val_statistics[~val_statistics.isnan()]

    ground_truth = torch.cat((torch.ones_like(train_statistics),torch.zeros_like(val_statistics))).flatten()
    predictions = torch.cat((-train_statistics,-val_statistics)).flatten()
    n_points = len(ground_truth)

    fpr, tpr, thresholds = roc_curve(ground_truth,predictions)
    roc_auc = auc(fpr, tpr)

    # Process FPRs
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
        df = pd.DataFrame([roc_auc,auc_se,tpr_at_fprs,tpr_se]).T
        df = df.rename(columns={0:"AUC",1:"AUC_SE"})
        df[[f'TPR@{fpr_val}' for fpr_val in fprs]] = pd.DataFrame(df[2].tolist(), index=df.index)
        df[[f'TPR@{fpr_val}' for fpr_val in fprs]] = pd.DataFrame(df[3].tolist(), index=df.index)
        df = df.drop(columns=[2,3])
        output = io.StringIO()
        df.to_csv(output,sep="\t")
        print(output.getvalue())
        df.to_csv(save_name+"_data.csv")
    if show_plot:
        plt.show()
    if ci:
        return roc_auc, tpr_at_fprs, auc_se, tpr_se
    else:
        return roc_auc, tpr_at_fprs


def plot_ROC_multiple(train_statistics_list, val_statistics_list, plot_title, labels, keep_first=None, ci=True, num_bootstraps=1000, fprs=None, log_scale=False, show_plot=True, save_name=None, lims=None, colors=None, bold_labels=None):
    '''
    Plots multiple ROC curves in a single plot

    Args:
        train_statistics_list (list[list[float]]): list of curves, each a list of train statistics
        val_statistics_list (list[list[float]]): list of curves, each a list of val statistics
        plot_title (str): title of the plot
        labels (list[str]): labels of each curve
        fprs (list[float]): return TPRs at given FPRs. If unspecified, calculates at every 0.1 increment
        keep_first (int): compute only for the first keep_first number of samples
        show_plot (bool): whether to show the plot
        save_name (str): save path for plot; does not save unless save_name is specified
        log_scale (bool): whether to plot on log-log scale
        lims (list): argument to xlim and ylim
        colors (list): list of colors to use
        bold_labels (list): list of indices to bold in legend
    
    Returns:
        roc_auc_map (dict[str,float]): map of labels to auc
        tpr_at_fprs_map (dict[str,list[float]]): map of labels to the tprs at the given fprs
    '''
    if fprs is None:
        fprs = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    if colors is None:
        colors = [mpl.colormaps["tab10"](i) for i in range(len(train_statistics_list))]

    roc_auc_map = {}
    tpr_at_fprs_map = {}
    auc_se_map = {}
    tpr_se_map = {}
    plt.figure(figsize=(7,7),dpi=300)
    plt.plot([0, 1], [0, 1], linestyle="--", c="k")
    for i, (train_statistics, val_statistics, label) in enumerate(zip(train_statistics_list, val_statistics_list,labels)):
        train_statistics = torch.as_tensor(train_statistics).flatten()[:keep_first]
        train_statistics = train_statistics[~train_statistics.isnan()]
        val_statistics = torch.as_tensor(val_statistics).flatten()[:keep_first]
        val_statistics = val_statistics[~val_statistics.isnan()]

        ground_truth = torch.cat((torch.ones_like(train_statistics),torch.zeros_like(val_statistics))).flatten()
        predictions = torch.cat((-train_statistics,-val_statistics)).flatten()
        n_points = len(ground_truth)

        fpr, tpr, thresholds = roc_curve(ground_truth,predictions)
        roc_auc = auc(fpr, tpr)
        
        # Compute CI
        if ci:
            fpr_range = np.linspace(0, 1, 2000)
            def auc_statistic(data,axis):
                ground_truth = data[0,0,:].T
                predictions = data[1,0,:].T
                fpr, tpr, thresholds = roc_curve(ground_truth, predictions)
                roc_auc = auc(fpr, tpr)
                tpr_range = np.interp(fpr_range,fpr,tpr)
                return np.array([[roc_auc]+tpr_range.tolist()]).T
            
            data = torch.cat((ground_truth[:,None],predictions[:,None]),dim=1)
            bootstrap_result = bootstrap((data,), auc_statistic, confidence_level=0.95, n_resamples=num_bootstraps, batch=1, method='percentile',axis=0)
        
        if not log_scale:
            plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:0.4f})',c=colors[i])
            plt.xlim([0,1] if lims is None else lims)
            plt.ylim([0,1] if lims is None else lims)
        else:
            plt.loglog(fpr, tpr, label=f'{label} (AUC = {roc_auc:0.4f})',c=colors[i])
            plt.xlim([10**(-int(np.log10(n_points))),1] if lims is None else lims)
            plt.ylim([10**(-int(np.log10(n_points))),1] if lims is None else lims)
        if ci:
            plt.fill_between(fpr_range,bootstrap_result.confidence_interval.low[1:],bootstrap_result.confidence_interval.high[1:],alpha=0.1,color=colors[i])
            auc_se_map[label] = bootstrap_result.standard_error[0]
            tpr_se_map[label] = [bootstrap_result.standard_error[np.max(np.argwhere(fpr_range<=fpr_val))] for fpr_val in fprs]        
        roc_auc_map[label] = roc_auc
        tpr_at_fprs_map[label] = [tpr[np.max(np.argwhere(fpr<=fpr_val))] for fpr_val in fprs]

    plt.title(plot_title)
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.minorticks_on()
    plt.grid(which="major",alpha=0.2)
    plt.grid(which="minor",alpha=0.1)
    texts = plt.legend().get_texts()
    for index in bold_labels:
        texts[index].set_fontweight('bold')
    if save_name is not None:
        if save_name[-4:] != ".png" and save_name[-4:] != ".pdf":
            save_name = save_name + ".png"
        plt.savefig(save_name, bbox_inches="tight")
    if show_plot:
        plt.show()
    return roc_auc_map, tpr_at_fprs_map, auc_se_map, tpr_se_map


def plot_ROC_files(files, plot_title, labels=None, keep_first=None, show_plot=True, save_name=None, log_scale=False, fprs=None):
    """
    Plots ROCs from saved statistic .pt files

    Args:
        files (list[str]): list of paths to pytorch files, each containing a tuple of (train_statistics, val_statistics)
        plot_title (str): title of the plot
        labels (list[str]): labels of each curve
        keep_first (int): compute only for the first keep_first number of samples
        show_plot (bool): whether to show the plot
        save_name (str): save path for plot; does not save unless save_name is specified
        log_scale (bool): whether to plot on log-log scale
        fprs (list[float]): return TPRs at given FPRs. If unspecified, calculates at every 0.1 increment
    
    Returns:
        roc_auc_map (dict[str,float]): map of labels to auc
        tpr_at_fprs_map (dict[str,list[float]]): map of labels to the tprs at the given fprs
    """
    train_statistics_list = []
    val_statistics_list = []
    for file in files:
        train_statistics, val_statistics = torch.load(file)
        train_statistics_list.append(train_statistics)
        val_statistics_list.append(val_statistics)
    if labels is None:
        labels = files
    plot_ROC_multiple(train_statistics_list, val_statistics_list, plot_title, labels, keep_first=keep_first, show_plot=show_plot, save_name=save_name, log_scale=log_scale, fprs=fprs)


def print_AUC(train_statistic, val_statistic):
    """
    Print the AUC given train and val stats.

    Args:
        train_statistics (list[float]): list of train statistics
        val_statistics (list[float]): list of val statistics
    
    Returns:
        roc_auc (float): ROC-AUC score
    """
    if torch.is_tensor(train_statistic):
        train_statistic = train_statistic.flatten()
    else:
        train_statistic = torch.tensor(train_statistic).flatten()

    train_statistic = train_statistic[~train_statistic.isnan()]
    if torch.is_tensor(val_statistic):
        val_statistic = val_statistic.flatten()
    else:
        val_statistic = torch.tensor(val_statistic).flatten()
    val_statistic = val_statistic[~val_statistic.isnan()]

    fpr, tpr, thresholds = roc_curve(torch.cat((torch.ones_like(train_statistic),torch.zeros_like(val_statistic))).flatten(),
                                    torch.cat((-train_statistic,-val_statistic)).flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc