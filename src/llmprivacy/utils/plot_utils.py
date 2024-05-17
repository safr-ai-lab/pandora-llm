import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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

def plot_ROC(train_statistics, val_statistics, plot_title, keep_first=None, show_plot=True, save_name=None, log_scale=False, fprs=None):
    '''
    Plots ROC curve with train and validation test statistics. Also saves TPRs at FPRs.
    
    **Note that we assume train statistic < test statistic. Negate before using if otherwise.**

    Args:
        train_statistics (list[float]): list of train statistics
        val_statistics (list[float]): list of val statistics
        plot_title (str): title of the plot
        keep_first (int): compute only for the first keep_first number of samples
        show_plot (bool): whether to show the plot
        save_name (str): save path for plot and scores (without extension); does not save unless save_name is specified
        log_scale (bool): whether to plot on log-log scale
        fprs (list[float]): return TPRs at given FPRs. If unspecified, calculates at every 0.1 increment
    
    Returns:
        auc (float): the ROC-AUC score
        tpr_at_fprs (list[float]): the tprs at the given fprs
    '''
    # Preprocess
    train_statistics = torch.as_tensor(train_statistics).flatten()[:keep_first]
    train_statistics = train_statistics[~train_statistics.isnan()]
    val_statistics = torch.as_tensor(val_statistics).flatten()[:keep_first]
    val_statistics = val_statistics[~val_statistics.isnan()]

    # Compute ROC
    fpr, tpr, thresholds = roc_curve(torch.cat((torch.ones_like(train_statistics),torch.zeros_like(val_statistics))).flatten(),
                                    torch.cat((-train_statistics,-val_statistics)).flatten())
    roc_auc = auc(fpr, tpr)
    
    # Process FPRs
    if fprs is None:
        fprs = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        # fprs = np.arange(0,1.1,0.1)
    tpr_at_fprs = [tpr[np.max(np.argwhere(fpr<=fpr_val))] for fpr_val in fprs]

    # Plot
    plt.figure()
    if not log_scale:
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.4f)' % roc_auc)
    else:
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.4f)' % roc_auc)
        plt.xscale("log",base=10,subs=list(range(11)))
        plt.yscale("log",base=10,subs=list(range(11)))
        plt.xlim(9e-4,1.1)
        plt.ylim(9e-4,1.1)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title(plot_title)
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if save_name is not None:
        plt.savefig(save_name+"_roc.png", bbox_inches="tight")
        with open(save_name+"_tfprs.txt", "w") as f:
            print(f"AUC of Experiment {plot_title}\n{roc_auc}\nTPRs\n{tpr_at_fprs}\nFPRs\n{fprs}")
            f.write(f"AUC of Experiment {plot_title}\n{roc_auc}\nTPRs\n{tpr_at_fprs}\nFPRs\n{fprs}")
    if show_plot:
        plt.show()
    return roc_auc, tpr_at_fprs

def plot_ROC_multiple(train_statistics_list, val_statistics_list, plot_title, labels, keep_first=None, show_plot=True, save_name=None, log_scale=False, fprs=None):
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
    
    Returns:
        roc_auc_map (dict[str,float]): map of labels to auc
        tpr_at_fprs_map (dict[str,list[float]]): map of labels to the tprs at the given fprs
    '''
    if fprs is None:
        fprs = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        # fprs = np.arange(0,1.1,0.1)
    
    roc_auc_map = {}
    tpr_at_fprs_map = {}
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--')
    for train_statistics, val_statistics, label in zip(train_statistics_list, val_statistics_list,labels):
        train_statistics = torch.as_tensor(train_statistics).flatten()[:keep_first]
        train_statistics = train_statistics[~train_statistics.isnan()]
        val_statistics = torch.as_tensor(val_statistics).flatten()[:keep_first]
        val_statistics = val_statistics[~val_statistics.isnan()]

        fpr, tpr, thresholds = roc_curve(torch.cat((torch.ones_like(train_statistics),torch.zeros_like(val_statistics))).flatten(),
                                        torch.cat((-train_statistics,-val_statistics)).flatten())
        roc_auc = auc(fpr, tpr)
        if not log_scale:
            plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:0.4f})')
        else:
            plt.loglog(fpr, tpr, label=f'{label} (AUC = {roc_auc:0.4f})')

        roc_auc_map[label] = roc_auc
        tpr_at_fprs_map[label] = [tpr[np.max(np.argwhere(fpr<=fpr_val))] for fpr_val in fprs]

    plt.title(plot_title)
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([0.001,1])
    plt.ylim([0.001,1])
    if save_name is not None:
        if save_name[-4:] != ".png":
            save_name = save_name + ".png"
        plt.savefig(save_name, bbox_inches="tight")
    if show_plot:
        plt.show()
    return roc_auc_map, tpr_at_fprs_map

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