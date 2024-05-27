import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import bootstrap
from sklearn.metrics import roc_curve, auc
import torch

def plot_histogram(train_statistics, val_statistics, plot_title, keep_first=None, bins=None, normalize=False, show_plot=True, save_name=None):
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

    if normalize:
        sigma, mu = torch.std_mean(torch.cat((train_statistics,val_statistics)))
        train_statistics = (train_statistics-mu)/sigma
        val_statistics = (val_statistics-mu)/sigma

    # Compute bins    
    if bins is None: # use max number of bins by fd and sturges rule
        train_len = len(train_statistics)
        train_iqr = np.subtract(*np.percentile(train_statistics, [75, 25]))
        train_binwidth = min(2.0 * train_iqr * train_len ** (-1.0 / 3.0), np.ptp(train_statistics)/(np.log2(train_len) + 1.0))
        train_bins = int(math.ceil(np.ptp(train_statistics)/train_binwidth))
        val_len = len(val_statistics)
        val_iqr = np.subtract(*np.percentile(val_statistics, [75, 25]))
        val_binwidth = min(2.0 * val_iqr * val_len ** (-1.0 / 3.0), np.ptp(val_statistics)/(np.log2(val_len) + 1.0))
        val_bins = int(math.ceil(np.ptp(val_statistics)/val_binwidth))
        bins = int(1.*max(train_bins,val_bins))

    train_min, train_max = train_statistics.min().item(), train_statistics.max().item()
    val_min, val_max = val_statistics.min().item(), val_statistics.max().item()
    train_bin_width = (train_max - train_min) / bins
    val_bin_width = (val_max - val_min) / bins
    bin_width = min(train_bin_width, val_bin_width)
    combined_min = min(train_min, val_min)
    combined_max = max(train_max, val_max)
    combined_bin_edges = np.arange(combined_min, combined_max + bin_width, bin_width)

    # Plot
    plt.figure()
    plt.hist(train_statistics, bins=combined_bin_edges, alpha=0.5, edgecolor='black', label='Train')
    plt.hist(val_statistics, bins=combined_bin_edges, alpha=0.5, edgecolor='black', label='Validation')
    plt.legend(loc='upper right')
    plt.xlabel('Normalized Attack Statistic' if normalize else 'Attack Statistic')
    plt.ylabel('Frequency')
    plt.title(plot_title)
    plt.minorticks_on()
    plt.grid(which="major",alpha=0.2)
    plt.grid(which="minor",alpha=0.1)
    if save_name is not None:
        plt.savefig(save_name+"_hist.png", bbox_inches="tight")
        plt.savefig(save_name+"_hist.pdf", bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()

def plot_histogram_plotly(train_statistics, val_statistics, plot_title, keep_first=None, bins=None, normalize=False, show_plot=True, save_name=None):
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

    if normalize:
        sigma, mu = torch.std_mean(torch.cat((train_statistics,val_statistics)))
        train_statistics = (train_statistics-mu)/sigma
        val_statistics = (val_statistics-mu)/sigma

    # Compute bins    
    if bins is None: # use max number of bins by fd and sturges rule
        train_len = len(train_statistics)
        train_iqr = np.subtract(*np.percentile(train_statistics, [75, 25]))
        train_binwidth = min(2.0 * train_iqr * train_len ** (-1.0 / 3.0), np.ptp(train_statistics)/(np.log2(train_len) + 1.0))
        train_bins = int(math.ceil(np.ptp(train_statistics)/train_binwidth))
        val_len = len(val_statistics)
        val_iqr = np.subtract(*np.percentile(val_statistics, [75, 25]))
        val_binwidth = min(2.0 * val_iqr * val_len ** (-1.0 / 3.0), np.ptp(val_statistics)/(np.log2(val_len) + 1.0))
        val_bins = int(math.ceil(np.ptp(val_statistics)/val_binwidth))
        bins = int(1.*max(train_bins,val_bins))

    train_min, train_max = train_statistics.min().item(), train_statistics.max().item()
    val_min, val_max = val_statistics.min().item(), val_statistics.max().item()
    train_bin_width = (train_max - train_min) / bins
    val_bin_width = (val_max - val_min) / bins
    bin_width = min(train_bin_width, val_bin_width)
    combined_min = min(train_min, val_min)
    combined_max = max(train_max, val_max)
    combined_bin_edges = np.arange(combined_min, combined_max + bin_width, bin_width)

    # Plot
    fig = make_subplots(rows=3, cols=1, row_heights=[0.1, 0.4, 0.55], shared_xaxes=True, vertical_spacing=0.02)

    # Rug plots
    fig.add_trace(go.Box(
        x=val_statistics.numpy(), 
        marker_symbol='line-ns-open', 
        marker_color='#ff7f0e',
        boxpoints='all',
        jitter=0,
        fillcolor='rgba(255,255,255,0)',
        line_color='rgba(255,255,255,0)',
        hoveron='points',
        showlegend=False,
        name='Validation'
    ), row=1, col=1)
    fig.add_trace(go.Box(
        x=train_statistics.numpy(), 
        marker_symbol='line-ns-open', 
        marker_color='#1f77b4',
        boxpoints='all',
        jitter=0,
        fillcolor='rgba(255,255,255,0)',
        line_color='rgba(255,255,255,0)',
        hoveron='points',
        showlegend=False,
        name='Train'
    ), row=1, col=1)

    # Violin plots
    fig.add_trace(go.Violin(
        x=val_statistics.numpy(), 
        line_color='#ff7f0e',
        box_visible=True,
        meanline_visible=True,
        showlegend=False,
        name='Validation',
    ), row=2, col=1)
    fig.add_trace(go.Violin(
        x=train_statistics.numpy(), 
        line_color='#1f77b4',
        box_visible=True,
        meanline_visible=True,
        showlegend=False,
        name='Train',
    ), row=2, col=1)

    # Histograms
    fig.add_trace(go.Histogram(
        x=train_statistics.numpy(), 
        nbinsx=len(combined_bin_edges)-1, 
        name='Train', 
        opacity=0.5, 
        marker_color='#1f77b4', 
        marker_line_color='black',
        marker_line_width=1.5,
        xbins=dict(start=combined_min, end=combined_max, size=bin_width)
    ), row=3, col=1)
    fig.add_trace(go.Histogram(
        x=val_statistics.numpy(), 
        nbinsx=len(combined_bin_edges)-1, 
        name='Validation', 
        opacity=0.5, 
        marker_color='#ff7f0e', 
        marker_line_color='black',
        marker_line_width=1.5,
        xbins=dict(start=combined_min, end=combined_max, size=bin_width)
    ), row=3, col=1)

    fig.update_layout(
        title=plot_title,
        width=800,
        height=800,
        xaxis3_title='Normalized Attack Statistic' if normalize else 'Attack Statistic',
        yaxis3_title='Frequency',
        yaxis=dict(range=[-1, 1.4], tickvals=[0.65,-0.35], ticktext=['Train','Validation'], tickmode='array'),
        xaxis1=dict(showticklabels=False, minor=dict(showgrid=True, ticklen=0)),
        xaxis2=dict(showticklabels=False, minor=dict(ticklen=0, showgrid=True)),
        xaxis3=dict(showticklabels=True, showgrid=True, minor=dict(ticklen=0, showgrid=True)),
        yaxis3=dict(showticklabels=True, minor=dict(ticklen=0, showgrid=True)),
        barmode='overlay'
    )

    if save_name is not None:
        fig.write_image(save_name + "_hist_plotly.png", scale=5)
        fig.write_image(save_name + "_hist_plotly.pdf", scale=5)
        fig.write_html(save_name + "_hist_plotly.html")
    if show_plot:
        fig.show()

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
    plt.close()
    if ci:
        return roc_auc, tpr_at_fprs, auc_se, tpr_se
    else:
        return roc_auc, tpr_at_fprs

def plot_ROC_plotly(train_statistics, val_statistics, plot_title, keep_first=None, ci=True, num_bootstraps=1000, fprs=None, log_scale=False, show_plot=True, save_name=None, lims=None, color='darkorange'):
    '''
    Plots ROC curve with train and validation test statistics. Also saves TPRs at FPRs. Uses plotly.
    
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
    color = mcolors.to_hex(color)
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
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='black'), showlegend=False))
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:0.4f}', line=dict(color=color),legendgroup=0))
    if ci:
        fig.add_trace(go.Scatter(
            x=fpr_range,
            y=bootstrap_result.confidence_interval.low[1:],
            fill=None,
            mode='lines',
            line=dict(color=color, width=0),
            legendgroup=0,
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=fpr_range,
            y=bootstrap_result.confidence_interval.high[1:],
            fill='tonexty',
            mode='lines',
            line=dict(color=color, width=0),
            showlegend=False,
            legendgroup=0,
            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:], 16)}, 0.1)'
        ))
    fig.update_layout(
        title=plot_title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(
            range=([-int(np.log10(n_points)), 0] if log_scale else [0,1]) if lims is None else lims,
            type='log' if log_scale else 'linear',
            constrain='domain',
            tickmode = 'linear',
            tick0 = -int(np.log10(n_points)) if log_scale else 0,
            dtick = 1 if log_scale else 0.2,
            minor=dict(ticks="inside", ticklen=0, showgrid=True)
        ),
        yaxis=dict(
            range=([-int(np.log10(n_points)), 0] if log_scale else [0,1]) if lims is None else lims,
            type='log' if log_scale else 'linear',
            scaleanchor='x',
            tickmode = 'linear',
            tick0 = -int(np.log10(n_points)) if log_scale else 0,
            dtick = 1 if log_scale else 0.2,
            minor=dict(ticks="inside", ticklen=0, showgrid=True)
        ),
        showlegend=True,
    )

    if save_name is not None:
        fig.write_image(save_name + "_roc_plotly.png")
        fig.write_image(save_name + "_roc_plotly.pdf")
        fig.write_html(save_name + "_roc_plotly.html")
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
    if save_name is not None:
        plt.savefig(save_name+"_roc.png", bbox_inches="tight")
        plt.savefig(save_name+"_roc.pdf", bbox_inches="tight")
        df = pd.DataFrame([roc_auc_map,auc_se_map,tpr_at_fprs_map,tpr_se_map]).T
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
        return roc_auc_map, tpr_at_fprs_map, auc_se_map, tpr_se_map
    else:
        return roc_auc_map, tpr_at_fprs_map

def plot_ROC_multiple_plotly(train_statistics_list, val_statistics_list, plot_title, labels, keep_first=None, ci=True, num_bootstraps=1000, fprs=None, log_scale=False, show_plot=True, save_name=None, lims=None, colors=None, bold_labels=[]):
    '''
    Plots multiple ROC curves in a single plot. Uses plotly

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
    colors = [mcolors.to_hex(color) for color in colors]

    roc_auc_map = {}
    tpr_at_fprs_map = {}
    auc_se_map = {}
    tpr_se_map = {}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='black'), showlegend=False))
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
        
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{label} (AUC = {roc_auc:0.4f})' if i not in bold_labels else f'<b>{label} (AUC = {roc_auc:0.4f})</b>', line=dict(color=colors[i]), legendgroup=i))

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
            fig.add_trace(go.Scatter(
                x=fpr_range,
                y=bootstrap_result.confidence_interval.low[1:],
                fill=None,
                mode='lines',
                line=dict(color=colors[i], width=0),
                showlegend=False,
                legendgroup=i,
            ))
            fig.add_trace(go.Scatter(
                x=fpr_range,
                y=bootstrap_result.confidence_interval.high[1:],
                fill='tonexty',
                mode='lines',
                line=dict(color=colors[i], width=0),
                showlegend=False,
                legendgroup=i,
                fillcolor=f'rgba({int(colors[i][1:3], 16)}, {int(colors[i][3:5], 16)}, {int(colors[i][5:], 16)}, 0.1)'
            ))
            auc_se_map[label] = bootstrap_result.standard_error[0]
            tpr_se_map[label] = [bootstrap_result.standard_error[np.max(np.argwhere(fpr_range<=fpr_val))] for fpr_val in fprs]        
        roc_auc_map[label] = roc_auc
        tpr_at_fprs_map[label] = [tpr[np.max(np.argwhere(fpr<=fpr_val))] for fpr_val in fprs]
    fig.update_layout(
        title=plot_title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(
            range=([-int(np.log10(n_points)), 0] if log_scale else [0,1]) if lims is None else lims,
            type='log' if log_scale else 'linear',
            constrain='domain',
            tickmode = 'linear',
            tick0 = -int(np.log10(n_points)) if log_scale else 0,
            dtick = 1 if log_scale else 0.2,
            minor=dict(ticks="inside", ticklen=0, showgrid=True)
        ),
        yaxis=dict(
            range=([-int(np.log10(n_points)), 0] if log_scale else [0,1]) if lims is None else lims,
            type='log' if log_scale else 'linear',
            scaleanchor='x',
            tickmode = 'linear',
            tick0 = -int(np.log10(n_points)) if log_scale else 0,
            dtick = 1 if log_scale else 0.2,
            minor=dict(ticks="inside", ticklen=0, showgrid=True)
        ),
        showlegend=True,
    )
    if save_name is not None:
        fig.write_image(save_name + "_roc_plotly.png",scale=5)
        fig.write_image(save_name + "_roc_plotly.pdf",scale=5)
        fig.write_html(save_name + "_roc_plotly.html")
        df = pd.DataFrame([roc_auc_map,auc_se_map,tpr_at_fprs_map,tpr_se_map]).T
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