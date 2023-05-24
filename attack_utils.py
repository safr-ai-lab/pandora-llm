import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

def mem_stats():
    '''
    Memory statistics for memory management
    '''
    t = torch.cuda.get_device_properties(0).total_memory / 1024**3
    r = torch.cuda.memory_reserved(0) / 1024**3
    a = torch.cuda.memory_allocated(0) / 1024**3
    print(f"Total Memory: {t:.2f} GB\n"
          f"Reserved Memory: {r:.2f} GB ({(100*(r/t)):.2f}%)\n"
          f"Remaining Memory: {t-r:.2f} GB ({(100*(t-r)/t):.2f}%)\n"
          f"---------------------------------\n"
          f"Allocated Memory: {a:.2f} GB ({(100*(a/t)):.2f}%)\n"
          f"Percent of Reserved Allocated: {(100*(a+1e-9)/(r+1e-9)):.2f}%\n")

def compute_input_ids_cross_entropy(model, input_ids, return_pt=True):
  mask  = (input_ids > 0).detach()                                     

  model.eval()

  with torch.no_grad():
    outputs = model(input_ids=input_ids.to(torch.long), attention_mask = mask)
    logits = outputs.logits
    del outputs
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

  loss_fn = CrossEntropyLoss()
  input_ids_without_first_token = input_ids[:, 1:].long()
  logits_without_last_token = logits[:, :-1, :]

  ans = []
  for i in range(len(logits_without_last_token)):
    length = len(input_ids_without_first_token[i,:])
    if len(torch.where(input_ids_without_first_token[i,:] == 0)[0]) > 0:
      length = torch.where(input_ids_without_first_token[i,:] == 0)[0].min()
    ce_loss = loss_fn(logits_without_last_token[i, :length], input_ids_without_first_token[i, :length])
    ans.append(ce_loss)

  ## Clean up 
  del logits, input_ids_without_first_token, logits_without_last_token
  torch.cuda.empty_cache()
  torch.cuda.synchronize()

  return torch.tensor(ans) if return_pt else ans 

def compute_dataloader_cross_entropy(model, dataloader, device=None, nbatches=None, samplelength=None, accelerator=None):    
    '''
    Computes dataloader cross entropy with additional support for specifying the full data loader and full sample length.
    Warning: using samplelength is discouraged
    '''
    if samplelength is not None:
        print("Warning: using sample length is discouraged. Please avoid using this parameter.")
    if accelerator is None:
        model.half()
        model.eval()
        model.to(device)
    losses = []
    for batchno, data_x in tqdm(enumerate(dataloader),total=len(dataloader)):
        if nbatches is not None and batchno >= nbatches:
            break
        with torch.no_grad():    
            ## Get predictions on training data 
            data_x = data_x["input_ids"]
            if samplelength is None:
                data_x = data_x.detach()                
            else:
                data_x = data_x[:,:samplelength].detach()
   
            ## Compute average log likelihood
            if accelerator is None:
                loss = compute_input_ids_cross_entropy(model, data_x.to(device)).detach().cpu()
            else:
                loss = compute_input_ids_cross_entropy(model, data_x, return_pt = False)

            losses.append(loss)

            del data_x
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    if accelerator is None:
        return torch.tensor(losses)
    else:
        losses = accelerator.gather_for_metrics(losses)
        losses = torch.cat([loss[0] for loss in losses])
        return losses


def plot_hist(train_perplexity, val_perplexity, show_plot = True, save_plot=False, plot_title = "Histogram", plot_name="hist.png"):
    
    # generate two sets of random values
    with torch.no_grad():
        valuestraining   = torch.flatten(train_perplexity) 
        valuesvalidation = torch.flatten(val_perplexity)

    ## Remove nan values (usually in validation set, i.e. really low prob)
    notnan = torch.logical_and(~valuestraining.isnan(), ~valuesvalidation.isnan())
    valuestraining = valuestraining[notnan]
    valuesvalidation = valuesvalidation[notnan]

    # create a figure and axis object
    fig, ax = plt.subplots()

    # plot a histogram of the first set of values with 20 bins
    ax.hist(valuestraining, bins=20, alpha=0.5, label='training')

    # plot a histogram of the second set of values with 20 bins
    ax.hist(valuesvalidation, bins=20, alpha=0.5, label='validation')

    # add a legend to the plot
    ax.legend(loc='upper right')

    # add labels and a title to the plot
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title(plot_title)

    # show the plot
    if save_plot:
        plt.savefig(plot_name)
    if show_plot:
        plt.show()

def plot_ROC(train_statistic,val_statistic,title,log_scale=False,show_plot=True,save_name=None):
    '''
    Plots ROC with train and validation test statistics. Note that we assume train statistic < test statistic. Negate before using if otherwise.
    '''
    train_statistic = torch.tensor(train_statistic).flatten()
    train_statistic = train_statistic[~train_statistic.isnan()]
    val_statistic = torch.tensor(val_statistic).flatten()
    val_statistic = val_statistic[~val_statistic.isnan()]

    fpr, tpr, thresholds = roc_curve(torch.cat((torch.ones_like(train_statistic),torch.zeros_like(val_statistic))).flatten(),
                                    torch.cat((-train_statistic,-val_statistic)).flatten())
    roc_auc = auc(fpr, tpr)
    plt.figure()
    if not log_scale:
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.4f)' % roc_auc)
    else:
        plt.loglog(fpr, tpr, color='darkorange', label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if save_name is not None:
        plt.savefig(save_name, bbox_inches="tight")
    if show_plot:
        plt.show()

def plot_ROC_multiple(train_statistics,val_statistics,title,labels,log_scale=False,show_plot=True,save_name=None):
    '''
    Plots multiple ROC curves in a single plot
    '''
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--')
    for train_statistic, val_statistic, label in zip(train_statistics, val_statistics,labels):
        train_statistic = torch.tensor(train_statistic).flatten()
        train_statistic = train_statistic[~train_statistic.isnan()]
        val_statistic = torch.tensor(val_statistic).flatten()
        val_statistic = val_statistic[~val_statistic.isnan()]

        fpr, tpr, thresholds = roc_curve(torch.cat((torch.ones_like(train_statistic),torch.zeros_like(val_statistic))).flatten(),
                                        torch.cat((-train_statistic,-val_statistic)).flatten())
        roc_auc = auc(fpr, tpr)
        if not log_scale:
            plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:0.4f})')
        else:
            plt.loglog(fpr, tpr, label=f'{label} (AUC = {roc_auc:0.4f})')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if save_name is not None:
        plt.savefig(save_name, bbox_inches="tight")
    if show_plot:
        plt.show()

def plot_ROC_files(files,title,labels=None,log_scale=False,show_plot=True,save_name=None):
    '''
    Plots ROCs from saved statistic .pt files
    '''
    train_statistics = []
    val_statistics = []
    for file in files:
        t_stat, v_stat = torch.load(file)
        train_statistics.append(t_stat)
        val_statistics.append(v_stat)
    if labels is None:
        labels = files
    plot_ROC_multiple(train_statistics,val_statistics,title,labels,log_scale=log_scale,show_plot=show_plot,save_name=save_name)
