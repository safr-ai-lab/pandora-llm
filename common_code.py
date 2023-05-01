import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from torch.nn import CrossEntropyLoss


## Memory statistics for memory management
def mem_stats():
    t = torch.cuda.get_device_properties(0).total_memory / 1024**3
    r = torch.cuda.memory_reserved(0) / 1024**3
    a = torch.cuda.memory_allocated(0) / 1024**3
    print(f"Total Memory: {t:.2f} GB\n"
          f"Reserved Memory: {r:.2f} GB ({(100*(r/t)):.2f}%)\n"
          f"Remaining Memory: {t-r:.2f} GB ({(100*(t-r)/t):.2f}%)\n"
          f"---------------------------------\n"
          f"Allocated Memory: {a:.2f} GB ({(100*(a/t)):.2f}%)\n"
          f"Percent of Reserved Allocated: {(100*(a+1e-9)/(r+1e-9)):.2f}%\n")

def compute_input_ids_cross_entropy(model, input_ids):
  mask  = (input_ids > 0).detach()                                     

  model.train(False)

  with torch.no_grad():
    outputs = model(input_ids=input_ids.to(torch.long), attention_mask = mask)
    logits = outputs.logits

  loss_fn = CrossEntropyLoss()
  input_len = input_ids.shape[-1] - 1
  input_ids_without_first_token = input_ids[:, 1:].long()
  logits_without_last_token = logits[:, :-1, :]

  # print(input_ids_without_first_token)
  ans = []
  for i in range(len(logits_without_last_token)):
    length = len(input_ids_without_first_token[i,:])
    if len(torch.where(input_ids_without_first_token[i,:] == 0)[0]) > 0:
      length = torch.where(input_ids_without_first_token[i,:] == 0)[0].min()
    # print(logits_without_last_token[i, :length].shape)
    # print(input_ids_without_first_token[i, :length].shape)
    # print(loss_fn(logits_without_last_token[i, :length], input_ids_without_first_token[i, :length]))
    ce_loss = loss_fn(logits_without_last_token[i, :length], input_ids_without_first_token[i, :length])
    ans.append(ce_loss)

  ## Clean up 
  del outputs, logits, input_ids_without_first_token, logits_without_last_token
  torch.cuda.empty_cache()
  torch.cuda.synchronize()

  return torch.Tensor(ans)

def compute_dataloader_cross_entropy(dataloader, nbatches, bs, device, model, samplelength):    
    cross_entropy = torch.zeros((nbatches, bs))
    for batchno, data_x in enumerate(dataloader):
        if batchno >= nbatches:
            break
        with torch.no_grad():       
            ## Get predictions on training data                       
            data_x = data_x[:,:samplelength].to(device).detach()
   
            ## Compute average log likelihood
            cross_entropy[batchno, :] = compute_input_ids_cross_entropy(model, data_x)

        if batchno % 50 == 0:
            print("batch no. ", batchno)  
            print("Memory after evaluation")
            mem_stats()
            print()
    return cross_entropy

def compute_dataloader_cross_entropy_v2(model, dataloader, device, nbatches=None, bs=1, samplelength=None):    
    model = model.to(device)
    if nbatches is None:
        cross_entropy = torch.zeros((len(dataloader), bs))
    else:
        cross_entropy = torch.zeros((nbatches, bs))
    for batchno, data_x in enumerate(dataloader):
        if nbatches is not None and batchno >= nbatches:
            break
        with torch.no_grad():       
            ## Get predictions on training data 
            data_x = data_x["input_ids"]
            if samplelength is None:
                data_x = data_x.to(device).detach()                  
            else:
                data_x = data_x[:,:samplelength].to(device).detach()
   
            ## Compute average log likelihood
            cross_entropy[batchno, :] = compute_input_ids_cross_entropy(model, data_x)
    return cross_entropy


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

def plot_ROC(train_diff, valid_diff, show_plot = True, save_plot = False, log_scale = False, plot_title = "ROC plot", plot_name = "ROC curve.png"):
    import matplotlib.pyplot as plt
    import numpy as np

    # generate two sets of random values
    with torch.no_grad():
        valuestraining   = torch.flatten(train_diff) 
        valuesvalidation = torch.flatten(valid_diff)

    notnan = torch.logical_and(~valuestraining.isnan(), ~valuesvalidation.isnan())
    valuestraining = valuestraining[notnan]
    valuesvalidation = valuesvalidation[notnan]

    ## Scale all values to be between 0 and 1
    st = min(valuestraining.min(),valuesvalidation.min())
    end = max(valuestraining.max(),valuesvalidation.max())
    print(st,end)

    y_scores =  torch.cat((valuestraining, valuesvalidation))
    y_scores = y_scores-min(y_scores)
    y_scores = y_scores/max(y_scores)
    y_true   = [1 for _ in range(len(valuestraining))] + [0 for _ in range(len(valuesvalidation))]

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Calculate the area under the ROC curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    if log_scale:
        plt.loglog(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    else:
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title(plot_title)

    if save_plot:
        plt.savefig(plot_name)
    if show_plot:
        plt.show()