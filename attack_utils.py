import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import pdb
from detect_gpt_utils import *
import timeit


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

def compute_input_ids_cross_entropy_batch(model, input_ids, return_pt=True):
  model.eval()

  mask_batch  = (input_ids > 0).detach() 

  with torch.no_grad():
    outputs = model(input_ids=input_ids.to(torch.long).squeeze(-1), attention_mask = mask_batch.squeeze(-1))
    logits = outputs.logits
    del outputs
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

  loss_fn = CrossEntropyLoss()

  # convert to long type as cross entropy will expect integer indices 
  input_ids_without_first_token = input_ids[:, 1:].long()
  # drop the last token because it is an EOS token?
  logits_without_last_token_batch = logits[:, :-1, :]

  ans = []
  # loop through each example in the batch 
  for i in range(len(logits_without_last_token_batch)):
    # only compute the cross entropy loss up until the first input_id = 0 (why? is this padding?)
    if len(torch.where(input_ids_without_first_token[i,:,:] == 0)[0]) > 0:
        length = torch.where(input_ids_without_first_token[i,:,:] == 0)[0].min()
    else: 
        length = len(input_ids_without_first_token[i,:,:])
    
    # truncate the logits & input_ids to length prior to computing CE loss
    ce_loss = loss_fn(logits_without_last_token_batch[i, :length, :], input_ids_without_first_token[i, :length].squeeze(-1))
    ans.append(ce_loss)

  ## Clean up 
  del logits, input_ids_without_first_token, logits_without_last_token_batch
  torch.cuda.empty_cache()
  torch.cuda.synchronize()

  return torch.mean(torch.tensor(ans)) if return_pt else np.mean(ans) 

def compute_dataloader_cross_entropy_batch(model, dataloader, device=None, nbatches=None, samplelength=None, accelerator=None, half=True, detect_args=None):    
    '''
    Computes dataloader cross entropy with additional support for specifying the full data loader and full sample length.
    Warning: using samplelength is discouraged
    '''
    if samplelength is not None:
        print("Warning: using sample length is discouraged. Please avoid using this parameter.")
    if accelerator is None:
        if half:
            print("Using model.half() ....")
            model.half()
        else:
            print("Not using model.half() ....")
        model.eval()
        model.to(device)

    losses = []
    model_name = 't5-small'
    mask_model = T5ForConditionalGeneration.from_pretrained(model_name)
    mask_model.to(device)
    mask_tokenizer = T5Tokenizer.from_pretrained(model_name)
    base_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
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
            

            # time this 
            start_pert = timeit.default_timer()
            data_x_batch = perturb_input_ids(data_x.squeeze(0).to(device), detect_args, base_tokenizer, mask_tokenizer, mask_model).unsqueeze(-1)
            end_pert = timeit.default_timer()
            elapsed_time = end_pert - start_pert 
            print(f'time to perturb input is {elapsed_time} seconds')

            ## Compute average log likelihood
            if accelerator is None:
                avg_perturbed_loss = compute_input_ids_cross_entropy_batch(model, data_x_batch.to(device)).detach().cpu()
                loss = compute_input_ids_cross_entropy(model, data_x.to(device)).detach().cpu()
                detect_gpt_score = loss - avg_perturbed_loss
            else:
                avg_perturbed_loss = compute_input_ids_cross_entropy_batch(model, data_x_batch, return_pt = False)
                loss = compute_input_ids_cross_entropy(model, data_x, return_pt = False)
                detect_gpt_score = loss - avg_perturbed_loss

            losses.append(detect_gpt_score)
            del data_x_batch, data_x
            
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # cleanup
    del mask_model, mask_tokenizer, base_tokenizer

    if accelerator is None:
        return torch.tensor(losses)
    else:
        losses = accelerator.gather_for_metrics(losses)
        losses = torch.cat([loss[0] for loss in losses])
        return losses


def compute_dataloader_cross_entropy(model, dataloader, device=None, nbatches=None, samplelength=None, accelerator=None, half=True):    
    '''
    Computes dataloader cross entropy with additional support for specifying the full data loader and full sample length.
    Warning: using samplelength is discouraged
    '''
    if samplelength is not None:
        print("Warning: using sample length is discouraged. Please avoid using this parameter.")
    if accelerator is None:
        if half:
            print("Using model.half() ....")
            model.half()
        else:
            print("Not using model.half() ....")
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
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.4f)' % roc_auc)
        plt.xscale("log",base=10,subs=list(range(11)))
        plt.yscale("log",base=10,subs=list(range(11)))
        # plt.xscale("symlog",base=10,subs=list(range(11)),linthresh=1e-3,linscale=0.25)
        # plt.yscale("symlog",base=10,subs=list(range(11)),linthresh=1e-3,linscale=0.25)
        plt.xlim(9e-4,1.1)
        plt.ylim(9e-4,1.1)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    print(f"AUC of Experiment {title}\n{roc_auc}")
    if save_name is not None:
        if "png" not in save_name:
            save_name = save_name + ".png"
        plt.savefig(save_name, bbox_inches="tight")
    if show_plot:
        plt.show()

def plot_ROC_multiple(train_statistics,val_statistics,title,labels,log_scale=False,show_plot=True,save_name=None,keep_first=1000):
    '''
    Plots multiple ROC curves in a single plot
    '''
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--')
    for train_statistic, val_statistic, label in zip(train_statistics, val_statistics,labels):
        train_statistic = torch.tensor(train_statistic).flatten()[:keep_first]
        assert(len(train_statistic)==keep_first)
        train_statistic = train_statistic[~train_statistic.isnan()]
        val_statistic = torch.tensor(val_statistic).flatten()[:keep_first]
        assert(len(val_statistic)==keep_first)
        val_statistic = val_statistic[~val_statistic.isnan()]

        fpr, tpr, thresholds = roc_curve(torch.cat((torch.ones_like(train_statistic),torch.zeros_like(val_statistic))).flatten(),
                                        torch.cat((-train_statistic,-val_statistic)).flatten())
        roc_auc = auc(fpr, tpr)
        if not log_scale:
            plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:0.4f})')
        else:
            plt.loglog(fpr, tpr, label=f'{label} (AUC = {roc_auc:0.4f})')
    print(title)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if save_name is not None:
        if "png" not in save_name:
            save_name = save_name + ".png"
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

def rademacher(shape):
    return 2*(torch.rand(shape) > 0.5) - 1



def flat_grad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g

def compute_point_probe(model, data_x, probe, device):                         
    probe_values = torch.zeros((data_x.size(0)))
    loss_fn = CrossEntropyLoss()

    for i in range(data_x.size(0)):
        input_ids = data_x[i:(i+1),:].to(device)
        mask  = (input_ids > 0).detach()       
        outputs =  model(input_ids=input_ids.to(torch.long), attention_mask = mask.to(device))
        logits = outputs.logits 

        length = len(input_ids[0,1:])
        if len(torch.where(input_ids[0,1:] == 0)[0]) > 0:
            length = torch.where(input_ids[0,1:] == 0)[0].min()

        ce_loss = loss_fn(logits[0,:length,:], input_ids[0, 1:length+1])
        ce_loss.backward(retain_graph=True)

        grad = flat_grad(ce_loss, model.parameters(), create_graph=True)  
        grad_v = flat_grad(grad.dot(probe), model.parameters(), retain_graph=True)  
        probe_values[i] = grad_v.dot(probe)
    return probe_values

def compute_dataloader_probe(model, dataloader, probe, device=None, nbatches=None, samplelength=None, accelerator=None, half=False):    
    '''
    Computes z^THz where H is the Hessian for a probe z
    Warning: using samplelength is discouraged
    '''
    if samplelength is not None:
        print("Warning: using sample length is discouraged. Please avoid using this parameter.")
    if accelerator is None:
        if half:
            model.half()
        model.to(device)

    trace_estimates = []
    for batchno, data_x in tqdm(enumerate(dataloader),total=len(dataloader)):
        if nbatches is not None and batchno >= nbatches:
            break
        ## Get predictions on training data 
        data_x = data_x["input_ids"]
        if samplelength is None:
            data_x = data_x.detach()                
        else:
            data_x = data_x[:,:samplelength].detach()

        ## Compute v^THv 
        if accelerator is None:
            est = compute_point_probe(model, data_x, probe, device).detach().cpu()
        else:
            est = compute_point_probe(model, data_x, probe, device).tolist()

        model.zero_grad()
        trace_estimates.append(est)
            
    
    if accelerator is None:
        return torch.tensor(trace_estimates)
    else:
        trace_estimates = accelerator.gather_for_metrics(trace_estimates)
        trace_estimates = torch.cat([loss[0] for loss in trace_estimates])
        return trace_estimates

def z_standardize_together(tensor1, tensor2):
    tensor_concat = torch.cat((tensor1,tensor2))
    tensor1 = (tensor1-tensor_concat.nanmean())/np.nanstd(tensor_concat)
    tensor2 = (tensor2-tensor_concat.nanmean())/np.nanstd(tensor_concat)
    return tensor1, tensor2

def approx_log_scale(x):
    return torch.log(1+torch.abs(x))*torch.sign(x)