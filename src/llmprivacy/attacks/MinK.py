from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM
from .Attack import MIA
from .LOSS import compute_input_ids_cross_entropy
from ..utils.plot_utils import plot_ROC_multiple, plot_ROC_multiple_plotly, plot_histogram, plot_histogram_plotly

####################################################################################################
# MAIN CLASS
####################################################################################################
class MinK(MIA):
    """
    Min-K thresholding attack
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)

    def compute_statistic_tokens(self, dataloader, num_batches=None, device=None, model_half=None, accelerator=None):
        """
        Compute the LOSS statistic per token for a given dataloader.

        Args:
            dataloader (DataLoader): input data to compute statistic over
            num_batches (Optional[int]): number of batches of the dataloader to compute over.
                If None, then comptues over whole dataloader
            device (Optional[str]): e.g. "cuda"
            model_half (Optional[bool]): whether to use model_half
            accelerator (Optional[Accelerator]): accelerator object
        Returns:
            torch.Tensor or list: loss of input IDs
        """
        if self.model is None:
            raise Exception("Please call .load_model() to load the model first.")
        if accelerator is not None:
            self.model, dataloader, = accelerator.prepare(self.model, dataloader)
        return compute_dataloader_cross_entropy_tokens(model=self.model,dataloader=dataloader,num_batches=num_batches,device=device,model_half=model_half)

    def attack_plot_ROC(self, train_statistics_tokens, val_statistics_tokens, title, k_range=None, log_scale=False, show_plot=False, save_name=None):
        """
        Generates and displays or saves a plot of the ROC curve for the membership inference attack.

        For Min-K, this method plots a series of ROC curves for differing values of k.

        This method uses the inputted statistics to create a ROC curve that 
        illustrates the performance of the attack. The plot can be displayed in a log scale, 
        shown directly, or saved to a file.

        Args:
            train_statistics (Iterable[float]): Statistics of the training set. Lower means more like train.
            val_statistics (Iterable[float]): Statistics of the validation set. Lower means more like train.
            title (str): The title for the ROC plot.
            k_range (list[int], optional): List of k values to plot.
                Defaults to a reasonable set of values: [0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
            log_scale (bool, optional): Whether to plot the ROC curve on a logarithmic scale. 
                Defaults to False.
            show_plot (bool, optional): Whether to display the plot. If False, the plot is not 
                shown but is saved directly to the file specified by `save_name`. Defaults to True.
            save_name (str, optional): The file name or path to save the plot image. If not 
                specified, the default name is generated by the given title with an 
                appropriate file extension. Defaults to None.
        """
        if save_name is None:
            save_name = title + ("_log" if log_scale else "")
        
        train_statistics = []
        val_statistics = []
        k_range = [0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        for k in k_range:
            train_stats = torch.tensor([torch.mean(torch.sort(ex)[0][:int(len(ex)*k)]) for ex in train_statistics_tokens])
            val_stats = torch.tensor([torch.mean(torch.sort(ex)[0][:int(len(ex)*k)]) for ex in val_statistics_tokens])
            train_statistics.append(train_stats)
            val_statistics.append(val_stats)
        plot_ROC_multiple(train_statistics, val_statistics, title, k_range, log_scale=log_scale, show_plot=show_plot, save_name=save_name)
        plot_ROC_multiple_plotly(train_statistics, val_statistics, title, k_range, log_scale=log_scale, show_plot=show_plot, save_name=save_name)
    
    def attack_plot_histogram(self, train_statistics_tokens, val_statistics_tokens, title, k_range=None, normalize=False, show_plot=False, save_name=None):
        """
        Generates and displays or saves a histogram of the statistics.

        For MinK, this method plots each histogram in a separate image.

        Args:
            train_statistics (Iterable[float]): Statistics of the training set. Lower means more like train.
            val_statistics (Iterable[float]): Statistics of the validation set. Lower means more like train.
            title (str): The title for the ROC plot.
            k_range (list[int], optional): List of k values to plot.
                Defaults to a reasonable set of values: [0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
            normalize (bool, optional): Whether to plot the histograms on a z-scored scale. 
                Defaults to False.
            show_plot (bool, optional): Whether to display the plot. If False, the plot is not 
                shown but is saved directly to the file specified by `save_name`. Defaults to True.
            save_name (str, optional): The file name or path to save the plot image. If not 
                specified, the default name is generated by the given title with an 
                appropriate file extension. Defaults to None.
        """
        if save_name is None:
            save_name = title + ("_z" if normalize else "")
        
        k_range = [0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        for k in k_range:
            train_statistics = torch.tensor([torch.mean(torch.sort(ex)[0][:int(len(ex)*k)]) for ex in train_statistics_tokens])
            val_statistics = torch.tensor([torch.mean(torch.sort(ex)[0][:int(len(ex)*k)]) for ex in val_statistics_tokens])
            plot_histogram(train_statistics, val_statistics, title+f"_{k}", normalize=normalize, show_plot=show_plot, save_name=save_name+f"_{k}")
            plot_histogram_plotly(train_statistics, val_statistics, title+f"_{k}", normalize=normalize, show_plot=show_plot, save_name=save_name+f"_{k}")

####################################################################################################
# HELPER FUNCTIONS
####################################################################################################

def compute_dataloader_cross_entropy_tokens(model, dataloader, device=None, num_batches=None, samplelength=None, accelerator=None, model_half=True):    
    '''
    Computes dataloader cross entropy with additional support for specifying the full data loader and full sample length.
    Warning: using samplelength is discouraged

    Args:
        model (transformers.AutoModelForCausalLM): HuggingFace model.
        dataloader (torch.utils.data.dataloader.DataLoader): DataLoader with tokens.
        device (str): CPU or GPU 
        nbatches (int): Number of batches to consider
        samplelength (int or NoneType): cut all samples to a given length
        accelerator (accelerate.Accelerator or NoneType): enable distributed training
        half (bool): use half precision floats for model

    Returns:
        torch.Tensor or list: loss of input IDs
    '''

    if samplelength is not None:
        print("Warning: using sample length is discouraged. Please avoid using this parameter.")
    if accelerator is None:
        if model_half:
            print("Using model.half() ....")
            model.half()
        else:
            print("Not using model.half() ....")
        model.eval()
        model.to(device)

    losses = []
    for batchno, data_x in tqdm(enumerate(dataloader),total=len(dataloader)):
        if num_batches is not None and batchno >= num_batches:
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
                loss = compute_input_ids_cross_entropy(model, data_x.to(device), return_pt=False, tokens=True).detach().cpu()
            else:
                loss = compute_input_ids_cross_entropy(model, data_x, return_pt=False, tokens=True)

            losses.append(loss)

            del data_x
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    if accelerator is None:
        return losses
    else:
        losses = accelerator.gather_for_metrics(losses)
        losses = torch.cat([loss[0] for loss in losses])
        return losses