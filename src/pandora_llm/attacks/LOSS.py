from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM
from .Attack import MIA

####################################################################################################
# MAIN CLASS
####################################################################################################
class LOSS(MIA):
    """
    LOSS thresholding attack
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None

    def compute_statistic(self, dataloader, num_batches=None, device=None, model_half=None, accelerator=None):
        """
        Compute the LOSS statistic for a given dataloader.

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
        return compute_dataloader_cross_entropy(model=self.model,dataloader=dataloader,num_batches=num_batches,device=device,model_half=model_half).cpu()

####################################################################################################
# HELPER FUNCTIONS
####################################################################################################

def compute_input_ids_cross_entropy(model: AutoModelForCausalLM, input_ids: torch.Tensor, return_pt: bool=True, tokens: bool=False):
    """
    Compute the cross-entropy loss between the logits from the model and provided input IDs.

    Args:
        model (transformers.AutoModelForCausalLM): HuggingFace model.
        input_ids (torch.Tensor): tensor of input IDs.
        return_pt (bool): Return tensor or list
        tokens (bool): Flag to consider only one token

    Returns:
        torch.Tensor or list: loss of input IDs
    """

    mask  = (input_ids > 0).detach()                                     

    model.eval()

    with torch.no_grad():
        outputs = model(input_ids=input_ids.to(torch.long), attention_mask = mask)
        logits = outputs.logits
        del outputs
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if tokens:
        loss_fn = CrossEntropyLoss(reduction="none")
    else:
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
    
    if tokens:
        return torch.tensor(ans) if return_pt else ans[0] 
    else:
        return torch.tensor(ans) if return_pt else ans 


def compute_dataloader_cross_entropy(model, dataloader, device=None, num_batches=None, samplelength=None, accelerator=None, model_half=True):    
    '''
    Computes dataloader cross entropy with additional support for specifying the full data loader and full sample length.
    Warning: using samplelength is discouraged

    Args:
        model (transformers.AutoModelForCausalLM): HuggingFace model.
        dataloader (torch.utils.data.dataloader.DataLoader): DataLoader with tokens.
        device (str): CPU or GPU 
        num_batches (int): Number of batches to consider
        samplelength (int or NoneType): cut all samples to a given length
        accelerator (accelerate.Accelerator or NoneType): enable distributed training
        half (bool): use half precision floats for model

    Returns:
        torch.Tensor: loss of input IDs
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
            if isinstance(data_x,dict):
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
