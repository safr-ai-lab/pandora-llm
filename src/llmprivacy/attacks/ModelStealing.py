from tqdm import tqdm
import torch
import torch.nn.functional as F
from einops import einsum
from transformers import AutoModelForCausalLM
from trak.projectors import CudaProjector, ProjectionType
from .Attack import MIA

####################################################################################################
# MAIN CLASS
####################################################################################################
class ModelStealing(MIA):
    """
    Model stealing attack
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_model_stealing(self, 
        svd_dataloader=None,
        project_file=None,
        proj_type="rademacher",
        proj_dim=512,
        proj_seed=229,
        model_half=False,
        device=None,
        saveas=None
    ):
        """
        Compute the embedding projection layer for the gray-box model-stealing attack

        Args:
            svd_dataloader (DataLoader): input data to compute statistic over
            projector (Dict) : dictionary of projector objects
            num_batches (Optional[int]): number of batches of the dataloader to compute over.
                If None, then comptues over whole dataloader
            
            model_half (Optional[bool]): whether to use model_half
            device (Optional[str]): e.g. "cuda"
            saveas (Optional[str]): fie to save 
        Returns:
            torch.Tensor or list: grad norm of input IDs
        """
       
        ####################################################################################################
        # OBTAIN PROJECTORS
        ####################################################################################################
        if project_file:
            print(f"USING {project_file} as projector")
            projectors = torch.load(project_file)
        else:
            print(f"COMPUTING basis change")
            projectors = {}
            dataloader_logits = compute_dataloader_logits_embedding(self.model, svd_dataloader, device, half=model_half).T.float().to(device)
        
            ## Generate matrix U @ torch.diag(S) which is equal to embedding projection up to symmetries
            U, S, _ = torch.linalg.svd(dataloader_logits,full_matrices=False)
            svd_embedding_projection_layer = U[:,:(next(self.model.parameters()).shape[1])] @ torch.diag(S[:(next(self.model.parameters()).shape[1])])
            
            ## Identify base change to convert regular gradients to gradients we can access
            projectors["svd_embedding_projection_layer"] = svd_embedding_projection_layer.clone().detach()        
            projectors["one_sided_grad_project"] = CudaProjector(50304, proj_dim, proj_seed, ProjectionType(proj_type), 'cuda', 8)
            if saveas:
                print(f"SAVING PROJECTORS AT {saveas}")
                torch.save(projectors, saveas)

        return projectors         
        
    def compute_dataloader_model_stealing(self, dataloader, projector, num_batches=None, device=None, model_half=None):
        '''
        Computes dataloader gradients with jl dimensionality reduction.
        Args:
            dataloader (torch.utils.data.dataloader.DataLoader): DataLoader of samples.
            projector (dict): dictionary of dimensionality reduction functions        
            num_batches (int, optional): Number of batches to consider
            device (str, optional): CPU or GPU 
            model_half (bool, optional): use half precision floats for model

        Returns:
            torch.Tensor or list: data for input IDs
        '''
        if self.model is None:
            raise Exception("Please call .load_model() to load the model first.")
        if projector is None or "svd_embedding_projection_layer" not in projector or "one_sided_grad_project" not in projector:
            raise Exception("Please call .compute_model_stealing() to compute projector object.")
        return compute_dataloader_basis_changes(model=self.model, dataloader=dataloader, projector=projector, device=device, nbatches=num_batches, half=model_half).cpu()


####################################################################################################
# HELPER FUNCTIONS
####################################################################################################

def compute_basis_change(model, input_ids, projector, device=None):
    """
    This computes the basis change for the last layer (Carlini et al. gray-box attack), and returns it
    with the norms of that layer.

    Args:
        model (transformers.AutoModelForCausalLM): HuggingFace model.
        input_ids (torch.Tensor): tensor of input IDs.
        projector (dict): dictionary of dimensionality reduction functions
        device (str, optional): CPU or GPU 
    
    Returns:
        torch.Tensor or list: data from input IDs
    """
    mask  = (input_ids > 0).detach()
    model.zero_grad()
    outputs = model(input_ids=input_ids.to(device),attention_mask=mask.to(device),labels=input_ids.to(device),output_hidden_states=True)
    outputs.logits.retain_grad()
    outputs.loss.backward()
    
    implied_latents = torch.linalg.lstsq(projector["svd_embedding_projection_layer"], outputs.logits[0].T).solution.T
    grad = einsum(outputs.logits.grad, implied_latents, "a b c, b d -> a c d")
    
    ## Norm and projected features
    L = torch.tensor([torch.norm(grad.flatten().view(-1,1).T,p=p) for  p in [float("inf"),1,2]])
    projected = projector["one_sided_grad_project"].project(grad[0,:,:].T.clone().to(device).contiguous(),1).flatten()
    
    del outputs, input_ids, mask
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return torch.concat((L.cpu(),projected.cpu()),dim=0).flatten()

def compute_dataloader_basis_changes(model, dataloader, projector, device=None, nbatches=None, half=True):
    '''
    Computes dataloader gradients with jl dimensionality reduction.
    Args:
        model (transformers.AutoModelForCausalLM): HuggingFace model.
        dataloader (torch.utils.data.dataloader.DataLoader): DataLoader of samples.
        projector (dict): dictionary of dimensionality reduction functions        
        device (str, optional): CPU or GPU 
        nbatches (int, optional): Number of batches to consider
        half (bool, optional): use half precision floats for model

    Returns:
        torch.Tensor or list: data for input IDs
    '''
    if "random_basis_change" in projector:
        projector["random_basis_change"].to(device)
    if "svd_embedding_projection_layer" in projector:
        projector["svd_embedding_projection_layer"].to(device)
        
    if half:
        print("Using model.half() ....")
        model.half()
        for key in projector.keys():
            projector[key].half()
    else:
        print("Not using model.half() ....")

    model.eval()
    model.to(device)

    grads = []
    for batchno, data_x in tqdm(enumerate(dataloader),total=len(dataloader)):
        if nbatches is not None and batchno >= nbatches:
            break
        ## Get predictions on data 
        if type(data_x) is dict:
            data_x = data_x["input_ids"]
        else:
            data_x = data_x[None,:]
        data_x = data_x.detach()                

        ## Compute features on input data
        grad = compute_basis_change(model, data_x, projector, device).detach().cpu()
        grads.append(grad)

        del data_x
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return torch.stack(grads)


def compute_input_ids_logits(model, input_ids, device=None):
    """
    Compute logits of last token in input ids

    Args:
        model (transformers.AutoModelForCausalLM): HuggingFace model.
        input_ids (torch.Tensor): tensor of input IDs.
        device (str, optional): CPU or GPU

    Returns:
        torch.Tensor: logits of last token in input ids
    
    """
    with torch.no_grad():
        
        mask  = (input_ids > 0).detach()
        outputs = model(input_ids=input_ids.to(device), attention_mask = mask.to(device),output_hidden_states=True)
        
        return outputs.logits[0,-1,:]


def compute_dataloader_logits_embedding(model, dataloader, device=None, nbatches=None, half=False):
    '''
    Computes logits of text in dataloader

    Args:
        model (transformers.AutoModelForCausalLM): HuggingFace model.
        dataloader (torch.utils.data.dataloader.DataLoader): DataLoader with tokens.
        device (str, optional): CPU or GPU 
        nbatches (int, optional): Number of batches to consider
        half (bool, optional): use half precision floats for model

    Returns:
        torch.Tensor or list: loss of input IDs
    '''
    if half:
        model.half()
    model.eval()
    model.to(device)

    losses = []
    for batchno, data_x in tqdm(list(enumerate(dataloader))[250:],total=len(dataloader)):
        if nbatches is not None and batchno >= nbatches:
            break
        ## Get predictions on data 
        if type(data_x) is dict:
            data_x = data_x["input_ids"]
        else:
            data_x = data_x[None,:]
        data_x = data_x.detach()                

        ## Compute features on input data
        
        loss = compute_input_ids_logits(model, data_x, device=device).detach().cpu()
        losses.append(loss)        

        del data_x, loss
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    
    return torch.stack(losses)
    