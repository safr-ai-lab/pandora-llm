import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from einops import einsum
from transformers import AutoModelForCausalLM
from .Attack import MIA
# from ..utils.attack_utils import *
# from ..utils.plot_utils import *
from trak.projectors import CudaProjector, ProjectionType

####################################################################################################
# MAIN CLASS
####################################################################################################
class ModelStealing(MIA):
    """
    Model stealing attack
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        
    def load_model(self):
        """
        Loads model into memory
        """
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, revision=self.model_revision, cache_dir=self.model_cache_dir)
        else:
            raise Exception("Model has already been loaded; please call .unload_model() first!")

    def unload_model(self):
        """
        Unloads model from memory
        """
        self.model = None

    def compute_model_stealing(self, 
        dataloader,
        method,
        svd_dataloader=None,
        proj_type="rademacher",
        proj_dim=512,
        proj_seed=229,
        num_batches=None,
        device=None, model_half=None, accelerator=None, max_length=None
    ):
        """
        Compute the JL projection of the gray-box model-stealing attack for a given dataloader.

        Args:
            dataloader (DataLoader): input data to compute statistic over
            method (str): the method, or a filename to load the projectors from
            svd_dataloader (DataLoader): svd dataloader
            proj_type (str): the projection type (either 'normal' or 'rademacher')
            proj_dim (int): the number of dimensions to project to
            proj_seed (int): the random seed to use in the projection
            num_batches (Optional[int]): number of batches of the dataloader to compute over.
                If None, then comptues over whole dataloader
            device (Optional[str]): e.g. "cuda"
            model_half (Optional[bool]): whether to use model_half
            accelerator (Optional[Accelerator]): accelerator object
        Returns:
            torch.Tensor or list: grad norm of input IDs
        """
        if self.model is None:
            raise Exception("Please call .load_model() to load the model first.")
        input_size = next(self.model.parameters()).shape[1] * next(self.model.parameters()).shape[0]
        projectors = {}

        ####################################################################################################
        # OBTAIN PROJECTORS
        ####################################################################################################
        if method=="no_rotation": # debugging, just use the godeye view (Id basis change)
            projectors["random_basis_change"] = torch.eye(next(self.model.parameters()).shape[1])
            projectors["embed_out"] = CudaProjector(input_size, proj_dim, 
                                                        proj_seed, ProjectionType(proj_type), 'cuda', 8)
        elif method=="randn_rotation": # N(0,I) basis change
            projectors["random_basis_change"] = torch.randn((next(self.model.parameters()).shape[1], next(self.model.parameters()).shape[1]))
            projectors["embed_out"] = CudaProjector(input_size, proj_dim, 
                                                        proj_seed, ProjectionType(proj_type), 'cuda', 8)
        elif method=="only_logits": # 
            input_size_only_logits = 2048 * 50304
            projectors["logits_project"] = CudaProjector(input_size_only_logits, proj_dim, 
                                                proj_seed, ProjectionType(self.project_type), 'cuda', 8) 
        else:
            dataloader_logits = compute_dataloader_logits_embedding(self.model, svd_dataloader, device, half=model_half).T.float().to(device)
            
            last_layer = [m for m in self.model.parameters()][-1]
            ## Generate matrix U @ torch.diag(S) which is equal to embedding projection up to symmetries
            U, S, _ = torch.linalg.svd(dataloader_logits,full_matrices=False)
            svd_embedding_projection_layer = U[:,:(next(self.model.parameters()).shape[1])] @ torch.diag(S[:(next(self.model.parameters()).shape[1])])
            
            ## Identify base change to convert regular gradients to gradients we can access
            projectors["svd_embedding_projection_layer"] = svd_embedding_projection_layer.clone().detach()
            
            if method=="one_sided_projection":
                projectors["one_sided_grad_project"] = CudaProjector(50304, 512, proj_seed, ProjectionType(proj_type), 'cuda', 8)
            else:
                projectors["grad_project"] = CudaProjector(input_size, proj_dim, 
                                                        proj_seed, ProjectionType(proj_type), 'cuda', 8)
            # if logits_and_embedding:
            #     projectors["logits_project"] = CudaProjector(2048*50304, proj_dim, 
            #                                             proj_seed, ProjectionType(proj_type), 'cuda', 8)
            #     projectors["hidden_project"] = CudaProjector(2048*next(self.model.parameters()).shape[1], proj_dim, 
            #                                             proj_seed, ProjectionType(proj_type), 'cuda', 8)
            # elif svd_to_real:
            #     projectors["basis_from_svd_to_real"] = torch.linalg.lstsq(last_layer, 
            #                                         projectors["svd_embedding_projection_layer"]).solution ## \tilde{W} \cdot A = W 
            #     print(torch.mean((projectors["svd_embedding_projection_layer"] @ projectors["basis_from_svd_to_real"]  - last_layer)**2))

            # # Identify base change to convert regular gradients to gradients we can access
            # base_change = torch.linalg.pinv(last_layer.float()).to('cpu') @ svd_embedding_projection_layer.to('cpu')
            # projectors["random_basis_change"] = torch.linalg.inv(base_change).T  # NOT A PROJECTOR, code just puts it in projectors dict for some reason

        ####################################################################################################
        # COMPUTE ON DATALOADER
        ####################################################################################################
        return compute_dataloader_basis_changes(model=self.model, dataloader=dataloader, projector=projectors, device=device, nbatches=num_batches, half=model_half).cpu() 



####################################################################################################
# HELPER FUNCTIONS
####################################################################################################
# TODO: needs a LOT of cleaning

def compute_basis_change(model, input_ids, projector, device=None):
    """
    This computes the basis change for the last layer (Carlini gray-box attack), and returns it
    with the norms of that layer.

    Args:
        model (transformers.AutoModelForCausalLM): HuggingFace model.
        input_ids (torch.Tensor): tensor of input IDs.
        projector (dict): dictionary of dimensionality reduction functions
        device (str): CPU or GPU 
    
    Returns:
        torch.Tensor or list: data from input IDs
    """
    mask  = (input_ids > 0).detach()
    model.zero_grad()
    outputs = model(input_ids=input_ids.to(device),attention_mask=mask.to(device),labels=input_ids.to(device),output_hidden_states=True)
    outputs.logits.retain_grad()
    outputs.loss.backward()
    
    param = model.get_parameter("embed_out.weight")

    ## Random basis change
    if "random_basis_change" in projector:
        copied_grad = param.grad.detach().clone()
        projector["random_basis_change"].to(device)
        grad = (copied_grad @ projector["random_basis_change"].to(device)).flatten().view(-1,1).T
                                    
        L = torch.tensor([torch.norm(grad,p=p) for p in [float("inf"),1,2]])    # get norms
        projected = projector["embed_out"].project(grad,1).to(device).flatten() # get projection of embedding_out layer
    elif "logits_project" in projector and "hidden_project" in projector:
        logits_grad  = F.pad(outputs.logits.grad, (0,0,0, 2048-outputs.logits.grad.shape[1],0,50304-outputs.logits.grad.shape[2]), 
                                "constant", 0).flatten()
        print(logits_grad.shape)

        logits = F.pad(outputs.logits[0], (0,0,0,2048-outputs.logits[0].shape[0]),"constant", 0).to(device)
        svd_embedding = torch.linalg.lstsq(projector["svd_embedding_projection_layer"].to(device), 
                        logits.T).solution.T.flatten()
        L = torch.tensor([torch.norm(logits_grad,p=p) for p in [float("inf"),1,2]]
                            +[torch.norm(svd_embedding,p=p) for p in [float("inf"),1,2]])    # get norms
        
        projected_logits_grad = projector["logits_project"].project(logits_grad.flatten().view(-1,1).T,1).flatten()
        projected_latents     = projector["hidden_project"].project(svd_embedding.flatten().view(-1,1).T,1).flatten()
        projected = torch.concat((projected_logits_grad.flatten(), projected_latents.flatten()))
    elif "logits_project" in projector and "hidden_project" not in projector:
        data = outputs.logits.grad.flatten()
        L = torch.tensor([torch.norm(data,p=p) for p in [float("inf"),1,2]])    # get norms
        projected = projector["vocab"].project(data.view(-1,1).T,1).to(device).flatten()
    elif "grad_project" in projector and "basis_from_svd_to_real" in projector:
        implied_latents = torch.linalg.lstsq(projector["svd_embedding_projection_layer"], outputs.logits[0].T).solution.T
        grad = einsum(outputs.logits.grad, implied_latents, "a b c, b d -> a c d")
        grad = grad @ projector["basis_from_svd_to_real"].T
        print(torch.mean((param.grad.detach().clone()-grad)**2))
        grad = grad.flatten().view(-1,1).T
        L = torch.tensor([torch.norm(grad,p=p) for  p in [float("inf"),1,2]])
        projected = projector["grad_project"].project(grad.flatten().view(-1,1).T,1).to(device).flatten()
    elif "grad_project" in projector:
        implied_latents = torch.linalg.lstsq(projector["svd_embedding_projection_layer"], outputs.logits[0].T).solution.T
        grad = einsum(outputs.logits.grad, implied_latents, "a b c, b d -> a c d").flatten().view(-1,1).T
        L = torch.tensor([torch.norm(grad,p=p) for  p in [float("inf"),1,2]])
        projected = projector["grad_project"].project(grad.flatten().view(-1,1).T,1).to(device).flatten()
    elif "one_sided_grad_project" in projector:
        implied_latents = torch.linalg.lstsq(projector["svd_embedding_projection_layer"], outputs.logits[0].T).solution.T
        grad = einsum(outputs.logits.grad, implied_latents, "a b c, b d -> a c d")
        L = torch.tensor([torch.norm(grad.flatten().view(-1,1).T,p=p) for  p in [float("inf"),1,2]])
        projected = projector["one_sided_grad_project"].project(grad[0,:,:].T.clone().to(device).contiguous(),1).flatten()
        print(torch.concat((L.cpu(),projected.cpu()),dim=0).flatten())
    else:
        assert False 
    
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
        device (str): CPU or GPU 
        nbatches (int): Number of batches to consider
        half (bool): use half precision floats for model

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
        device (str): CPU or GPU

    Returns:
        torch.Tensor: logits of last token in input ids
    
    """
    mask  = (input_ids > 0).detach()
    outputs = model(input_ids=input_ids.to(device), attention_mask = mask.to(device),output_hidden_states=True)
    return outputs.logits[0,-1,:]


def compute_dataloader_logits_embedding(model, dataloader, device=None, nbatches=None, half=False):
    '''
    Computes logits of text in dataloader

    Args:
        model (transformers.AutoModelForCausalLM): HuggingFace model.
        dataloader (torch.utils.data.dataloader.DataLoader): DataLoader with tokens.
        device (str): CPU or GPU 
        nbatches (int): Number of batches to consider
        half (bool): use half precision floats for model

    Returns:
        torch.Tensor or list: loss of input IDs
    '''
    if half:
        model.half()
    model.eval()
    model.to(device)

    losses = []
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
        loss = compute_input_ids_logits(model, data_x, device=device).detach().cpu()
        losses.append(loss)

        del data_x
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return torch.stack(losses)