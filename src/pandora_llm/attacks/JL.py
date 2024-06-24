import subprocess
import math
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import AutoModelForCausalLM
from trak.projectors import CudaProjector, ProjectionType, BasicProjector
from .Attack import MIA

####################################################################################################
# MAIN CLASS
####################################################################################################
class JL(MIA):
    """
    JL features attack
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_jl(self, 
        dataloader,
        proj_x_to,
        proj_each_layer_to,
        proj_type,
        proj_seed=229,
        num_batches=None,
        device=None, model_half=None, accelerator=None, max_length=None
    ):
        """
        Compute the JL projection of the gradients for a given dataloader.

        Args:
            dataloader (DataLoader): input data to compute statistic over
            proj_x_to (int): the number of dimensions to project embedding gradient to
            proj_each_layer_to (int): the number of dimensions to project each layer to
            proj_type (str): the projection type (either 'normal' or 'rademacher')
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
        print(f"In total, the number of features will be: {sum(1 for _ in self.model.named_parameters()) * proj_each_layer_to}.")

        # Retrieve embedding layer
        if accelerator is not None:
            self.model.gradient_checkpointing_enable() # TODO
            self.model, dataloader, = accelerator.prepare(self.model, dataloader)
            if accelerator.is_main_process:
                subprocess.call(["python", "model_embedding.py",
                    "--model_name", self.model_name,
                    "--model_revision", self.model_revision,
                    "--model_cache_dir", self.model_cache_dir,
                    "--save_path", "results/JL/embedding.pt",
                    "--model_half" if model_half else ""
                    ]
                )
            accelerator.wait_for_everyone()
            embedding_layer = torch.load("results/JL/embedding.pt")
            self.model.train()
        else:
            embedding_layer = self.model.get_input_embeddings().weight
        
        # Project each type of data with a JL dimensionality reduction
        projectors = {}                 
        projectors["x"] = BasicProjector(
            grad_dim=next(self.model.parameters()).shape[1]*2048,
            proj_dim=proj_x_to,
            seed=proj_seed,
            proj_type=ProjectionType(proj_type),
            device=device,
            block_size=1
        ) #TODO replace with CudaProjector
        
        for i, (name,param) in enumerate(self.model.named_parameters()):
            projectors[(i,name)] = BasicProjector(
                grad_dim=math.prod(param.size()),
                proj_dim=proj_each_layer_to,
                seed=proj_seed,
                proj_type=ProjectionType(proj_type),
                device=device,
                block_size=1
            )
    
        return compute_dataloader_jl(model=self.model,embedding_layer=embedding_layer,projector=projectors,dataloader=dataloader,nbatches=num_batches,device=device,half=model_half).cpu() 
        
    def compute_jl_balanced(self, 
        dataloader,
        proj_x_to,
        proj_group_to,
        proj_type,
        proj_seed=229,
        num_splits=8,
        num_batches=None,
        device=None, model_half=None, accelerator=None, max_length=None
    ):
        """
        Compute the JL projection of the gradients for a given dataloader, more equally distributed across num_split splits

        Args:
            dataloader (DataLoader): input data to compute statistic over
            proj_x_to (int): the number of dimensions to project embedding gradient to
            proj_group_to (int): the number of dimensions to project each group to
            proj_type (str): the projection type (either 'normal' or 'rademacher')
            proj_seed (int): the random seed to use in the projection
            num_splits (int): how many splits of parameters to compute JL over
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
        print(f"In total, the number of features will be: {num_splits * proj_group_to}.")

        # Retrieve embedding layer
        if accelerator is not None:
            self.model.gradient_checkpointing_enable() # TODO
            self.model, dataloader, = accelerator.prepare(self.model, dataloader)
            if accelerator.is_main_process:
                subprocess.call(["python", "model_embedding.py",
                    "--model_name", self.model_name,
                    "--model_revision", self.model_revision,
                    "--model_cache_dir", self.model_cache_dir,
                    "--save_path", "results/JL/embedding.pt",
                    "--model_half" if model_half else ""
                    ]
                )
            accelerator.wait_for_everyone()
            embedding_layer = torch.load("results/JL/embedding.pt")
            self.model.train()
        else:
            embedding_layer = self.model.get_input_embeddings().weight
        
        ## JL balanced Features
        sizes = []
        for i, (name,param) in enumerate(self.model.named_parameters()):
            sizes.append(math.prod(param.size()))
        
        def balanced_partition(sizes, num_groups):
            # Pair each size with its original index
            sizes_with_indices = list(enumerate(sizes))
            # Sort sizes in descending order while keeping track of indices
            sizes_with_indices.sort(key=lambda x: x[1], reverse=True)
            
            # Initialize groups and their sums
            groups = [[] for _ in range(num_groups)]
            group_sums = [0] * num_groups
            group_indices = [[] for _ in range(num_groups)]
            
            # Assign each size to the group with the smallest current sum
            for index, size in sizes_with_indices:
                min_index = group_sums.index(min(group_sums))
                groups[min_index].append(size)
                group_indices[min_index].append(index)
                group_sums[min_index] += size

            return groups, group_sums, group_indices
        groups, sums, indices = balanced_partition(sizes, num_splits)
        print(f"Split groups: {groups}")
        print(f"Split sums: {sums}")
        print(f"Split group indices: {indices}")

        projectors = {}
        for i in range(num_splits):
            projectors[i] = CudaProjector(
                grad_dim=sums[i], 
                proj_dim=proj_group_to,
                seed=proj_seed,
                proj_type=ProjectionType(proj_type),
                device='cuda',
                max_batch_size=32,
            )
        
        projectors["x"] = BasicProjector(
            grad_dim=next(self.model.parameters()).shape[1]*2048,
            proj_dim=proj_x_to,
            seed=proj_seed,
            proj_type=ProjectionType(proj_type),
            device='cuda',
            block_size=1,
        )
        return compute_dataloader_jl_balanced(model=self.model, embedding_layer=embedding_layer, dataloader=dataloader, projector=projectors, indices=indices, device=device, nbatches=num_batches, half=model_half).cpu() 

####################################################################################################
# HELPER FUNCTIONS
####################################################################################################

def compute_input_ids_grad_jl_balanced(model, embedding_layer, input_ids, projector, group_indices, device=None):
    """
    Compute JL of gradients grouped by indices
    """

    mask  = (input_ids > 0).detach()
    input_embeds=Variable(embedding_layer[input_ids.cpu()],requires_grad=True)

    ## Get gradient with respect to x    
    model.zero_grad()
    outputs = model(inputs_embeds=input_embeds.to(device),attention_mask=mask.to(device),labels=input_ids.to(device))    
    outputs.loss.backward()
    x_grad = input_embeds.grad.detach().to(device)
    x_grad = F.pad(x_grad, (0,0,0, 2048-x_grad.shape[1],0,next(model.parameters()).shape[1]-x_grad.shape[2]),"constant", 0).flatten().view(-1,1).T
    all_grads = projector["x"].project(x_grad,1).to(device).flatten()

    ## Get gradient with respect to theta
    model.zero_grad()
    outputs = model(input_ids=input_ids.to(device),attention_mask=mask.to(device),labels=input_ids.to(device))
    outputs.loss.backward()

    for j, indexarr in enumerate(group_indices):
        catelems = []
        for i, (name,param) in enumerate(model.named_parameters()):
            if i in indexarr:
                grad = param.grad.flatten()
                # print(f"Grad shape: {grad.shape}")
                catelems.append(grad)
        result_tensor = torch.cat(catelems, dim=0).view(-1,1).T # pre transform: torch.Size([128188416])
        # print(f"Res Tens Shape: {result_tensor.shape}")
        all_grads = torch.concat((all_grads, projector[j].project(result_tensor,1).flatten()),dim=0)
    
    del outputs, input_embeds, input_ids, mask
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return all_grads

def compute_dataloader_jl_balanced(model, embedding_layer, dataloader, projector, indices, device=None, nbatches=None, half=True):
    '''
    Computes dataloader gradients with jl dimensionality reduction.

    Args:
        model (transformers.AutoModelForCausalLM): HuggingFace model.
        embedding_layer (torch.nn.parameter.Parameter): computes embeddings from tokens, useful for taking grad wrt x
        dataloader (torch.utils.data.dataloader.DataLoader): DataLoader of samples.
        projector (dict): dictionary of dimensionality reduction functions        
        device (str): CPU or GPU 
        nbatches (int): Number of batches to consider
        half (bool): use half precision floats for model

    Returns:
        torch.Tensor or list: data for input IDs
    '''
    if half:
        print("Using model.half() ....")
        model.half()
    else:
        print("Not using model.half() ....")
    model.eval()
    model.to(device)
    # if "random_basis_change" in projector:
    #     projector["random_basis_change"] = projector["random_basis_change"].to(device).half()

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
        grad = compute_input_ids_grad_jl_balanced(model, embedding_layer, data_x, projector, indices, device=device).detach().cpu()
        grads.append(grad)

        del data_x
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return torch.stack(grads)

def compute_input_ids_grad_jl(model, embedding_layer, input_ids,  projector, device=None):
    """
    Compute JL of gradients with respect x and/or theta

    Args:
        model (transformers.AutoModelForCausalLM): HuggingFace model.
        embedding_layer (torch.nn.parameter.Parameter): computes embeddings from tokens, useful for taking grad wrt x
        input_ids (torch.Tensor): tensor of input IDs.
        projector (dict): dictionary of dimensionality reduction functions
        device (str, optional): CPU or GPU 
                
    Returns:
        torch.Tensor or list: data from input IDs
    """

    mask  = (input_ids > 0).detach()
    input_embeds=Variable(embedding_layer[input_ids.cpu()],requires_grad=True)

    ## Get gradient with respect to x    
    model.zero_grad()
    outputs = model(inputs_embeds=input_embeds.to(device),attention_mask=mask.to(device),labels=input_ids.to(device))    
    outputs.loss.backward()
    x_grad = input_embeds.grad.detach().to(device)
    x_grad = F.pad(x_grad, (0,0,0, 2048-x_grad.shape[1],0,next(model.parameters()).shape[1]-x_grad.shape[2]),"constant", 0).flatten().view(-1,1).T
    all_grads = projector["x"].project(x_grad,1).to(device).flatten()

    ## Get gradient with respect to theta
    model.zero_grad()
    outputs = model(input_ids=input_ids.to(device),attention_mask=mask.to(device),labels=input_ids.to(device))
    outputs.loss.backward()

    for i, (name,param) in enumerate(model.named_parameters()):
        grad = param.grad.flatten().view(-1,1).T
        all_grads = torch.concat((all_grads, projector[(i,name)].project(grad,1).flatten()),dim=0)
    
    del outputs, input_embeds, input_ids, mask
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return all_grads

def compute_dataloader_jl(model, embedding_layer, dataloader, projector, device=None, nbatches=None, half=True):
    '''
    Computes dataloader gradients with jl dimensionality reduction.
    
    Args:
        model (transformers.AutoModelForCausalLM): HuggingFace model.
        embedding_layer (torch.nn.parameter.Parameter): computes embeddings from tokens, useful for taking grad wrt x
        dataloader (torch.utils.data.dataloader.DataLoader): DataLoader of samples.
        projector (dict): dictionary of dimensionality reduction functions        
        device (str): CPU or GPU 
        nbatches (int): Number of batches to consider
        half (bool): use half precision floats for model

    Returns:
        torch.Tensor or list: data for input IDs
    '''
    if half:
        print("Using model.half() ....")
        model.half()
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
        grad = compute_input_ids_grad_jl(model, embedding_layer, data_x, projector, device=device).detach().cpu()
        grads.append(grad)

        del data_x
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return torch.stack(grads)