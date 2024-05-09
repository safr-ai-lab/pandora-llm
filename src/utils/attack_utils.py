import torch
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from tqdm import tqdm
# from detect_gpt_utils import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.autograd import Variable
from deepspeed.utils import safe_get_full_grad
import psutil
import subprocess
import torch.nn.functional as F

def compute_input_ids_cross_entropy(model: AutoModelForCausalLM, input_ids: torch.Tensor, return_pt: bool=True, token: bool=False):
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

    if token:
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
    
    if token:
        return torch.tensor(ans) if return_pt else ans[0] 
    else:
        return torch.tensor(ans) if return_pt else ans 

def compute_dataloader_cross_entropy_tokens(model, dataloader, device=None, nbatches=None, samplelength=None, accelerator=None, half=True):    
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
                loss = compute_input_ids_cross_entropy(model, data_x.to(device), return_pt = False,tokens=True).detach().cpu()
            else:
                loss = compute_input_ids_cross_entropy(model, data_x, return_pt = False,tokens=True)

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

def compute_input_ids_cross_entropy_batch(model, input_ids, return_pt=True):
    """
    Compute the cross entropy over an entire batch of inputs
    Args:
        model (transformers.AutoModelForCausalLM): HuggingFace model.
        input_ids (torch.Tensor): tensor of input IDs.
        return_pt (bool): Return tensor or list
        
    Returns:
        torch.Tensor or list: loss of input IDs
    """
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
    return torch.mean(torch.tensor(ans),dim=-1) if return_pt else sum(ans)/len(ans)

def compute_dataloader_cross_entropy_batch(model, dataloader, device=None, nbatches=None, samplelength=None, accelerator=None, half=True, detect_args=None):    
    '''
    Computes dataloader cross entropy of different models for DetectGPT-based attack. 
    Warning: using samplelength is discouraged
    Args:
        model (transformers.AutoModelForCausalLM): HuggingFace model.
        dataloader (torch.utils.data.dataloader.DataLoader): DataLoader with tokens.
        device (str): CPU or GPU 
        nbatches (int): Number of batches to consider
        samplelength (int or NoneType): cut all samples to a given length
        accelerator (accelerate.Accelerator or NoneType): enable distributed training
        half (bool): use half precision floats for model
        detect_args (dict): config for DetectGPT

    Returns:
        torch.Tensor or list: loss of input IDs
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

    base_model_name = "EleutherAI/pythia-70m-deduped"
    mask_model_name = 't5-small'
    if accelerator is None:
        mask_model = T5ForConditionalGeneration.from_pretrained(mask_model_name)
        mask_tokenizer = T5Tokenizer.from_pretrained(mask_model_name)
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        mask_model.eval()
        mask_model.to(device)
    else:
        # mask_model = T5ForConditionalGeneration.from_pretrained(mask_model_name)
        # mask_tokenizer = T5Tokenizer.from_pretrained(mask_model_name)
        # base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        # mask_model.eval()
        # mask_model.to(accelerator.device)
        # detect_args["device"] = accelerator.device
        torch.save(detect_args,f"DetectGPT/detect_args_{accelerator.device}.pt")

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
            
            if accelerator is None:
                data_x_batch = perturb_input_ids(data_x.squeeze(0).to(device), detect_args, base_tokenizer, mask_tokenizer, mask_model).unsqueeze(-1)
            else:
                # data_x_batch = perturb_input_ids(data_x.squeeze(0), detect_args, base_tokenizer, mask_tokenizer, mask_model).unsqueeze(-1)
                
                torch.save(data_x,f"DetectGPT/base_input_ids_{accelerator.device}.pt")
                subprocess.call(["python", "perturb_input_ids.py",
                    "--base_model_name", f"{base_model_name}",
                    "--mask_model_name", f"{mask_model_name}",
                    "--data_x", f"DetectGPT/base_input_ids_{accelerator.device}.pt",
                    "--detect_args", f"DetectGPT/detect_args_{accelerator.device}.pt",
                    "--save_path", f"DetectGPT/perturbed_input_ids_{accelerator.device}.pt",
                    # "--accelerate",
                    # "--model_half" if self.config["model_half"] else ""
                    ]
                )
                data_x_batch = torch.load(f"DetectGPT/perturbed_input_ids_{accelerator.device}.pt")
                # data_x_batch = torch.concatenate([data_x for _ in range(2)],dim=0).unsqueeze(2)

            ## Compute average log likelihood
            if accelerator is None:
                avg_perturbed_loss = compute_input_ids_cross_entropy_batch(model, data_x_batch.to(device)).detach().cpu()
                loss = compute_input_ids_cross_entropy(model, data_x.to(device)).detach().cpu()
                detect_gpt_score = loss - avg_perturbed_loss
            else:
                perturbed_losses = 0
                for pert_num in range(data_x_batch.shape[0]):
                    perturbed_losses += compute_input_ids_cross_entropy(model, data_x_batch[pert_num].T.to(accelerator.device), return_pt = False)[0]
                # avg_perturbed_loss = compute_input_ids_cross_entropy_batch(model, data_x_batch.to(accelerator.device), return_pt = False)
                loss = compute_input_ids_cross_entropy(model, data_x, return_pt = False)[0]
                # detect_gpt_score = [loss[0]-avg_perturbed_loss]
                detect_gpt_score = [loss-perturbed_losses/data_x_batch.shape[0]]

            losses.append(detect_gpt_score)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    if accelerator is None:
        return torch.tensor(losses)
    else:
        losses = accelerator.gather_for_metrics(losses)
        losses = torch.cat([loss[0] for loss in losses])
        return losses


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

def compute_input_ids_all_norms(model, embedding_layer, input_ids, ps, extraction_mia, device=None, accelerator=None):
    """
    Compute norms of gradients with respect x, theta
    
    Args:
        model (transformers.AutoModelForCausalLM): HuggingFace model.
        embedding_layer (torch.nn.parameter.Parameter): computes embeddings from tokens, useful for taking grad wrt x
        input_ids (torch.Tensor): tensor of input IDs.
        ps (list): gradient norm types
        extraction_mia (bool): flag to cut samples for extraction mia data collection
        device (str): CPU or GPU 
        accelerator (accelerate.Accelerator or NoneType): enable distributed training
        
    Returns:
        torch.Tensor or list: gradient norms of input ID
    """
    
    if extraction_mia:
        print("cutting to 100")
        input_ids = input_ids[:,:100]
        print(input_ids.shape)


    ## Compute gradient with respect to x
    mask  = (input_ids > 0).detach()
    input_embeds=Variable(embedding_layer[input_ids.cpu()],requires_grad=True)
    if accelerator is not None:
        model.zero_grad()
        outputs = model(inputs_embeds=input_embeds.to(accelerator.device), attention_mask = mask.to(accelerator.device), labels=input_ids.to(accelerator.device))
    else:
        model.zero_grad()
        outputs = model(inputs_embeds=input_embeds.to(device),attention_mask=mask.to(device),labels=input_ids.to(device))
    if accelerator is not None:
        accelerator.backward(outputs.loss)
    else:
        outputs.loss.backward()
    x_grad = input_embeds.grad.detach()

    ## Compute gradients with respect to different layers
    mask  = (input_ids > 0).detach()
    if accelerator is not None:
        model.zero_grad()
        outputs = model(input_ids=input_ids.to(accelerator.device), attention_mask = mask.to(accelerator.device), labels=input_ids.to(accelerator.device))
    else:
        model.zero_grad()
        outputs = model(input_ids=input_ids.to(device),attention_mask=mask.to(device),labels=input_ids.to(device))
    if accelerator is not None:
        accelerator.backward(outputs.loss)
    else:
        outputs.loss.backward()
    
    layer_norms = {p:[] for p in ps}
    for i, (name,param) in enumerate(model.named_parameters()):
        if accelerator is None:
            grad = param.grad.flatten()
        else:
            grad = safe_get_full_grad(param).flatten()
        
        ## Append all norms of layers to dictionary
        for p in ps:
            layer_norms[p].append(torch.norm(grad,p=p))
    
    del outputs, input_embeds, input_ids, mask
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    ## Compute norms of entire gradients and append norms of each layer
    output = torch.tensor([torch.norm(x_grad,p=p,dim=(1,2)) for p in ps] + [torch.norm(torch.tensor(layer_norms[p]),p=p) for p in ps])
    output = torch.concat((output, torch.tensor([l for p in ps for l in layer_norms[p]])))
    return output

def compute_dataloader_all_norms(model, embedding_layer, dataloader, ps, extraction_mia, device=None, nbatches=None, samplelength=None, accelerator=None, half=True):
    '''
    Computes gradient norms of text in dataloader.
    Warning: using samplelength is discouraged
    
    Args:
        model (transformers.AutoModelForCausalLM): HuggingFace model.
        embedding_layer (torch.nn.parameter.Parameter): computes embeddings from tokens, useful for taking grad wrt x
        dataloader (torch.utils.data.dataloader.DataLoader): DataLoader of samples.
        ps (list): gradient norm types
        extraction_mia (bool): flag to cut samples for extraction mia data collection
        device (str): CPU or GPU 
        nbatches (int): Number of batches to consider
        samplelength (int or NoneType): cut all samples to a given length
        accelerator (accelerate.Accelerator or NoneType): enable distributed training
        half (bool): use half precision floats for model

    Returns:
        torch.Tensor: norms of gradients for input IDs
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
        ## Get predictions on data 
        if type(data_x) is dict:
            data_x = data_x["input_ids"]
        else:
            data_x = data_x[None,:]
        print(data_x.shape)
        if samplelength is None:
            data_x = data_x.detach()                
        else:
            data_x = data_x[:,:samplelength].detach()

        ## Compute norms on data_x
        if accelerator is None:
            loss = compute_input_ids_all_norms(model, embedding_layer, data_x, ps, extraction_mia, 
                                               device=device,accelerator=accelerator).detach().cpu()
        else:
            loss = compute_input_ids_all_norms(model, embedding_layer, data_x, ps, extraction_mia, 
                                               device=device,accelerator=accelerator).to(accelerator.device)

        losses.append(loss)

        del data_x
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    if accelerator is not None:
        losses = accelerator.gather_for_metrics(losses)
        losses = torch.cat(losses)
    
    return torch.stack(losses)

def compute_input_ids_jl(model, embedding_layer, input_ids,  projector, last_layer, device=None):
    """
    Compute JL of gradients with respect x and/or theta
    Args:
        model (transformers.AutoModelForCausalLM): HuggingFace model.
        embedding_layer (torch.nn.parameter.Parameter): computes embeddings from tokens, useful for taking grad wrt x
        input_ids (torch.Tensor): tensor of input IDs.
        projector (dict): dictionary of dimensionality reduction functions
        last_layer (bool): flag to only use JL features of embedding projection gradient
        device (str): CPU or GPU 
                
    Returns:
        torch.Tensor or list: data from input IDs
    """
    
    if last_layer:
        mask  = (input_ids > 0).detach()
        model.zero_grad()
        outputs = model(input_ids=input_ids.to(device),attention_mask=mask.to(device),labels=input_ids.to(device))
        outputs.loss.backward()
        
        for i, (name,param) in enumerate(model.named_parameters()):
            if name == "embed_out.weight":
                copied_grad = param.grad.detach().clone()
                grad = (copied_grad @ projector["random_basis_change"]).flatten().view(-1,1).T
                L = torch.tensor([torch.norm(grad,p=p) for p in [float("inf"),1,2]])
                projected = projector["embed_out"].project(grad,1).to(device).flatten()
                
                del outputs, input_ids, mask
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                return torch.concat((L.cpu(),projected.cpu()),dim=0).flatten()

    else:
        mask  = (input_ids > 0).detach()
        input_embeds=Variable(embedding_layer[input_ids.cpu()],requires_grad=True)

        ## Get gradient with respect to x    
        model.zero_grad()
        outputs = model(inputs_embeds=input_embeds.to(device),attention_mask=mask.to(device),labels=input_ids.to(device))    
        outputs.loss.backward()
        x_grad = input_embeds.grad.detach().to(device)
        x_grad = F.pad(x_grad, (0,0,0, 2048-x_grad.shape[1],0,2048-x_grad.shape[2]), "constant", 0).flatten().view(-1,1).T
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

def compute_dataloader_jl(model, embedding_layer, dataloader, projector, last_layer, device=None, nbatches=None, half=True):
    '''
    Computes dataloader gradients with jl dimensionality reduction.
    Args:
        model (transformers.AutoModelForCausalLM): HuggingFace model.
        embedding_layer (torch.nn.parameter.Parameter): computes embeddings from tokens, useful for taking grad wrt x
        dataloader (torch.utils.data.dataloader.DataLoader): DataLoader of samples.
        projector (dict): dictionary of dimensionality reduction functions        
        last_layer (bool): flag to only use JL features of embedding projection features
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
    if "random_basis_change" in projector:
        projector["random_basis_change"] = projector["random_basis_change"].to(device).half()

    losses = []
    for batchno, data_x in tqdm(enumerate(dataloader),total=len(dataloader)):
        if nbatches is not None and batchno >= nbatches:
            break
        ## Get predictions on data 
        if type(data_x) is dict:
            data_x = data_x["input_ids"]
        else:
            data_x = data_x[None,:]
        print(data_x.shape)
        data_x = data_x.detach()                

        ## Compute features on input data
        loss = compute_input_ids_jl(model, embedding_layer, data_x, projector, last_layer, device=device).detach().cpu()
        losses.append(loss)

        del data_x
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return torch.stack(losses)


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
    outputs = model(input_ids=input_ids.to(device), attention_mask = mask.to(device))
    return outputs.logits[0,-1,:]


def compute_dataloader_logits_embedding(model, dataloader, device=None, nbatches=None, half=True):
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

def rademacher(shape):
    """
    Generates rademacher noise
    """
    return 2*(torch.rand(shape) > 0.5) - 1

def flat_grad(y, x, retain_graph=False, create_graph=False):
    """
    Flattens gradient into one vectoe
    """
    if create_graph:
        retain_graph = True

    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g

def z_standardize_together(tensor1, tensor2):
    """
    Standardize two arrays together
    """
    tensor_concat = torch.cat((tensor1,tensor2))
    tensor1 = (tensor1-tensor_concat.nanmean())/np.nanstd(tensor_concat)
    tensor2 = (tensor2-tensor_concat.nanmean())/np.nanstd(tensor_concat)
    return tensor1, tensor2

def approx_log_scale(x):
    """
    Log-scale equivalent for potentially negative values
    """
    return torch.log(1+torch.abs(x))*torch.sign(x)

def scipy_csr_matrix_to_torch_sparse(scipy_csr_mat):
    ## Convert to coo
    coo = scipy_csr_mat.tocoo()

    ## From coo to pytorch
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)

    shape = coo.shape
    return torch.sparse_coo_tensor(i, v, torch.Size(shape))


def split_pt_into_dict(pt_file, only_x=False, only_theta=False, divideby = 10000000):
    '''
    Convert .pt of norms into dictionary. Divide L1 norms by divideby 
    for numerical overflow issues. 
    '''    
    train_stat, val_stat = torch.load(pt_file)
    tstat_dict = {}
    valstat_dict = {}
    if only_x or (not only_x and not only_theta):
        tstat_dict["linf_x"] = train_stat[:,0]
        tstat_dict["l1_x"] = train_stat[:,1]
        tstat_dict["l2_x"] = train_stat[:,2]
        valstat_dict["linf_x"] = val_stat[:,0]
        valstat_dict["l1_x"] = val_stat[:,1]
        valstat_dict["l2_x"] = val_stat[:,2]

    #separators
    s1 = 6 
    total_vector_len = train_stat.shape[1]
    s2 = s1+(train_stat.shape[1]-s1) // 3
    s3 = s1+((train_stat.shape[1]-s1) // 3) * 2
    s4 = total_vector_len

    if only_theta or (not only_x and not only_theta):
        tstat_dict["linf_layers"] = train_stat[:,s1:s2]
        tstat_dict["l1_layers"] = train_stat[:,s2:s3]
        tstat_dict["l2_layers"] = train_stat[:,s3:s4]

        tstat_dict["linf_theta"] = tstat_dict["linf_layers"].abs().max(dim=1).values
        tstat_dict["l1_theta"] = (tstat_dict["l1_layers"]/divideby).abs().sum(axis=1)
        tstat_dict["l2_theta"] = (tstat_dict["l2_layers"]**2).sum(dim=1)

        valstat_dict["linf_layers"] = val_stat[:,s1:s2]
        valstat_dict["l1_layers"] = val_stat[:,s2:s3]
        valstat_dict["l2_layers"] = val_stat[:,s3:s4]

        valstat_dict["linf_theta"] = valstat_dict["linf_layers"].abs().max(dim=1).values
        valstat_dict["l1_theta"] = (valstat_dict["l1_layers"]/divideby).abs().sum(axis=1)
        valstat_dict["l2_theta"] = (valstat_dict["l2_layers"]**2).sum(dim=1)


    return tstat_dict, valstat_dict

def split_unsplit_pt(pt_file):
    '''
    Convert pt_file into dictionary and then .pt file
    '''
    train_stat, val_stat = split_pt_into_dict(pt_file)
    return torch.cat(list(train_stat.values()),axis=0), torch.cat(list(val_stat.values()),axis=0)

# def mem_stats():
#     """
#     Memory statistics for memory management
#     """
#     t = torch.cuda.get_device_properties(0).total_memory / 1024**3
#     r = torch.cuda.memory_reserved(0) / 1024**3
#     a = torch.cuda.memory_allocated(0) / 1024**3
#     print(f"CPU RAM: {psutil.virtual_memory()[3]/1e9}/{psutil.virtual_memory()[4]/1e9:.2f} ({psutil.virtual_memory()[2]:.2f}%)\n"
#           f"Total Memory: {t:.2f} GB\n"
#           f"Reserved Memory: {r:.2f} GB ({(100*(r/t)):.2f}%)\n"
#           f"Remaining Memory: {t-r:.2f} GB ({(100*(t-r)/t):.2f}%)\n"
#           f"---------------------------------\n"
#           f"Allocated Memory: {a:.2f} GB ({(100*(a/t)):.2f}%)\n"
#           f"Percent of Reserved Allocated: {(100*(a+1e-9)/(r+1e-9)):.2f}%\n")
    
# def compute_dataloader_probe(model, dataloader, probe, device=None, nbatches=None, samplelength=None, accelerator=None, half=False):    
#     '''
#     Computes z^THz where H is the Hessian for a probe z
#     Warning: using samplelength is discouraged
#     '''
#     if samplelength is not None:
#         print("Warning: using sample length is discouraged. Please avoid using this parameter.")
#     if accelerator is None:
#         if half:
#             model.half()
#         model.to(device)

#     trace_estimates = []
#     for batchno, data_x in tqdm(enumerate(dataloader),total=len(dataloader)):
#         if nbatches is not None and batchno >= nbatches:
#             break
#         ## Get predictions on training data 
#         data_x = data_x["input_ids"]
#         if samplelength is None:
#             data_x = data_x.detach()                
#         else:
#             data_x = data_x[:,:samplelength].detach()

#         ## Compute v^THv 
#         if accelerator is None:
#             est = compute_point_probe(model, data_x, probe, device).detach().cpu()
#         else:
#             est = compute_point_probe(model, data_x, probe, device).tolist()

#         model.zero_grad()
#         trace_estimates.append(est)
            
    
#     if accelerator is None:
#         return torch.tensor(trace_estimates)
#     else:
#         trace_estimates = accelerator.gather_for_metrics(trace_estimates)
#         trace_estimates = torch.cat([loss[0] for loss in trace_estimates])
#         return trace_estimates

# def compute_point_probe(model, data_x, probe, device):    
#     """
#     For a model, computes the v^THv with vector products, where H is Hessian of loss
#     at data_x with respect to the parameters and v is the probe.
#     """                     
#     probe_values = torch.zeros((data_x.size(0)))
#     loss_fn = CrossEntropyLoss()

#     for i in range(data_x.size(0)):
#         input_ids = data_x[i:(i+1),:].to(device)
#         mask  = (input_ids > 0).detach()       
#         outputs =  model(input_ids=input_ids.to(torch.long), attention_mask = mask.to(device))
#         logits = outputs.logits 

#         length = len(input_ids[0,1:])
#         if len(torch.where(input_ids[0,1:] == 0)[0]) > 0:
#             length = torch.where(input_ids[0,1:] == 0)[0].min()

#         ce_loss = loss_fn(logits[0,:length,:], input_ids[0, 1:length+1])
#         ce_loss.backward(retain_graph=True)

#         grad = flat_grad(ce_loss, model.parameters(), create_graph=True)  
#         grad_v = flat_grad(grad.dot(probe), model.parameters(), retain_graph=True)  
#         probe_values[i] = grad_v.dot(probe)
#     return probe_values

