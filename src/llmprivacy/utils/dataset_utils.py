import math
from itertools import groupby
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Tuple
    
def collate_fn(batch, tokenizer, max_length):
    """
    Apply tokenizer to all elements of batch and pad to length.

    Args:
        batch (list of str): batch of texts to tokenize
        tokenizer (AutoTokenizer): the tokenizer
        max_length (int): max_length of model

    Returns:
        dict: the tokenized results
            input_ids (long tensor): input ids
            labels (long tensor): the input ids again (the causal lm objective; shifting occurs automatically in forward)
            attention_mask (long tensor): the attention mask

    """
    tokens = [tokenizer.encode(example, return_tensors="pt", truncation=True, max_length=max_length) for example in batch]
    max_length = max([t.size(1) for t in tokens])
    tokens_padded = [torch.cat([t, t.new_zeros(t.size(0), max_length-t.size(1))], dim=1) for t in tokens]
    tokens_padded = torch.cat(tokens_padded, dim=0)
    return {
        "input_ids":tokens_padded,
        "labels":tokens_padded,
        "attention_mask": (tokens_padded>0).int()
    }

def process_domain_specific_data(dataset, seed=229, num_splits=1,window=100):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    
    # Get tokens for everything, and add EOS_token between examples
    tokens = [tokenizer.encode(example, return_tensors="pt", truncation=True) for example in dataset]
    collated_docs_with_eos_split = []
    for item in tqdm(tokens):
        collated_docs_with_eos_split += item.tolist()[0] + [tokenizer.eos_token_id]

    # Turn tokens back into strings. 
    dataset = []
    for i in tqdm(range(int(math.ceil(len(collated_docs_with_eos_split) / window)))):
        dataset.append(tokenizer.decode(collated_docs_with_eos_split[window * i:window * (i+1)]))
    splits = [dataset[i * len(dataset)//num_splits : (i+1) * len(dataset) // num_splits] for i in range(num_splits)]

    return splits


def load_train_pile_random(number=1000, percentage=None, start_index=0, seed=229, num_splits=1, deduped=True, unpack=False, min_length=20):
    """
    Load train pile samples from random deduped sampler.

    NOTE: min_length is only supported during unpacking

    Args:
        number (int): Number of samples
        percentage (float): Percentage of total samples (if number not specified). Default to None
        start_index (int): What index to start counting samples from. Default to 0
        seed (int): Random seed
        num_splits (int): Number of splits to separate data into. Usually set to 1 in most methods
        deduped (bool): Whether to load the deduped (True) or duped (False) Pile
        unpack (bool): Unpacks samples 
        min_length (int): Filter out sequences that are below this minimum number of tokens 
    Returns:
        list[list[str]]: List of splits, each split is a list of text samples.
    """
    if deduped and unpack:
        raise NotImplementedError("Deduped pile train random sampler does not have EOS tokens, so cannot unpack.")
    if deduped:
        dataset = load_dataset("EleutherAI/pile-deduped-pythia-random-sampled",split="train",revision="d2c194e").shuffle(seed=seed)
    else:
        dataset = load_dataset("EleutherAI/pile-duped-pythia-random-sampled",split="train",revision="d2c194e").shuffle(seed=seed)
    clip_len = number if percentage is None else int(len(dataset)*percentage)
    if not (1<=clip_len<=len(dataset)):
        raise IndexError(f"Number or percentage out of bounds. You specified {clip_len} samples but there are only {len(dataset)} samples.")
    
    def unpack(examples):
        chunks = []
        for sample in examples:
            result = []
            for k, group in groupby(sample, lambda y: y == tokenizer.eos_token_id):
                input_ids= list(group)
                if (not k) and len(input_ids)>min_length:
                    result.append(tokenizer.decode(input_ids))
            chunks.extend(result)
        return chunks

    dataset = dataset.select(range(start_index,start_index+clip_len))
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")

    if unpack:
        clip_len = number if percentage is None else int(len(dataset)*percentage)
        dataset = dataset.map(lambda x: {"text": unpack(x["tokens"])},remove_columns=["index","tokens","is_memorized"],batched=True).shuffle(seed=seed)
        dataset = dataset.select(range(clip_len))["text"]
    else:
        dataset = dataset.map(lambda x: {"text": tokenizer.decode(x["tokens"])},remove_columns=["index","tokens","is_memorized"])["text"]
    splits = [dataset[i * len(dataset)//num_splits : (i+1) * len(dataset) // num_splits] for i in range(num_splits)]

    return splits

def load_val_pile(number=1000, percentage=None, start_index=0, seed=229, num_splits=1, window=2048, compensation_factor=2.):
    """
    Loads the validation pile (NOT deduped), does an exact match deduplication and shuffling, 
    packs samples into 2048-sized chunks, and returns the specified number of splits. 

    Args:
        number (int): Number of samples
        percentage (float): Percentage of total samples (if number not specified). Default to None
        start_index (int): What index to start counting samples from. Default to 0
        seed (int): Random seed
        num_splits (int): Number of splits to separate data into. Usually set to 1 in most methods
        window (int): number of tokens to pack up to. If not packing, set to 0.
        compensation_factor (float): when packing, sample this times more samples to compensate for packing. Default to 2.
    
    Returns:
        list[list[str]]: List splits, each split is a list of text samples.
    """
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation").shuffle(seed=seed)
    clip_len = number if percentage is None else int(len(dataset)*percentage)

    if window==0: # No packing
        dataset = dataset.select(range(start_index,start_index+clip_len))
        dataset = list(dict.fromkeys(entry["text"] for entry in dataset))[:clip_len]
    else:
        # Use a multiple of clip_len to ensure enough samples after packing
        if not (1<=clip_len*compensation_factor<=len(dataset)):
            raise IndexError(f"Number or percentage out of bounds. You specified {clip_len} samples but there are only {len(dataset)} samples.")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
        dataset = dataset.select(range(start_index,start_index+int(clip_len*compensation_factor)))
        dataset = dataset.map(lambda x: {"tokens": tokenizer(x["text"])["input_ids"]}, remove_columns=["text","meta"])["tokens"]

        # Get tokens for everything, and add EOS_token between examples
        collated_docs_with_eos_split = []
        for item in tqdm(dataset):
            collated_docs_with_eos_split += item + [tokenizer.eos_token_id]

        # Turn tokens back into strings. 
        dataset = []
        for i in tqdm(range(int(math.ceil(len(collated_docs_with_eos_split) / window)))):
            dataset.append(tokenizer.decode(collated_docs_with_eos_split[window * i:window * (i+1)]))
        dataset = dataset[:clip_len]
        if len(dataset)!=clip_len:
            print("WARNING: Packing resulted in less samples than expected!!!")
    splits = [dataset[i * len(dataset)//num_splits : (i+1) * len(dataset) // num_splits] for i in range(num_splits)]

    return splits

def load_data_from_pt_files(list_of_filenames):
    """
    Loads data from different .pt files and combines them into one file 
    Args:
        list_of_filenames (list[str]): Number of samples
        
    Returns:
        dict[torch.Tensor]: List of data from all pt_files
    """

    ## Load data across different files
    data = {key: None for key in torch.load(list_of_filenames[0]).keys()}
    for file in list_of_filenames:
        curr_file = torch.load(file)
        if set(curr_file.keys()) != set(data):
            raise Exception("different files have different keys")
        for key in curr_file.keys():
            if data[key] is None:
                data[key] = curr_file[key].clone()
            else:
                data[key] = torch.concat((data[key], curr_file[key]))
    
    ## Apply random permutation to data and sort
    randperm = torch.randperm(len(list(data.values())[0]))
    for key in data.keys():
        data[key] = torch.tensor(data[key])[randperm]
    return data


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

def replace_by_colmeans(tensor: torch.tensor) -> torch.tensor:
    """
    Replace nan values by column means in a given tensor.
    """
    mask = ~torch.isfinite(tensor)
    tensor[mask] = torch.nan

    # Compute the mean of each column, ignoring non-finite values
    col_means = tensor.nanmean(dim=1)
    
    # Replace the non-finite values with the column means
    tensor[mask] = torch.broadcast_to(col_means.reshape(-1,1), mask.shape)[mask]
    return tensor 

def tstat_vstat_colmeans(pt_file: str) -> Tuple[torch.tensor, torch.tensor]:
    """ 
    Replace nan values by column means for train and valid data
    """
    tstat, vstat = torch.load(pt_file)
    tstat = replace_by_colmeans(tstat)
    vstat = replace_by_colmeans(vstat)
    return tstat, vstat
