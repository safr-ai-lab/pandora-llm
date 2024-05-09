import math
from itertools import groupby
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
    
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

def load_train_pile_random(number=1000, percentage=None, seed=229, num_splits=1, deduped=True, unpack=False, min_length=20):
    """
    Load train pile samples from random deduped sampler.

    NOTE: min_length is only supported during unpacking

    Args:
        number (int): Number of samples
        percentage (float): Percentage of total samples (if number not specified). Default to None
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

    dataset = dataset.select(range(clip_len))
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")

    if unpack:
        clip_len = number if percentage is None else int(len(dataset)*percentage)
        dataset = dataset.map(lambda x: {"text": unpack(x["tokens"])},remove_columns=["index","tokens","is_memorized"],batched=True).shuffle(seed=seed)
        dataset = dataset.select(range(clip_len))["text"]
    else:
        dataset = dataset.map(lambda x: {"text": tokenizer.decode(x["tokens"])},remove_columns=["index","tokens","is_memorized"])["text"]
    splits = [dataset[i * len(dataset)//num_splits : (i+1) * len(dataset) // num_splits] for i in range(num_splits)]

    return splits

def load_val_pile(number=1000, percentage=None, seed=229, num_splits=1, window=2048):
    """
    Loads the validation pile (NOT deduped), does an exact match deduplication and shuffling, 
    packs samples into 2048-sized chunks, and returns the specified number of splits. 

    Args:
        number (int): Number of samples
        percentage (float): Percentage of total samples (if number not specified). Default to None
        seed (int): Random seed
        num_splits (int): Number of splits to separate data into. Usually set to 1 in most methods
        window (int): number of tokens to pack up to. If not packing, set to 0.
    
    Returns:
        list[list[str]]: List splits, each split is a list of text samples.
    """
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation").shuffle(seed=seed)
    clip_len = number if percentage is None else int(len(dataset)*percentage)

    if window==0: # No packing
        dataset = dataset.select(range(clip_len))
        dataset = list(dict.fromkeys(entry["text"] for entry in dataset))[:clip_len]
    else:
        # Use twice clip_len to ensure enough samples after packing
        if not (1<=clip_len*2<=len(dataset)):
            raise IndexError(f"Number or percentage out of bounds. You specified {clip_len} samples but there are only {len(dataset)} samples.")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
        dataset = dataset.select(range(clip_len*2))
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
