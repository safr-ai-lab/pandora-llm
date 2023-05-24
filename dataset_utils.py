from datasets import load_dataset
import torch
from transformers import AutoTokenizer
from itertools import groupby

def collate_fn(batch,tokenizer,length):
    tokens = [tokenizer.encode(example, return_tensors="pt", truncation=True, max_length=length) for example in batch]
    max_length = max([t.size(1) for t in tokens])
    tokens_padded = [torch.cat([t, t.new_zeros(t.size(0), max_length - t.size(1))], dim=1) for t in tokens]
    tokens_padded = torch.cat(tokens_padded, dim=0)
    return {
        "input_ids":tokens_padded,
        "labels":tokens_padded,
        "attention_mask": torch.tensor(tokens_padded>0,dtype=int)
    }

def load_train_pile_random_deduped(number=1000, percentage=None, seed=229, num_splits=2):
    '''
    Loads the random sample from training pile
    '''
    # print("WARNING: NOT UNPACKABLE BY PYTHIA BUG") 

    if num_splits % 2 !=0:
        print(f"Warning! Shadow models requires an even number of splits! You specified {num_splits} splits.")

    # Process clip length
    dataset = load_dataset("EleutherAI/pile-deduped-pythia-random-sampled",split="train").shuffle(seed=seed)
    clip_len = number if percentage is None else int(len(dataset)*percentage)
    if not (1<=clip_len<=len(dataset)):
        raise IndexError(f"Number or percentage out of bounds. You specified {clip_len} samples but there are only {len(dataset)} samples.")
    
    # Clip samples
    dataset = dataset.select(range(clip_len))
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    dataset = dataset.map(lambda x: {"text": tokenizer.decode(x["tokens"])},remove_columns=["index","tokens","is_memorized"])["text"]
    splits = [dataset[i * len(dataset)//num_splits : (i+1) * len(dataset) // num_splits] for i in range(num_splits)]

    return splits

def load_train_pile_random_duped(number=1000, percentage=None, seed=229, unpack=False, num_splits=2):
    '''
    Loads the random sample from duped training pile, unpacking the sequences (i.e., splitting by EOS token)
    '''
    def unpack(examples):
        chunks = []
        for sample in examples:
            result = []
            for k, group in groupby(sample, lambda y: y == tokenizer.eos_token_id):
                input_ids= list(group)
                if (not k) and len(input_ids)>20:
                    result.append(tokenizer.decode(input_ids))
            chunks.extend(result)
        return chunks
    
    if num_splits % 2 !=0:
        print(f"Warning! Shadow models requires an even number of splits! You specified {num_splits} splits.")
    
    # Process clip length 
    dataset = load_dataset("EleutherAI/pile-duped-pythia-random-sampled",split="train").shuffle(seed=seed)
    clip_len = number if percentage is None else int(len(dataset)*percentage)
    if not (1<=clip_len<=len(dataset)):
        raise IndexError(f"Number or percentage out of bounds. You specified {clip_len} samples but there are only {len(dataset)} samples.")
    
    # Clip samples
    dataset = dataset.select(range(clip_len))
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    clip_len = number if percentage is None else int(len(dataset)*percentage)

    # Unpack, and clip again. 
    dataset = dataset.map(lambda x: {"text": unpack(x["tokens"])},remove_columns=["index","tokens","is_memorized"],batched=True).shuffle(seed=seed)
    dataset = dataset.select(range(clip_len))["text"]
    splits = [dataset[i * len(dataset)//num_splits : (i+1) * len(dataset) // num_splits] for i in range(num_splits)]

    return splits

def load_train_pile_ordered(number=1000, percentage=None, seed=229, num_splits=2):
    '''
    Loads sample from training pile ordered as seen by Pythia
    '''
    if num_splits % 2 !=0:
        print(f"Warning! If using shadow models, use an even number of splits! You specified {num_splits} splits.")
    
    raise NotImplementedError()

    # dataset = load_dataset("EleutherAI/pile-deduped-pythia-random-sampled", split=f"train[:{percentage*100}%]")
    # dataset = list(dict.fromkeys(entry["text"] for entry in dataset))
    # splits = [dataset[i * len(dataset)//num_splits : (i+1) * len(dataset) // num_splits] for i in range(num_splits)]

    # return splits


def load_val_pile(number=1000, percentage=None, seed=229, num_splits=2):
    '''
    Loads the validation pile (NOT deduped), does an exact match deduplication and shuffling, and returns the specified number of splits
    '''
    if num_splits % 2 !=0:
        print(f"Warning! If using shadow models, use an even number of splits! You specified {num_splits} splits.")
    
    dataset = load_dataset("the_pile_val.py", split="validation").shuffle(seed=seed)
    clip_len = number if percentage is None else int(len(dataset)*percentage)
    if not (1<=clip_len<=len(dataset)):
        raise IndexError(f"Number or percentage out of bounds. You specified {clip_len} samples but there are only {len(dataset)} samples.")
    dataset = list(dict.fromkeys(entry["text"] for entry in dataset))[:clip_len]
    splits = [dataset[i * len(dataset)//num_splits : (i+1) * len(dataset) // num_splits] for i in range(num_splits)]

    return splits

def preprocess_val_data(tokenizer, window, save=True):
    """
    Get val data, essence we concat all documents in shuffled order together, with a single EOD token separating each doc from the next...
    
    Split back into strings for one 2048-token chunk, then the second block of 2048 tokens (non-overlapping), and so forth. 
    """
    validation = load_val_pile(number=0, percentage=1, seed=229, num_splits=1)[0]

    # Get tokens for everything, and add EOS_token between examples
    collated_docs_with_eos_split = []
    for idx, item in enumerate(validation):
        if idx % 1000 == 0:
            print(f"{(idx/1000)} / 214")
        data = item + tokenizer.eos_token
        collated_docs_with_eos_split += tokenizer(data)['input_ids']
    
    def round_up(num):
        if int(num) < num:
            return int(num) + 1
        return num

    # Turn tokens back into strings. 
    dataset = []
    for i in range(round_up(len(collated_docs_with_eos_split) / window)):
        dataset.append(tokenizer.decode(collated_docs_with_eos_split[window * i:window * (i+1)]))
    if save:  
        torch.save(dataset, "val_data.pt")

    return dataset