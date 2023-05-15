from datasets import load_dataset
import torch

def load_val_pile(percentage=0.025,seed=229,num_splits=2):
    '''
    Loads the validation pile (NOT deduped), does an exact match deduplication and shuffling, and returns the specified number of splits
    '''
    if num_splits % 2 !=0:
        print(f"Warning! Shadow models requires an even number of splits! You specified {num_splits} splits.")
    
    dataset = list(dict.fromkeys(entry["text"] for entry in load_dataset("the_pile_val.py", split="validation").shuffle(seed=229)))[:int(len(dataset)*percentage)]

    splits = [dataset[i * len(dataset)//num_splits : (i+1) * len(dataset) // num_splits] for i in range(num_splits)]

    return splits

# def collate_fn(batch):
#     tokens = [tokenizer.encode(example, return_tensors="pt", truncation=True,max_length=model.config.max_position_embeddings) for example in batch]
#     max_length = max([t.size(1) for t in tokens])
#     tokens_padded = [torch.cat([t, t.new_zeros(t.size(0), max_length - t.size(1))], dim=1) for t in tokens]
#     tokens_padded = torch.cat(tokens_padded, dim=0)
#     return {
#         "input_ids":tokens_padded,
#         "labels":tokens_padded,
#         "attention_mask": torch.tensor(tokens_padded>0,dtype=int)
#     }