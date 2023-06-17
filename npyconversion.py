import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig

def process_user_npy(npypaths, tokenizer, name, number=1000, seq_length=2048, save=False):
    """
    Load a user npy file as input data. Assumes everything is packed and uniform size. 
    """
    nps = []
    for nppath in npypaths:
        nps.append(np.load(nppath))
    
    finaldata = np.concatenate(nps)
    dl = []
    dataset = []

    for i in range(number):
        string = tokenizer.decode(finaldata[i][:seq_length])
        dataset.append(string)

        dicti = {}
        dicti['input_ids'] = torch.tensor([finaldata[i][:seq_length]])
        dicti['labels'] = torch.tensor([finaldata[i][:seq_length]])
        dicti['attention_mask'] = torch.tensor([np.ones(seq_length)])
        dl.append(dicti)

    if save:
        torch.save(dataset, f"{name}_dataset.pt")
        torch.save(dl, f"{name}_dl.pt")
        
    return dl, dataset

model_title = "pythia-160m-deduped"
model_name = "EleutherAI/" + model_title

max_length = AutoConfig.from_pretrained(model_name).max_position_embeddings
tokenizer = AutoTokenizer.from_pretrained(model_name)

npypaths = ["batch0_bs1024.npy", "batch99_bs1024.npy", "batch199_bs1024.npy", "batch299_bs1024.npy", "batch399_bs1024.npy", "batch499_bs1024.npy", "batch599_bs1024.npy", "batch699_bs1024.npy", "batch799_bs1024.npy", "batch899_bs1024.npy", "batch999_bs1024.npy"]

for npy in npypaths:
    batchno = npy[5:npy.index("_")]
    process_user_npy([npy], tokenizer, f"batch{batchno}", save=True)