import re
import math
import subprocess
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from .Attack import MIA
from .LOSS import compute_input_ids_cross_entropy

####################################################################################################
# MAIN CLASS
####################################################################################################
class DetectGPT(MIA):
    """
    DetectGPT thresholding attack
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_statistic(self, dataloader, num_batches=None, device=None, model_half=None, accelerator=None, detect_args=None):
        """
        Compute the DetectGPT statistic for a given dataloader.

        Args:
            dataloader (DataLoader): input data to compute statistic over
            num_batches (Optional[int]): number of batches of the dataloader to compute over.
                If None, then comptues over whole dataloader
            device (Optional[str]): e.g. "cuda"
            model_half (Optional[bool]): whether to use model_half
            accelerator (Optional[Accelerator]): accelerator object
            detect_args (Optional[dict]): detectgpt args
        Returns:
            torch.Tensor or list: loss of input IDs
        """
        if self.model is None:
            raise Exception("Please call .load_model() to load the model first.")
        if accelerator is not None:
            self.model, dataloader, = accelerator.prepare(self.model, dataloader)
        return compute_dataloader_cross_entropy_batch(model=self.model,dataloader=dataloader,num_batches=num_batches,device=device,model_half=model_half,detect_args=detect_args).cpu()

####################################################################################################
# MAIN CLASS
####################################################################################################

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

def compute_dataloader_cross_entropy_batch(model, dataloader, device=None, num_batches=None, samplelength=None, accelerator=None, model_half=True, detect_args=None):    
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
        if model_half:
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
        if num_batches is not None and batchno >= num_batches:
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

PATTERN = re.compile(r"<extra_id_\d+>")
SPLIT_LEN = 64
BSIZE_MULT = 32


def pad_sequences_to_length(model_output, desired_length, pad_token_id):
    """
    Pads each sequence in the model's output tensor to a certain length.

    Parameters:
    model_output (torch.Tensor): The output tensor from the model
    desired_length (int): The desired length for each sequence
    pad_token_id (int): The token ID to use for padding

    Returns:
    torch.Tensor: The padded output tensor
    """
    # Pad sequences to the desired length
    padded_output = F.pad(model_output, pad=(0, desired_length - model_output.shape[1]), mode='constant', value=pad_token_id)
    
    return padded_output

def split_text(text, maxlen=64):
    """
    Splits a given text into 64-word or maxlen-word chunks (depends on spaces for word boundaries)
    Input: text (str)
    Output: chunks (list of str, each str max maxlen)
    """
    text = text.split(' ')
    if len(text) <= maxlen:
        return ' '.join(text)
    else:
        num_chunks = int(np.ceil(len(text) / maxlen))
        chunks = [text[i*maxlen:(i+1)*maxlen] for i in range(num_chunks)]
        return [' '.join(s) for s in chunks]

def mask_text(text, args, ceil_pct=False):
    """
    This function masks ceil_pct percent of the words in text

    text: the chunk of text to mask (str)
    args:
        buffer_size: 1 (int),
        pct_words_masked: .2,
        span_length: 2,
    
    return: masked chunk of text (str)
    """
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    # Not sure where this computation comes from?
    n_spans = args["pct_words_masked"] * len(tokens) / (args["span_length"] + args["buffer_size"] * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - args["span_length"])
        end = start + args["span_length"]
        search_start = max(0, start - args["buffer_size"])
        search_end = min(len(tokens), end + args["buffer_size"])
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    tokens = tokens + [mask_string]
    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    #assert num_filled == n_masks + 1, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def count_masks(texts):
    """
    Returns a list with the lengths of all extra_id tokens
    """
    return [len([x for x in text.split() if x.startswith("<extra_id_")])-1 for text in texts]


def chunk_list(lst, sizes):
    """
    Defines a list of lists where the length of element i is decided by sizes[i]
    """
    iterator = iter(lst)
    return [[next(iterator) for _ in range(size)] for size in sizes]


def get_batch_outputs(flattened_texts, mask_tokenizer, mask_model, args, num_texts=BSIZE_MULT):
    """
    This function gets generations from the mask_model 

    flattened_texts: list of strs (masked chunks of text)
    mask_tokenizer: tokenizer of model that will fill in mask (e.g., T5-small)
    mask_model: model that will fill in mask (e.g., T5-small)
    args: 
    - mask_top_p: top p arg for decoding/sampling from mask model
    - device: cpu or cuda

    return: 2-tuple:
    - batch_outputs: a list of generations
    - max_len_gen: the maximum length generation
    """
    seq_len = len(flattened_texts)
    max_len_gen = 0
    seq_index = 0
    batch_outputs = []
    while seq_index < seq_len:
        flattened_texts_part = flattened_texts[seq_index:min([seq_len, seq_index + BSIZE_MULT])]
        tokenized_texts = mask_tokenizer(flattened_texts_part, return_tensors="pt", padding=True).to(args["device"])
        part_outputs = mask_model.generate(**tokenized_texts, do_sample=True, top_p=args["mask_top_p"], num_return_sequences=1, num_beams=1, max_length=3*SPLIT_LEN)
        max_len_gen = max([max_len_gen, part_outputs.size()[1]])
        batch_outputs.append(part_outputs)
        seq_index += BSIZE_MULT
    return batch_outputs, max_len_gen

def replace_masks_extract_fills(texts, mask_tokenizer, mask_model, args, printflag=False):
    """
    texts: doubly nested list of texts that are chunked and masked (list of texts which are list of masked str)
    mask_tokenizer: tokenizer of model that will fill in mask (e.g., T5-small)
    mask_model: model that will fill in mask (e.g., T5-small)

    return: triply nested list of [list of texts which are lists of chunks which are lists of texts in between mask tokens]
    """
    # prepare the inputs
    flattened_texts = [c for text in texts for c in text] # list of strs (chunks of masked texts)
    
    batch_outputs, max_len_gen = get_batch_outputs(flattened_texts, mask_tokenizer, mask_model, args, num_texts=BSIZE_MULT)
        
    # pad all the outputs and concatenate 
    outputs = pad_sequences_to_length(batch_outputs[0], desired_length=max_len_gen, pad_token_id=mask_tokenizer.pad_token_id)
    for k in range(1, len(batch_outputs)):
        next_padded_output = pad_sequences_to_length(batch_outputs[k], desired_length=max_len_gen, pad_token_id=mask_tokenizer.pad_token_id)
        outputs = torch.cat([outputs, next_padded_output], dim=0)

    # decode all the outputs
    decoded_text = mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

    filled_text = [x.replace("<pad>", "").replace("</s>", "").strip() for x in decoded_text]
    
    # reshape the list to the original structure (list of texts which are lists of chunks with masked words)
    filled_texts = list(chunk_list(filled_text, [len(lst) for lst in texts]))
    
    # return the text in between each matched mask token
    extracted_fills = [[PATTERN.split(chunk)[1:-1] for chunk in text] for text in filled_texts]
    
    # remove whitespace around each fill
    extracted_fills = [[[fill.strip() for fill in chunk] for chunk in text] for text in extracted_fills]
    
    return extracted_fills

def apply_extracted_fills(masked_texts, extracted_fills, printflag=False):
    """
    masked_texts: doubly nested list of texts that are chunked and masked (list of texts which are list of masked str)
    extracted_fills: triply nested list of [list of texts which are lists of chunks which are lists of texts in between mask tokens]
    """
    # split masked text into tokens, only splitting on spaces (not newlines)
    texts = []
    for text_id in range(len(masked_texts)):
      if printflag:
        print(f'filling in text {text_id}')
      masked_text = masked_texts[text_id]
      extracted_fill = extracted_fills[text_id]
      n_expected = count_masks(masked_text)

      # replace each mask token with the corresponding fill
      text = ''
      for idx, (masked_chunk, fills, n) in enumerate(zip(masked_text, extracted_fill, n_expected)):
          # handle case where there are no masks
        #   print(idx, (masked_chunk, fills, n))

          if n == -1: 
              continue
          masked_chunk = masked_chunk.split(' ')
          if printflag:
            print(f'filling in chunk {idx} of text {text_id}')
          if len(fills) < n:
              print('insufficient # of fills on chunk')
              for fill_idx in range(n):
                  masked_chunk[masked_chunk.index(f"<extra_id_{fill_idx}>")] = ''
                  # remove the last mask
              masked_chunk[masked_chunk.index(f"<extra_id_{n}>")] = ''
          else:
              for fill_idx in range(n):
                masked_chunk[masked_chunk.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]
              # remove the last mask
              masked_chunk[masked_chunk.index(f"<extra_id_{n}>")] = ''
          
          text += ' '.join(masked_chunk)
      texts.append(text)
    return texts

def perturb_input_ids(input_id, args, base_tokenizer, mask_tokenizer, masked_model): 
    """
    Generates perturbed text with masked_model providing replacements 
    """
    with torch.no_grad():
        # texts = [base_tokenizer.decode(input_id) for _ in range(args["num_perts"])]
        # masked_texts = [[mask_text (chunk, args, False) for chunk in split_text(text, SPLIT_LEN)] for text in texts]
        
        texts = [[base_tokenizer.decode(input_id[i*SPLIT_LEN:(i+1)*SPLIT_LEN]) for i in range(math.ceil(len(input_id)/SPLIT_LEN))] for _ in range(args["num_perts"])]
        masked_texts = [[mask_text (chunk, args, False) for chunk in text] for text in texts]
                
        extracted_fills = replace_masks_extract_fills(masked_texts,mask_tokenizer, masked_model, args)
        perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [[mask_text (chunk, args, False) for chunk in split_text(text, SPLIT_LEN)]for idx,text in enumerate(texts) if idx in idxs]
            extracted_fills = replace_masks_extract_fills(masked_texts, mask_tokenizer, masked_model, args)
            new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
        # convert back to input_ids 
        max_length=args["model_max_length"]
        tokens = [base_tokenizer.encode(x, return_tensors="pt", truncation=True, max_length=max_length) for x in perturbed_texts]
        tokens_padded = [torch.cat([t, t.new_zeros(t.size(0), max_length - t.size(1))], dim=1) for t in tokens]
        tokens_padded = torch.cat(tokens_padded, dim=0)
        return tokens_padded