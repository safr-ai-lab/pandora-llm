import numpy as np
import re
import random
import argparse
import datetime
import os
import json
import functools
from dataset_utils import *
import pdb 
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoConfig
import time

PATTERN = re.compile(r"<extra_id_\d+>")
SPLIT_LEN = 64

def split_text(text, maxlen=64):
    text = text.split(' ')
    if len(text) <= maxlen:
      return ' '.join(text)
    else:

      num_chunks = int(np.ceil(len(text) / maxlen))
      chunks = [text[i*maxlen:(i+1)*maxlen] for i in range(num_chunks)]
      return [' '.join(s) for s in chunks]

def mask_text (text, args, ceil_pct=False):


    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

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
    return [len([x for x in text.split() if x.startswith("<extra_id_")])-1 for text in texts]


# replace each masked span with a sample from T5 mask_model
def replace_masks_extract_fills(texts, mask_tokenizer, mask_model, args, printflag=False):
    filled_texts = []
    for num in range(len(texts)):
      if printflag:
         print(f'filling masked text {num +1}')
      n_expected = count_masks(texts[num])
      stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
      tokens = mask_tokenizer(texts[num], return_tensors="pt", padding=True).to(args["device"])
      outputs = mask_model.generate(**tokens, do_sample=True, top_p=args["mask_top_p"], num_return_sequences=1, eos_token_id=stop_id, max_length=3*SPLIT_LEN)
      decoded_text =  mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)
      filled_text = [x.replace("<pad>", "").replace("</s>", "").strip() for x in decoded_text]
      filled_texts.append(filled_text)
      
    
    # return the text in between each matched mask token
    extracted_fills = [[PATTERN.split(chunk)[1:-1] for chunk in text] for text in filled_texts]
    # remove whitespace around each fill
    extracted_fills = [[[fill.strip() for fill in chunk] for chunk in text] for text in extracted_fills]
    return extracted_fills

def chunk_list(lst, sizes):
    iterator = iter(lst)
    return [[next(iterator) for _ in range(size)] for size in sizes]

def replace_masks_extract_fills_vec(texts, mask_tokenizer, mask_model, args, printflag=False):
    # prepare the inputs
    tokenized_texts = mask_tokenizer(texts, return_tensors="pt", padding=True).to(args["device"])

    # compute maximum number of expected tokens for each text
    n_expected = [count_masks(text) for text in texts]
    #max_n_expected = max(n_expected)
    #stop_id = mask_tokenizer.encode(f"<extra_id_{max_n_expected}>")[0]
    
    # generate the outputs in a single call
    outputs = mask_model.generate(**tokenized_texts, do_sample=True, top_p=args["mask_top_p"], num_return_sequences=1, max_length=3*SPLIT_LEN)
    
    # decode the outputs
    decoded_text = mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)
    filled_text = [x.replace("<pad>", "").replace("</s>", "").strip() for x in decoded_text]
    
    # reshape the list to the original structure
    filled_texts = list(chunk_list(filled_text, n_expected))
    
    # return the text in between each matched mask token
    extracted_fills = [[PATTERN.split(chunk)[1:-1] for chunk in text] for text in filled_texts]
    
    # remove whitespace around each fill
    extracted_fills = [[[fill.strip() for fill in chunk] for chunk in text] for text in extracted_fills]
    
    return extracted_fills

def apply_extracted_fills(masked_texts, extracted_fills, printflag=False):
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

def convert(s):

    # initialization of string to ""
    new = ""

    # traverse in the string
    for x in s:
        new += x

    # return string
    return new


def perturb_texts_(texts, args, mask_tokenizer, masked_model, ceil_pct=False):

    masked_texts = [[mask_text (chunk, args, False) for chunk in split_text(text, SPLIT_LEN)] for text in texts]
    extracted_fills = replace_masks_extract_fills(masked_texts, mask_tokenizer, masked_model, args)
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

    return convert(perturbed_texts)


def perturb_input_ids(input_id, args, base_tokenizer, mask_tokenizer, masked_model): 
    texts = [base_tokenizer.decode(input_id) for _ in range(args["num_perts"])]
    masked_texts = [[mask_text (chunk, args, False) for chunk in split_text(text, SPLIT_LEN)] for text in texts]
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
    # convert backt to input_ids 
    max_length=args["model_max_length"]
    tokens = [base_tokenizer.encode(x, return_tensors="pt", truncation=True, max_length=max_length) for x in perturbed_texts]
    tokens_padded = [torch.cat([t, t.new_zeros(t.size(0), max_length - t.size(1))], dim=1) for t in tokens]
    tokens_padded = torch.cat(tokens_padded, dim=0)
    return tokens_padded


if __name__ == "__main__":
    model_name = 't5-small'
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model.to('cuda')
    args = {'buffer_size':1, 'device':'cuda', 'mask_top_p': 10, 'pct_words_masked':.2, 'span_length':2, 'num_perts': 5, 'truncation_length':10000}

    texts = 'In A Midsummer Night’s Dream, residents of Athens mix with fairies from a local forest, with comic results. In the city, Theseus, Duke of Athens, is to marry Hippolyta, queen of the Amazons. Bottom the weaver and his friends rehearse in the woods a play they hope to stage for the wedding celebrations. Four young Athenians are in a romantic tangle. Lysander and Demetrius love Hermia; she loves Lysander and her friend Helena loves Demetrius. Hermia’s father, Egeus, commands Hermia to marry Demetrius, and Theseus supports the father’s right. All four young Athenians end up in the woods, where Robin Goodfellow, who serves the fairy king Oberon, puts flower juice on the eyes of Lysander, and then Demetrius, unintentionally causing both to love Helena. Oberon, who is quarreling with his wife, Titania, uses the flower juice on her eyes. She falls in love with Bottom, who now, thanks to Robin Goodfellow, wears an ass’s head.As the lovers sleep, Robin Goodfellow restores Lysander’s love for Hermia, so that now each young woman is matched with the man she loves. Oberon disenchants Titania and removes Bottom’s ass’s head. The two young couples join the royal couple in getting married, and Bottom rejoins his friends to perform the play.'
    perturbed_text = perturb_texts_(texts, args, tokenizer, masked_model=model)
    print(perturbed_text)
 # test perturb ids 
    mod_size = '70m'
    model_title = f"pythia-{mod_size}" + "-deduped" 
    model_name = "EleutherAI/" + model_title
    model_revision = 'step98000'
    model_cache_dir = "./"+ model_title + "/"+model_revision 
    base_tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = AutoConfig.from_pretrained(model_name).max_position_embeddings
    input_ids = collate_fn(texts, base_tokenizer, 1024)
    x = input_ids["input_ids"][1]
    perturbed_x_inputs = perturb_input_ids(x, args, base_tokenizer, tokenizer, model, ceil_pct=False)


