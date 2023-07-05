import matplotlib.pyplot as plt
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
import random
import argparse
import datetime
import os
import json
import functools

import time

# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")

def tokenize_and_mask(text, args, ceil_pct=False):

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
    
    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


# replace each masked span with a sample from T5 mask_model
def replace_masks(texts, mask_tokenizer, mask_model, args):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(args["device"])
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=args["mask_top_p"], num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills

def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts

def convert(s):
 
    # initialization of string to ""
    new = ""
 
    # traverse in the string
    for x in s:
        new += x
 
    # return string
    return new


def perturb_texts_(texts, args, base_tokenizer, masked_model, ceil_pct=False):
    masked_texts = [tokenize_and_mask(x, args, ceil_pct) for x in texts]
    raw_fills = replace_masks(masked_texts, base_tokenizer, masked_model, args)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
        masked_texts = [tokenize_and_mask(x, args, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = replace_masks(masked_texts, base_tokenizer, masked_model, args)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1

    return convert(perturbed_texts)

if __name__ == "__main__":
    
    
    args = {'buffer_size':1, 'device':'cuda', 'mask_top_p': 10, 'pct_words_masked':.2, 'span_length':2}
    model_name = 't5-small'
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model.to(args['device'])

    texts = ['Four young Athenians are in a romantic tangle. Lysander and Demetrius love Hermia; she loves Lysander and her friend Helena loves Demetrius.', ' Hermia’s father, Egeus, commands Hermia to marry Demetrius, and Theseus supports the father’s right. All four young Athenians end up in the woods, where Robin Goodfellow, who serves the fairy king Oberon, puts flower juice on the eyes of Lysander, and then Demetrius, unintentionally causing both to love Helena. Oberon, who is quarreling with his wife, Titania, uses the flower juice on her eyes. She falls in love with Bottom, who now, thanks to Robin Goodfellow, wears an ass’s head.']
    perturbed_text = perturb_texts_(texts, args, base_tokenizer=tokenizer, masked_model=model)
    print(perturbed_text)