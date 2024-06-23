from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
from transformers.generation.utils import GenerationMixin, GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList

def generate_suffixes(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    prefix_length: int,
    generation_config: GenerationConfig,
    num_generations: int,
    accelerate: bool,
) -> np.ndarray:
    """
    Generates from the model using the first `prefix_length` tokens from each sample in the dataloader.

    Returns a numpy array of shape (num_samples,num_generations,prefix_length+suffix_length)
    """
    generations = []
    if not accelerate:
        device = next(model.parameters()).device
    for trial in range(num_generations):
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"][:,:prefix_length]
            with torch.no_grad():
                generated_tokens = model.generate(
                    inputs=input_ids if accelerate else input_ids.to(device),
                    generation_config=generation_config
                ).cpu().detach()
                generated_tokens = torch.cat([generated_tokens, torch.full((1, generation_config.min_length - generated_tokens.shape[1]), generation_config.pad_token_id, dtype=torch.long)], dim=1)
                generations.extend(generated_tokens.numpy())    
    return np.array(generations).reshape(num_generations,len(dataloader.dataset),generation_config.max_length).transpose(1,0,2)

def calculate_sentence_probability(
    logits: torch.FloatTensor,
    input_ids: torch.LongTensor,
    condition_from_index: Optional[int] = 0,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
    **kwargs,
) -> torch.FloatTensor:
    """
    Calculates the probability that a sentence is decoded from the logits with given decoding strategy
    Operates in log space
    Uses logic of transformers/generation/utils.py
    """
    if generation_config is None:
        generation_config = GenerationConfig()
    generation_config.update(**kwargs)
    generation_config.validate()

    if generation_config.pad_token_id is None:
        raise ValueError("No pad_token set")

    batch_size = input_ids.size(0)
    sentence_probability = torch.zeros(batch_size)

    logits_processor = GenerationMixin()._get_logits_processor(generation_config=generation_config,
            input_ids_seq_length=input_ids.size(1),
            encoder_input_ids=None,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
    )
    logits_warper = GenerationMixin()._get_logits_warper(generation_config=generation_config)
    for pos in range(condition_from_index,input_ids.size(1) - 1):
        # Extract logits for the current token
        current_logits = logits[:, pos, :]
        prev_input_ids = input_ids[:,:(pos+1)]
        current_input_ids = input_ids[:,(pos+1)]
        current_logits = logits_processor(prev_input_ids, current_logits)
        current_logits = logits_warper(prev_input_ids, current_logits)
        if generation_config.do_sample:
            # Sample
            probs = F.log_softmax(current_logits, dim=-1)
        else:
            # Greedy decoding
            probs = torch.full((current_logits.shape[0],current_logits.shape[1]),-float("Inf"))
            probs[:,torch.argmax(current_logits, dim=-1)] = 0

        # Ignore Padding
        probs[:,generation_config.pad_token_id] = 0
        sentence_probability += probs.gather(1,current_input_ids.unsqueeze(1)).squeeze(-1)

    return torch.exp(sentence_probability)

def compute_dataloader_suffix_probability(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    prefix_length: int,
    generation_config: GenerationConfig,
) -> np.ndarray:
    device = next(model.parameters()).device
    probabilities = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            logits = model(batch["input_ids"].to(device)).logits.cpu()
            probs = calculate_sentence_probability(logits.cpu(),batch["input_ids"].cpu(),condition_from_index=prefix_length,generation_config=generation_config)
            probabilities.extend(probs.tolist())
    return np.array(probabilities)