from generation_utils import generate_suffixes, compute_actual_prob
import argparse
from transformers.generation.utils import GenerationMixin, GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments, Trainer
from datasets import load_dataset
import os
import torch
from itertools import groupby
from tqdm import tqdm
from torch.utils.data import DataLoader

"""
This code is used for creating the pretrain data that is used to train the classifier (specific for MIA ranking).
- Get first n_samples of train points in 229 shuffle. For each point, add TRUE suffix to pt files + one GENERATION. 

python pretrain_train_data_gen.py --mod_size 1b --checkpoint step98000 --k 50 --x 50 --n_samples 10
"""

parser = argparse.ArgumentParser()

# Automatic
parser.add_argument("--top_k", type=int, default=24, help="Top-k filtering")
parser.add_argument("--top_p", type=float, default=0.8, help="Top-p (nucleus) filtering")
parser.add_argument("--typical_p", type=float, default=0.9, help="Typical-p filtering")
parser.add_argument("--temperature", type=float, default=0.58, help="Temperature for generation")
parser.add_argument("--repetition_penalty", type=float, default=1.04, help="Repetition penalty")
parser.add_argument("--bs", type=int, default=1, help="Batch size")
parser.add_argument('--deduped', type=bool, default=True, required=False, help='Use deduped models')
parser.add_argument("--seed", type=int, default=229, help="seed for randomness")
parser.add_argument("--start_from")
# To specify
parser.add_argument('--mod_size', action="store", type=str, required=True, help='Pythia Model Size')
parser.add_argument('--checkpoint', action="store", type=str, required=False, help='Model revision. If not specified, use last checkpoint.')
parser.add_argument('--k', action="store", type=int, default=50, required=False, help="For generation, do K-length prefix and X-length suffix where K+X <= 100.")
parser.add_argument('--x', action="store", type=int, default=50, required=False, help="For generation, do K-length prefix and X-length suffix where K+X <= 100.")
parser.add_argument('--n_samples', action="store", type=int, required=True, help='Number of prefixes to run generations on.')
parser.add_argument("--name", action="store", type=str, required=False, help="name for final files, in addition to mod size, parameters, k/x, etc.")
args = parser.parse_args()

# Model configs
seed = 229
device = "cuda" if torch.cuda.is_available() else "cpu"
model_revision = args.checkpoint
model_title = f"pythia-{args.mod_size}" + ("-deduped" if args.deduped else "")
model_name = "EleutherAI/" + model_title
model_cache_dir = "./"+ model_title + ("/"+model_revision if args.checkpoint else "")

model = AutoModelForCausalLM.from_pretrained(model_name, revision=model_revision, cache_dir=model_cache_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # not set by default

# title
if args.name:
    title_str = args.name
else:
    title_str = f"samp={args.n_samples}_size={args.mod_size}_k={args.k}_x={args.x}_ckpt={args.checkpoint}_topk={args.top_k}_top.p={args.top_p}_typical.p={args.typical_p}_rep.pen={args.repetition_penalty}_deduped={args.deduped}"

num = args.n_samples
k = args.k
x = args.x
min_length = 100 # need every sample to be at least this many tokens

# Make the dataset
def unpack(examples):
    chunks = []
    for sample in examples:
        result = []
        for k, group in groupby(sample, lambda y: y == tokenizer.eos_token_id):
            input_ids= list(group)
            if (not k) and len(input_ids)>min_length:
                result.append(input_ids)
        chunks.extend(result)
    return chunks

dataset = load_dataset("EleutherAI/pile-deduped-pythia-random-sampled",split="train",revision="d2c194e46b51d810f8aba4b0c4598af105d8b3ab").shuffle(seed=seed)
clip_len = num
if not (1<=clip_len<=len(dataset)):
    raise IndexError(f"Number or percentage out of bounds. You specified {clip_len} samples but there are only {len(dataset)} samples.")

# Clip samples
dataset = dataset.select(range(clip_len))
rep = dataset.map(lambda x: {"tokens": unpack(x["tokens"])},remove_columns=["index","tokens","is_memorized"],batched=True) # .shuffle(seed=seed)
dataset = rep.select(range(clip_len))["tokens"]

# Things to return
data_toks = []
data_strs = []
labels = []

# Generation config
generation_config = GenerationConfig()
generation_config.update(**{
    "min_length":100,
    "max_length":100,
    "do_sample":True,
    "pad_token_id":tokenizer.pad_token_id,
    "top_k": args.top_k,
    "top_p": args.top_p,
    "typical_p": args.typical_p,
    "temperature": args.temperature,
    "repetition_penalty": args.repetition_penalty,
})

# Make `Generation` directory
directory_path = "Generation"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print(f"Directory '{directory_path}' created successfully.")
else:
    print(f"Directory '{directory_path}' already exists. Using it!")

print(f"Dataset Length: {len(dataset)}")

i = 0
# Generation
with open(f"data_summary_{title_str}.txt", "w") as textfile:
    textfile.write("gen_token_hamming01\tgen_tok_prob\ttrue_tok_prob\tprefix\tsuffix\tgen_suffix\tprefix_tok\tsuffix_tok\tgen_suffix_tok\n")
    for sample in dataset:
        print(f"Processing sample {i}... ")
        prefix_tok = sample[:k]
        prefix_str = tokenizer.decode(sample[:k])
        true_suffix_tok = sample[k:k+x]
        true_suffix_str = tokenizer.decode(sample[k:k+x])

        prefixes = torch.tensor([prefix_tok])
        suffixes = torch.tensor([true_suffix_tok])

        # generation
        generations = generate_suffixes(
            model=model,
            prefixes=DataLoader(prefixes,batch_size=1),
            generation_config=generation_config,
            trials=1,
            accelerate=False
        )
        
        generation_tok = generations[0][k:k+x]
        generation_str = tokenizer.decode(generation_tok)
        gen_suffixes = torch.tensor([generation_tok])

        # get probability of true suffix 
        print("Getting true suffix probability...")
        probabilities = compute_actual_prob(prefixes, suffixes, args, model, tokenizer, device, title=f"train_{k}_{x}_{num}_probs", kx_tup=(k, x))
        true_prob = probabilities[0]

        # get probability of generation
        print("Getting generated suffix probability...")
        probabilities = compute_actual_prob(prefixes, gen_suffixes, args, model, tokenizer, device, title=f"train_{k}_{x}_{num}_probs", kx_tup=(k, x))
        gen_prob = probabilities[0]

        # compute hamming01 between true suffix and generated one
        hamming01 = sum(g == t for g, t in zip(generation_tok, true_suffix_tok)) / x

        pre_rep = prefix_str.replace("\t", "[tab]").replace("\n", "[newline]")
        suf_rep = true_suffix_str.replace("\t", "[tab]").replace("\n", "[newline]")
        gen_rep = generation_str.replace("\t", "[tab]").replace("\n", "[newline]")
        textfile.write(f"{hamming01}\t{gen_prob}\t{true_prob}\t{pre_rep}\t{suf_rep}\t{gen_rep}\t{prefix_tok}\t{true_suffix_tok}\t{generation_tok}\n")
        
        # info
        data_toks.append(sample[:k+x])
        data_strs.append(prefix_str + true_suffix_str) 
        labels.append(1)
        data_toks.append(list(prefix_tok) + list(generation_tok))
        data_strs.append(prefix_str+generation_str)
        labels.append(0)
        i += 1

# Output pt files
    
torch.save(data_toks, f"tokens_{title_str}.pt")
torch.save(data_strs, f"strs_{title_str}.pt")
torch.save(labels, f"labels_{title_str}.pt")

