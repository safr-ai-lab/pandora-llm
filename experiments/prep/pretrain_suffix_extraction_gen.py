from pandora_llm.utils.generation_utils import generate_suffixes, compute_actual_prob
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
- Get first n_samples + args.after of train points in 229 shuffle. Set after to 150k. 
- For each point, add TRUE suffix to pt files + one GENERATION. 

python pretrain_suffix_extraction_gen.py --mod_size 1b --checkpoint step98000 --k 50 --x 50 --n_samples 2000 --n_gen 20 --after 150000
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

# To specify
parser.add_argument('--mod_size', action="store", type=str, required=True, help='Pythia Model Size')
parser.add_argument('--checkpoint', action="store", type=str, required=False, help='Model revision. If not specified, use last checkpoint.')
parser.add_argument('--k', action="store", type=int, default=50, required=False, help="For generation, do K-length prefix and X-length suffix where K+X <= 100.")
parser.add_argument('--x', action="store", type=int, default=50, required=False, help="For generation, do K-length prefix and X-length suffix where K+X <= 100.")
parser.add_argument('--n_samples', action="store", type=int, required=True, help='Number of prefixes to run generations on.')
parser.add_argument("--name", action="store", type=str, required=False, help="name for final files, in addition to mod size, parameters, k/x, etc.")
parser.add_argument("--after", action="store", type=int, required=False, default=0, help="if you want to create separate chunks, specify this guy to filter after a certain index.")
parser.add_argument("--n_gen", action="store", type=int, required=False, default=0, help="num other suffix gens to do per sample")
args = parser.parse_args()

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
min_length = 100

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
if args.after:
    dataset = dataset.select(range(args.after, args.after + clip_len))
else:
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

print(dataset)
print(f"Dataset Length: {len(dataset)}")
print(type(dataset))

i = 0

suffixes = []
suffix_strs = []
suffix_probs = []
true_suffix_toks = []
true_suffix_strs = []
prefix_toks = []
prefix_strs = []

for sample in dataset:
    samp_suffixes = []
    samp_suffixes_strs = []
    samp_suffixes_probs = []

    print(f"Processing sample {i}... ")
    prefix_tok = sample[:k]
    prefix_str = tokenizer.decode(sample[:k])
    prefixes = torch.tensor([prefix_tok])

    prefix_toks.append(prefix_tok)
    prefix_strs.append(prefix_str)
    true_suffix_tok = sample[k:k+x]
    true_suffix_toks.append(true_suffix_tok)
    true_suffix_str = tokenizer.decode(sample[k:k+x])
    true_suffix_strs.append(true_suffix_str)

    for j in range(args.n_gen):

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

        # get probability of generation
        probabilities = compute_actual_prob(prefixes, gen_suffixes, args, model, tokenizer, device, title=f"train_{k}_{x}_{num}_probs", kx_tup=(k, x))
        gen_prob = probabilities[0]
        print(f"Getting generated suffix probability... {gen_prob}")
        i += 1

        samp_suffixes.append(list(generation_tok))
        samp_suffixes_strs.append(generation_str)
        samp_suffixes_probs.append(gen_prob)
    
    suffixes.append(samp_suffixes)
    suffix_strs.append(samp_suffixes_strs)
    suffix_probs.append(samp_suffixes_probs)

torch.save(suffixes, f"{args.n_gen}_generations_toks_{title_str}.pt")
torch.save(suffix_strs, f"{args.n_gen}_generations_strs_{title_str}.pt")
torch.save(suffix_probs, f"{args.n_gen}_generations_probs_{title_str}.pt")
torch.save(true_suffix_toks, f"true_suffix_tok_{title_str}.pt")
torch.save(true_suffix_strs, f"true_suffix_strs_{title_str}.pt")
torch.save(prefix_toks, f"prefix_tok_{title_str}.pt")
torch.save(prefix_strs, f"prefix_strs_{title_str}.pt")