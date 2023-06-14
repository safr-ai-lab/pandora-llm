import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, GPTNeoXForCausalLM, AutoTokenizer, AutoConfig
from attack_utils import *
from dataset_utils import *
from LOSS import LOSS
from MoPe import MoPe
import time
import argparse
from accelerate import Accelerator
import os
import csv

"""
Sample command line prompt (no acceleration)
python run_generation.py --mod_size 70m --deduped --checkpoint step98000 --n_samples 1000
Sample command line prompt (with acceleration)
accelerate launch run_generation.py --accelerate --mod_size 70m --deduped --checkpoint step98000 --n_samples 1000
"""

def write_results(tokenizer,prefixes,suffixes,guesses,save_name):
    with open(save_name,"w") as f:
        for row in range(len(prefixes)):
            prefix = tokenizer.decode(prefixes[row])
            guess = tokenizer.decode(guesses[row])
            suffix = tokenizer.decode(suffixes[row])
            f.write(f"Example {row}\n")
            f.write("Prefix: "+prefix.replace("\n","")+"\n")
            f.write("Suffix: "+suffix.replace("\n","")+"\n")
            f.write("Guess: "+guess.replace("\n","")+"\n\n")

def generate(attack,prefixes,suffix_length,bs,device,method="loss"):
    generations = []
    losses = []
    for off in tqdm(range(0, len(prefixes), bs)):
        prompt_batch = prefixes[off:off+bs]
        prompt_batch = np.stack(prompt_batch, axis=0)
        input_ids = torch.tensor(prompt_batch, dtype=torch.int64)
        with torch.no_grad():
            # 1. Generate outputs from the model
            generated_tokens = model.generate(
                input_ids.to(device),
                max_length=prefixes.shape[1]+suffix_length,
                do_sample=True, 
                top_k=10,
                top_p=1,
                pad_token_id=50256
            ).cpu().detach()

            # 2. Compute each sequence's probability, excluding EOS and SOS.
            outputs = model(
                generated_tokens.cuda(),
                labels=generated_tokens.cuda(),
            )
            logits = outputs.logits.cpu().detach()
            logits = logits[:, :-1].reshape((-1, logits.shape[-1])).float()
            loss_per_token = torch.nn.functional.cross_entropy(
                logits, generated_tokens[:, 1:].flatten(), reduction='none')
            loss_per_token = loss_per_token.reshape((-1, prefixes.shape[1]+suffix_length - 1))[:,-suffix_length-1:-1]
            likelihood = loss_per_token.mean(1)
            
            generations.extend(generated_tokens.numpy())
            losses.extend(likelihood.numpy())
        
    return np.atleast_2d(generations), np.atleast_2d(losses).reshape((len(generations), -1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefixes', action="store", required=True, help='.npy file of prefixes')
    parser.add_argument('--suffixes', action="store", required=True, help='.npy file of suffixes')
    parser.add_argument('--mod_size', action="store", type=str, required=True, help='Pythia Model Size')
    parser.add_argument('--deduped', action="store_true", required=False, help='Use deduped models')
    parser.add_argument('--checkpoint', action="store", type=str, required=False, help='Model revision. If not specified, use last checkpoint.')
    parser.add_argument('--n_samples', action="store", type=int, required=True, help='Dataset size')
    parser.add_argument('--n_iterations', action="store", type=int, required=False, help='How many generations per prompt')
    parser.add_argument('--suffix_length', action="store", type=int, required=False, default=50, help='Length of suffix')
    parser.add_argument('--seed', action="store", type=int, required=False, default=229, help='Seed')
    parser.add_argument('--bs', action="store", type=int, required=False, default=1, help='Batch size')
    parser.add_argument('--accelerate', action="store_true", required=False, help='Use accelerate')
    parser.add_argument('--model_half', action="store_true", required=False, help='Use half precision (fp16). 1 for use; 0 for not.')
    args = parser.parse_args()

    accelerator = Accelerator() if args.accelerate else None

    ## Other parameters
    model_revision = args.checkpoint
    seed = args.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_title = f"pythia-{args.mod_size}" + ("-deduped" if args.deduped else "")
    model_name = "EleutherAI/" + model_title
    model_cache_dir = "./"+ model_title + ("/"+model_revision if args.checkpoint else "")

    ## Load model and training and validation dataset
    start = time.perf_counter()

    prefixes = np.load(args.prefixes).astype(np.int64)[-args.n_samples:]
    suffixes = np.load(args.suffixes).astype(np.int64)[-args.n_samples:]

    print("Prefixes,Suffixes",prefixes.shape,suffixes.shape)

    # model = GPTNeoXForCausalLM.from_pretrained(model_name, revision=model_revision, cache_dir=model_cache_dir).half().eval().to(device)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").half().eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

    model_name = "EleutherAI/gpt-neo-1.3B"
    model_revision = None
    model_cache_dir = None
    attack = MoPe(model_name, model_revision=model_revision, cache_dir=model_cache_dir)

    all_generations, all_losses = [], []
    for trial in range(args.n_iterations):
        # generations, losses = attack.generate(prefixes,args.suffix_length,args.bs,device)
        generations, losses = attack.generate(prefixes,args.suffix_length,args.bs,device,tokenizer,2,0.005)
        all_generations.append(generations)
        all_losses.append(losses)
    generations = np.stack(all_generations, axis=1)
    losses = np.stack(all_losses, axis=1)

    # print("Generations,Losses",generations.shape,losses.shape)

    for generations_per_prompt in [1, 10, 100]:
        limited_generations = generations[:, :generations_per_prompt, :]
        limited_losses = losses[:, :generations_per_prompt, :]

        # print("Limited G,L",limited_generations.shape,limited_losses.shape)
        
        axis0 = np.arange(generations.shape[0])
        axis1 = limited_losses.argmin(1).reshape(-1)

        # print("Axes 0,1",axis0,axis1)

        guesses = limited_generations[axis0, axis1, -args.suffix_length:]
        batch_losses = limited_losses[axis0, axis1]
        
        # print("Guesses","Batch L",guesses.shape)
        
        with open("guess%d.csv"%generations_per_prompt, "w") as file_handle:
            print("Writing out guess with", generations_per_prompt)
            writer = csv.writer(file_handle)
            writer.writerow(["Example ID", "Suffix Guess"])

            order = np.argsort(batch_losses.flatten())
            
            # Write out the guesses
            for example_id, guess in zip(order, guesses[order]):
                row_output = [
                    example_id, str(list(guesses[example_id])).replace(" ", "")
                ]
                writer.writerow(row_output)
            
            print(f"Accuracy {np.sum(np.all(guesses==suffixes, axis=-1)) / 100}%")

        write_results(tokenizer,prefixes,suffixes,guesses,"results.txt")
    return

    ## Run LOSS attack

    config_loss = {
        "training_dl": training_dataloader,
        "validation_dl": validation_dataloader,
        "bs": args.bs,
        "nbatches": args.n_samples,
        "samplelength": args.sample_length,
        "device": device,
        "accelerator": accelerator,
        "model_half": args.model_half
    }

    end = time.perf_counter()

    if accelerator is None or accelerator.is_main_process:
        print(f"- Code initialization time was {end-start} seconds.")

    start = time.perf_counter()

    LOSSer = LOSS(model_name, model_revision=model_revision, cache_dir=model_cache_dir)

    LOSSer.inference(config_loss)
    LOSSer.save()

    LOSSer.attack_plot_ROC(log_scale = False, show_plot=False)
    LOSSer.attack_plot_ROC(log_scale = True, show_plot=False)

    end = time.perf_counter()

    if accelerator is None or accelerator.is_main_process:
        print(f"- LOSS at {args.mod_size} and {args.n_samples} samples took {end-start} seconds.")

if __name__ == "__main__":
    main()