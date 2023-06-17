import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mod_size', action="store", type=str, required=True, help='Pythia Model Size')
    args = parser.parse_args()
    
    perturb_amts = [0.05, 0.01, 0.005, 0.001]
    N = 20

    for p in perturb_amts:
        os.system(f"python run_mope.py --mod_size {args.mod_size} --n_models {N} --sigma {p} --pack --n_samples 2000 --deduped --checkpoint step98000 --model_half --noise 1")

if __name__ == "__main__":
    main()