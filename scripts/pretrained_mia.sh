#!/bin/bash
# python run_loss.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --n_samples 1000 --pack --seed 229
python run_mink.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --n_samples 1000 --pack --seed 229
# python run_grad.py --mod_size 70m --deduped --checkpoint step98000 --n_samples 200 --pack --seed 229
# python run_zliblora.py --mod_size 70m --deduped --checkpoint step98000 --n_samples 200 --pack --seed 229
# python run_mope.py --mod_size 70m --deduped --checkpoint step98000 --n_samples 200 --pack --seed 229
