#!/bin/bash
python experiments/run_loss.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 1000 --pack --seed 229
# python experiments/run_mink.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 1000 --pack --seed 229
# python experiments/run_mope.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 1000 --pack --seed 229 \
#     --num_models 2 --noise_stdev 0.005 --noise_type gaussian
# python experiments/run_zlib.py --dataset_name pile-deduped --num_samples 1000 --pack --seed 229
# python experiments/run_gradnorm.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 1000 --pack --seed 229
# python experiments/run_flora.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 1000 --pack --seed 229
# python experiments/run_zliblora.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 1000 --pack --seed 229
# python experiments/run_alora.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 1000 --pack --seed 229
# python experiments/run_detectgpt.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 1000 --pack --seed 229 --num_perts 2
