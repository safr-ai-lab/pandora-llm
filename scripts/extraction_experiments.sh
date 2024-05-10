#!/bin/bash
python experiments/extraction/run_loss.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 1000 --pack --seed 229
