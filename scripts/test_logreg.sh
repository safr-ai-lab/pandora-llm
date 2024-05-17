#!/bin/bash
python experiments/mia/run_logreg.py --feature_set data --model_name EleutherAI/pythia-160m-deduped --model_revision step98000 --clf_num_samples 15000 --mia_num_samples 2000 \
    --clf_pos_features /n/holylabs/LABS/sneel_lab/Lab/jgwang/final_repo/experiments/prep/Data/jllayers_train_model=EleutherAI-pythia-160m-deduped_samp=15000_seed=229_projseed=229_half=False_start=0.pt \
    --clf_neg_features /n/holylabs/LABS/sneel_lab/Lab/jgwang/final_repo/experiments/prep/Data/jllayers_val_model=EleutherAI-pythia-160m-deduped_samp=15000_seed=229_projseed=229_half=False_start=0.pt \
    --mia_train_features /n/holylabs/LABS/sneel_lab/Lab/jgwang/final_repo/experiments/prep/Data/jllayers_train_model=EleutherAI-pythia-160m-deduped_samp=2000_seed=229_projseed=229_half=False_start=150000.pt \
    --mia_val_features /n/holylabs/LABS/sneel_lab/Lab/jgwang/final_repo/experiments/prep/Data/jllayers_val_model=EleutherAI-pythia-160m-deduped_samp=2000_seed=229_projseed=229_half=False_start=150000.pt
