#!/bin/bash

# python experiments/mia/run_gradnorm.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 1000 --start_index 0 --pack --seed 229
# python experiments/mia/run_gradnorm.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 1000 --start_index 10000 --pack --seed 229
# python experiments/mia/run_jl.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 1000 --start_index 0 --pack --seed 229 \
#     --compute_jl --compute_balanced_jl --compute_model_stealing
# python experiments/mia/run_jl.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 1000 --start_index 10000 --pack --seed 229 \
#     --compute_jl --compute_balanced_jl --compute_model_stealing
python experiments/mia/run_logreg.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --clf_num_samples 1000 --mia_num_samples 1000 --seed 229 \
    --feature_set x_grad_inf theta_grad_inf layerwise_grad_inf x_grad_2 theta_grad_2 layerwise_grad_2 x_grad_1 theta_grad_1 layerwise_grad_1 balanced_jl \
    --clf_pos_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=1000_S=10000_seed=229_train.pt results/JL/JL_EleutherAI-pythia-70m-deduped_step98000_N=1000_S=10000_seed=229_train.pt \
    --clf_neg_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=1000_S=10000_seed=229_val.pt results/JL/JL_EleutherAI-pythia-70m-deduped_step98000_N=1000_S=10000_seed=229_val.pt  \
    --mia_train_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229_train.pt results/JL/JL_EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229_train.pt  \
    --mia_val_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229_val.pt results/JL/JL_EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229_val.pt  
python experiments/mia/run_nn.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --clf_num_samples 1000 --mia_num_samples 1000 --seed 229 \
    --feature_set x_grad_inf theta_grad_inf layerwise_grad_inf x_grad_2 theta_grad_2 layerwise_grad_2 x_grad_1 theta_grad_1 layerwise_grad_1 balanced_jl \
    --clf_pos_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=1000_S=10000_seed=229_train.pt results/JL/JL_EleutherAI-pythia-70m-deduped_step98000_N=1000_S=10000_seed=229_train.pt \
    --clf_neg_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=1000_S=10000_seed=229_val.pt results/JL/JL_EleutherAI-pythia-70m-deduped_step98000_N=1000_S=10000_seed=229_val.pt  \
    --mia_train_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229_train.pt results/JL/JL_EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229_train.pt  \
    --mia_val_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229_val.pt results/JL/JL_EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229_val.pt  