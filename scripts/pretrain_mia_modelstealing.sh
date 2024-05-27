####################################################################################################
# (Supervised) MIAs in the Pretrained Setting (Grad Norms)
####################################################################################################
####################################################################################################
# Generate Model Stealing Features
####################################################################################################
# For training supervised classifier
python experiments/mia/run_modelstealing.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --method one_sided_projection --project_type normal --num_samples 10000 --start_index 150000 --pack --seed 229
# For evaluating supervised classifier
python experiments/mia/run_modelstealing.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --method one_sided_projection --project_type normal --num_samples 2000 --start_index 0 --pack --seed 229

####################################################################################################
# Model Stealing Attack
####################################################################################################
python experiments/mia/run_logreg.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --clf_num_samples 10000 --mia_num_samples 2000 --seed 229 \
    --feature_set x_grad_inf x_grad_2 x_grad_1 theta_grad_inf theta_grad_2 theta_grad_1 layerwise_grad_inf layerwise_grad_2 layerwise_grad_1 \
    --clf_pos_features results/ModelStealing/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_train.pt \
    --clf_neg_features results/ModelStealing/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_val.pt \
    --mia_train_features results/ModelStealing/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_train.pt \
    --mia_val_features results/ModelStealing/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_val.pt
python experiments/mia/run_nn.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --clf_num_samples 10000 --mia_num_samples 2000 --seed 229 \
    --feature_set x_grad_inf x_grad_2 x_grad_1 theta_grad_inf theta_grad_2 theta_grad_1 layerwise_grad_inf layerwise_grad_2 layerwise_grad_1 \
    --clf_pos_features results/ModelStealing/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_train.pt \
    --clf_neg_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_val.pt \
    --mia_train_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_train.pt \
    --mia_val_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_val.pt