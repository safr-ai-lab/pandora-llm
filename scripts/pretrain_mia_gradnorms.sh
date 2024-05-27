####################################################################################################
# (Supervised) MIAs in the Pretrained Setting (Grad Norms)
####################################################################################################
####################################################################################################
# Generate Gradient Norm Features
####################################################################################################
# For training supervised classifier
python experiments/mia/run_gradnorm.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 10000 --start_index 150000 --pack --seed 229
# For evaluating supervised classifier
python experiments/mia/run_gradnorm.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 2000 --start_index 0 --pack --seed 229

####################################################################################################
# Full Supervised Attack
####################################################################################################
# All gradient information
python experiments/mia/run_logreg.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --clf_num_samples 10000 --mia_num_samples 2000 --seed 229 \
    --feature_set x_grad_inf x_grad_2 x_grad_1 theta_grad_inf theta_grad_2 theta_grad_1 layerwise_grad_inf layerwise_grad_2 layerwise_grad_1 \
    --clf_pos_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_train.pt \
    --clf_neg_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_val.pt \
    --mia_train_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_train.pt \
    --mia_val_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_val.pt
python experiments/mia/run_nn.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --clf_num_samples 10000 --mia_num_samples 2000 --seed 229 \
    --feature_set x_grad_inf x_grad_2 x_grad_1 theta_grad_inf theta_grad_2 theta_grad_1 layerwise_grad_inf layerwise_grad_2 layerwise_grad_1 \
    --clf_pos_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_train.pt \
    --clf_neg_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_val.pt \
    --mia_train_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_train.pt \
    --mia_val_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_val.pt

####################################################################################################
# Feature Ablation
####################################################################################################
# Gradient only wrt x
python experiments/mia/run_logreg.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --clf_num_samples 10000 --mia_num_samples 2000 --seed 229 \
    --feature_set x_grad_inf x_grad_2 x_grad_1 \
    --clf_pos_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_train.pt \
    --clf_neg_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_val.pt \
    --mia_train_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_train.pt \
    --mia_val_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_val.pt
python experiments/mia/run_nn.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --clf_num_samples 10000 --mia_num_samples 2000 --seed 229 \
    --feature_set x_grad_inf x_grad_2 x_grad_1 \
    --clf_pos_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_train.pt \
    --clf_neg_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_val.pt \
    --mia_train_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_train.pt \
    --mia_val_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_val.pt

# Gradient only wrt theta
python experiments/mia/run_logreg.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --clf_num_samples 10000 --mia_num_samples 2000 --seed 229 \
    --feature_set theta_grad_inf theta_grad_2 theta_grad_1 \
    --clf_pos_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_train.pt \
    --clf_neg_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_val.pt \
    --mia_train_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_train.pt \
    --mia_val_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_val.pt
python experiments/mia/run_nn.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --clf_num_samples 10000 --mia_num_samples 2000 --seed 229 \
    --feature_set theta_grad_inf theta_grad_2 theta_grad_1 \
    --clf_pos_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_train.pt \
    --clf_neg_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_val.pt \
    --mia_train_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_train.pt \
    --mia_val_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_val.pt

# Gradient only wrt layerwise theta
python experiments/mia/run_logreg.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --clf_num_samples 10000 --mia_num_samples 2000 --seed 229 \
    --feature_set layerwise_grad_inf layerwise_grad_2 layerwise_grad_1 \
    --clf_pos_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_train.pt \
    --clf_neg_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_val.pt \
    --mia_train_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_train.pt \
    --mia_val_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_val.pt
python experiments/mia/run_nn.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --clf_num_samples 10000 --mia_num_samples 2000 --seed 229 \
    --feature_set layerwise_grad_inf layerwise_grad_2 layerwise_grad_1 \
    --clf_pos_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_train.pt \
    --clf_neg_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_val.pt \
    --mia_train_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_train.pt \
    --mia_val_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_val.pt

####################################################################################################
# Train Data Ablation
####################################################################################################
# Iterate over clf_num_samples

for NUM_TRAIN in 100 200 300 400 500 1000 1500 2000 2500 5000 7500 10000
do
    python experiments/mia/run_logreg.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --clf_num_samples ${NUM_TRAIN} --mia_num_samples 2000 --seed 229 \
        --feature_set x_grad_inf x_grad_2 x_grad_1 theta_grad_inf theta_grad_2 theta_grad_1 layerwise_grad_inf layerwise_grad_2 layerwise_grad_1 \
        --clf_pos_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_train.pt \
        --clf_neg_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_val.pt \
        --mia_train_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_train.pt \
        --mia_val_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_val.pt
    python experiments/mia/run_nn.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --clf_num_samples ${NUM_TRAIN} --mia_num_samples 2000 --seed 229 \
        --feature_set x_grad_inf x_grad_2 x_grad_1 theta_grad_inf theta_grad_2 theta_grad_1 layerwise_grad_inf layerwise_grad_2 layerwise_grad_1 \
        --clf_pos_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_train.pt \
        --clf_neg_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_val.pt \
        --mia_train_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_train.pt \
        --mia_val_features results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_val.pt
done
