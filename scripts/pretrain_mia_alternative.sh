####################################################################################################
# (Supervised) MIAs in the Pretrained Setting (Alternative Features)
####################################################################################################
for MODEL_SIZE in 70m
do 
    ####################################################################################################
    # Higher Resolution Grad Norms
    ####################################################################################################
    # For training supervised classifier
    python experiments/mia/run_gradnorm.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --num_samples 10000 --start_index 150000 --pack --seed 229
    # For evaluating supervised classifier
    python experiments/mia/run_gradnorm.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --num_samples 2000 --start_index 0 --pack --seed 229
    python experiments/mia/run_logreg.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --clf_num_samples 10000 --mia_num_samples 2000 --seed 229 \
        --feature_set highres_grad \
        --clf_pos_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229_train.pt \
        --clf_neg_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229_val.pt \
        --mia_train_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229_train.pt \
        --mia_val_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229_val.pt
    python experiments/mia/run_nn.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --clf_num_samples 10000 --mia_num_samples 2000 --seed 229 \
        --feature_set highres_grad \
        --clf_pos_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229_train.pt \
        --clf_neg_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229_val.pt \
        --mia_train_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229_train.pt \
        --mia_val_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229_val.pt
    ####################################################################################################
    # JL Transformed Grads
    ####################################################################################################
    # For training supervised classifier
    python experiments/mia/run_jl.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --compute_balanced_jl --proj_type normal --num_samples 10000 --start_index 150000 --pack --seed 229
    # For evaluating supervised classifier
    python experiments/mia/run_jl.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --compute_balanced_jl --proj_type normal --num_samples 2000 --start_index 0 --pack --seed 229
    python experiments/mia/run_logreg.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --clf_num_samples 10000 --mia_num_samples 2000 --seed 229 \
        --feature_set balanced_jl \
        --clf_pos_features results/JL/JL_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229/JL_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229_train.pt \
        --clf_neg_features results/JL/JL_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229/JL_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229_val.pt \
        --mia_train_features results/JL/JL_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229/JL_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229_train.pt \
        --mia_val_features results/JL/JL_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229/JL_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229_val.pt
    python experiments/mia/run_nn.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --clf_num_samples 10000 --mia_num_samples 2000 --seed 229 \
        --feature_set balanced_jl \
        --clf_pos_features results/JL/JL_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229/JL_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229_train.pt \
        --clf_neg_features results/JL/JL_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229/JL_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229_val.pt \
        --mia_train_features results/JL/JL_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229/JL_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229_train.pt \
        --mia_val_features results/JL/JL_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229/JL_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229_val.pt
done