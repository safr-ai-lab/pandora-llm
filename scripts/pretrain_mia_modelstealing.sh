####################################################################################################
# (Supervised) MIAs in the Pretrained Setting (Model Stealing)
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
    --feature_set model_stealing \
    --clf_pos_features results/ModelStealing/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_method=one_sided_projection/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_method=one_sided_projection_train.pt \
    --clf_neg_features results/ModelStealing/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_method=one_sided_projection/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_method=one_sided_projection_val.pt \
    --mia_train_features results/ModelStealing/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_method=one_sided_projection/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_method=one_sided_projection_train.pt \
    --mia_val_features results/ModelStealing/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_method=one_sided_projection/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_method=one_sided_projection_val.pt
python experiments/mia/run_nn.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --clf_num_samples 10000 --mia_num_samples 2000 --seed 229 \
    --feature_set model_stealing \
    --clf_pos_features results/ModelStealing/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_method=one_sided_projection/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_method=one_sided_projection_train.pt \
    --clf_neg_features results/ModelStealing/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_method=one_sided_projection/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=10000_S=150000_seed=229_method=one_sided_projection_val.pt \
    --mia_train_features results/ModelStealing/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_method=one_sided_projection/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_method=one_sided_projection_train.pt \
    --mia_val_features results/ModelStealing/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_method=one_sided_projection/ModelStealing_EleutherAI-pythia-70m-deduped_step98000_N=2000_S=0_seed=229_method=one_sided_projection_val.pt