#!/bin/bash
#SBATCH --job-name pretrain100mia                   # Job name
#SBATCH --time 3-00:00                              # Runtime limit in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=20000                                 # Memory pool for all cores in MB (see also --mem-per-cpu)
#SBATCH --output err_out/%x/%x_%A_%a.out            # File to which STDOUT will be written, %x inserts job name, %A inserts job id, %a inserts array id
#SBATCH --error err_out/%x/%x_%A_%a.err             # File to which STDERR will be written, %x inserts job name, %A inserts job id, %a inserts array id
#SBATCH --partition gpu,seas_gpu,gpu_requeue        # Which partitions
#SBATCH --ntasks 1                                  # How many tasks (to run parallely)
#SBATCH --gres gpu:nvidia_a100-sxm4-80gb:1          # What kind / how many GPUs
#SBATCH --requeue                                   # Allow to be requeued
#SBATCH --account sneel_lab                         # Who to charge compute under
#SBATCH --mail-type ALL                             # When to mail
#SBATCH --mail-user jasonwang1@college.harvard.edu  # Who to mail
#SBATCH --array 1-1                                 # Submit a job array of 1-N jobs
#SBATCH --comment "comment"                         # Comment

# Setup conda environment
# export HF_DATASETS_CACHE="/n/holylabs/LABS/sneel_lab/Lab/"
# export TRANSFORMERS_CACHE="/n/holylabs/LABS/sneel_lab/Lab/"
conda init bash
source activate llm-privacy
module load cuda
conda list

# Run jobs
MODELS=("EleutherAI/pythia-1.4b-deduped")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID-1]}
MODELPATH=${MODEL//\//-}

echo "JOB #${SLURM_ARRAY_TASK_ID}"
echo "MODEL=${MODEL}"
echo "MODELPATH=${MODELPATH}"

python experiments/mia/run_gradnorm.py --model_name ${MODEL} --model_revision step98000 --num_samples 15000 --start_index 0 --pack --seed 229 --max_length 100
python experiments/mia/run_logreg.py --model_name ${MODEL} --model_revision step98000 --clf_num_samples 15000 --mia_num_samples 2000 --seed 229 --tag "100" \
    --feature_set x_grad_inf theta_grad_inf layerwise_grad_inf x_grad_2 theta_grad_2 layerwise_grad_2 x_grad_1 theta_grad_1 layerwise_grad_1 \
    --clf_pos_features results/GradNorm/GradNorm_${MODELPATH}_step98000_N=15000_S=0_seed=229_trunc=100_train.pt \
    --clf_neg_features results/GradNorm/GradNorm_${MODELPATH}_step98000_N=15000_S=0_seed=229_trunc=100_val.pt \
    --mia_train_features results/GradNorm/GradNorm_${MODELPATH}_step98000_N=15000_S=0_seed=229_trunc=100_train.pt \
    --mia_val_features results/GradNorm/GradNorm_${MODELPATH}_step98000_N=15000_S=0_seed=229_trunc=100_val.pt
python experiments/mia/run_nn.py --model_name ${MODEL} --model_revision step98000 --clf_num_samples 15000 --mia_num_samples 2000 --seed 229 --tag "100" \
    --feature_set x_grad_inf theta_grad_inf layerwise_grad_inf x_grad_2 theta_grad_2 layerwise_grad_2 x_grad_1 theta_grad_1 layerwise_grad_1 \
    --clf_pos_features results/GradNorm/GradNorm_${MODELPATH}_step98000_N=15000_S=0_seed=229_trunc=100_train.pt \
    --clf_neg_features results/GradNorm/GradNorm_${MODELPATH}_step98000_N=15000_S=0_seed=229_trunc=100_val.pt \
    --mia_train_features results/GradNorm/GradNorm_${MODELPATH}_step98000_N=15000_S=0_seed=229_trunc=100_train.pt \
    --mia_val_features results/GradNorm/GradNorm_${MODELPATH}_step98000_N=15000_S=0_seed=229_trunc=100_val.pt