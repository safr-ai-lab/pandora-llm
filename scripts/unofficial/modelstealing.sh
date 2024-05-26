#!/bin/bash
#SBATCH --job-name modelstealing                    # Job name
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
#SBATCH --array 1-7                                 # Submit a job array of 1-N jobs
#SBATCH --comment "comment"                         # Comment

# Setup conda environment
export HF_DATASETS_CACHE="/n/holylabs/LABS/sneel_lab/Lab/"
export TRANSFORMERS_CACHE="/n/holylabs/LABS/sneel_lab/Lab/"
conda init bash
source activate llm-privacy
module load cuda
conda list

# Run jobs
MODELS=("EleutherAI/pythia-70m-deduped" "EleutherAI/pythia-160m-deduped" "EleutherAI/pythia-410m-deduped" "EleutherAI/pythia-1b-deduped" "EleutherAI/pythia-1.4b-deduped" "EleutherAI/pythia-2.8b-deduped" "EleutherAI/pythia-6.9b-deduped")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID-1]}
MODELPATH=${MODEL//\//-}

if [ "$SLURM_ARRAY_TASK_ID" -le 3 ]; then
  SAMP=20000
else
  SAMP=15000
fi

echo "JOB #${SLURM_ARRAY_TASK_ID}"
echo "MODEL=${MODEL}"
echo "MODELPATH=${MODELPATH}"

python experiments/mia/run_logreg.py --model_name ${MODEL} --model_revision step98000 --clf_num_samples ${SAMP} --mia_num_samples 2000 --seed 229 \
    --feature_set model_stealing \
    --clf_pos_features experiments/prep/Data/carliniGray_train_model=${MODELPATH}_samp=${SAMP}_seed=229_projseed=229_half=False_start=0.pt \
    --clf_neg_features experiments/prep/Data/carliniGray_val_model=${MODELPATH}_samp=${SAMP}_seed=229_projseed=229_half=False_start=0.pt \
    --mia_train_features experiments/prep/Data/carliniGray_train_model=${MODELPATH}_samp=2000_seed=229_projseed=229_half=False_start=150000.pt \
    --mia_val_features experiments/prep/Data/carliniGray_val_model=${MODELPATH}_samp=2000_seed=229_projseed=229_half=False_start=150000.pt
python experiments/mia/run_nn.py --model_name ${MODEL} --model_revision step98000 --clf_num_samples ${SAMP} --mia_num_samples 2000 --seed 229 \
    --feature_set model_stealing \
    --clf_pos_features experiments/prep/Data/carliniGray_train_model=${MODELPATH}_samp=${SAMP}_seed=229_projseed=229_half=False_start=0.pt \
    --clf_neg_features experiments/prep/Data/carliniGray_val_model=${MODELPATH}_samp=${SAMP}_seed=229_projseed=229_half=False_start=0.pt \
    --mia_train_features experiments/prep/Data/carliniGray_train_model=${MODELPATH}_samp=2000_seed=229_projseed=229_half=False_start=150000.pt \
    --mia_val_features experiments/prep/Data/carliniGray_val_model=${MODELPATH}_samp=2000_seed=229_projseed=229_half=False_start=150000.pt