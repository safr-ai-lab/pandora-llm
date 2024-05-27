#!/bin/bash
#SBATCH --job-name grad70m                          # Job name
#SBATCH --time 1-00:00                              # Runtime limit in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=20000                                 # Memory pool for all cores in MB (see also --mem-per-cpu)
#SBATCH --output results/err_out/%x/%x_%A_%a.out    # File to which STDOUT will be written, %x inserts job name, %A inserts job id, %a inserts array id
#SBATCH --error results/err_out/%x/%x_%A_%a.err     # File to which STDERR will be written, %x inserts job name, %A inserts job id, %a inserts array id
#SBATCH --partition gpu,seas_gpu,gpu_requeue        # Which partitions
#SBATCH --ntasks 1                                  # How many tasks (to run parallely)
#SBATCH --gres gpu:nvidia_a100-sxm4-80gb:1          # What kind / how many GPUs
#SBATCH --requeue                                   # Allow to be requeued
#SBATCH --account sneel_lab                         # Who to charge compute under
#SBATCH --mail-type ALL                             # When to mail
#SBATCH --mail-user jasonwang1@college.harvard.edu  # Who to mail
#SBATCH --array 1-1                                 # Submit a job array of 1-N jobs
#SBATCH --comment "comment"                         # Comment

conda init bash
source activate llm-privacy
module load cuda
conda list

python experiments/mia/run_gradnorm.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 10000 --start_index 150000 --pack --seed 229
python experiments/mia/run_gradnorm.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 2000 --start_index 0 --pack --seed 229